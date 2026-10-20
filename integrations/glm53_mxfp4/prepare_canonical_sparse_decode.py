"""Freeze a decode-order correction only after the exact native operation is implicated."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

from .audit_drafter_feature_export_numpy import arrays
from .drafter_fp8_roundtrip import digest


def prepare(template: Path, evidence: Path, review: Path, output: Path) -> dict[str, Any]:
    """Require equal consumed scores and selected membership across all independently reviewed pairs."""
    checked = json.loads(review.read_bytes())
    if (checked["status"] != "reviewed" or not checked["production_restored"]
            or checked["plan_sha256"] != digest(evidence / "plan.json") or len(checked["pairs"]) != 12):
        raise ValueError("Require completed independent first-decode review on both ranks")
    for pair in checked["pairs"]:
        first, topk = pair["first_different_result"], pair["decode_topk"]
        if (not pair["complete_input_ids_and_positions_exact"] or first is None
                or first["name"] != "layer3.native_decode_topk#0/output"
                or not first["matching_consumed_inputs"] or not topk["consumed_scores"]["exact"]
                or topk["consumed_scores"]["nonfinite_pairs"]
                or topk["parameters"]["next_n"] != 1 or topk["parameters"]["rows"] != 8):
            raise ValueError("Do not correct ordering before isolating the native decode selection")
        for name in ("pool_selection", "expanded_token_selection"):
            selected = topk[name]
            if selected["different_multiset_rows"] or selected["different_order_rows"] != 8:
                raise ValueError("Require order-only drift on all eight rows")
            for inventory in selected["inventories"]:
                if any(inventory[k] for k in ("non_sentinel_negative", "causal_or_extent_violations", "duplicate_entries")):
                    raise ValueError("Invalid or duplicate native selection requires a separate correction")
    manifest = json.loads((template / "source-manifest.json").read_bytes())
    if not all(digest(template / name) == sha for name, sha in manifest.items()):
        raise ValueError("Frozen decode trace sources changed")
    plan = json.loads((template / "plan.json").read_bytes())
    if plan["revision"] != "drafter-long-decode-v1":
        raise ValueError("Use the exact independently reviewed decode trace")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest:
        shutil.copyfile(template / name, output / name)
    for name in ("canonical_sparse_decode.py", "check_canonical_sparse_decode.py", "check_canonical_sparse_prefill.py"):
        (output / name).write_text(Path(__file__).with_name(name).read_text().replace("from .", "from "))
    shutil.copyfile(review, output / "parent-decode-review.json")
    case = plan["calibration_cases"][0]
    for rank in (0, 1):
        path = evidence / f"ct-r{rank}" / f"context-{case['id']}-capture1-0001-r{rank}.safetensors"
        meta = json.loads(path.with_suffix(".json").read_bytes())
        if digest(path) != meta["payload_sha256"]:
            raise ValueError("Native component operands changed")
        values = arrays(path)
        events = meta["decode_components"]["events"]

        def fields(name: str, event_list: list[dict[str, Any]] = events,
                   data: dict[str, np.ndarray] = values) -> dict[str, np.ndarray]:
            event = next(e for e in event_list if e["name"] == name)
            return {f["path"]: data[f["payload_key"]] for f in event["tensors"]}

        inputs = fields("layer3.native_decode_topk#0/input")
        indexer = next(e["name"] for e in events if e["name"].endswith(".indexer.indexer_op#0/output"))
        tensors = {"scores": inputs["/scores"], "lengths": inputs["/seq_lens"],
                   "native_pools": fields("layer3.native_decode_topk#0/output")[""],
                   "native_tokens": fields(indexer)[""],
                   "seq_lens": (values["target_positions"] + 1).astype(np.int32)}
        save_file({k: torch.from_numpy(np.array(v, copy=True)) for k, v in tensors.items()},
                  str(output / f"decode-fixture-r{rank}.safetensors"))
    plan.update(
        revision="drafter-canonical-decode-v1", private_port=8035,
        purpose="Qualify canonical decode selection then repeat the native full model with both ordering fixes.",
        parent_decode_plan_sha256=digest(evidence / "plan.json"), parent_decode_review_sha256=digest(review),
        source_sha256=digest(Path(__file__)),
        notes=[
            "Keep the verified canonical prefill policy; add selected-index sorting only for rows8/next_n1/k512 decode.",
            "Other decode shapes retain native behavior; no C8/C16 or long-context qualification.",
            "Before graph capture, replay real scores and mixed padding on both GPUs.",
            "Poison unused columns with large values andNaN; selected membership must remain exact.",
            "Six interleaved CUDA-graph timing trials per fixture with clocks/thermals and unchanged2% repeat control.",
            "Remove all per-operation graph observers for the full-model repeat; retain passive feature/logit capture.",
            "Same three warmups, six long requests,128 output tokens, original drafter and seven drafts.",
            "Restore production on every exit; no training, promotion, sealed-test access or serving-speed claim.",
        ],
    )
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in ("prepare.py", "control.py", "measure_features.py"):
        path = output / name
        path.write_text(path.read_text().replace("8034", "8035").replace("29707", "29708")
                        .replace("mxwave-drafter-long-decode-v1", "mxwave-drafter-canonical-decode-v1"))
    path = output / "trial_plugin.py";text = path.read_text()
    anchor = "        import drafter_long_decode_probe\n        drafter_long_decode_probe.arm_capture(worker)"
    if text.count(anchor) != 1:
        raise ValueError("Unexpected graph observation arm hook")
    text = text.replace(anchor, '''        import canonical_sparse_decode
        import check_canonical_sparse_decode
        worker.mxwave_decode_component = check_canonical_sparse_decode.qualify(
            pathlib.Path(f"/trial/decode-fixture-r{rank}.safetensors"),
            pathlib.Path(f"/traces/decode-component-r{rank}.json"))
        worker.mxwave_decode_policy = canonical_sparse_decode.install()''')
    anchor = '        import drafter_long_decode_probe\n        report["context_probe"] = drafter_long_decode_probe.install(worker)'
    if text.count(anchor) != 1:
        raise ValueError("Unexpected graph payload-retention hook")
    text = text.replace(anchor, '        report["context_probe"] = drafter_context_probe.install(worker)\n'
                        '        report["canonical_sparse_decode"] = worker.mxwave_decode_policy\n'
                        '        report["canonical_decode_component"] = worker.mxwave_decode_component')
    path.write_text(text)
    path = output / "trial_boot.py";text = path.read_text()
    text = text.replace("        import drafter_long_decode_probe\n        assert callable(drafter_long_decode_probe.arm_capture)\n",
                        "        import canonical_sparse_decode\n        import check_canonical_sparse_decode\n"
                        "        assert callable(canonical_sparse_decode.install)\n        assert callable(check_canonical_sparse_decode.qualify)\n")
    path.write_text(text)
    for path in output.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    frozen = {p.name: digest(p) for p in sorted(output.iterdir())
              if p.is_file() and p.name != "source-manifest.json"}
    (output / "source-manifest.json").write_text(json.dumps(frozen, indent=2) + "\n")
    return {"plan_sha256": digest(output / "plan.json"), "frozen_files": len(frozen),
            "private_port": 8035, "requests": 6, "training_eligible": False}


def main() -> None:
    """Prepare the bounded correction after the independent archived decode trace is reviewed."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "evidence", "review", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template, args.evidence, args.review, args.output), indent=2))


if __name__ == "__main__":
    main()
