"""Freeze a bounded native top-k ordering correction and full-model repeat check."""

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
from .review_drafter_sparse_prefill import event_fields


def prepare(template: Path, evidence: Path, review: Path, output: Path) -> dict[str, Any]:
    """Require verified order-only evidence and preserve target, drafter and repeat gates."""
    checked = json.loads(review.read_bytes())
    if (checked["status"] != "reviewed" or not checked["production_restored"]
            or checked["plan_sha256"] != digest(evidence / "plan.json")
            or len(checked["pairs"]) != 12):
        raise ValueError("Require independently reviewed sparse-prefill evidence on both ranks")
    for pair in checked["pairs"]:
        if not pair["all_explicit_indexer_inputs_exact"]:
            raise ValueError("Indexer input drift must be separated before changing order")
        for chunk in pair["chunks"]:
            pools = chunk["pool_selection"]
            if (not chunk["consumed_score_inputs_exact"] or not chunk["consumed_scores"]["exact"]
                    or chunk["consumed_scores"]["nonfinite_pairs"]
                    or pools["different_multiset_rows"] or not pools["different_order_rows"]):
                raise ValueError("Require finite equal scores and order-only selected-index drift")
            for inventory in pools["inventories"]:
                if any(inventory[k] for k in ("non_sentinel_negative", "causal_or_extent_violations",
                                              "duplicate_entries")):
                    raise ValueError("Invalid or duplicate native selections require a separate fix")
    manifest = json.loads((template / "source-manifest.json").read_bytes())
    if not all(digest(template / name) == sha for name, sha in manifest.items()):
        raise ValueError("Frozen long-repeat sources changed")
    plan = json.loads((template / "plan.json").read_bytes())
    if plan["revision"] != "drafter-long-repeat-v1":
        raise ValueError("Require the uninstrumented long-repeat control template")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest:
        shutil.copyfile(template / name, output / name)
    for name in ("canonical_sparse_prefill.py", "check_canonical_sparse_prefill.py"):
        text = Path(__file__).with_name(name).read_text()
        (output / name).write_text(text.replace("from .canonical_sparse_prefill", "from canonical_sparse_prefill"))
    shutil.copyfile(review, output / "parent-sparse-review.json")
    case = plan["calibration_cases"][0]
    for rank in (0, 1):
        path = evidence / f"ct-r{rank}" / f"context-{case['id']}-capture1-0000-r{rank}.safetensors"
        meta = json.loads(path.with_suffix(".json").read_bytes())
        if digest(path) != meta["payload_sha256"]:
            raise ValueError("Retained component operands changed")
        values = arrays(path)
        run = {"trace": meta["sparse_prefill"], "values": values}
        inputs = event_fields(run, "sparse_prefill.topk#0/input")
        tensors = {"scores": inputs["/0"], "starts": inputs["/1"], "ends": inputs["/2"],
                   "native_pools": event_fields(run, "sparse_prefill.topk#0/output")[""],
                   "native_tokens": event_fields(run, "language_model.model.layers.3.self_attn.indexer.indexer_op#0/output")[""],
                   "seq_lens": (values["target_positions"] + 1).astype(np.int32)}
        save_file({k: torch.from_numpy(np.array(v, copy=True)) for k, v in tensors.items()},
                  str(output / f"sparse-fixture-r{rank}.safetensors"))
    plan.update(
        revision="drafter-canonical-prefill-v1", private_port=8033,
        purpose="Qualify canonical selected pool order, then repeat the full long-prompt target.",
        parent_sparse_plan_sha256=digest(evidence / "plan.json"), parent_review_sha256=digest(review),
        client_deadline_seconds=1200, outage_seconds=3300,
        source_sha256=digest(Path(__file__)),
        notes=[
            "Native selection and all weights remain fixed; sort only selected int32 pool indices.",
            "Scope: native prefill k512, at most4096 rows. Decode operator unchanged.",
            "GPU startup gates both ranks: real retained scores, mixed padding, token-set and causal checks.",
            "Five component row sizes8/16/64/128/3973;32 native/canonical and32 graph repeats each.",
            "Six interleaved trials at16/128/3973 rows, clock/thermal records, unchanged2% repeat gate.",
            "Then three warmups and native-before/four feature captures/native-after,128 output tokens.",
            "No per-operation observation in the full-model check; all decision logits retained.",
            "No training, sealed-test access, promotion or end-to-end speed claim; restore on every exit.",
        ],
    )
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in ("prepare.py", "control.py", "measure_features.py"):
        path = output / name
        path.write_text(path.read_text().replace("8030", "8033").replace("29703", "29706")
                        .replace("mxwave-drafter-long-repeat-v1", "mxwave-drafter-canonical-prefill-v1"))
    path = output / "trial_plugin.py"
    text = path.read_text()
    anchor = "        import drafter_head_probe\n        drafter_head_probe.arm_capture(worker)"
    if text.count(anchor) != 1:
        raise ValueError("Unexpected pre-warmup load hook")
    text = text.replace(anchor, '''        import torch.distributed
        import canonical_sparse_prefill
        import check_canonical_sparse_prefill
        rank = torch.distributed.get_rank()
        worker.mxwave_sparse_component = check_canonical_sparse_prefill.qualify(
            pathlib.Path(f"/trial/sparse-fixture-r{rank}.safetensors"),
            pathlib.Path(f"/traces/sparse-component-r{rank}.json"))
        worker.mxwave_sparse_policy = canonical_sparse_prefill.install()
''' + anchor)
    anchor = '        report["trial"] = worker.mxwave_trial_snapshot()'
    if text.count(anchor) != 1:
        raise ValueError("Unexpected startup report hook")
    text = text.replace(anchor, anchor + '\n        report["canonical_sparse_prefill"] = worker.mxwave_sparse_policy\n'
                        '        report["canonical_sparse_component"] = worker.mxwave_sparse_component')
    path.write_text(text)
    path = output / "trial_boot.py"
    text = path.read_text()
    anchor = "        import verify_capture_tokenizer\n"
    if text.count(anchor) != 1:
        raise ValueError("Unexpected preflight tokenizer check")
    text = text.replace(anchor, "        import canonical_sparse_prefill\n"
                        "        import check_canonical_sparse_prefill\n"
                        "        assert callable(canonical_sparse_prefill.install)\n"
                        "        assert callable(check_canonical_sparse_prefill.qualify)\n" + anchor)
    path.write_text(text)
    for path in output.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    frozen = {p.name: digest(p) for p in sorted(output.iterdir())
              if p.is_file() and p.name != "source-manifest.json"}
    (output / "source-manifest.json").write_text(json.dumps(frozen, indent=2) + "\n")
    return {"plan_sha256": digest(output / "plan.json"), "frozen_files": len(frozen),
            "private_port": 8033, "requests": 6, "training_eligible": False}


def main() -> None:
    """Prepare the reviewable experiment without changing any service."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "evidence", "review", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template, args.evidence, args.review, args.output), indent=2))


if __name__ == "__main__":
    main()
