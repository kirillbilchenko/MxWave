"""Freeze a CT-only native draft-length screen with the existing restoration controller."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest
from .prepare_pretrained_drafter import replace_one

REVISION = "ct-draft-length-v1"
DEPTHS = {"s7a": 7, "s3": 3, "s5": 5, "s0": 0, "s7b": 7}


def configure(command: list[str], depth: int) -> list[str]:
    """Change only native speculative configuration; zero removes the drafter entirely."""
    if depth not in (0, 3, 5, 7) or command.count("--speculative-config") != 1:
        raise ValueError("Expected the frozen DFlash command and an allowed depth")
    result = command.copy()
    at = result.index("--speculative-config")
    config = json.loads(result[at + 1])
    if config["method"] != "dflash" or config["model"] != "/draft":
        raise ValueError("Unexpected drafter configuration")
    if not depth:
        del result[at:at + 2]
    else:
        config["num_speculative_tokens"] = depth
        # This is a C1 screen. Preserve the higher-concurrency policy wherever possible.
        config["num_speculative_tokens_per_batch_size"] = [
            [lo, hi, min(n, depth)]
            for lo, hi, n in config["num_speculative_tokens_per_batch_size"]
        ]
        result[at + 1] = json.dumps(config)
    return result


def prepare(template: Path, output: Path) -> dict[str, Any]:
    """Prepare CPU-checkable files without changing production or starting a model."""
    manifest = json.loads((template / "source-manifest.json").read_bytes())
    if any(digest(template / name) != sha for name, sha in manifest.items()):
        raise ValueError("Frozen serving template changed")
    plan = json.loads((template / "plan.json").read_bytes())
    if plan["revision"] != "drafter-pretrained-modal-v1" or plan["sealed_test_opened"]:
        raise ValueError("Expected the reviewed pretrained development screen")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest:
        shutil.copyfile(template / name, output / name)
    plan.update(
        revision=REVISION, arms=list(DEPTHS), depths=DEPTHS,
        modes={arm: "ct_mxfp8" for arm in DEPTHS},
        variants={arm: "modal_fp8" for arm in DEPTHS},
        purpose="CT + Modal FP8 native 7/3/5/off/7 draft-length screen, no production arm.",
        parent_plan_sha256=digest(template / "plan.json"),
        trials=6, requests=48, private_port=8042, outage_seconds=4500,
        client_deadline_seconds=900, max_model_len=8192,
        speed_control_relative_tolerance=0.02, sealed_test_opened=False,
        notes=[
            "Same CT MXFP4 experts, MXFP8 KDA, exact vocabulary/scales and target policies.",
            "Same Modal FP8 checkpoint at depths3/5/7; depth0 unloads speculation entirely.",
            "Same eight development prompts, six shuffled seed42 cycles, 256 output tokens.",
            "Fixed C1, temperature0, prefix cache off, 4096 prefill budget,16 slots,6GiB KV/rank.",
            "First/final seven-draft loads bracket load drift; retain all failures and outputs.",
            "Output differences are recorded; no assumption of target batch invariance.",
            "Actual native counter depths and startup graph/config evidence are required.",
            "Existing independent restoration watchdog; no training, sealed test or promotion.",
        ],
    )
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in ("prepare.py", "control.py"):
        path = output / name
        path.write_text(path.read_text().replace("8040", "8042").replace("29713", "29715")
                        .replace("mxwave-drafter-pretrained-modal-v1", "mxwave-" + REVISION))
    source = Path(__file__).read_text()
    # Export only our independent argv transformation into the detached preparation script.
    function = source[source.index("def configure("):source.index("\n\ndef prepare(")]
    replace_one(output / "prepare.py", "\ndef main():", "\n" + function + "\n\ndef main():")
    replace_one(output / "prepare.py", '            command = entry["command"]',
                '            command = configure(entry["command"], plan["depths"][arm])\n'
                '            entry["command"] = command\n'
                '            env["MXWAVE_DRAFT_DEPTH"] = str(plan["depths"][arm])')
    replace_one(output / "prepare.py",
                '    assert len({json.dumps(result["launches"][a][0]["command"]) '
                'for a in plan["arms"]}) == 1',
                '    assert result["launches"]["s7a"][0]["command"] == '
                'result["launches"]["s7b"][0]["command"]\n'
                '    normalized = set()\n'
                '    for arm in plan["arms"]:\n'
                '        cmd = result["launches"][arm][0]["command"].copy()\n'
                '        if "--speculative-config" in cmd:\n'
                '            at = cmd.index("--speculative-config"); del cmd[at:at + 2]\n'
                '        normalized.add(json.dumps(cmd))\n'
                '    assert len(normalized) == 1, "Target serving settings differ"')
    replace_one(output / "trial_boot.py",
                '        assert args.max_num_seqs == 16 and args.speculative_config["method"] '
                '== "dflash"',
                '        assert args.max_num_seqs == 16\n'
                '        depth = int(os.environ["MXWAVE_DRAFT_DEPTH"])\n'
                '        if depth:\n'
                '            assert args.speculative_config["method"] == "dflash"\n'
                '            assert args.speculative_config["num_speculative_tokens"] == depth\n'
                '        else:\n'
                '            assert args.speculative_config is None')
    replace_one(output / "pretrained_startup.py", '    expected=plan["drafter_files"][variant]',
                '    depth=int(os.environ["MXWAVE_DRAFT_DEPTH"])\n'
                '    if depth == 0:\n'
                '        assert worker.vllm_config.speculative_config is None\n'
                '        assert worker.model_runner.speculator is None\n'
                '        result={"status":"installed","variant":"none","depth":0,\n'
                '                "rank":get_tensor_model_parallel_rank(),"drafter_loaded":False}\n'
                '        (Path("/traces")/f"pretrained-startup-r{result[\'rank\']}.json").write_text(\n'
                '            json.dumps(result,indent=2)+"\\n")\n'
                '        return result\n'
                '    assert worker.vllm_config.speculative_config.num_speculative_tokens == depth\n'
                '    assert worker.model_runner.speculator.num_speculative_steps == depth\n'
                '    expected=plan["drafter_files"][variant]')
    replace_one(output / "pretrained_startup.py", '"variant":variant,"rank":rank,',
                '"variant":variant,"rank":rank,"depth":depth,"drafter_loaded":True,')
    worker = output / "trial_worker.py"
    text = worker.read_text()
    start = text.index("        assert speculator is not None")
    end = text.index('        mode = os.environ["MXWAVE_TRIAL_ARM"]', start)
    block = text[start:end]
    text = text[:start] + (
        '        depth = int(os.environ["MXWAVE_DRAFT_DEPTH"])\n'
        '        if not depth:\n'
        '            assert speculator is None and r.vllm_config.speculative_config is None\n'
        '            draft = None\n'
        '            same_head = same_embedding = None\n'
        '        else:\n' + ''.join('    ' + line + '\n' for line in block.splitlines())
    ) + text[end:]
    for old, new in (
        ('str(draft.model._fused_kv_weight.dtype)',
         'str(draft.model._fused_kv_weight.dtype) if draft is not None else None'),
        ('draft.model._fused_kv_weight.numel()',
         'draft.model._fused_kv_weight.numel() if draft is not None else 0'),
        ('type(draft.model.quant_config).__name__',
         'type(draft.model.quant_config).__name__ if draft is not None else None'),
    ):
        if text.count(old) != 1:
            raise ValueError("Unexpected target snapshot template")
        text = text.replace(old, new)
    worker.write_text(text)
    measurement = Path(__file__).with_name("measure_ct_draft_lengths.py")
    shutil.copyfile(measurement, output / "measure_features.py")
    (output / "measure_features.py").write_text(
        (output / "measure_features.py").read_text().replace("from .mtp_metrics", "from mtp_metrics"))
    shutil.copyfile(Path(__file__).with_name("mtp_metrics.py"), output / "mtp_metrics.py")
    # The launcher updates this status after finalization; it cannot be an immutable payload.
    replace_one(output / "finalize.py", '"queue.log", "finalize-queue.log"):',
                '"queue.log", "finalize-queue.log", "run-prepared-status.json"):')
    for path in output.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    frozen = {p.name: digest(p) for p in sorted(output.iterdir())
              if p.is_file() and p.name != "source-manifest.json"}
    (output / "source-manifest.json").write_text(json.dumps(frozen, indent=2) + "\n")
    return {"status": "frozen_not_launched", "plan_sha256": digest(output / "plan.json"),
            "arms": DEPTHS, "requests_per_arm": 48, "sources": len(frozen)}


def main() -> None:
    """Prepare the bounded experiment; never start or stop serving here."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template, args.output), indent=2))


if __name__ == "__main__":
    main()
