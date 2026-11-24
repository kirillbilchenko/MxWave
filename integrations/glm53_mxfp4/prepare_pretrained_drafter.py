"""Prepare a bounded pretrained-drafter screen after the active capture restores serving."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest
from .prepare_context_native import _MEASURE

REVISION = "drafter-pretrained-modal-v1"

_STARTUP = '''"""Record actual native configuration and verify the replicated context weight."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path

def install(worker):
    import torch
    from safetensors import safe_open
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import get_eagle3_aux_layers_from_config
    plan=json.loads(Path("/trial/plan.json").read_bytes())
    variant=os.environ["MXWAVE_DRAFTER_VARIANT"]
    expected=plan["drafter_files"][variant]
    for name,sha in expected.items():
        with (Path("/draft")/name).open("rb") as stream:
            assert hashlib.file_digest(stream,"sha256").hexdigest()==sha,(variant,name)
    model=worker.model_runner.speculator.model
    config=model.config
    expected_config=json.loads(Path("/draft/config.json").read_bytes())
    assert type(model).__name__=="DFlash2Qwen3ForCausalLM"
    assert len(model.model.layers)==expected_config["num_hidden_layers"]
    layers=get_eagle3_aux_layers_from_config(worker.vllm_config.speculative_config)
    assert list(layers)==[i+1 for i in expected_config["dflash_config"]["target_layer_ids"]]
    assert model.model.fc.weight.shape==(4096,4096*len(layers))
    with safe_open("/draft/model.safetensors",framework="pt",device="cpu") as weights:
        fc=weights.get_tensor("fc.weight")
        assert torch.equal(fc,model.model.fc.weight.detach().cpu())
    if variant=="modal":
        assert model.model.quant_config is None
        assert all(layer.self_attn.qkv_proj.weight.dtype==torch.bfloat16
                   for layer in model.model.layers)
        assert bool(config.is_causal)
    if variant=="modal_fp8":
        assert type(model.model.quant_config).__name__=="Fp8Config"
        assert bool(config.is_causal)
    calls={}
    manager=worker.model_runner.speculator.query_cudagraph_manager
    original=manager.run_fullgraph
    rank=get_tensor_model_parallel_rank()
    def replay(descriptor):
        result=original(descriptor)
        key=(descriptor.num_reqs,descriptor.num_tokens)
        calls[key]=calls.get(key,0)+1
        if calls[key]==1 or calls[key]%64==0:
            print(json.dumps({"native_draft_replay":list(key),"count":calls[key],"rank":rank}),flush=True)
        return result
    manager.run_fullgraph=replay
    result={"status":"installed","variant":variant,"rank":rank,
            "checkpoint_files_verified":True,"fc_native_bitwise_equal":True,
            "draft_layers":len(model.model.layers),"aux_layers":list(layers),
            "context_width":model.model.fc.weight.shape[1],"causal":bool(config.is_causal),
            "sliding_window":config.sliding_window,
            "quant_config":type(model.model.quant_config).__name__,
            "observation":"Startup checks and graph counts only; no feature capture."}
    (Path("/traces")/f"pretrained-startup-r{rank}.json").write_text(json.dumps(result,indent=2)+"\\n")
    return result
'''

_QUEUE = '''"""Wait for the existing capture and restoration; never interrupt either."""
from __future__ import annotations
import hashlib,json,os,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).parent
PARENT=ROOT.parent/"drafter-full-capture-v1"

def status(phase,**extra):
    result={"phase":phase,"pid":os.getpid(),"updated_epoch":time.time(),**extra}
    path=ROOT/"queue.json";temp=path.with_suffix(".incomplete")
    temp.write_text(json.dumps(result,indent=2)+"\\n");temp.replace(path)

def main():
    plan=json.loads((ROOT/"plan.json").read_bytes())
    sources=json.loads((ROOT/"source-manifest.json").read_bytes())
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==sha for n,sha in sources.items())
    started=time.time()
    while True:
        parent=json.loads((PARENT/"controller.json").read_bytes())
        assert hashlib.sha256((PARENT/"plan.json").read_bytes()).hexdigest()==plan["wait_capture_sha256"]
        assert parent["phase"] not in ("failed_restored","preparation_failed"), "Capture failed; do not auto-start another outage"
        assert time.time()-started<14*3600, "Capture wait expired; no serving changes"
        post=PARENT/"postprocess-complete.json"
        runner=PARENT/"run-prepared-status.json"
        if parent["phase"]=="complete_restored" and not parent["service_paused"] and post.exists() and runner.exists():
            done=json.loads(runner.read_bytes())
            assert done["step"]=="finalize" and done["returncode"]==0
            assert json.loads(post.read_bytes())["status"]=="verified"
            assert parent["restoration"]["health_200"] and parent["restoration"]["originals_running"]
            break
        status("waiting_for_verified_capture_and_restoration",capture_phase=parent["phase"])
        time.sleep(30)
    for step in ("prepare","run-prepared"):
        status(step)
        with (ROOT/(step+"-queue-launch.log")).open("xb") as log:
            result=subprocess.run(["python3","-S","-B","-u",str(ROOT/(step+".py"))],
                                  stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,(step,result.returncode)
    controller=json.loads((ROOT/"controller.json").read_bytes())
    assert controller["phase"]=="complete_restored" and not controller["service_paused"]
    done=json.loads((ROOT/"run-prepared-status.json").read_bytes())
    assert done["step"]=="finalize" and done["returncode"]==0 and done["control_returncode"]==0
    status("complete_retained_requires_review",training_started=False)

if __name__=="__main__":
    try:main()
    except BaseException as error:
        status("failed",error=repr(error));raise
'''


def replace_one(path: Path, old: str, new: str) -> None:
    """Apply an exact, single-match revision to a frozen template."""
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f"Template anchor changed: {path.name}: {old!r}")
    path.write_text(text.replace(old, new))


def prepare(template: Path, audit: Path, capture: Path, output: Path) -> dict[str, Any]:
    """Freeze a Modal/original comparison without loading a GPU or changing serving."""
    manifest = json.loads((template / "source-manifest.json").read_bytes())
    if not all(digest(template / name) == sha for name, sha in manifest.items()):
        raise ValueError("Qualified template changed")
    download = json.loads((audit / "modal-download.json").read_bytes())
    checkpoint = audit / "modal-checkpoint"
    fp8_checkpoint = audit / "modal-fp8-checkpoint"
    fp8_review = json.loads((fp8_checkpoint / "conversion-review.json").read_bytes())
    if (fp8_review["source_sha256"] != download["sha256"]
            or fp8_review["exported_sha256"] != digest(fp8_checkpoint / "model.safetensors")
            or fp8_review["quantized_tensors"] != 42
            or fp8_review["training_performed"] or fp8_review["target_weights_changed"]):
        raise ValueError("Unverified FP8 drafter conversion")
    if (download["status"] != "verified"
            or download["sha256"] != digest(checkpoint / "model.safetensors")):
        raise ValueError("Pretrained checkpoint differs")
    installed = [json.loads((audit / f"installed-r{rank}.json").read_bytes()) for rank in (0, 1)]
    if installed[0] != installed[1] or installed[0]["context_input_width"] != 24576:
        raise ValueError("Installed configuration helpers differ")
    native = audit.parent / "drafter-preflight-v1/original"
    original_config = json.loads((native / "config.json").read_bytes())
    if original_config["dflash_config"]["target_layer_ids"] != [5, 14, 24, 33, 42]:
        raise ValueError("Current drafter changed")
    corpus = json.loads((capture / "plan.json").read_bytes())
    validation = [row for row in corpus["calibration_cases"] if row["split"] == "validation"]
    cases = [row for domain, count in (("code", 4), ("chat", 2), ("explanation", 2))
             for row in [r for r in validation if r["domain"] == domain][:count]]
    if len(cases) != 8 or len({row["id"] for row in cases}) != 8:
        raise ValueError("Missing development domains")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest:
        shutil.copyfile(template / name, output / name)
    plan = json.loads((output / "plan.json").read_bytes())
    plan.update(
        revision=REVISION, arms=["a1", "b1", "c1", "a2"],
        modes={arm: "ct_mxfp8" for arm in ("a1", "b1", "c1", "a2")},
        variants={"a1": "original", "b1": "modal", "c1": "modal_fp8", "a2": "original"},
        purpose="Pretrained DFlash2 development screen before paying adaptation cost.",
        private_port=8040, prefill_budget=4096, kv_bytes_per_rank=6 * 1024**3,
        calibration_cases=cases, control_cases=[], requests=48, output_tokens=256,
        warmup_case=next(row for row in corpus["calibration_cases"] if row["split"] == "train"),
        outage_seconds=3600, client_deadline_seconds=900,
        wait_capture_sha256=digest(capture / "plan.json"), sealed_test_opened=False,
        drafter_files={
            "original": {name: digest(native / name)
                         for name in ("model.safetensors", "config.json")},
            "modal": {name: digest(checkpoint / name)
                      for name in ("model.safetensors", "config.json")},
            "modal_fp8": {name: digest(fp8_checkpoint / name)
                          for name in ("model.safetensors", "config.json")},
        },
        pretrained_revision=download["revision"],
        notes=[
            "Start only after the current full capture verifies and original production restores.",
            "Same CT MXFP8 target, canonical sparse/mHC/expert policies, C1, fixed seven drafts.",
            "Original FP8 / pretrained Modal BF16 / Modal FP8 / original FP8.",
            "Only Modal backbone linear precision changes in the derived FP8 candidate.",
            "Eight frozen development prompts, six seed-42 shuffled cycles per load, two warmups.",
            "No training or feature capture; startup configuration/FC checks and graph counts only.",
            "Record output identity, rounds, accepted drafts, end-to-end TPS, TTFT and resources.",
            "Repeat tokens/counters required within each load; compare output identity across arms.",
            "Baseline load controls must be within 2%; one candidate load cannot qualify final speed.",
            "Reuse existing target and original 500k-service restoration watchdog; 60-minute bound.",
            "No sealed prompts, C8/C16, long-context, MTP or automatic training/promotion in this screen.",
        ],
    )
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in ("prepare.py", "control.py"):
        path = output / name
        path.write_text(path.read_text().replace("8035", "8040").replace("29708", "29713")
                        .replace("drafter-canonical-decode-v1", REVISION))
    path = output / "prepare.py"
    replace_one(path, 'env.update(MXWAVE_TRIAL_ARM=mode, VLLM_SERVER_DEV_MODE="1")',
                'env.update(MXWAVE_TRIAL_ARM=mode, VLLM_SERVER_DEV_MODE="1",\n'
                '                       MXWAVE_DRAFTER_VARIANT=plan["variants"][arm])')
    replace_one(path, '                if mount[1] == "/trial":',
                '                if mount[1] == "/draft" and plan["variants"][arm] != "original":\n'
                f'                    mount[0] = str(ROOT.parent / ("{REVISION}-"\n'
                '                                      + plan["variants"][arm] + "-checkpoint"))\n'
                '                if mount[1] == "/trial":')
    # Each arm needs CLI/config validation because layer counts and precision differ.
    replace_one(path, '        if mode in result["cpu_checks"]:\n            continue\n', '')
    replace_one(path, 'result["cpu_checks"][f"{mode}-r{rank}"]',
                'result["cpu_checks"][f"{arm}-r{rank}"]')
    path = output / "trial_plugin.py"
    replace_one(path, '        import drafter_head_probe\n        drafter_head_probe.arm_capture(worker)\n', '')
    replace_one(path, '        import drafter_context_probe\n'
                '        report["context_probe"] = drafter_context_probe.install(worker)',
                '        import pretrained_startup\n'
                '        report["context_probe"] = pretrained_startup.install(worker)')
    (output / "pretrained_startup.py").write_text(_STARTUP)
    (output / "measure_features.py").write_text(_MEASURE.replace('8020', '8040'))
    (output / "wait_for_capture.py").write_text(_QUEUE)
    for path in output.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    frozen = {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()}
    (output / "source-manifest.json").write_text(json.dumps(frozen, indent=2) + "\n")
    return {"status": "frozen_not_launched", "sources": len(frozen),
            "plan_sha256": digest(output / "plan.json"), "prompts": 8, "requests_per_arm": 48,
            "arms": plan["arms"], "sealed_test_opened": False}


def main() -> None:
    """Prepare the bounded screen locally without contacting serving hosts."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "audit", "capture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template, args.audit, args.capture, args.output), indent=2))


if __name__ == "__main__":
    main()
