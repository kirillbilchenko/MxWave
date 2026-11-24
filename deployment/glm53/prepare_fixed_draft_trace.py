"""Prepare the paired DFlash trace with only BF16 convolution exclusion metadata corrected."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess

from control_stock_trial import ORIGINAL, RELEASE, identity, inspect, run, write


def main() -> None:
    """Validate the corrected metadata on CPU before any second GPU window."""
    root, previous = RELEASE / "stock-draft-trace-v2", RELEASE / "stock-draft-trace-v1"
    fixed = RELEASE / "drafter-modal-vllm032-bf16-conv-v1"
    checkpoint = RELEASE.parent / "drafter-pretrained-modal-v1-modal_fp8-checkpoint"
    if (root / "prepared.json").exists():
        raise FileExistsError("Preserve the frozen preparation")
    plan = json.loads((previous / "prepared.json").read_bytes())
    for rank in (0, 1):
        if identity(inspect(rank, ORIGINAL[rank])) != plan["originals"][rank]:
            raise RuntimeError("Production identity changed")
    for name in ("profile-plan.json", "profile_worker.py", "profile_client.py",
                 "profile_boot.py", "control_stock_trial.py", "draft_trace_client.py"):
        shutil.copyfile(previous / name, root / name)
    shutil.copyfile(root / "draft_trace_client.py", root / "smoke_client.py")
    source = (previous / "cpu_preflight.py").read_text()
    marker = "from vllm.entrypoints.cli.serve import cmd_init"
    if source.count(marker) != 1:
        raise RuntimeError("Unknown native CLI preflight")
    correction = """
spec_index = command.index('--speculative-config') + 1
specification = json.loads(command[spec_index])
assert specification['model'] == '/drafter'
specification['model'] = '/drafter-fixed'
command[spec_index] = json.dumps(specification)
env['MXWAVE_DRAFT_REQUIRE_BF16_CONV'] = '1'
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.models.qwen3_dflash2 import DFlashGroupedConv
from unittest.mock import patch
from types import SimpleNamespace
fixed_config = json.loads(Path('/drafter-fixed/config.json').read_text())
fixed_quant = Fp8Config.from_config(fixed_config['quantization_config'])
cpu_conv_checks = []
for layer_id in range(45, 51):
    for side in ('attention_conv', 'mlp_conv'):
        prefix = f'model.layers.{layer_id}.{side}'
        with patch('vllm.distributed.parallel_state._TP',
                   SimpleNamespace(rank_in_group=0, world_size=2)):
            module = DFlashGroupedConv(4096, 2, 16, 6, torch.bfloat16, prefix, fixed_quant)
        proj = module.kernel_projection
        assert proj.weight.dtype == torch.bfloat16
        assert type(proj.quant_method).__name__ == 'UnquantizedLinearMethod'
        cpu_conv_checks.append(prefix + '.kernel_projection')
        del module, proj
    assert not is_layer_skipped(f'model.layers.{layer_id}.self_attn.q_proj',
                               fixed_quant.ignored_layers)
(root / f'conv-cpu-rank{rank}.json').write_text(json.dumps(cpu_conv_checks, indent=2)+'\\n')
"""
    source = source.replace(marker, correction + "\n" + marker)
    (root / "cpu_preflight.py").write_text(source)
    (root / "reports").mkdir(exist_ok=True)
    run(1, ["mkdir", "-p", str(root), str(root / "reports")])
    subprocess.run(["scp", "-i", "/home/kirya/.ssh/id_glm_cluster",
                    *(str(p) for p in root.glob("*.py")), str(root / "profile-plan.json"),
                    "kirya@192.168.192.2:" + str(root) + "/"], check=True, timeout=60)
    for rank in (0, 1):
        receipt = run(rank, ["python3", "-S", str(root / "prepare_drafter_compat.py"),
                             "--source", str(checkpoint), "--output", str(fixed)], timeout=120)
        (root / f"compatibility-rank{rank}.json").write_text(receipt)
        image = plan["images"][rank]
        run(rank, ["mkdir", "-p", str(root / "cache")])
        run(rank, ["docker", "run", "--rm", "--network", "none",
                   "-v", str(previous / "cache") + ":/seed:ro",
                   "-v", str(root / "cache") + ":/cache:rw", "--entrypoint", "python3",
                   image, "-S", "-c",
                   "import shutil; shutil.copytree('/seed', '/cache', dirs_exist_ok=True)"],
            timeout=180)
        server = plan["servers"][rank]
        server["name"] = f"mxwave-glm53-stock-draft-trace-v2-r{rank}"
        server["argv"] = [v.replace(str(previous), str(root)).replace(
            f"mxwave-glm53-stock-draft-trace-v1-r{rank}", server["name"]) for v in server["argv"]]
        server["argv"][2:2] = ["-v", str(fixed) + ":/drafter-fixed:ro"]
        cpu = list(server["argv"])
        cpu.remove("-d")
        for flag in ("--name", "--gpus"):
            index = cpu.index(flag)
            del cpu[index:index + 2]
        cpu[cpu.index("--network") + 1] = "none"
        for flag in ("--memory", "--memory-swap"):
            cpu[cpu.index(flag) + 1] = "12g"
        cpu = [v.replace(":/profile:ro", ":/profile:rw").replace(
            "/profile/profile_boot.py", "/profile/cpu_preflight.py") for v in cpu]
        cpu[2:2] = ["--rm", "--cpus", "4"]
        try:
            cpu_output = run(rank, cpu, timeout=240)
        except RuntimeError as exc:
            with (root / f"cpu-preflight-rank{rank}-failure.log").open("a") as failure:
                failure.write(str(exc) + "\n")
            raise
        (root / f"cpu-preflight-rank{rank}.log").write_text(cpu_output)
        if rank == 1:
            for name in ("cpu-preflight-rank1.json", "launch-rank1.json", "conv-cpu-rank1.json"):
                (root / name).write_text(run(1, ["cat", str(root / name)]))
    component = plan["component"]
    component["argv"] = [v.replace("draft-trace-gpu-v1", "draft-trace-gpu-v2")
                         for v in component["argv"]]
    component["name"] = "mxwave-glm53-draft-trace-gpu-v2"
    plan["owned"] = [{"rank": 1, "name": component["name"]}] + [
        {"rank": s["rank"], "name": s["name"]} for s in plan["servers"]]
    plan["scope"] = "V1 paired trace; metadata-only BF16 conv correction; no new weight values"
    plan["derived_cache_seed"] = str(previous / "cache")
    plan["sources"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in root.glob("*.py")}
    plan["cpu_preflights"] = [json.loads((root / f"cpu-preflight-rank{r}.json").read_bytes())
                              for r in (0, 1)]
    plan["corrected_drafter"] = str(fixed)
    write(root / "prepared.json", plan)
    print(json.dumps({"status": "prepared", "root": str(root),
                      "sha256": hashlib.sha256((root / "prepared.json").read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
