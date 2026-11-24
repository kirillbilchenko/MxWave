"""Prepare a short native DFlash trace from the completed BF16-cache isolation trial."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess

from control_stock_trial import ORIGINAL, RELEASE, identity, inspect, run, write


def main() -> None:
    """Freeze observer-only launches after both official-image CPU preflights pass."""
    root = RELEASE / "stock-draft-trace-v1"
    previous = RELEASE / "stock-profile-v3"
    if (root / "prepared.json").exists():
        raise FileExistsError("Trace preparation is immutable")
    plan = json.loads((previous / "prepared.json").read_bytes())
    for rank in (0, 1):
        if identity(inspect(rank, ORIGINAL[rank])) != plan["originals"][rank]:
            raise RuntimeError("Production identity changed")
    for name in ("profile-plan.json", "profile_worker.py", "profile_client.py",
                 "profile_boot.py", "control_stock_trial.py"):
        shutil.copyfile(previous / name, root / name)
    shutil.copyfile(root / "draft_trace_client.py", root / "smoke_client.py")
    source = (previous / "cpu_preflight.py").read_text()
    old = "'profile_worker.ProfileWorker'"
    if source.count(old) != 1:
        raise RuntimeError("Unknown CPU preflight observer binding")
    source = source.replace(old, "'draft_trace_worker.DraftTraceWorker'")
    source += "\nfrom draft_trace_worker import DraftTraceWorker, tensor_report\n"
    source += "assert callable(DraftTraceWorker.mxwave_draft_trace_start)\n"
    source += "assert tensor_report(torch.tensor([1., 2.]), 1)['preview'] == [1.]\n"
    source += "assert {'input_batch', 'is_profile', 'aux_hidden_states'} <= set(\n"
    source += "    inspect.signature(DFlash2Speculator.propose).parameters)\n"
    (root / "cpu_preflight.py").write_text(source)
    (root / "reports").mkdir(exist_ok=True)
    run(1, ["mkdir", "-p", str(root), str(root / "reports")])
    subprocess.run(["scp", "-i", "/home/kirya/.ssh/id_glm_cluster",
                    *(str(p) for p in root.glob("*.py")), str(root / "profile-plan.json"),
                    "kirya@192.168.192.2:" + str(root) + "/"], check=True, timeout=60)
    for rank in (0, 1):
        image = plan["images"][rank]
        run(rank, ["mkdir", "-p", str(root / "cache")])
        run(rank, ["docker", "run", "--rm", "--network", "none",
                   "-v", str(previous / "cache") + ":/seed:ro",
                   "-v", str(root / "cache") + ":/cache:rw", "--entrypoint", "python3",
                   image, "-S", "-c",
                   "import shutil; shutil.copytree('/seed', '/cache', dirs_exist_ok=True)"],
            timeout=180)
        server = plan["servers"][rank]
        server["name"] = f"mxwave-glm53-stock-draft-trace-v1-r{rank}"
        server["argv"] = [v.replace(str(previous), str(root)).replace(
            f"mxwave-glm53-stock-profile-v3-r{rank}", server["name"]) for v in server["argv"]]
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
        (root / f"cpu-preflight-rank{rank}.log").write_text(run(rank, cpu, timeout=240))
        if rank == 1:
            for name in ("cpu-preflight-rank1.json", "launch-rank1.json"):
                (root / name).write_text(run(1, ["cat", str(root / name)]))
    component = plan["component"]
    component["argv"] = [v.replace("profile-gpu-v3", "draft-trace-gpu-v1")
                         for v in component["argv"]]
    component["name"] = "mxwave-glm53-draft-trace-gpu-v1"
    plan["owned"] = [{"rank": 1, "name": component["name"]}] + [
        {"rank": s["rank"], "name": s["name"]} for s in plan["servers"]]
    plan["scope"] = "Two prompts, native vs draft-eager; first eight boundaries; no speed claim"
    plan["derived_cache_seed"] = str(previous / "cache")
    plan["sources"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in root.glob("*.py")}
    plan["cpu_preflights"] = [json.loads((root / f"cpu-preflight-rank{r}.json").read_bytes())
                              for r in (0, 1)]
    write(root / "prepared.json", plan)
    print(json.dumps({"status": "prepared", "root": str(root),
                      "sha256": hashlib.sha256((root / "prepared.json").read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
