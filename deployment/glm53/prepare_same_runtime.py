"""Freeze and launch the final same-production-runtime comparison without overlay edits."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import time
import urllib.parse
from pathlib import Path
from typing import Any

from control_stock_trial import ORIGINAL, identity, inspect, run, write
from prepare_quality_clone import DockerConnection

BASE = Path("/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15")


def command_for(command: list[str], mode: str, port: int, master: int) -> list[str]:
    """Preserve all serving flags except private ports, observation and spec-off quality."""
    result = list(command)
    for flag, value in (("--port", str(port)), ("--master-port", str(master))):
        result[result.index(flag) + 1] = value
    if mode == "gsm8k":
        index = result.index("--speculative-config")
        del result[index:index + 2]
    if "--worker-extension-cls" in result:
        raise ValueError("Do not replace an existing production worker extension")
    return result + ["--worker-extension-cls", "same_runtime_plugin.SameRuntimeWorker"]


def create_container(config: dict[str, Any], name: str) -> str:
    """Create an unstarted container through Docker without exposing private environment."""
    if subprocess.run(["docker", "inspect", name], capture_output=True,
                      check=False).returncode == 0:
        raise FileExistsError(name)
    connection = DockerConnection("localhost", timeout=60)
    connection.request("POST", "/containers/create?name=" + urllib.parse.quote(name),
                       body=json.dumps(config).encode(),
                       headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    payload = json.loads(response.read())
    if response.status != 201:
        raise RuntimeError(f"Container creation failed: HTTP {response.status}")
    return str(payload["Id"])


def clone(root: Path, arm: str, mode: str, trial: int, rank: int) -> dict[str, Any]:
    """Clone production; only B's model and explicitly selected CT adapters differ."""
    original = json.loads(subprocess.check_output(["docker", "inspect", ORIGINAL[rank]]))[0]
    stage = f"{mode}-{arm}-{trial}"
    name = f"mxwave-glm53-{root.name}-{stage}-r{rank}"
    config = copy.deepcopy(original["Config"])
    config.pop("Hostname", None)
    config["Image"] = original["Image"]
    config["HostConfig"] = copy.deepcopy(original["HostConfig"])
    port = 8050 + (arm == "B") + (2 if mode == "gsm8k" else 0)
    config["Cmd"] = command_for(config["Cmd"], mode, port, 29740 + port - 8050)
    env = dict(value.split("=", 1) for value in config["Env"])
    prior_env = dict(env)
    env.update(MXWAVE_COMPARE_ARM=arm, VLLM_SERVER_DEV_MODE="1")
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":/trial:/kda-runtime:/ct-code"
    binds = config["HostConfig"]["Binds"]
    reports = root / "reports" / stage
    reports.mkdir(parents=True, exist_ok=True)
    binds.extend([str(root) + ":/trial:ro", str(reports) + ":/reports:rw",
                  str(BASE / "full-model-v1/kda-mxfp8-speed-v1") + ":/kda-runtime:ro",
                  str(BASE / "full-model-v1/full-c16admissionv1-src") + ":/ct-code:ro"])
    if arm == "B":
        candidate_env = json.loads((root / "adapter-environment.json").read_bytes())[str(rank)]
        required = {"MXWAVE_MOE_MODE", "MXWAVE_MOE_RAW_LIBRARY", "MXWAVE_MOE_ALIGNED_LIBRARY",
                    "MXWAVE_MOE_RECORD_CODEC", "MXWAVE_MOE_DISPATCH",
                    "MXWAVE_GLM53_ATTENTION_MAPPING", "MXWAVE_GLM53_KDA_MODE",
                    "MXWAVE_GLM53_KDA_SOURCE", "MXWAVE_ZIPSERV_LIBRARY",
                    "MXWAVE_GLM53_VOCAB_MODE", "MXWAVE_GLM53_VOCAB_ARTIFACTS",
                    "MXWAVE_GLM53_VOCAB_MANIFEST_SHA256"}
        env.update({key: candidate_env[key] for key in required})
        env["PYTHONPATH"] += ":/experiment/triple-v2/src/integrations/zipserv"
        model_mounts = [i for i, value in enumerate(binds) if value.split(":")[1] == "/model"]
        if len(model_mounts) != 1:
            raise ValueError("Require exactly one production model bind")
        binds[model_mounts[0]] = str(BASE / "full-model-v1/source") + ":/model:ro"
        binds.extend([str(BASE) + ":/audit:ro",
                      "/home/kirya/local-spark/experiments/zipserv-vllm031-2026-10-14:/experiment:ro"])
    config["Env"] = [key + "=" + value for key, value in env.items()]
    created_id = create_container(config, name)
    created = json.loads(subprocess.check_output(["docker", "inspect", created_id]))[0]
    if created["State"]["Running"] or created["Config"]["Cmd"] != config["Cmd"]:
        raise RuntimeError("Prepared clone must be stopped with the exact frozen command")
    if any(env.get(key) != value for key, value in prior_env.items() if key != "PYTHONPATH"):
        raise RuntimeError("A production overlay/environment flag was changed")
    unchanged = set(original["Config"]) - {"Hostname", "Image", "Cmd", "Env"}
    if any(original["Config"][key] != created["Config"][key] for key in unchanged):
        raise RuntimeError("Unexpected production config change")
    for host in (config["HostConfig"], created["HostConfig"]):
        host["OomKillDisable"] = bool(host.get("OomKillDisable"))
        host["Dns"] = host.get("Dns") or []
        host["Binds"] = sorted(host["Binds"])
    if created["HostConfig"] != config["HostConfig"]:
        raise RuntimeError("Unexpected clone host configuration change")
    result = {"rank": rank, "name": name, "argv": ["docker", "start", created_id],
              "identity": identity(created), "port": port, "arm": arm, "mode": mode,
              "command": config["Cmd"], "all_production_flags_preserved": True,
              "environment_additions": sorted(set(env) - set(prior_env)),
              "admission_window_ms": env.get("GLM_IDLE_COALESCE_MS"),
              "dense_head": True, "drafter_mount_unchanged": True}
    write(root / f"clone-{stage}-r{rank}.json", result)
    if trial == 0:
        cpu = copy.deepcopy(config)
        cpu["Entrypoint"] = ["python3"]
        check = (
            "import json; from vllm import platforms; from vllm.platforms.cpu import CpuPlatform; "
            "platforms.current_platform=CpuPlatform(); "
            "from vllm.plugins import load_general_plugins; load_general_plugins(); "
            "from vllm.v1.worker.gpu_worker import Worker; "
            "assert getattr(Worker.load_model, '_mxwave_same_runtime', False); "
            "from vllm.entrypoints.cli.serve import cmd_init; "
            "from vllm.utils.argparse_utils import FlexibleArgumentParser; "
            "p=FlexibleArgumentParser(); sub=p.add_subparsers(dest='subparser'); "
            "commands=cmd_init(); [c.subparser_init(sub) for c in commands]; "
            "a=p.parse_args(" + repr(config["Cmd"]) + "); "
            "next(c for c in commands if c.name==a.subparser).validate(a); "
            "assert a.max_model_len==500000 and a.max_num_seqs==32; "
            "assert a.max_num_batched_tokens==6919 and a.kv_cache_memory_bytes==6442450944; "
            "print('MXWAVE_SAME_RUNTIME_CPU_PASSED')")
        cpu["Cmd"] = ["-c", check]
        cpu["HostConfig"]["DeviceRequests"] = []
        cpu["HostConfig"]["NetworkMode"] = "none"
        cpu["HostConfig"]["Memory"] = 8 * 1024**3
        cpu["HostConfig"]["MemorySwap"] = 8 * 1024**3
        cpu["Env"] = [e for e in cpu["Env"] if not e.startswith("NVIDIA_VISIBLE_DEVICES=")]
        cpu["Env"] += ["CUDA_VISIBLE_DEVICES=", "NVIDIA_VISIBLE_DEVICES=void"]
        cpu_id = create_container(cpu, name + "-cpu")
        run_result = subprocess.run(["docker", "start", "-a", cpu_id], capture_output=True,
                                    timeout=240, check=False)
        log = run_result.stdout + run_result.stderr
        (root / f"cpu-{stage}-r{rank}.log").write_bytes(log)
        status = json.loads(subprocess.check_output(["docker", "inspect", cpu_id]))[0]
        if status["State"]["ExitCode"] != 0 or b"MXWAVE_SAME_RUNTIME_CPU_PASSED" not in log:
            raise RuntimeError(f"CPU overlay/adapter validation failed: {name}; retained log")
        subprocess.run(["docker", "rm", cpu_id], check=True, capture_output=True)
    return result


def prepare(root: Path) -> None:
    """Freeze both arms, protocol and watchdog before pausing any serving process."""
    if (root / "prepared.json").exists():
        raise FileExistsError("Never reuse a prepared experiment")
    previous = BASE / "full-model-v1/glm53-package-v1/deployment-quick-v1"
    current = BASE / "full-model-v1/glm53-package-v1/c1-runtime-profile-v1"
    expected = json.loads((current / "controller.json").read_bytes())["originals"]
    if [identity(inspect(r, ORIGINAL[r])) for r in (0, 1)] != expected:
        raise RuntimeError("Exact production changed since restoration")
    workloads = json.loads((previous / "workloads.json").read_bytes())
    workloads.pop("production_repeat_prompt_ids", None)
    workloads.update(salt=root.name, long_context=[])
    wanted = {f"{d}-short-c{c}" for d in ("code", "chat") for c in (1, 8)}
    if {c["name"] for c in workloads["cells"]} != wanted:
        raise RuntimeError("Require exactly four final-screen cells")
    write(root / "workloads.json", workloads)
    old_adapter = json.loads((BASE / "full-model-v1/kda-mxfp8-speed-v1/prepared.json").read_bytes())
    write(root / "adapter-environment.json", {
        str(rank): {key: value for key, value in
                    old_adapter["launches"]["b1"][rank]["environment"].items()
                    if key.startswith("MXWAVE_")} for rank in (0, 1)})
    for name in ("gsm8k-protocol.json", "gsm8k_benchmark.py", "tokenizer-identities.json"):
        (root / name).write_bytes((previous / name).read_bytes())
    from compare_client import validate_quality_protocol

    quality = json.loads((root / "gsm8k-protocol.json").read_bytes())
    validate_quality_protocol(quality)
    if quality["questions_per_arm"] != 300:
        raise RuntimeError("Require the same frozen 300-question subset")
    metadata = root / "mxwave_same_runtime-0.0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: mxwave-same-runtime\nVersion: 0.0.0\n")
    (metadata / "entry_points.txt").write_text(
        "[vllm.general_plugins]\nmxwave_same_runtime = same_runtime_plugin:install\n")
    run(1, ["mkdir", "-p", str(root)])
    subprocess.run(["scp", "-r", "-i", "/home/kirya/.ssh/id_glm_cluster",
                    *[str(p) for p in root.iterdir() if p.is_file() or p == metadata],
                    "kirya@192.168.192.2:" + str(root) + "/"], check=True, timeout=90)
    stages, owned = [], []
    for mode, trial, arm in ([('timing', i, a) for i in range(3) for a in "AB"]
                             + [('gsm8k', 0, a) for a in "AB"]):
        servers = []
        for rank in (0, 1):
            row = json.loads(run(rank, ["python3", "-S", str(root / "prepare_same_runtime.py"),
                "--root", str(root), "--clone", "--arm", arm, "--mode", mode,
                "--trial", str(trial), "--rank", str(rank)], timeout=360))
            servers.append(row)
            write(root / f"clone-{mode}-{arm}-{trial}-r{rank}.json", row)
        owned.extend(servers)
        stages.append({"name": f"{mode}-{arm}-{trial}", "kind": "same_runtime_clone",
                       "arm": arm, "mode": mode, "trial": trial, "port": servers[0]["port"],
                       "servers": servers, "client_timeout_seconds": 1800})
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob("*")) if p.is_file()
             and (p.suffix in (".py", ".json", ".txt") or p.name == "METADATA")
             and "reports" not in p.parts
             and p.name not in {"preparation-status.json", "launch.json"}}
    write(root / "prepared.json", {"status": "passed", "originals": expected,
        "stages": stages, "owned": owned, "files": files, "outage_seconds": 7200,
        "abort_on_stage_error": True, "promotion": "None", "quality_questions_per_arm": 300,
        "decision": {"c1_min_ratio_both_domains": .97, "c8_min_ratio_both_domains": 1.0,
                     "repeat_spread_limit": .02, "c1_stop_below": .95,
                     "quality_screen": "B correct >= A correct; zero errors/truncations",
                     "inconclusive": "3–5% deficit, unstable controls or mixed cells: retain production",
                     "statistical_equivalence_claim": False},
        "notes": ["All production flags/overlays and Inco draft policy are preserved.",
                  "Actual production admission flag is 4 ms, not the previously assumed 10 ms.",
                  ("CT KDA conversion and format eligibility can bypass some overlay dispatches; "
                   "retain activation reports, never claim identical kernel paths."),
                  "No overlay modifications, no fallback settings and no promotion."]})


def main() -> None:
    """Prepare detached on Spark, then run the bounded controller only after CPU checks pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--clone", action="store_true")
    parser.add_argument("--arm", choices=("A", "B"))
    parser.add_argument("--mode", choices=("timing", "gsm8k"))
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.clone:
        print(json.dumps(clone(args.root, args.arm, args.mode, args.trial, args.rank)))
        return
    status = args.root / "preparation-status.json"
    write(status, {"phase": "preparing", "epoch": time.time()})
    try:
        prepare(args.root)
        write(status, {"phase": "prepared", "epoch": time.time()})
        if args.launch:
            subprocess.run(["python3", "-S", str(args.root / "compare_control.py"),
                            "--root", str(args.root)], check=True)
    except BaseException as exc:
        write(status, {"phase": "stopped", "error": repr(exc), "epoch": time.time(),
                       "policy": "No workaround; controller restores if an outage began"})
        raise


if __name__ == "__main__":
    main()
