"""Hand over to the tested CT profile on explicit request, with startup rollback.

This is an operational tool for the existing custom production runtime. It does
not change the portable stock-vLLM recipe or the recorded qualification outcome.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

import control_stock_trial as c
from prepare_same_runtime import create_container

SOURCE = c.RELEASE / "same-runtime-final-v3"


def tested_identity(value: dict[str, Any]) -> dict[str, str]:
    """Reproduce the source preparer's documented Docker default/order normalization."""
    normalized = copy.deepcopy(value)
    host = normalized["HostConfig"]
    host["OomKillDisable"] = bool(host.get("OomKillDisable"))
    host["Dns"] = host.get("Dns") or []
    host["Binds"] = sorted(host["Binds"])
    return c.identity(normalized)


def live_config(value: dict[str, Any], root: Path) -> dict[str, Any]:
    """Keep the tested model/settings; change listener, reports and dev endpoints only."""
    config = copy.deepcopy(value["Config"])
    config.pop("Hostname", None)
    config["Image"] = value["Image"]
    command = config["Cmd"]
    if command[command.index("--port") + 1] != "8051":
        raise ValueError("Require the tested timing-B listener")
    command[command.index("--port") + 1] = "8000"
    config["Env"] = [e for e in config["Env"] if not e.startswith("VLLM_SERVER_DEV_MODE=")]
    config["Env"].append("VLLM_SERVER_DEV_MODE=0")
    config["HostConfig"] = copy.deepcopy(value["HostConfig"])
    binds = config["HostConfig"]["Binds"]
    reports = [i for i, b in enumerate(binds) if b.split(":")[1] == "/reports"]
    if len(reports) != 1:
        raise ValueError("Require one report mount; never overwrite benchmark evidence")
    binds[reports[0]] = str(root / "reports") + ":/reports:rw"
    return config


def prepare_rank(root: Path, rank: int) -> None:
    """Clone the exact successful stopped B container and check imports without a GPU."""
    plan = json.loads((root / "source-prepared.json").read_bytes())
    source = next(e for e in plan["owned"]
                  if e["rank"] == rank and "timing-B-2" in e["name"])
    value = json.loads(subprocess.check_output(["docker", "inspect", source["name"]]))[0]
    if tested_identity(value) != source["identity"] or value["State"]["Running"]:
        raise RuntimeError("Successful source container changed or is running")
    plugin = SOURCE / "same_runtime_plugin.py"
    if hashlib.sha256(plugin.read_bytes()).hexdigest() != plan["files"][plugin.name]:
        raise RuntimeError("Frozen runtime plugin changed")
    receipt_path = root / f"clone-r{rank}.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_bytes())
        current = json.loads(subprocess.check_output(["docker", "inspect", receipt["name"]]))[0]
        cpu_status = json.loads((root / f"cpu-r{rank}.json").read_bytes())
        if (c.identity(current) != receipt["identity"] or current["State"]["Running"]
                or receipt["source_identity"] != source["identity"]
                or cpu_status["status"] != "passed"):
            raise RuntimeError("Cannot reuse an incomplete or changed CPU preparation")
        return
    (root / "reports").mkdir(exist_ok=False)
    config = live_config(value, root)
    name = f"mxwave-glm53-{root.name}-r{rank}"
    container = create_container(config, name)
    created = json.loads(subprocess.check_output(["docker", "inspect", container]))[0]
    for key in set(config) - {"HostConfig", "Image"}:
        if created["Config"][key] != config[key]:
            raise RuntimeError(f"Unexpected clone field: {key}")
    wanted_host, actual_host = copy.deepcopy(config["HostConfig"]), copy.deepcopy(created["HostConfig"])
    for host in (wanted_host, actual_host):
        host["OomKillDisable"] = bool(host.get("OomKillDisable"))
        host["Dns"] = host.get("Dns") or []
        host["Binds"] = sorted(host["Binds"])
    if actual_host != wanted_host or created["State"]["Running"]:
        raise RuntimeError("Clone host settings changed or clone unexpectedly running")
    receipt = {"rank": rank, "name": name, "identity": c.identity(created),
               "source_identity": source["identity"], "source_raw_identity": c.identity(value),
               "command": config["Cmd"],
               "plugin_sha256": plan["files"][plugin.name]}
    c.write(root / f"clone-r{rank}.json", receipt)
    cpu = copy.deepcopy(config)
    cpu["Entrypoint"] = ["python3"]
    cpu["Cmd"] = ["-c", (
        "import os; from vllm import platforms; from vllm.platforms.cpu import CpuPlatform; "
        "platforms.current_platform=CpuPlatform(); "
        "from vllm.plugins import load_general_plugins; load_general_plugins(); "
        "from vllm.v1.worker.gpu_worker import Worker; "
        "assert getattr(Worker.load_model, '_mxwave_same_runtime', False); "
        "from same_runtime_plugin import scale_codec_contract; "
        "assert scale_codec_contract()['encoding']=='sparse_palette'; "
        "assert os.environ['VLLM_SERVER_DEV_MODE']=='0'; print('CT_LIVE_CPU_PASSED')")]
    cpu["HostConfig"].update(DeviceRequests=[], NetworkMode="none", Memory=8 * 1024**3,
                              MemorySwap=8 * 1024**3)
    cpu["Env"] = [e for e in cpu["Env"] if not e.startswith("NVIDIA_VISIBLE_DEVICES=")]
    cpu["Env"] += ["CUDA_VISIBLE_DEVICES=", "NVIDIA_VISIBLE_DEVICES=void"]
    cpu_id = create_container(cpu, name + "-cpu")
    result = subprocess.run(["docker", "start", "-a", cpu_id], capture_output=True,
                            timeout=180, check=False)
    log = result.stdout + result.stderr
    (root / f"cpu-r{rank}.log").write_bytes(log)
    status = json.loads(subprocess.check_output(["docker", "inspect", cpu_id]))[0]
    if status["State"]["ExitCode"] != 0 or b"CT_LIVE_CPU_PASSED" not in log:
        raise RuntimeError("CT CPU preflight failed; no production outage started")
    subprocess.run(["docker", "rm", cpu_id], capture_output=True, check=True)
    c.write(root / f"cpu-r{rank}.json", {"status": "passed"})


def prepare(root: Path) -> None:
    """Prepare both stopped clones and freeze restoration identities before handover."""
    if (root / "prepared.json").exists():
        raise FileExistsError("Use a fresh handover directory")
    prior = json.loads((SOURCE / "prepared.json").read_bytes())
    originals = [c.identity(c.inspect(r, c.ORIGINAL[r])) for r in (0, 1)]
    if originals != prior["originals"]:
        raise RuntimeError("Original production identities changed")
    entries = []
    source_plan = {"owned": [e for e in prior["owned"] if "timing-B-2" in e["name"]],
                   "files": {"same_runtime_plugin.py": prior["files"]["same_runtime_plugin.py"]},
                   "prepared_sha256": hashlib.sha256((SOURCE / "prepared.json").read_bytes()).hexdigest()}
    for rank in (0, 1):
        c.run(rank, ["python3", "-S", "-c",
                     ("import sys; from pathlib import Path; p=Path(sys.argv[1]); "
                      "assert not p.exists() or p.read_text()==sys.argv[2]; p.write_text(sys.argv[2])"),
                     str(root / "source-prepared.json"), json.dumps(source_plan, indent=2) + "\n"])
        c.run(rank, ["python3", "-S", str(root / Path(__file__).name), "--root", str(root),
                     "--prepare-rank", str(rank)], timeout=240)
        entries.append(json.loads(c.run(rank, ["cat", str(root / f"clone-r{rank}.json")])))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.py")}
    c.write(root / "prepared.json", {"status": "passed", "originals": originals,
            "owned": entries, "files": files, "source": str(SOURCE),
            "user_selected": True, "benchmark_qualification_passed": False,
            "changes_from_tested_B": ["API port 8051 to 8000", "isolated report directory",
                                      "disable development endpoints", "new container hostname"]})


def watch(root: Path, pid: int, start: str) -> None:
    """Restore on startup failure/deadline; stop watching after successful user handover."""
    c.write(root / "watchdog.json", {"pid": os.getpid(), "parent": pid, "ready": True})
    while True:
        state = json.loads((root / "controller.json").read_bytes())
        if state["phase"] in (*c.TERMINAL, "complete_live"):
            return
        dead = c.ticks(pid) != start
        expired = time.time() > state.get("deadline_epoch", float("inf"))
        if state["service_paused"] and (dead or expired):
            if not dead:
                os.kill(pid, signal.SIGTERM)
            for attempt in range(3):
                try:
                    c.restore(root)
                    return
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(10)
        if dead:
            return
        time.sleep(3)


def smoke(root: Path) -> None:
    """Verify authenticated generation; retain this as readiness, not a quality benchmark."""
    key = Path("/home/kirya/local-spark/runtime/secrets/llm_api_key").read_text().strip()
    body = {"model": "glm-5.3-flash", "messages": [
        {"role": "user", "content": "What is 2 + 2? Reply briefly."}],
        "temperature": 0, "seed": 42, "max_tokens": 128}
    request = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + key,
                                               "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    c.write(root / "smoke.json", result)
    if not result.get("choices") or result.get("usage", {}).get("completion_tokens", 0) <= 0:
        raise RuntimeError("CT failed generation readiness check")


def start(root: Path) -> None:
    """Switch an idle service under a detached watchdog and leave CT live after verification."""
    plan = json.loads((root / "prepared.json").read_bytes())
    if plan["status"] != "passed" or (root / "controller.json").exists():
        raise RuntimeError("Require a fresh passed preparation")
    for name, digest in plan["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Prepared handover source changed")
    for entry in plan["owned"]:
        if c.identity(c.inspect(entry["rank"], entry["name"])) != entry["identity"]:
            raise RuntimeError("Prepared CT container changed")
    if [c.identity(c.inspect(r, c.ORIGINAL[r])) for r in (0, 1)] != plan["originals"]:
        raise RuntimeError("Original service changed since preparation")
    c.production_models()
    for _ in range(2):
        with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=10) as response:
            metrics = response.read().decode()
        values = [float(line.rsplit(" ", 1)[1]) for line in metrics.splitlines()
                  if re.match(r"vllm:num_requests_(running|waiting)(\{| )", line)]
        if not values or any(values):
            raise RuntimeError("Service is busy; no handover started")
        time.sleep(3)
    state = {"phase": "prepared", "pid": os.getpid(), "started_epoch": time.time(),
             "service_paused": False, "originals": plan["originals"], "owned": plan["owned"],
             "user_selected": True, "benchmark_qualification_passed": False}
    path = root / "controller.json"
    c.write(path, state)

    def terminated(signum: int, frame: Any) -> None:
        raise RuntimeError(f"Handover interrupted: signal {signum}")

    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    with (root / "watchdog.log").open("xb") as output:
        watcher = subprocess.Popen(["python3", "-S", str(Path(__file__).resolve()),
            "--root", str(root), "--watch", str(os.getpid()), str(c.ticks(os.getpid()))],
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        while not (root / "watchdog.json").exists():
            if watcher.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Watchdog did not initialize")
            time.sleep(.2)
        state.update(service_paused=True, phase="stopping_production",
                     deadline_epoch=time.time() + 1500, watchdog_pid=watcher.pid)
        c.write(path, state)
        for rank in (0, 1):
            c.run(rank, ["docker", "stop", "--timeout", "30", c.ORIGINAL[rank]])
        state["phase"] = "loading_ct"
        c.write(path, state)
        for entry in reversed(plan["owned"]):
            c.run(entry["rank"], ["docker", "start", entry["name"]], timeout=120)
        while not c.health(8000):
            if time.time() > state["deadline_epoch"] - 150:
                raise RuntimeError("CT startup exceeded its deadline")
            if not all(c.inspect(e["rank"], e["name"])["State"]["Running"]
                       for e in plan["owned"]):
                raise RuntimeError("CT rank exited during startup")
            time.sleep(5)
        reports = []
        for rank in (0, 1):
            raw = c.run(rank, ["cat", str(root / "reports" / f"startup-r{rank}.json")])
            report = json.loads(raw)
            if (report["arm"] != "B" or report["max_model_len"] != 500000
                    or not report["speculation"] or report["draft_layers"] != 5
                    or report["experts"]["layers_seen"] != 42
                    or len(report["kda"]["layers"]) != 68):
                raise RuntimeError("Loaded CT profile did not match the tested setup")
            reports.append(report)
        c.write(root / "startup.json", reports)
        models = c.production_models()
        smoke(root)
        c.retain_logs(root, plan["owned"])
        with (root / "restore.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = json.loads(path.read_bytes())
            if current["phase"] != "loading_ct" or time.time() >= state["deadline_epoch"]:
                raise RuntimeError("Restoration already owns this handover")
            if any(c.inspect(r, c.ORIGINAL[r])["State"]["Running"] for r in (0, 1)):
                raise RuntimeError("Original service unexpectedly running")
            if [c.identity(c.inspect(r, c.ORIGINAL[r])) for r in (0, 1)] != plan["originals"]:
                raise RuntimeError("Original rollback identities changed")
            state.update(phase="complete_live", service_paused=False, models=models,
                         finished_epoch=time.time(), active_profile="tested_CT_Inco")
            c.write(path, state)
    except BaseException as exc:
        state.update(error=repr(exc), phase="failed")
        c.write(path, state)
        if state["service_paused"]:
            c.restore(root)
        else:
            state["phase"] = "preparation_failed"
            c.write(path, state)
        raise


def main() -> None:
    """Prepare, launch, watch or roll back the explicitly requested CT service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-rank", type=int, choices=(0, 1))
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--start", action="store_true")
    modes.add_argument("--watch", nargs=2)
    modes.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    if args.prepare_rank is not None:
        prepare_rank(args.root, args.prepare_rank)
    elif args.prepare:
        prepare(args.root)
    elif args.start:
        start(args.root)
    elif args.watch:
        watch(args.root, int(args.watch[0]), args.watch[1])
    else:
        c.restore(args.root)


if __name__ == "__main__":
    main()
