"""Prepare two trace-enabled deployments; preserve original containers and all settings."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import urllib.parse
from pathlib import Path
from typing import Any

from control_stock_trial import ORIGINAL, identity, inspect, run, write
from prepare_quality_clone import DockerConnection

PROFILER = {"profiler": "torch", "torch_profiler_dir": "/c1-traces",
            "torch_profiler_with_stack": False, "torch_profiler_use_gzip": True,
            "torch_profiler_record_shapes": True, "detailed_trace_annotation": True}


def profile_command(command: list[str], *, production: bool) -> list[str]:
    """Add native instrumentation and private ports without changing serving policy."""
    result = list(command)
    if "--profiler-config" in result:
        raise ValueError("Preserve existing profiler configuration rather than overwrite it")
    for flag, value in (("--port", "8048" if production else "8046"),
                        ("--master-port", "29730" if production else "29731")):
        result[result.index(flag) + 1] = value
    if production:
        result[result.index("--host") + 1] = "127.0.0.1"
    result += ["--profiler-config", json.dumps(PROFILER)]
    return result


def clone(original: str, name: str, trace_dir: Path) -> dict[str, Any]:
    """Create a stopped production clone with only ports and profiling changed."""
    value = json.loads(subprocess.check_output(["docker", "inspect", original]))[0]
    config = copy.deepcopy(value["Config"])
    config["Cmd"] = profile_command(config["Cmd"], production=True)
    config["Image"] = value["Image"]
    config.pop("Hostname", None)
    config["HostConfig"] = copy.deepcopy(value["HostConfig"])
    config["HostConfig"]["Binds"].append(str(trace_dir) + ":/c1-traces:rw")
    trace_dir.mkdir(parents=True, exist_ok=True)
    if subprocess.run(["docker", "inspect", name], capture_output=True,
                      check=False).returncode == 0:
        raise FileExistsError("Never reuse an existing profile clone")
    connection = DockerConnection("localhost", timeout=60)
    connection.request("POST", "/containers/create?name=" + urllib.parse.quote(name),
                       body=json.dumps(config).encode(),
                       headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    payload = json.loads(response.read())
    if response.status != 201:
        raise RuntimeError(f"Profile clone creation failed: HTTP {response.status}")
    created = json.loads(subprocess.check_output(["docker", "inspect", payload["Id"]]))[0]
    for key in set(value["Config"]) - {"Cmd", "Image", "Hostname"}:
        if created["Config"].get(key) != value["Config"].get(key):
            raise RuntimeError(f"Unexpected clone config change: {key}")
    if created["State"]["Running"] or created["Config"]["Cmd"] != config["Cmd"]:
        raise RuntimeError("Clone must remain stopped with its exact planned command")
    expected_host, actual_host = config["HostConfig"], copy.deepcopy(created["HostConfig"])
    for host in (expected_host, actual_host):
        host["OomKillDisable"] = bool(host.get("OomKillDisable"))
        host["Dns"] = host.get("Dns") or []
        host["Binds"] = sorted(host["Binds"])
    if actual_host != expected_host:
        raise RuntimeError("Unexpected clone host configuration change")
    prior_mounts = sorted(value["Mounts"], key=lambda m: m["Destination"])
    preserved = sorted((m for m in created["Mounts"] if m["Destination"] != "/c1-traces"),
                       key=lambda m: m["Destination"])
    if prior_mounts != preserved:
        raise RuntimeError("Original production mounts changed")
    return {"id": created["Id"], "name": name, "identity": identity(created),
            "allowed_changes": ["private ports/loopback API", "native profiler",
                                "trace output mount", "container hostname"],
            "private_config_and_serving_settings_preserved": True}


def prepare(root: Path, previous: Path) -> None:
    """Freeze existing candidate settings and a production clone with identical tracing."""
    if (root / "prepared.json").exists() or (root / "controller.json").exists():
        raise FileExistsError("Preserve existing preparation or controller")
    old = json.loads((previous / "prepared.json").read_bytes())
    for name, digest in old["files"].items():
        if hashlib.sha256((previous / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Previous frozen preparation changed: {name}")
    boot = json.loads((root / "boot-originals.json").read_bytes())
    originals = [identity(inspect(r, ORIGINAL[r])) for r in (0, 1)]
    if originals != boot["identities"]:
        raise RuntimeError("Production changed since the verified reboot normalization")
    workloads = json.loads((previous / "workloads.json").read_bytes())
    workloads["cells"] = [c for c in workloads["cells"] if c["name"].endswith("-short-c1")]
    if {c["name"] for c in workloads["cells"]} != {"code-short-c1", "chat-short-c1"}:
        raise RuntimeError("Require exactly the two final-screen C1 prompts")
    workloads["salt"] = root.name
    write(root / "workloads.json", workloads)
    source_stage = next(s for s in old["stages"] if s["arm"] == "B" and s["mode"] == "timing")
    servers_b = copy.deepcopy(source_stage["servers"])
    for rank, server in enumerate(servers_b):
        plan = json.loads((previous / f"launch-timing-rank{rank}.json").read_bytes())
        before = copy.deepcopy(plan)
        plan["command"] = profile_command(plan["command"], production=False)
        write(root / f"launch-timing-rank{rank}.json", plan)
        write(root / f"candidate-config-delta-rank{rank}.json", {
            "previous_sha256": hashlib.sha256(
                (previous / f"launch-timing-rank{rank}.json").read_bytes()).hexdigest(),
            "environment_unchanged": before["environment"] == plan["environment"],
            "changes": ["private collective port", "native profiler only"],
        })
        server["name"] = f"mxwave-glm53-{root.name}-b-r{rank}"
        server["argv"] = [x.replace(str(previous), str(root)).replace(
            source_stage["servers"][rank]["name"], server["name"]) for x in server["argv"]]
        server["argv"][2:2] = ["-v", str(root / "traces-B") + ":/c1-traces:rw"]
    for rank in (0, 1):
        run(rank, ["mkdir", "-p", str(root), str(root / "reports"),
                   str(root / "traces-A"), str(root / "traces-B"), str(root / "cache")])
    subprocess.run(["scp", "-i", "/home/kirya/.ssh/id_glm_cluster",
                    *map(str, root.glob("*.py")), *map(str, root.glob("launch-*.json")),
                    "kirya@192.168.192.2:" + str(root) + "/"], check=True, timeout=60)
    servers_a = []
    for rank in (0, 1):
        name = f"mxwave-glm53-{root.name}-a-r{rank}"
        result = json.loads(run(rank, ["python3", "-S", str(root / "prepare_c1_profile.py"),
                                      "--clone", ORIGINAL[rank], "--name", name,
                                      "--root", str(root)]))
        write(root / f"clone-A-rank{rank}.json", result)
        servers_a.append({"rank": rank, "name": name,
                          "argv": ["docker", "start", result["id"]]})
        image = next(x for x in servers_b[rank]["argv"] if x.startswith("sha256:"))
        run(rank, ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python3",
                   "-v", str(previous / "cache") + ":/seed:ro",
                   "-v", str(root / "cache") + ":/cache:rw", image, "-S", "-c",
                   "import shutil; shutil.copytree('/seed', '/cache', dirs_exist_ok=True)"],
            timeout=180)
        check = ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python3",
                 image, "-c", "from vllm.config.profiler import ProfilerConfig; "
                 "import json; p=ProfilerConfig(**json.loads(" + repr(json.dumps(PROFILER)) +
                 ")); print(p.profiler)"]
        run(rank, check, timeout=180)
    stages = [{"name": f"profile-{a}", "arm": a, "kind": "profile_clone", "mode": "timing",
               "trial": 0, "port": port, "servers": servers, "client_timeout_seconds": 1200}
              for a, port, servers in (("A", 8048, servers_a), ("B", 8046, servers_b))]
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted([*root.glob("*.py"), *root.glob("*.json")])}
    write(root / "prepared.json", {
        "status": "passed", "originals": originals, "stages": stages,
        "client_script": "c1_profile_client.py",
        "owned": servers_a + servers_b, "files": files, "outage_seconds": 5400,
        "scope": "Two native trace deployments; code/chat C1 only, no tuning or quality run",
        "profiling": PROFILER, "serving_policy_changed": False,
        "quality_qualified": False, "speed_qualified": False, "promotion": "None",
        "final_action": "Restore exact original production and leave nodes powered on",
        "previous_final_preparation_sha256": hashlib.sha256(
            (previous / "prepared.json").read_bytes()).hexdigest(),
    })
    print(json.dumps({"status": "prepared", "root": str(root), "stages": 2}))


def main() -> None:
    """Prepare on Spark1 or create an unstarted production clone on either rank."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--clone")
    parser.add_argument("--name")
    args = parser.parse_args()
    if args.clone:
        if not args.name:
            parser.error("Clone requires a name")
        print(json.dumps(clone(args.clone, args.name, args.root / "traces-A")))
    else:
        if args.previous is None:
            parser.error("Preparation requires the previous frozen run")
        prepare(args.root, args.previous)


if __name__ == "__main__":
    main()
