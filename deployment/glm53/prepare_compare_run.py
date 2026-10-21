"""Prepare the two-arm comparison without stopping production or allocating GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from control_stock_trial import ORIGINAL, RELEASE, identity, inspect, run, write


def remove_pair(arguments: list[str], flag: str) -> None:
    """Remove exactly one explicit option/value pair."""
    index = arguments.index(flag)
    del arguments[index:index + 2]


def prepare(root: Path, *, images_path: Path | None = None, drafter: Path | None = None) -> None:
    """Freeze CPU-validated commands, private quality clones and every input hash."""
    if (root / "prepared.json").exists():
        raise FileExistsError("Never replace an existing comparison preparation")
    predecessor = json.loads((RELEASE / "stock-trial-v7/prepared.json").read_bytes())
    selected_images = predecessor["images"]
    if images_path is not None:
        image_record = json.loads(images_path.read_bytes())
        selected_images = image_record["images"]
        if (len(selected_images) != 2
                or any(not image.startswith("sha256:") for image in selected_images)):
            raise ValueError("Require two immutable candidate image IDs")
        shutil.copyfile(images_path, root / "image-selection.json")
    if not root.name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in root.name):
        raise ValueError("Experiment directory needs a Docker-safe lowercase name")
    container_prefix = "mxwave-glm53-" + root.name
    originals = [identity(inspect(rank, ORIGINAL[rank])) for rank in (0, 1)]
    for rank in (0, 1):
        if originals[rank] != predecessor["originals"][rank]:
            raise RuntimeError("Original production changed")
    remote_files = [str(p) for p in root.glob("*.py")]
    run(1, ["mkdir", "-p", str(root), str(root / "reports")])
    subprocess.run(["scp", "-i", "/home/kirya/.ssh/id_glm_cluster", *remote_files,
                    "kirya@192.168.192.2:" + str(root) + "/"], check=True, timeout=60)
    (root / "reports").mkdir(exist_ok=True)
    templates = []
    for rank, server in enumerate(predecessor["servers"]):
        old_image, image = predecessor["images"][rank], selected_images[rank]
        argv = list(server["argv"])
        image_index = argv.index(old_image)
        prefix = [part.replace(str(RELEASE / "stock-trial-v7"), str(root))
                  for part in argv[:image_index]]
        remove_pair(prefix, "--name")
        if drafter is not None:
            mounts = [i for i, part in enumerate(prefix) if part.endswith(":/drafter:ro")]
            if len(mounts) != 1:
                raise RuntimeError("Require exactly one read-only drafter mount")
            prefix[mounts[0]] = str(drafter) + ":/drafter:ro"
        prefix += ["-v", str(root) + ":/trial:ro", "--entrypoint", "python3"]
        templates.append((prefix, image))
        # Caches are isolated from prior retained trials and reused only sequentially.
        run(rank, ["mkdir", "-p", str(root / "cache")])
        run(rank, ["docker", "run", "--rm", "--network", "none",
                   "-v", str(RELEASE / "stock-trial-v7/cache") + ":/seed:ro",
                   "-v", str(root / "cache") + ":/cache:rw", "--entrypoint", "python3",
                   image, "-S", "-c",
                   "import shutil; shutil.copytree('/seed', '/cache', dirs_exist_ok=True)"],
            timeout=180)
        cpu = list(prefix)
        cpu.remove("-d")
        cpu += ["--rm", "--cpus", "4"]
        remove_pair(cpu, "--gpus")
        cpu[cpu.index("--network") + 1] = "none"
        for flag in ("--memory", "--memory-swap"):
            cpu[cpu.index(flag) + 1] = "12g"
        cpu = [part.replace(":/trial:ro", ":/trial:rw") for part in cpu]
        output = run(rank, [*cpu, image, "/trial/compare_preflight.py", str(rank)], timeout=240)
        (root / f"cpu-preflight-rank{rank}.log").write_text(output)
        if rank == 1:
            for name in ("cpu-preflight-rank1.json", "launch-timing-rank1.json",
                         "launch-gsm8k-rank1.json"):
                (root / name).write_text(run(1, ["cat", str(root / name)]))
    clones = []
    for rank in (0, 1):
        name = f"{container_prefix}-quality-a-r{rank}"
        result = json.loads(run(rank, ["python3", "-S", str(root / "prepare_quality_clone.py"),
                                      "--original", ORIGINAL[rank], "--name", name]))
        write(root / f"quality-clone-rank{rank}.json", result)
        clones.append({"rank": rank, "name": name, "argv": ["docker", "start", result["id"]]})

    def candidate(stage: str, mode: str) -> list[dict[str, Any]]:
        result = []
        for rank, (prefix, image) in enumerate(templates):
            name = f"{container_prefix}-{stage}-r{rank}"
            result.append({"rank": rank, "name": name,
                           "argv": [*prefix, "--name", name, image,
                                    "/trial/compare_boot.py", mode, str(rank), stage]})
        return result

    production = [{"rank": rank, "name": inspect(rank, ORIGINAL[rank])["Name"].lstrip("/")}
                  for rank in (0, 1)]
    stages = []
    owned = list(clones)
    compatibility = candidate("compatibility", "timing")
    owned += compatibility
    stages.append({"name": "compatibility-B", "mode": "compatibility", "arm": "B",
                   "kind": "candidate", "port": 8046, "servers": compatibility,
                   "client_timeout_seconds": 600})
    for trial in range(6):
        for arm in ("A", "B"):
            name = f"timing-{arm}-{trial}"
            servers = production if arm == "A" else candidate(f"b{trial}", "timing")
            if arm == "B":
                owned += servers
            stages.append({"name": name, "mode": "timing", "arm": arm, "trial": trial,
                           "kind": "production" if arm == "A" else "candidate",
                           "port": 8000 if arm == "A" else 8046, "servers": servers,
                           "client_timeout_seconds": 3600})
    for arm in ("A", "B"):
        servers = clones if arm == "A" else candidate("quality-b", "gsm8k")
        if arm == "B":
            owned += servers
        stages.append({"name": f"gsm8k-{arm}", "mode": "gsm8k", "arm": arm,
                       "kind": "quality", "port": 8047 if arm == "A" else 8046,
                       "servers": servers, "client_timeout_seconds": 7200})
    inputs = ("workloads.json", "gsm8k-protocol.json", "tokenizer-identities.json")
    for name in inputs:
        if not (root / name).exists():
            raise RuntimeError(f"Missing frozen input: {name}")
    for rank in (0, 1):
        check = json.loads((root / f"cpu-preflight-rank{rank}.json").read_bytes())
        if check["status"] != "passed" or check["gpu_used"]:
            raise RuntimeError("Both no-GPU preflights must pass")
    selected = [*root.glob("*.py"), *(root / name for name in inputs),
                *root.glob("launch-*.json"), *root.glob("cpu-preflight-*.json"),
                *root.glob("image-selection.json")]
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(selected)}
    plan = {"status": "passed", "originals": originals, "stages": stages, "owned": owned,
            "outage_seconds": 36000, "files": files, "speculative_quality": False,
            "full_gsm8k_questions_per_arm": 1319, "qualification_passed": False,
            "timing_failure_policy": "Retain failed/partial timing, skip remaining timing, "
                                     "attempt both independent full GSM8K arms, restore originals",
            "cache_control": "Unique per-request salts; same salts in both arms, warmup separate; "
                             "verify positive same-salt reuse and zero fresh-salt hits on GPU",
            "known_failures": ["stock self-repeat KL", "prefix output identity"],
            "drafter_correction": "Pinned config-only BF16 conv exclusions; weight bytes unchanged",
            "compatibility_policy": "Check 500k, actual allocation, positive draft acceptance and "
                "prefix reuse before timing; retain output identity failures as failed quality gates",
            "missing_production_optimizations": predecessor["missing_production_optimizations"],
            "comparison": "Complete deployments; unavailable production optimizations disclosed"}
    write(root / "prepared.json", plan)
    print(json.dumps({"status": "prepared", "stages": len(stages), "lease_hours": 10,
                      "prepared_sha256": hashlib.sha256((root / "prepared.json").read_bytes())
                      .hexdigest()}))


def main() -> None:
    """Prepare one explicitly named private experiment directory on Spark1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--drafter", type=Path)
    args = parser.parse_args()
    prepare(args.root, images_path=args.images, drafter=args.drafter)


if __name__ == "__main__":
    main()
