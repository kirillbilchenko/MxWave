"""Freeze the user's final reduced comparison from the validated broad preparation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from compare_client import validate_quality_protocol
from control_stock_trial import ORIGINAL, TERMINAL, identity, inspect, run, write


def reduced_inputs(workloads: dict[str, Any], protocol: dict[str, Any],
                   repeat_prompt: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select fixed workloads and an answer-independent 300-question sample."""
    workloads, protocol = copy.deepcopy(workloads), copy.deepcopy(protocol)
    validate_quality_protocol(protocol)
    if protocol["questions_per_arm"] != 1319:
        raise ValueError("Derive the screen from the full frozen benchmark")
    wanted = {f"{domain}-short-c{c}" for domain in ("code", "chat") for c in (1, 8)}
    workloads["cells"] = [cell for cell in workloads["cells"] if cell["name"] in wanted]
    if len(workloads["cells"]) != 4 or {c["name"] for c in workloads["cells"]} != wanted:
        raise ValueError("Require all four short code/chat C1/C8 cells exactly once")
    if not 3900 <= len(repeat_prompt) <= 4600:
        raise ValueError("Require the frozen 4k-class repeat prompt")
    workloads.update(salt="glm53-stock-quick-v1", long_context=[],
                     production_repeat_prompt_ids=repeat_prompt,
                     repeat_scope="One production self-repeat screen; no full-vocabulary KL")
    workloads["banks"] = {domain: {"short": workloads["banks"][domain]["short"]}
                          for domain in ("code", "chat")}
    indices = sorted(sorted(range(1319), key=lambda i: hashlib.sha256(
        f"glm53-quick-quality-v1:{i}".encode()).digest())[:300])
    lookup = {case["source_index"]: case for case in protocol["cases"]}
    protocol.update(cases=[lookup[i] for i in indices], questions_per_arm=300,
                    selection="sha256-source-index-v1", selected_source_indices=indices,
                    cache_salt_prefix="glm53-stock-quick-gsm8k-v1",
                    interpretation="Screening subset, not a full GSM8K benchmark or promotion gate")
    validate_quality_protocol(protocol)
    return workloads, protocol


def prepare(root: Path, predecessor: Path, repeat_plan: Path) -> None:
    """Reuse immutable validated launches; create fresh results, caches and quality clones."""
    if (root / "prepared.json").exists() or root == predecessor:
        raise FileExistsError("Require a new preparation directory")
    if not root.name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in root.name):
        raise ValueError("Require a Docker-safe directory name")
    old = json.loads((predecessor / "prepared.json").read_bytes())
    state = json.loads((predecessor / "controller.json").read_bytes())
    if state["phase"] not in TERMINAL or state["service_paused"]:
        raise RuntimeError("Wait for the broad comparison to finish exact restoration")
    for name, digest in old["files"].items():
        if hashlib.sha256((predecessor / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Predecessor preparation changed: {name}")
    for rank in (0, 1):
        if identity(inspect(rank, ORIGINAL[rank])) != old["originals"][rank]:
            raise RuntimeError("Production identity changed")
    repeat = json.loads(repeat_plan.read_bytes())
    prompt = next(case["token_ids"] for case in repeat["cases"] if case["name"] == "short")
    workloads, protocol = reduced_inputs(
        json.loads((predecessor / "workloads.json").read_bytes()),
        json.loads((predecessor / "gsm8k-protocol.json").read_bytes()), prompt)
    write(root / "workloads.json", workloads)
    write(root / "gsm8k-protocol.json", protocol)
    inherited = ["tokenizer-identities.json", "image-selection.json",
                 *[f"cpu-preflight-rank{r}.json" for r in (0, 1)],
                 *[f"launch-{mode}-rank{r}.json" for mode in ("timing", "gsm8k") for r in (0, 1)]]
    for name in inherited:
        shutil.copyfile(predecessor / name, root / name)
    run(1, ["mkdir", "-p", str(root), str(root / "reports")])
    (root / "reports").mkdir(exist_ok=True)
    subprocess.run(["scp", "-i", "/home/kirya/.ssh/id_glm_cluster",
                    *map(str, root.glob("*.py")), *[str(root / name) for name in inherited],
                    "kirya@192.168.192.2:" + str(root) + "/"], check=True, timeout=60)

    clones = []
    for rank in (0, 1):
        name = f"mxwave-glm53-{root.name}-quality-a-r{rank}"
        result = json.loads(run(rank, ["python3", "-S", str(root / "prepare_quality_clone.py"),
                                      "--original", ORIGINAL[rank], "--name", name]))
        write(root / f"quality-clone-rank{rank}.json", result)
        clones.append({"rank": rank, "name": name, "argv": ["docker", "start", result["id"]]})

    stages, owned = [], list(clones)
    for stage in old["stages"]:
        if not (stage["mode"] == "gsm8k"
                or stage["mode"] == "timing" and stage["trial"] < 3):
            continue
        stage = copy.deepcopy(stage)
        if stage["arm"] == "B":
            for server in stage["servers"]:
                server["name"] = server["name"].replace(predecessor.name, root.name)
                server["argv"] = [part.replace(str(predecessor), str(root)).replace(
                    "mxwave-glm53-" + predecessor.name, "mxwave-glm53-" + root.name)
                    for part in server["argv"]]
            owned.extend(stage["servers"])
        elif stage["mode"] == "gsm8k":
            stage["servers"] = clones
        stage["client_timeout_seconds"] = 1800 if stage["mode"] == "timing" else 3600
        stages.append(stage)
    if len(stages) != 8:
        raise RuntimeError("Require exactly six timing arms and two quality arms")
    # Reuse only closed compilation caches; launch JSONs and image IDs remain identical.
    candidate = next(stage for stage in stages if stage["arm"] == "B")
    for rank, server in enumerate(candidate["servers"]):
        image = next(part for part in server["argv"] if part.startswith("sha256:"))
        run(rank, ["mkdir", "-p", str(root / "cache")])
        run(rank, ["docker", "run", "--rm", "--network", "none",
                   "-v", str(predecessor / "cache") + ":/seed:ro",
                   "-v", str(root / "cache") + ":/cache:rw", "--entrypoint", "python3",
                   image, "-S", "-c",
                   "import shutil; shutil.copytree('/seed', '/cache', dirs_exist_ok=True)"],
            timeout=180)
    evidence = {"predecessor": str(predecessor),
                "prepared_sha256": hashlib.sha256((predecessor / "prepared.json").read_bytes())
                .hexdigest(), "inherited_files": {name: hashlib.sha256((root / name).read_bytes())
                .hexdigest() for name in inherited},
                "compatibility_result_sha256": hashlib.sha256(
                    (predecessor / "compatibility-B/result.json").read_bytes()).hexdigest(),
                "reuse": "Same verified launch commands, immutable images and corrected drafter; "
                         "only scope, result directory, cache salts and owned container names change",
                "sparse_investigation": "Cancelled by user; no additional GPU diagnostics"}
    write(root / "preparation-provenance.json", evidence)
    selected = [*root.glob("*.py"), *root.glob("*.json")]
    plan = {**old, "stages": stages, "owned": owned, "outage_seconds": 10800,
            "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(selected)},
            "full_gsm8k_questions_per_arm": None, "quality_questions_per_arm": 300,
            "timing_trials_per_arm": 3, "quality_scope": "Frozen answer-independent subset",
            "timing_failure_policy": "Retain partial results, skip later timing if invalid, "
                                     "attempt both 300-question arms within the same three-hour lease",
            "final_action": "Restore exact production, verify backups, then power off both Sparks",
            "promotion": "None"}
    write(root / "prepared.json", plan)
    print(json.dumps({"status": "prepared", "stages": len(stages), "lease_hours": 3,
                      "questions_per_arm": 300, "cells_per_timing_arm": 4,
                      "prepared_sha256": hashlib.sha256((root / "prepared.json").read_bytes())
                      .hexdigest()}))


def main() -> None:
    """Prepare a fresh reduced comparison without allocating GPUs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--repeat-plan", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.predecessor, args.repeat_plan)


if __name__ == "__main__":
    main()
