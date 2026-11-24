"""Compare the frozen compressed target with and without native depth-two MTP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import run_graph_quality as graph


def launch(rank: int, revision: str, label: str, original: dict[str, Any],
           vocabulary: int, manifest_sha: str) -> tuple[list[str], dict[str, Any]]:
    """Keep target precision/storage/transport identical; add only native MTP to the second arm."""
    if label not in ("control", "candidate"):
        raise ValueError("Unknown MTP pilot arm")
    argv, meta = graph.launch(rank, revision, "candidate", original, vocabulary, manifest_sha)
    stage = "raw" if label == "control" else "aligned"
    name = f"mxwave-glm53-full-{revision}-{stage}-r{rank}"
    argv[argv.index("--name") + 1] = name
    meta["name"] = name
    depth = 0 if label == "control" else 2
    for command in (argv, meta["command"]):
        command[command.index("--worker-extension-cls") + 1] = "mtp_pilot_worker.MtpPilotWorker"
        if depth:
            command.extend(["--speculative-config", json.dumps({
                "method": "mtp", "num_speculative_tokens": depth,
                "use_local_argmax_reduction": True})])
    extra = {"MXWAVE_MTP_DEPTH": str(depth), "MXWAVE_MTP_PILOT": "1"}
    meta["environment"].update(extra)
    for key, value in extra.items():
        at = argv.index("--entrypoint")
        argv[at:at] = ["-e", key + "=" + value]
    old = graph.quality.c.ROOT / f"run-{revision}/kda"
    new = graph.quality.c.ROOT / f"run-{revision}/{stage}"
    mounts = []
    for source, target, access in meta["mounts"]:
        if target in ("/cache", "/traces"):
            replacement = str(new / target.lstrip("/"))
            graph.quality.c.run(rank, ["mkdir", "-p", replacement])
            index = argv.index(f"{source}:{target}:{access}")
            argv[index] = f"{replacement}:{target}:{access}"
            source = replacement
        mounts.append((source, target, access))
    meta["mounts"] = mounts
    meta["pilot"] = {"depth": depth, "target": "same compressed paired-speed candidate",
                     "separate_arm_cache_and_traces": True, "old_cache_root": str(old)}
    return argv, meta


def configure() -> None:
    """Reuse the existing idle guard, source binding and independent restoration watchdog."""
    graph.configure()
    quality = graph.quality
    quality.launch = launch
    quality.PREPARED_NAME = "mtp-pilot-v2-prepared.json"
    quality.CLIENT_SCRIPT = "measure_mtp_pilot.py"
    quality.ANALYZE_SCRIPT = "analyze_mtp_pilot.py"
    quality.CLIENT_WAIT_SECONDS = 1200
    quality.OUTAGE_SECONDS = 5400
    quality.CLIENT_MEMORY = "4g"
    quality.SANDBOX_TASKS = False
    quality.ENTRY_SCRIPT = Path(__file__).resolve()


if __name__ == "__main__":
    configure()
    graph.quality.main()
