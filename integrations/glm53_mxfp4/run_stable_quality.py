"""Matched quality evaluation under the component-qualified stable-dispatch experiment."""

from __future__ import annotations

from typing import Any

import run_quality_model as quality

_launch = quality.launch


def launch(*args: Any, **kwargs: Any) -> tuple[list[str], dict[str, Any]]:
    """Use identical canonical dispatch and workspace settings in both reference/candidate."""
    argv, meta = _launch(*args, **kwargs)
    for command in (argv, meta["command"]):
        command[command.index("--worker-extension-cls") + 1] = "stable_dispatch.StableWorker"
    for key, value in {
        "MXWAVE_STABLE_MOE_DISPATCH": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }.items():
        meta["environment"][key] = value
        at = argv.index("--entrypoint")
        argv[at:at] = ["-e", key + "=" + value]
    return argv, meta


if __name__ == "__main__":
    quality.launch = launch
    quality.main()
