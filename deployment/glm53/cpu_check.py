"""Check the installed wheel in an isolated official image; never allocate a GPU."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path


def main() -> None:
    """Validate upstream imports and plugin installation without loading model weights."""
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("Run this check without GPU access")
    import argparse

    from mxwave_glm53.cli import doctor, launch_command
    from mxwave_glm53.plugin import check_upstream

    checked = doctor(Path("/release/runtime"))
    if checked["status"] != "passed":
        raise RuntimeError(checked)
    source_hashes = check_upstream()
    from vllm import platforms
    from vllm.platforms.cpu import CpuPlatform

    platforms.current_platform = CpuPlatform()
    os.environ.update({
        "MXWAVE_GLM53_ENABLE": "1", "MXWAVE_GLM53_BUNDLE": "/release/runtime",
        "MXWAVE_GLM53_MODEL": "/source/source",
        "MXWAVE_GLM53_DRAFTER": "/source/drafter-pretrained-modal-v1-modal_fp8-checkpoint",
        "MXWAVE_GLM53_REPORTS": "/release/cpu-reports", "VLLM_PLUGINS": "mxwave_glm53",
    })
    entries = importlib.metadata.entry_points(group="vllm.general_plugins", name="mxwave_glm53")
    if len(entries) != 1:
        raise RuntimeError("Missing or duplicated installed plugin entry point")
    register = next(iter(entries)).load()
    register()
    register()
    launch = argparse.Namespace(
        bundle=Path("/release/runtime"), model=Path("/source/source"),
        drafter=Path("/source/drafter-pretrained-modal-v1-modal_fp8-checkpoint"),
        reports=Path("/release/cpu-reports"), node_rank=0, master_addr="192.168.192.1",
        master_port=29719, port=8046, host="127.0.0.1", max_model_len=8192, kv_cache_gib=6,
    )
    command, environment = launch_command(launch)
    from vllm.entrypoints.cli.serve import cmd_init
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    sub = parser.add_subparsers(dest="subparser")
    commands = cmd_init()
    for item in commands:
        item.subparser_init(sub)
    parsed = parser.parse_args(command[3:])
    next(item for item in commands if item.name == parsed.subparser).validate(parsed)
    if parsed.speculative_config["num_speculative_tokens"] != 5:
        raise RuntimeError("Selected native draft depth changed")
    launch.max_model_len = 500000
    launch.max_num_seqs = 32
    launch.max_num_batched_tokens = 6919
    launch.prefix_caching = True
    launch.graph_sizes = [1, 2, 4, 6, 8, 12, 16, 18, 24, 32]
    launch.vocab_mode = "embedding"
    capacity_command, capacity_environment = launch_command(launch)
    capacity_args = parser.parse_args(capacity_command[3:])
    next(item for item in commands if item.name == capacity_args.subparser).validate(capacity_args)
    assert capacity_args.enable_prefix_caching and capacity_args.max_model_len == 500000
    assert capacity_args.max_num_seqs == 32 and capacity_args.max_num_batched_tokens == 6919
    assert capacity_environment["MXWAVE_GLM53_VOCAB_MODE"] == "embedding"
    from vllm.v1.worker.gpu_worker import Worker

    from mxwave_glm53._runtime.kda_mxfp8_runtime import method_type
    from mxwave_glm53._runtime.model_plugin import _INSTALLED
    from mxwave_glm53._runtime.vocab_adapter import VocabMethod

    if not _INSTALLED or not getattr(Worker.load_model, "_mxwave_glm53", False):
        raise RuntimeError("Runtime startup hooks were not installed")
    result = {
        "status": "passed", "scope": "CPU import/entrypoint/API check only",
        "gpu_used": False, "installation": checked, "upstream_source_hashes": source_hashes,
        "kda_method": method_type().__name__, "vocabulary_method": VocabMethod.__name__,
        "cli_validated": True, "launch_command": command, "launch_environment": environment,
        "capacity_cli_validated": True, "capacity_command": capacity_command,
        "capacity_environment": capacity_environment,
        "gpu_model_qualification": "pending",
    }
    Path("/release/cpu-check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
