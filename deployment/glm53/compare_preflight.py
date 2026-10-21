"""Validate both native candidate launches and cache-salt handling without GPUs."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path


def preflight(rank: int) -> None:
    """Freeze serving and speculation-off commands against the installed stock parser."""
    import torch
    from vllm import platforms
    from vllm.entrypoints.cli.serve import cmd_init
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.platforms.cpu import CpuPlatform
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.core import kv_cache_utils

    from mxwave_glm53.cli import doctor, launch_command

    if torch.cuda.is_available():
        raise RuntimeError("Preparation must run without GPU access")
    platforms.current_platform = CpuPlatform()
    installation = doctor(Path("/runtime"))
    if installation["status"] != "passed":
        raise RuntimeError("Pinned installation check failed")
    args = argparse.Namespace(
        bundle=Path("/runtime"), model=Path("/model"), drafter=Path("/drafter"),
        reports=Path("/reports"), node_rank=rank, master_addr="192.168.192.1",
        master_port=29719, port=8046, host="127.0.0.1", max_model_len=500000,
        kv_cache_gib=6, max_num_seqs=32, max_num_batched_tokens=6919,
        graph_sizes=[1, 2, 4, 6, 8, 12, 16, 18, 24, 32], prefix_caching=True,
        vocab_mode="embedding", gpu_memory_utilization=.85, block_size=2304,
        load_format="runai_streamer", loader_memory_mib=3072, loader_concurrency=8,
        draft_attention_backend="TRITON_ATTN")
    command, env = launch_command(args)
    command += ["--chat-template", "/model/chat_template.jinja", "--enable-prompt-tokens-details"]
    env["VLLM_SERVER_DEV_MODE"] = "1"
    launches = {}
    for mode in ("timing", "gsm8k"):
        selected = list(command)
        if mode == "gsm8k":
            index = selected.index("--speculative-config")
            del selected[index:index + 2]
        parser = FlexibleArgumentParser()
        sub = parser.add_subparsers(dest="subparser")
        commands = cmd_init()
        for item in commands:
            item.subparser_init(sub)
        parsed = parser.parse_args(selected[3:])
        next(item for item in commands if item.name == parsed.subparser).validate(parsed)
        if (parsed.max_model_len != 500000 or parsed.max_num_seqs != 32
                or parsed.max_num_batched_tokens != 6919 or not parsed.enable_prefix_caching):
            raise RuntimeError("Capacity settings changed")
        if mode == "gsm8k" and parsed.speculative_config is not None:
            raise RuntimeError("Quality arm must not load a drafter")
        if mode == "timing" and parsed.speculative_config["num_speculative_tokens"] != 5:
            raise RuntimeError("Selected drafter policy changed")
        launch = {"command": selected, "environment": env}
        target = Path("/trial") / f"launch-{mode}-rank{rank}.json"
        target.write_text(json.dumps(launch, indent=2) + "\n")
        launches[mode] = hashlib.sha256(target.read_bytes()).hexdigest()
    if (not all("cache_salt" in cls.model_fields for cls in
                (ChatCompletionRequest, CompletionRequest))
            or "request.cache_salt" not in inspect.getsource(kv_cache_utils)):
        raise RuntimeError("Native cache-salt protocol/hash contract is unavailable")
    receipt = {"status": "passed", "gpu_used": False, "rank": rank, "launches": launches,
               "installation": installation, "cache_salt_protocol_and_hashing_verified": True}
    (Path("/trial") / f"cpu-preflight-rank{rank}.json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "passed", "rank": rank, "gpu_used": False}))


if __name__ == "__main__":
    preflight(int(sys.argv[1]))
