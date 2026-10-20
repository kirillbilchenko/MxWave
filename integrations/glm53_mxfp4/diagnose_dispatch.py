"""Qualify the exact diagnostic RPC on a tiny live engine from the pinned image."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from diagnose_mxfp8 import write


def arm(worker: Any) -> dict[str, Any]:
    """Count real graph replay calls without modifying model weights or graph contents."""
    manager = worker.model_runner.cudagraph_manager
    original = manager.run_fullgraph
    worker._mxwave_probe_replays = 0

    def run(descriptor: Any) -> Any:
        worker._mxwave_probe_replays += 1
        return original(descriptor)

    manager.run_fullgraph = run
    return {"runner": type(worker.model_runner).__module__ + "." +
                      type(worker.model_runner).__name__, "captured_graphs": len(manager.graphs)}


def count(worker: Any) -> int:
    """Return the number of actual full-graph replays in this private tiny engine."""
    return int(worker._mxwave_probe_replays)


def main() -> None:
    """Require FULL/NONE/FULL descriptors and observed replay/zero-replay/replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve prior tiny-engine qualification")
    state: dict[str, Any] = {"status": "running", "started_epoch": time.time(), "phases": [],
        "code_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("diagnostic_worker.py", "diagnose_checker.py", "diagnose_dispatch.py")},
        "scope": "Tiny TP1 Llama, actual pinned GPUModelRunner; not GLM quality or TP2 proof"}
    try:
        import diagnose_checker

        saved = sys.argv
        checker_path = args.report.parent / "checker-report.json"
        sys.argv = [saved[0], "--source", str(args.source), "--report", str(checker_path)]
        try:
            diagnose_checker.main()
        finally:
            sys.argv = saved
        state["checker"] = json.loads(checker_path.read_bytes())
        gc.collect()
        torch.cuda.empty_cache()
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from vllm import LLM, SamplingParams

        model = args.report.parent / "tiny-model"
        model.mkdir()
        config = {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": 64,
            "max_position_embeddings": 64, "rms_norm_eps": 1e-5,
            "hidden_act": "silu", "tie_word_embeddings": False, "dtype": "bfloat16"}
        (model / "config.json").write_text(json.dumps(config))
        write(args.report, state)
        llm = LLM(model=str(model), load_format="dummy", skip_tokenizer_init=True,
            dtype="bfloat16", max_model_len=64, max_num_seqs=2, max_num_batched_tokens=16,
            kv_cache_memory_bytes=16 * 1024**2, gpu_memory_utilization=.01,
            enable_prefix_caching=False, enable_chunked_prefill=True,
            enable_flashinfer_autotune=False,
            worker_extension_cls="diagnostic_worker.DiagnosticWorker",
            compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 2], "max_cudagraph_capture_size": 2})
        state["engine"] = llm.collective_rpc(arm)
        if state["engine"][0]["runner"] != "vllm.v1.worker.gpu.model_runner.GPUModelRunner":
            raise RuntimeError("Probe did not instantiate the actual full-model runner class")
        sampling = SamplingParams(temperature=0, seed=42, max_tokens=4, min_tokens=4,
                                  ignore_eos=True)
        for label, eager in (("graphs_before", False), ("eager", True),
                             ("graphs_after", False)):
            dispatch = llm.collective_rpc("mxwave_set_eager", args=(eager,))
            if any(row["status"] != "complete" for row in dispatch):
                raise RuntimeError(f"Tiny-engine switch failed: {dispatch}")
            before = llm.collective_rpc(count)[0]
            outputs = llm.generate([{"prompt_token_ids": [1, 2, 3, 4]}], sampling,
                                   use_tqdm=False)
            replays = llm.collective_rpc(count)[0] - before
            result = outputs[0].outputs[0]
            phase = {"label": label, "dispatch": dispatch, "actual_graph_replays": replays,
                     "token_ids": list(result.token_ids)}
            state["phases"].append(phase)
            write(args.report, state)
            if len(result.token_ids) != 4 or (replays == 0) != eager:
                raise RuntimeError("Observed execution did not follow graph/eager dispatch")
        if not all(phase["token_ids"] == state["phases"][0]["token_ids"]
                   for phase in state["phases"]):
            raise RuntimeError("Tiny-engine graph/eager greedy outputs changed")
        state.update(status="complete", passed=True, finished_epoch=time.time(),
                     cuda_peak_bytes=torch.cuda.max_memory_allocated())
        write(args.report, state)
    except BaseException as exc:
        state.update(status="failed", passed=False, error=repr(exc), finished_epoch=time.time())
        write(args.report, state)
        raise


if __name__ == "__main__":
    main()
