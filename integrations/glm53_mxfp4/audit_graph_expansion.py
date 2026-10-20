"""Audit pinned native candidate construction and dispatch on CPU without capturing graphs."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from graph_capture_policy import candidates_for_sizes

NATIVE_SHA = "c183937e6eb5b9c28c79d98fb4c64f562e7649d5f6d65743e6640b2f378ecf9f"


def main() -> None:
    """Replace only PP/pool discovery; native candidate/dispatch methods execute unchanged."""
    import torch
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu import cudagraph_utils as native

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assert not args.report.exists()
    assert hashlib.sha256(Path(inspect.getfile(native)).read_bytes()).hexdigest() == NATIVE_SHA
    original_pp, original_platform = native.get_pp_group, native.current_platform
    cases = []
    try:
        native.get_pp_group = lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True)
        native.current_platform = SimpleNamespace(get_global_graph_pool=lambda: None)
        for captures, max_seqs in (([1, 4], 8), ([1, 2, 4, 8], 8),
                                   ([1, 2, 4, 8, 16], 8), ([1, 2, 4, 8, 16], 16)):
            config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=max_seqs),
                compilation_config=SimpleNamespace(cudagraph_capture_sizes=captures,
                                                   max_cudagraph_capture_size=max(captures)),
                parallel_config=SimpleNamespace(data_parallel_size=1, tensor_parallel_size=2),
                speculative_config=None)
            manager = native.CudaGraphManager(config, torch.device("cpu"),
                CUDAGraphMode.FULL_DECODE_ONLY, decode_query_len=1)
            manager._graphs_captured = True
            values = {}
            for policy, allowed in (("native", set(captures)), ("baseline_subset", {1, 4})):
                full = manager._candidates
                manager._candidates = candidates_for_sizes(manager, allowed)
                rows = {}
                for n in (1, 2, 3, 4, 8, 16):
                    d = manager.dispatch(n, n, uniform_token_count=1, num_active_loras=0)
                    rows[str(n)] = {"mode": str(d.cg_mode), "tokens": d.num_tokens,
                                    "requests": d.num_reqs}
                values[policy] = rows
                manager._candidates = full
            assert values["baseline_subset"]["2"] == {"mode": "FULL", "tokens": 4, "requests": 4}
            assert values["baseline_subset"]["8"]["mode"] == "NONE"
            if 8 in captures:
                assert values["native"]["2"]["tokens"] == 2
                assert values["native"]["8"]["mode"] == "FULL"
            assert values["native"]["16"]["mode"] == ("FULL" if max_seqs == 16 else "NONE")
            cases.append({"capture_sizes": captures, "max_num_seqs": max_seqs, "dispatch": values})
    finally:
        native.get_pp_group, native.current_platform = original_pp, original_platform
    assert not torch.cuda.is_initialized()
    result = {"status": "passed", "cuda_initialized": False, "model_constructed": False,
        "native_candidate_builder_and_dispatch_called": True,
        "native_module_sha256": NATIVE_SHA, "graph_pool_and_PP_discovery_substituted": True,
        "cases": cases, "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("graph_capture_policy.py"))},
        "limits": ["No CUDA graphs are captured/replayed and no GPU memory is measured.",
                   "Native descriptor policy does not establish numerical correctness or speed.",
                   "The max-seqs-16 case tests descriptor eligibility only, not cache/model capacity."]}
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}))


if __name__ == "__main__":
    main()
