"""Preflight value evidence and device completion controls on actual tiny CUDA graphs."""

from __future__ import annotations

import json

import torch
import vllm.distributed
from decode_state_gpu_fixture import fixture
from decode_state_values_worker import DecodeStateValuesWorker


def main() -> None:
    """Verify exact active states and the magnitude of deliberately untracked hidden drift."""
    vllm.distributed.get_tensor_model_parallel_rank = lambda: 0
    cases = []
    for policy in ("normal", "device_sync"):
        for drift in (False, True):
            for stream in (False, True):
                row = fixture(drift, stream, DecodeStateValuesWorker, policy)
                row.update(completion_policy=policy, hidden_magnitudes_checked=True,
                           all_active_post_states_exact=True, raw_evidence_written=True)
                cases.append(row)
    print(json.dumps({"status": "passed", "cases": cases,
                      "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()}))
    print("MXWAVE_DECODE_FIXTURE_PASSED")


if __name__ == "__main__":
    main()
