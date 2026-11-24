"""Reproduce counted sparse-prefix ordering with synthetic indices and no model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_sparse_conversion import NATIVE_SHA256, convert_stable


def main() -> None:
    """Compare native and padded mapping membership/order across real CUDA graph replays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", choices=(1, 2, 4, 8, 16), type=int, default=[1, 8, 16])
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    if args.runs < 2:
        raise ValueError("Require at least two graph replays")
    import vllm.v1.attention.backends.mla.sparse_utils as helper

    actual = hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest()
    if actual != NATIVE_SHA256:
        raise ValueError("This reproducer targets the recorded native helper source")
    native = helper.triton_convert_req_index_to_global_index
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    output = {"native_source_sha256": actual, "rows": [], "model_weights_required": False}
    for count in args.rows:
        requests = torch.arange(count, dtype=torch.int32, device="cuda")
        table = (1 + torch.arange(144, dtype=torch.int32, device="cuda")).repeat(count, 1)
        indices = torch.full((count, 2176), -1, dtype=torch.int32, device="cuda")
        columns = (torch.arange(259, device="cuda") * 127) % 2176
        indices[:, columns] = torch.arange(259, dtype=torch.int32, device="cuda") % 258
        expected = []
        for row in indices.cpu().numpy():
            selected = row[row >= 0]
            expected.append(((1 + selected // 64) * 64 + selected % 64).astype(np.int32))
        modes = {}
        for mode in ("native", "stable"):
            def operation(mode=mode, requests=requests, table=table, indices=indices):
                kwargs = {"BLOCK_SIZE": 64, "NUM_TOPK_TOKENS": 2176, "return_valid_counts": True}
                if mode == "native":
                    return native(requests, table, indices, **kwargs)
                return convert_stable(native, requests, table, indices, **kwargs)

            for _ in range(3):
                operation()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                slots, valid = operation()
            hashes = []
            for _ in range(args.runs):
                graph.replay()
                torch.cuda.synchronize()
                result = slots.cpu().numpy()
                assert np.array_equal(valid.cpu().numpy(), np.full(count, 259))
                for row, values in zip(result, expected, strict=True):
                    assert np.array_equal(np.sort(row[:259]), np.sort(values))
                    assert np.all(row[259:] == -1)
                    if mode == "stable":
                        assert np.array_equal(row[:259], values)
                hashes.append(hashlib.sha256(result.tobytes()).hexdigest())
            modes[mode] = {"distinct_outputs": len(set(hashes)), "replays": args.runs,
                           "membership_counts_and_invalid_tails_exact": True}
            if mode == "stable":
                assert len(set(hashes)) == 1
        output["rows"].append({"count": count, "modes": modes})
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
