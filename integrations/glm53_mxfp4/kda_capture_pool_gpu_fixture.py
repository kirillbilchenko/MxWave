"""Exercise real CUDA startup/capture stream lifetimes with tiny scratch writes and native GEMM."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import kda_capture_pool
import kda_runtime
import torch


def main() -> None:
    """Reproduce the third-stream guard, then keep two disjoint slots across capture and serving."""
    torch.set_num_threads(1)
    torch.manual_seed(70316)
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("Require the same SM121 worker architecture")
    properties = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(256 * 1024**2 / properties.total_memory)
    device = torch.device("cuda:0")
    pool = kda_runtime.pool_for(device, 32 * 32)
    weights = [torch.randn((32, 32), dtype=torch.bfloat16, device=device) for _ in range(2)]
    x = torch.randn((16, 32), dtype=torch.bfloat16, device=device)
    expected = [torch.nn.functional.linear(x, weight) for weight in weights]
    torch.cuda.synchronize()
    stream_b, stream_c, stream_d = [torch.cuda.Stream() for _ in range(3)]

    def operation(index: int) -> tuple[torch.Tensor, int]:
        slot, lock = kda_runtime._slot(pool)
        with lock:
            scratch = pool[slot].view(32, 32)
            scratch.copy_(weights[index])
            return torch.nn.functional.linear(x, scratch), slot

    first, _ = operation(0)
    with torch.cuda.stream(stream_b):
        second, _ = operation(1)
    torch.cuda.synchronize()
    if not torch.equal(first.view(torch.uint8), expected[0].view(torch.uint8)) or not torch.equal(
            second.view(torch.uint8), expected[1].view(torch.uint8)):
        raise RuntimeError("Tiny eager scratch operation differs from its native GEMM reference")
    rejected = False
    with torch.cuda.stream(stream_c):
        try:
            kda_runtime._slot(pool)
        except ValueError as exc:
            if "two inference streams" not in str(exc):
                raise
            rejected = True
    if not rejected:
        raise RuntimeError("The original unretired three-stream failure did not reproduce")

    class Descriptor:
        num_tokens = 16

    class Manager:
        def capture(self) -> None:
            self.graphs = {}
            with torch.cuda.stream(stream_d):
                operation(0)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream_c):
                self.output, self.slot = operation(0)
            self.graphs[Descriptor()] = self.graph

    # The real native manager is source-bound by install(); this tiny manager
    # substitutes model input construction while keeping actual CUDA streams.
    kda_capture_pool.patch(Manager, kda_runtime)
    manager = Manager()
    manager.capture()
    # Capture records the scratch copy/GEMM; execute them before inspecting
    # whether later eager work preserves the graph-owned weight bytes.
    manager.graph.replay()
    torch.cuda.synchronize()
    output, eager_slot = operation(1)
    torch.cuda.synchronize()
    if manager.slot == eager_slot or not torch.equal(pool[manager.slot].view(32, 32).view(torch.uint8),
                                                    weights[0].view(torch.uint8)):
        raise RuntimeError("Retired warmup slot aliases or overwrote graph-owned storage")
    if not torch.equal(output.view(torch.uint8), expected[1].view(torch.uint8)):
        raise RuntimeError("Post-capture eager operation changed")
    for _ in range(5):
        manager.graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(manager.output.view(torch.uint8), expected[0].view(torch.uint8)):
            raise RuntimeError("Captured output differs after graph replay and eager reuse")
        operation(1)
    torch.cuda.synchronize()
    storage = kda_runtime.report()
    if (storage["pool_bytes"] != 4096 or storage["pool_streams"] != [2]
            or storage["captured_pool_streams"] != [1]):
        raise RuntimeError("Tiny fixture exceeded its two-slot ownership contract")
    result = {"status": "passed", "tiny_eager_outputs_bits_exact": True,
        "tiny_graph_outputs_bits_exact": True, "graph_replays_checked": 5,
        "initialization_graph_replays": 1,
        "third_unretired_stream_refused": True, "graph_and_eager_slots_distinct": True,
        "captured_weight_bytes_survive_eager_reuse": True, "pool_slots": 2,
        "tiny_pool_bytes": storage["pool_bytes"], "maximum_real_pool_bytes": 256 * 1024**2,
        "startup_transitions": kda_capture_pool.report()["transitions"],
        "target_kernel_qualification": False,
        "scope": "Actual CUDA streams, graph capture/replay and tiny BF16 GEMM; native compressed kernels are unchanged and not called here.",
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("kda_runtime.py", "kda_capture_pool.py", Path(__file__).name)}}
    print(json.dumps(result))
    print("MXWAVE_DECODE_FIXTURE_PASSED")


if __name__ == "__main__":
    main()
