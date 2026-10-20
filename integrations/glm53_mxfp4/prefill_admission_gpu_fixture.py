"""Verify that metadata observation preserves eager outputs and excludes captured graph replays."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from prefill_weight_audit import WeightAudit


def main() -> None:
    """Use tiny native BF16 GPU linears; this fixture does not qualify the target KDA kernel."""
    torch.manual_seed(42)
    runtime = SimpleNamespace(apply=lambda layer, x: x @ layer.weight.T, FUSED_MAX_TOKENS=8)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(32, 32, bias=False,
                device="cuda", dtype=torch.bfloat16) for _ in range(2)])
            for i, layer in enumerate(self.layers):
                layer.mxwave_kda_qualification = {"prefix": f"fixture.{i}"}
                layer.mxwave_kda_metadata = (32, 32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = runtime.apply(layer, x)
            return x

    model = Model().eval()
    x = torch.randn((16, 32), dtype=torch.bfloat16, device="cuda")
    with torch.inference_mode():
        expected = model(x)
        observer = WeightAudit(model, runtime, capture=torch.cuda.is_current_stream_capturing,
                               compiling=torch.compiler.is_compiling)
        observer.start()
        actual = model(x)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        small = x[:4].clone()
        expected_small = model(small)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                model(small)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        before = len(observer.events)
        with torch.cuda.graph(graph):
            captured = model(small)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured.view(torch.int16), expected_small.view(torch.int16))
        assert len(observer.events) == before
        result = observer.stop()
        assert all(len(e["calls"]) == 2 and not e["duplicate_weight_prefixes"]
                   for e in result["events"])
        assert all(r["mode"] == "reconstruction" for r in result["events"][0]["calls"])
        assert torch.equal(model(x).view(torch.int16), expected.view(torch.int16))
    print(json.dumps({"status": "passed", "tiny_eager_outputs_bits_exact": True,
        "tiny_graph_outputs_bits_exact": True, "graph_capture_and_replays_excluded": True,
        "observer_methods_restored": True, "target_kernel_qualification": False}))
    print("MXWAVE_DECODE_FIXTURE_PASSED")


if __name__ == "__main__":
    main()
