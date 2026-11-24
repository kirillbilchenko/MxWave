"""Check dynamic graph-written MLA observations with a tiny authored CUDA call chain."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import torch
from mla_internal_capture import MLAInternalCapture


def main() -> None:
    """Require exact replayed operands, planned KV bytes and removed hooks before a model probe."""
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 1):
        raise ValueError("Require the pinned Spark SM121 device")
    torch.cuda.set_per_process_memory_fraction(256 * 1024**2 / props.total_memory)
    device = torch.device("cuda", 0)
    q = torch.ones(4, 32, 2, dtype=torch.bfloat16, device=device)
    latent = torch.full((4, 512), 17, dtype=torch.uint8, device=device)
    rope = torch.empty(4, 32, 0, dtype=torch.bfloat16, device=device)
    slots = torch.arange(4, dtype=torch.int64, device=device)
    topk = torch.arange(8, dtype=torch.int32, device=device).view(4, 2)
    valid_counts = torch.full((4,), 2, dtype=torch.int32, device=device)
    requests = torch.arange(4, dtype=torch.int32, device=device)
    block_table = torch.zeros(4, 1, dtype=torch.int32, device=device)
    scale = torch.tensor(1., device=device)

    class Wrapper:
        def __init__(self) -> None:
            self._plan_info = [42, 99]
            self._kv_indices_buf = torch.zeros(8, dtype=torch.int32, device=device)
            self._qo_indptr_buf = torch.arange(5, dtype=torch.int32, device=device)
            self._kv_indptr_buf = self._qo_indptr_buf * 2
            self._kv_len_arr_buf = torch.full((4,), 2, dtype=torch.int32, device=device)

        def run(self, query: torch.Tensor, qpe: torch.Tensor, cache: torch.Tensor,
                kpe: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            return query * .5

    wrapper = Wrapper()

    def convert(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return topk.clone(), valid_counts.clone()

    backend = SimpleNamespace(
        _get_sm90_state=lambda: SimpleNamespace(wrapper=wrapper),
        triton_convert_req_index_to_global_index=convert)

    class Impl:
        dcp_world_size = 1

        def do_kv_cache_update(self, value: torch.Tensor, kpe: torch.Tensor,
                              cache: torch.Tensor, mapping: torch.Tensor,
                              dtype: str, kscale: torch.Tensor) -> None:
            cache.index_copy_(0, mapping, value.view(4, 1, 512))

    class Target:
        qk_rope_head_dim, kv_lora_rank, num_heads = 0, 512, 32

        def __init__(self) -> None:
            self.impl = Impl()
            self.W_UK_T = torch.full((32, 2, 512), .5, dtype=torch.bfloat16, device=device)
            self.W_UV = torch.full((32, 512, 2), 1 / 512, dtype=torch.bfloat16, device=device)
            self.cache = torch.zeros(8, 1, 512, dtype=torch.uint8, device=device)

        def forward(self, query: torch.Tensor, value: torch.Tensor,
                    qpe: torch.Tensor) -> torch.Tensor:
            self.impl.do_kv_cache_update(value, qpe, self.cache, slots, "fp8_e4m3", scale)
            projected = torch.bmm(query.transpose(0, 1), self.W_UK_T).transpose(0, 1)
            selected, _ = backend.triton_convert_req_index_to_global_index(
                requests, block_table, topk, return_valid_counts=True)
            wrapper._kv_indices_buf.copy_(selected.reshape(-1).clamp_min(0))
            attended = wrapper.run(projected, qpe, self.cache, self.cache[..., :0], ckv_scale=.5)
            output = torch.empty(32, 4, 2, dtype=torch.bfloat16, device=device)
            torch.bmm(attended.transpose(0, 1), self.W_UV, out=output)
            return output.transpose(0, 1)

    target = Target()
    model = SimpleNamespace(named_modules=lambda: [
        ("language_model.model.layers.3.self_attn.attn", target)])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            target.forward(q, latent, rope)
    stream.synchronize()
    trace = MLAInternalCapture()
    original_bmm = torch.bmm
    try:
        trace.install(model, backend)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = target.forward(q, latent, rope)
    finally:
        trace.restore()
    assert torch.bmm is original_bmm and not trace.methods
    torch.cuda.synchronize()
    events = {event["name"]: event for event in trace.events}
    assert len(events) == 14

    def values(name: str) -> dict[str, torch.Tensor]:
        return {tensor["path"]: tensor["value"] for tensor in events[name]["tensors"]}

    for index in range(5):
        q.fill_(index + 1)
        latent.fill_(17 + index)
        topk.copy_(torch.arange(8, dtype=torch.int32, device=device).roll(index).view(4, 2))
        valid_counts.copy_(torch.tensor([2, 1, 2, 0], dtype=torch.int32, device=device))
        wrapper._kv_len_arr_buf.copy_(torch.tensor([2, 2, 1, 2],
                                                 dtype=torch.int32, device=device))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, torch.full_like(output, (index + 1) * .5))
        assert torch.equal(values("query_bmm/input")[""], q.transpose(0, 1))
        selected = values("attention/selected_cache")
        ids = topk.cpu().long()
        expected = target.cache.cpu().view(8, 512)[ids]
        mask = torch.tensor([[True, True], [True, True], [True, False], [True, True]])
        expected[~mask] = 0
        assert torch.equal(selected["/bytes"].cpu(), expected)
        assert torch.equal(selected["/valid"].cpu(), mask)
        assert torch.equal(values("sparse_convert/output")["/valid_counts"], valid_counts)
        assert bool((values("kv_write/after")["/bytes"] == 17 + index).all())
    metadata = trace.metadata()
    assert metadata["python_wrappers_removed"]
    assert metadata["combined_copy_staging_bytes"] + 8 * 1024**2 <= 32 * 1024**2
    print(json.dumps({"status": "passed", "authored_chain_only": True,
        "real_model_probed": False, "replays_checked": 5, "native_calls_not_replaced": True,
        "dynamic_query_and_selected_cache_bytes_exact": True,
        "planned_prefix_keeps_slot_zero_reads": True,
        "python_hooks_removed_before_replay": True,
        "copy_and_staging": metadata, "cuda_peak_bytes": torch.cuda.max_memory_allocated()}))
    print("MXWAVE_DECODE_FIXTURE_PASSED")


if __name__ == "__main__":
    main()
