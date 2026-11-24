"""Compare two head methods on identical live draft states without changing proposals."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def temporary_head(head: Any, weight: Any, method: Any) -> Iterator[None]:
    """Restore both head attributes even when a diagnostic projection fails."""
    old_weight, old_method = head.weight, head.quant_method
    try:
        head.weight, head.quant_method = weight, method
        yield
    finally:
        head.weight, head.quant_method = old_weight, old_method


def write(path: Path, value: Any) -> None:
    """Publish a JSON record atomically."""
    temporary = path.with_suffix(".incomplete")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_sources(plan: dict[str, Any]) -> dict[str, str]:
    """Verify installed APIs that this probe calls, without copying their implementation."""
    import vllm

    root = Path(vllm.__file__).parent
    actual = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in plan["probe_vllm_sources"]
    }
    assert actual == plan["probe_vllm_sources"], actual
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
        DFlash2Speculator,
        _selector_walk_kernel,
    )

    assert callable(UnquantizedLinearMethod.apply)
    assert callable(DFlash2Qwen3ForCausalLM.compute_candidates)
    assert callable(DFlash2Speculator._sample_path)
    assert hasattr(_selector_walk_kernel, "__getitem__")
    return actual


class ProjectionCapture:
    """Retain a projection before the installed top-k routine masks padding in place."""

    def __init__(self, method: Any) -> None:
        self.method = method
        self.raw: Any = None

    def apply(self, layer: Any, x: Any, bias: Any = None) -> Any:
        """Delegate the actual projection and retain a separate immutable tensor."""
        result = self.method.apply(layer, x, bias)
        self.raw = result.detach().clone()
        return result


def tensor_bytes(tensor: Any) -> bytes:
    """Get exact contiguous CPU bytes, including BF16 payloads."""
    import torch

    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def arm_capture(worker: Any) -> None:
    """Keep references to actual graph tensors before capture, without adding GPU work."""
    import torch

    spec = worker.model_runner.speculator
    draft = spec.model
    assert not hasattr(spec, "_mxwave_head_taps")
    original_candidates = draft.compute_candidates
    original_sample = spec._sample_path
    original_generate = spec._generate_draft
    taps: dict[str, Any] = {"graphs": {}, "original_candidates": original_candidates}
    spec._mxwave_head_taps = taps

    def generate(num_reqs: int, num_tokens_padded: int, *args: Any, **kwargs: Any) -> Any:
        taps["capturing"] = (
            (num_reqs, num_tokens_padded) if torch.cuda.is_current_stream_capturing() else None
        )
        try:
            return original_generate(num_reqs, num_tokens_padded, *args, **kwargs)
        finally:
            taps["capturing"] = None

    def candidates(hidden: Any) -> Any:
        ids, values = original_candidates(hidden)
        key = taps.get("capturing")
        if key is not None:
            taps["graphs"].setdefault(key, {}).update(hidden=hidden, ids=ids, values=values)
        return ids, values

    def sample(candidate_ids: Any, scores: Any, num_reqs: int) -> Any:
        result = original_sample(candidate_ids, scores, num_reqs)
        key = taps.get("capturing")
        if key is not None:
            assert key[0] == num_reqs
            taps["graphs"][key].update(candidate_ids=candidate_ids, scores=scores)
        return result

    spec._generate_draft = generate
    draft.compute_candidates = candidates
    spec._sample_path = sample


def install(worker: Any) -> dict[str, Any]:
    """Install bounded shadow diagnostics after startup profiling and graph capture."""
    import torch
    import vocab_runtime
    from mxwave_zipserv import runtime
    from safetensors.torch import save_file
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import _selector_walk_kernel

    root = Path("/traces")
    plan = json.loads(Path("/trial/plan.json").read_bytes())
    sources = validate_sources(plan)
    rank = get_tensor_model_parallel_rank()
    spec = worker.model_runner.speculator
    draft = spec.model
    target = getattr(worker.model_runner.model, "language_model", worker.model_runner.model)
    head = draft.lm_head
    assert head is target.lm_head and head.weight is None
    packed_method = head.quant_method
    assert type(packed_method).__name__ == "VocabMethod"
    assert spec.num_speculative_steps == 7 and spec.selector_top_k == 16
    before = torch.cuda.memory_allocated()
    dense = runtime.decompress(vocab_runtime.packed_head(head))
    digest = hashlib.sha256()
    for chunk in dense.split(1024):
        digest.update(tensor_bytes(chunk))
    qualification = head.mxwave_vocab_qualification
    assert digest.hexdigest() == qualification["source_sha256"]
    weight = torch.nn.Parameter(dense, requires_grad=False)
    dense_method = UnquantizedLinearMethod()
    taps = spec._mxwave_head_taps
    assert (1, 8) in taps["graphs"], sorted(taps["graphs"])
    original_candidates = taps["original_candidates"]
    pending: dict[str, Any] = {}
    counts: dict[str, int] = {}

    def project(hidden: Any, method: Any, selected_weight: Any) -> tuple[Any, Any, Any]:
        capture = ProjectionCapture(method)
        with temporary_head(head, selected_weight, capture):
            ids, values = original_candidates(hidden)
        assert capture.raw is not None and head.weight is None
        assert head.quant_method is packed_method
        return ids.clone(), values.clone(), capture.raw

    @torch.inference_mode()
    def candidates(hidden: Any, actual: dict[str, Any]) -> Any:
        marker = json.loads((root / "probe-marker.json").read_bytes())
        if not marker["active"]:
            assert not pending
            return original_candidates(hidden)
        assert not torch.cuda.is_current_stream_capturing()
        assert not pending and tuple(hidden.shape) == (7, 4096)
        case = marker["case"]
        assert case in ("code_256", "explanation_256")
        index = counts.get(case, 0)
        assert index < plan["max_frames_per_case"]
        order = ("packed", "dense", "dense_repeat", "packed_repeat")
        if index % 2:
            order = ("dense", "packed", "packed_repeat", "dense_repeat")
        results = {}
        for label in order:
            is_dense = label.startswith("dense")
            results[label] = project(
                hidden, dense_method if is_dense else packed_method, weight if is_dense else None
            )
        pending.update(
            case=case, index=index, hidden=hidden.clone(), results=results, order=list(order),
            actual=actual,
        )
        return results["packed"][:2]

    @torch.inference_mode()
    def sample(candidate_ids: Any, scores: Any, num_reqs: int) -> None:
        if not pending:
            return
        assert num_reqs == 1 and tuple(candidate_ids.shape) == (1, 7, 16)
        frame = dict(pending)
        pending.clear()
        results = frame.pop("results")
        actual = frame.pop("actual")
        packed_ids, packed_values, packed_raw = results["packed"]
        dense_ids, dense_values, dense_raw = results["dense"]
        request_index = spec.sample_idx_mapping[0].item()
        assert request_index >= 0 and spec.temperature[request_index].item() == 0
        anchors = spec.input_buffers.input_ids[spec._anchor_indices[:1]]
        alternate_scores = draft.model.candidate_selector(
            dense_ids.view_as(candidate_ids),
            dense_values.view_as(candidate_ids),
            frame["hidden"].view(1, 7, -1),
            anchors,
        )
        tokens = torch.empty_like(spec.draft_tokens[:1])
        realized = torch.empty_like(spec._selector_scores[:1])
        _selector_walk_kernel[(1,)](
            alternate_scores.contiguous(), dense_ids.contiguous(),
            spec.sample_pos, spec.sample_idx_mapping, spec.temperature, spec.seeds,
            tokens, realized, num_steps=7, top_k=16, BLOCK_K=16,
            SAMPLE_PROBABILISTIC=spec.draft_logits is not None,
            USE_FP64=spec.use_fp64_gumbel, num_warps=1,
        )
        tensors = {
            "hidden": frame.pop("hidden"), "anchors": anchors,
            "packed_ids": packed_ids, "packed_values": packed_values,
            "dense_ids": dense_ids, "dense_values": dense_values,
            "packed_scores": scores, "dense_scores": alternate_scores,
            "packed_proposal": actual["proposal"], "dense_proposal": tokens,
            "packed_realized": actual["realized"], "dense_realized": realized,
            "sample_positions": spec.sample_pos[:7],
            "graph_ids": actual["ids"], "graph_values": actual["values"],
            "graph_scores": actual["scores"],
        }
        packed_shadow_scores = draft.model.candidate_selector(
            packed_ids.view_as(candidate_ids),
            packed_values.view_as(candidate_ids),
            tensors["hidden"].view(1, 7, -1),
            anchors,
        )
        tensors["packed_shadow_scores"] = packed_shadow_scores
        tensors["after_proposal"] = spec.draft_tokens[:1].clone()
        tensors["after_realized"] = spec._selector_scores[:1].clone()
        repeats = {}
        for label in ("packed", "dense"):
            for position, kind in enumerate(("ids", "values", "raw")):
                value = results[label][position]
                repeated = results[label + "_repeat"][position]
                repeats[label + "_" + kind] = tensor_bytes(value) == tensor_bytes(repeated)
                if kind != "raw":
                    tensors[label + "_" + kind + "_repeat"] = repeated
        raw_delta = packed_raw.float() - dense_raw.float()
        frame.update(
            schema="mxwave-draft-head-frame-v1", rank=rank,
            graph_num_tokens=8, graph_num_reqs=1,
            graph_shadow_exact={
                "ids": torch.equal(actual["ids"], packed_ids),
                "values": torch.equal(actual["values"], packed_values),
                "scores": torch.equal(actual["scores"], packed_shadow_scores),
            },
            actual_outputs_untouched={
                "proposal": torch.equal(spec.draft_tokens[:1], actual["proposal"]),
                "realized_scores": torch.equal(spec._selector_scores[:1], actual["realized"]),
            },
            repeats=repeats,
            raw_shape=list(packed_raw.shape),
            raw_bit_differences=int(
                (packed_raw.view(torch.int16) != dense_raw.view(torch.int16)).sum().item()
            ),
            raw_max_abs=float(raw_delta.abs().max().item()),
            raw_finite=bool(
                torch.isfinite(packed_raw).all().item() and torch.isfinite(dense_raw).all().item()
            ),
            raw_sha256={
                label: hashlib.sha256(tensor_bytes(values[2])).hexdigest()
                for label, values in results.items()
            },
            full_raw_retained=frame["index"] < 2 or frame["index"] % 8 == 0,
        )
        if frame["full_raw_retained"]:
            tensors.update(packed_raw=packed_raw, dense_raw=dense_raw)
        stored = {}
        frame["tensor_dtypes"] = {}
        for name, tensor in tensors.items():
            value = tensor.detach().contiguous().cpu()
            frame["tensor_dtypes"][name] = str(value.dtype)
            stored[name] = value.view(torch.int16) if value.dtype == torch.bfloat16 else value
        stem = f"frame-{frame['case']}-{frame['index']:04d}-r{rank}"
        path = root / (stem + ".safetensors")
        assert not path.exists()
        save_file(stored, str(path))
        path.chmod(0o644)
        frame["payload_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame["payload_bytes"] = path.stat().st_size
        write(root / (stem + ".json"), frame)
        counts[frame["case"]] = frame["index"] + 1
        write(root / f"probe-progress-r{rank}.json", {"counts": counts})

    manager = spec.query_cudagraph_manager
    original_replay = manager.run_fullgraph

    @torch.inference_mode()
    def replay(descriptor: Any) -> Any:
        result = original_replay(descriptor)
        marker = json.loads((root / "probe-marker.json").read_bytes())
        if marker["active"]:
            assert descriptor.num_tokens == 8 and descriptor.num_reqs == 1
            graph = taps["graphs"][(1, 8)]
            actual = {
                key: graph[key].clone() for key in ("hidden", "ids", "values", "scores")
            }
            actual.update(
                proposal=spec.draft_tokens[:1].clone(),
                realized=spec._selector_scores[:1].clone(),
            )
            candidates(actual["hidden"], actual)
            sample(actual["ids"].view(1, 7, 16), actual["scores"], 1)
        return result

    manager.run_fullgraph = replay
    report = {
        "status": "installed", "rank": rank, "sources": sources,
        "dense_head_source_sha256": digest.hexdigest(),
        "dense_head_matches_prepared_source": True, "shape": list(dense.shape),
        "extra_cuda_allocated": torch.cuda.memory_allocated() - before,
        "native_path_returned": "packed", "diagnostic_memory_only": True,
        "observation": "pre-capture tensor references; shadow work after native FULL replay",
        "captured_graphs": [list(key) for key in sorted(taps["graphs"])],
        "full_logits_capture": "first two frames and every eighth; all top-k and proposals",
    }
    write(root / f"probe-installed-r{rank}.json", report)
    return report
