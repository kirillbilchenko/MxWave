"""Retain bounded native DFlash2 states for a separate selector calibration pilot."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def install(worker: Any) -> dict[str, Any]:
    """Observe rank-zero graph outputs; keep all native proposal outputs untouched."""
    import torch
    from drafter_context_probe import retain_frame
    from drafter_head_probe import tensor_bytes, validate_sources, write
    from safetensors.torch import save_file
    from vllm.distributed import get_tensor_model_parallel_rank

    root = Path("/traces")
    plan = json.loads(Path("/trial/plan.json").read_bytes())
    sources = validate_sources(plan)
    rank = get_tensor_model_parallel_rank()
    spec = worker.model_runner.speculator
    assert spec.num_speculative_steps == 7 and spec.selector_top_k == 16
    taps = spec._mxwave_head_taps
    assert (1, 8) in taps["graphs"]
    selector = spec.model.model.candidate_selector
    weights = {
        "projection": selector.hidden_projection.weight,
        "predecessor": selector.predecessor_codebook,
        "successor": selector.successor_codebook,
    }
    inventory = {
        name: {
            "sha256": hashlib.sha256(tensor_bytes(value)).hexdigest(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in weights.items()
    }
    assert all(value.dtype == torch.bfloat16 for value in weights.values())
    if rank == 0:
        path = root / "selector-original.safetensors"
        assert not path.exists()
        save_file({name: value.detach().cpu() for name, value in weights.items()}, str(path))
        path.chmod(0o644)
    write(root / f"selector-inventory-r{rank}.json", inventory)
    counts: dict[str, int] = {}
    allowed = {row["id"] for row in plan["calibration_cases"]}
    manager = spec.query_cudagraph_manager
    original = manager.run_fullgraph

    @torch.inference_mode()
    def replay(descriptor: Any) -> Any:
        result = original(descriptor)
        if rank:
            return result
        marker = json.loads((root / "probe-marker.json").read_bytes())
        if not marker["active"]:
            return result
        case = marker["case"]
        assert case in allowed
        assert descriptor.num_reqs == 1 and descriptor.num_tokens == 8
        index = counts.get(case, 0)
        assert index < plan["max_frames_per_case"]
        assert shutil.disk_usage(root).free > 2 * 1024**3
        graph = taps["graphs"][(1, 8)]
        tensors = {name: graph[name] for name in ("hidden", "ids", "values", "scores")}
        tensors.update(
            anchors=spec.input_buffers.input_ids[spec._anchor_indices[:1]],
            proposal=spec.draft_tokens[:1],
            sample_positions=spec.sample_pos[:7],
        )
        retain_frame(
            root / f"calibration-{case}-{index:04d}.safetensors", tensors,
            {
                "schema": "mxwave-selector-calibration-frame-v1",
                "case": case, "index": index, "rank": rank,
                "native_full_graph": True, "timing_claim": False,
            },
        )
        counts[case] = index + 1
        write(root / "probe-progress-r0.json", {"counts": counts})
        return result

    manager.run_fullgraph = replay
    report = {
        "status": "installed", "rank": rank, "sources": sources,
        "weights": inventory, "capture_rank": 0,
        "captured_graphs": [list(key) for key in sorted(taps["graphs"])],
        "native_outputs_unchanged": True,
        "scope": "New calibration prompts; no updates or speed measurements",
    }
    write(root / f"probe-installed-r{rank}.json", report)
    return report
