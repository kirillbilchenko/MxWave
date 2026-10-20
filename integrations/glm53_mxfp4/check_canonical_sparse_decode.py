"""Qualify real and heavily padded decode selection with graph-based latency controls."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .canonical_sparse_decode import canonical_decode
from .check_canonical_sparse_prefill import telemetry


def qualify(fixture: Path, output: Path) -> dict[str, Any]:
    """Check membership, unused-score exclusion, graph repeat and paired graph execution cost."""
    from vllm.models.glm5next.nvidia.ops.kpool_compress import expand_pools_and_append_tail

    if output.exists():
        raise FileExistsError(output)
    raw = load_file(str(fixture))
    native = torch.ops._C.top_k_per_row_decode
    if getattr(native, "_mxwave_canonical_decode", False):
        raise ValueError("Qualify both native and corrected operation before installation")
    reports, retained = [], {}
    for variant in ("retained", "mixed_padding"):
        scores = raw["scores"].cuda()
        lengths = raw["lengths"].reshape(-1).cuda()
        tokens = raw["seq_lens"].reshape(-1).cuda()
        if variant == "mixed_padding":
            lengths = torch.tensor([0, 1, 7, 32, 511, 512, 993, 995], dtype=torch.int32, device="cuda")
            tokens = lengths * 4 + torch.arange(8, device="cuda", dtype=torch.int32) % 4
        actual = torch.empty((8, 512), dtype=torch.int32, device="cuda")
        arguments = (scores, 1, lengths, actual, 8, scores.stride(0), scores.stride(1), 512)

        def call(corrected: bool, args: tuple[Any, ...] = arguments) -> None:
            args[3].fill_(-1)
            if corrected:
                canonical_decode(native, *args)
            else:
                native(*args)

        call(False)
        native_reference = actual.clone()
        expected = native_reference.sort(-1).values
        if variant == "retained" and not torch.equal(expected, raw["native_pools"].cuda().sort(-1).values):
            raise ValueError("Native replay changes retained selected membership")
        for _ in range(32):
            call(False)
            if not torch.equal(actual.sort(-1).values, expected):
                raise ValueError("Native selection membership does not repeat")
            call(True)
            if not torch.equal(actual, expected):
                raise ValueError("Canonical decode selection does not repeat")
        # Prove the explicit next_n=1 bounds exclude all unused score columns.
        original_scores = scores.clone()
        unused = torch.arange(scores.shape[1], device="cuda")[None, :] >= lengths[:, None]
        for fill in (1.0e30, float("nan")):
            scores.copy_(original_scores).masked_fill_(unused, fill)
            call(True)
            if not torch.equal(actual, expected):
                raise ValueError("Unused score columns affect native decode selection")
        scores.copy_(original_scores)
        expanded = expand_pools_and_append_tail(expected.to(torch.int64), tokens, 4)
        expanded = torch.nn.functional.pad(expanded, (0, raw["native_tokens"].shape[1] - expanded.shape[1]), value=-1)
        if (torch.any(expanded < -1) or torch.any(expanded >= tokens[:, None])
                or (variant == "retained" and not torch.equal(expanded.sort(-1).values,
                                                              raw["native_tokens"].cuda().sort(-1).values))):
            raise ValueError("Canonical expansion changes retained tokens, masking or causality")
        graphs = []
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(20):
                call(False)
                call(True)
        torch.cuda.current_stream().wait_stream(stream)
        for corrected in (False, True):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(100):
                    call(corrected)
            for _ in range(32):
                graph.replay()
                checked = actual if corrected else actual.sort(-1).values
                if not torch.equal(checked, expected):
                    raise ValueError("Native/canonical graph replay changes selected output")
            retained[variant + ("_canonical_graph" if corrected else "_native_graph")] = actual.cpu()
            graphs.append(graph)

        def elapsed(graph: Any) -> float:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(32):
                graph.replay()
            end.record()
            end.synchronize()
            return begin.elapsed_time(end) / 3200

        trials = []
        for trial in range(6):
            before = telemetry()
            order = [0, 1, 0] if trial % 2 == 0 else [1, 0, 1]
            ms = [elapsed(graphs[index]) for index in order]
            drift = abs(ms[2] / ms[0] - 1)
            trials.append({"order": ["canonical" if i else "native" for i in order],
                           "milliseconds_per_call": ms, "repeat_relative_drift": drift,
                           "repeat_control_passed": drift <= 0.02,
                           "before": before, "after": telemetry()})
        deltas = []
        for trial in trials:
            if trial["repeat_control_passed"]:
                a, b, c = trial["milliseconds_per_call"]
                delta = b - (a + c) / 2
                deltas.append(delta if trial["order"][1] == "canonical" else -delta)
        for name, tensor in {"scores": scores, "lengths": lengths, "seq_lens": tokens,
                             "native_reference": native_reference, "canonical_pools": expected,
                             "expanded_tokens": expanded}.items():
            retained[variant + "_" + name] = tensor.detach().cpu().contiguous()
        reports.append({"variant": variant, "native_membership_repeats": 32,
                        "canonical_exact_repeats": 32, "native_graph_replays": 32,
                        "canonical_graph_exact_replays": 32, "calls_per_graph": 100,
                        "unused_columns_poison_tests": ["large_finite", "nan"],
                        "trials": trials, "stable_trials": len(deltas),
                        "median_added_ms_stable_trials": statistics.median(deltas) if deltas else None})
        del graphs, graph, actual, expected, original_scores, arguments, call
    payload = output.with_suffix(".safetensors")
    save_file(retained, str(payload))
    payload.chmod(0o644)
    result = {"status": "passed", "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
              "payload_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(), "fixtures": reports,
              "repeat_limit": 0.02, "serving_overhead_qualified": False,
              "timing_scope": "native/canonical CUDA graphs,100 calls per replay,32 replays per sample"}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
