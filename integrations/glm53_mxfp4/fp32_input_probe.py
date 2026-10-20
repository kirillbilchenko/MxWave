"""Require integrated routing, exact parent outputs and graph-owned scratch before model use."""

from __future__ import annotations

from typing import Any

from fp32_input_policy import policy


def protocol() -> dict[str, Any]:
    """Freeze the real input/output fixtures, fallback sizes and graph/stream stress coverage."""
    return {"revision": "fp32inputv1", "ranks": [0, 1], "layers": [0, 22, 44],
        "kinds": ["input", "output"], "tokens": [1, 4, 8, 16],
        "activation_cases": ["787", "797", "1909", "basis"],
        "fallback_tokens": [0, 2, 3, 5, 6, 7, 9, 15, 17, 54, 256],
        "compiled_tokens": [8, 16], "compiled_layers": [0], "graph_replays": 8,
        "two_stream_replays_per_graph": 32, "maximum_capture_extra_bytes": 16 * 1024**2,
        "CUDA_allocation_cap_bytes": 2 * 1024**3, "container_memory_bytes": 12 * 1024**3,
        "component_timeout_seconds": 900, "minimum_head_free_bytes": 2 * 1024**3,
        "maximum_payload_bytes_per_rank": 256 * 1024**2,
        "routing": policy(), "model_loaded": False, "performance_measured": False,
        "parent_exact_word_screen_passed": False,
        "model_quality_or_serving_speed_qualified": False, "automatic_promotion_allowed": False}


def assess(report: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on missing routes, aliasing, allocation growth or changed parent output words."""
    p = protocol()
    rank = report["rank"]
    if report["status"] != "complete" or report["protocol"] != p or rank not in p["ranks"]:
        raise ValueError("Incomplete or altered integration protocol")
    expected = [f"kda-{kind}-l{layer}-r{rank}" for layer in p["layers"] for kind in p["kinds"]]
    if [c["name"] for c in report["cases"]] != expected:
        raise ValueError("Missing integrated source fixtures")
    known = fallback = compiled = streams = 0

    def graph(row: dict[str, Any]) -> None:
        if not row["graph_matches_eager"] or row["graph_retained_bytes"] != 0:
            raise ValueError("Graph changes bytes or retains allocations during replay")
        if not 0 <= row["capture_extra_bytes"] <= p["maximum_capture_extra_bytes"]:
            raise ValueError("Graph capture duplicates scratch storage")

    for case in report["cases"]:
        is_input = "-input-" in case["name"]
        if not all(case[k] for k in ("decoded_weight_bits_exact", "packed_buffers_unchanged",
                                    "outer_pool_guards_unchanged")):
            raise ValueError("Source storage or outer pool guards changed")
        activations = p["activation_cases"] if is_input else ["787"]
        if [(x["tokens"], x["activation"]) for x in case["known"]] != [
            (n, a) for n in p["tokens"] for a in activations
        ]:
            raise ValueError("Missing retained activation coverage")
        for row in case["known"]:
            if (not row["matches_parent"] or row["output_sha256"] != row["parent_sha256"]
                    or row["graph_output_sha256"] != row["parent_sha256"]):
                raise ValueError("Integrated outputs differ from prior native words")
            graph(row)
            known += 1
        if is_input:
            if not case["FP32_workspace_suffix_guards_unchanged"]:
                raise ValueError("FP32 workspace overrun")
            if [x["tokens"] for x in case["fallback"]] != p["fallback_tokens"]:
                raise ValueError("Missing untested-row fallback coverage")
            for row in case["fallback"]:
                if (not row["matches_qualified"] or row["actual"]["sha256"] != row["qualified"]["sha256"]
                        or row["graph_output_sha256"] != row["qualified"]["sha256"]):
                    raise ValueError("Qualified fallback changed")
                graph(row)
                fallback += 1
            want = p["compiled_tokens"] if "-l0-" in case["name"] else []
            if [x["tokens"] for x in case["compiled"]] != want:
                raise ValueError("Missing compiled integration coverage")
            for row in case["compiled"]:
                if (not row["matches_parent"] or row["output_sha256"] != row["parent_sha256"]
                        or row["graph_output_sha256"] != row["parent_sha256"]):
                    raise ValueError("Compilation changes FP32 output words")
                graph(row)
                compiled += 1
            chain = case["compiled_chain"]
            if want:
                if (chain["physical_rows"] != [8, 17, 8]
                        or len(chain["expected_output_sha256"]) != 3
                        or chain["output_sha256"] != chain["expected_output_sha256"]
                        or chain["graph_output_sha256"] != chain["expected_output_sha256"]):
                    raise ValueError("Compiled FP32/native/FP32 pool reuse changes output")
                graph(chain)
            elif chain is not None:
                raise ValueError("Unexpected compiled chain scope")
            stress = case["stream_stress"]
            if (stress["replays_per_graph"] != p["two_stream_replays_per_graph"]
                    or stress["owned_slots"] != [0, 1] or stress["captured_streams"] != 2
                    or len(stress["parent_sha256"]) != 2
                    or stress["output_sha256"] != stress["parent_sha256"]
                    or not all(stress[k] for k in ("outputs_match_parent", "third_stream_rejected",
                                                  "workspace_suffix_guards_unchanged"))
                    or stress["replay_retained_bytes"] != 0):
                raise ValueError("Captured stream scratch aliases, grows or loses ownership")
            streams += 1
        elif case["fallback"] or case["compiled"] or "stream_stress" in case:
            raise ValueError("Unexpected output-route experiment")
    if report["CUDA_peak_bytes"] > p["CUDA_allocation_cap_bytes"]:
        raise ValueError("CUDA allocation cap exceeded")
    return {"diagnostic_complete": True, "integration_invariants_passed": True,
        "rank": rank, "known_parent_comparisons": known, "fallback_comparisons": fallback,
        "compiled_shapes": compiled, "two_stream_stress_fixtures": streams,
        "parent_exact_word_screen_passed": False, "parent_input_timing_screen_passed": False,
        "model_quality_or_serving_speed_qualified": False, "automatic_promotion_allowed": False}
