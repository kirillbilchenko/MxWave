"""Attribute combined-adapter kernels through CPU external ids and model-only graph evidence."""

from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from analyze_full_profiles import gpu_steps, kernel_group
from analyze_kda_profiles import matched_steps, shape_samples
from compare_full_model import trace_events

CUSTOM = {"mxwave_glm53::kda_linear": "kda", "mxwave_glm53::moe_gemm": "expert_scales",
          "mxwave_glm53::embedding_rows": "embedding", "mxwave_zipserv::linear": "head"}


def attribute(path: Path, *, model_only_graphs: bool = False) -> dict[str, Any]:
    """Retain unresolved kernels; classify graph BF16 only with the pinned runner assumption."""
    cpu = {}
    scopes: dict[tuple[int, int], list[dict[str, Any]]] = collections.defaultdict(list)
    durations: dict[str, list[float]] = collections.defaultdict(list)
    with gzip.open(path, "rt") as stream:
        for event in trace_events(stream):
            if event.get("cat") != "cpu_op" or event.get("ph") != "X":
                continue
            external = event.get("args", {}).get("External id")
            if external is not None:
                cpu[external] = {key: event[key] for key in ("name", "pid", "tid", "ts", "dur")}
            if event["name"] in CUSTOM:
                scopes[(event["pid"], event["tid"])].append(event)
                durations[event["name"]].append(float(event["dur"]))
    starts = {}
    for key, values in scopes.items():
        values.sort(key=lambda e: e["ts"])
        starts[key] = [e["ts"] for e in values]
    groups: dict[str, dict[str, Any]] = {}
    evidence = collections.Counter()
    with gzip.open(path, "rt") as stream:
        for event in trace_events(stream):
            if event.get("cat") != "kernel" or event.get("ph") != "X":
                continue
            name, args = event["name"], event.get("args", {})
            lower = name.lower()
            component = None
            operation = cpu.get(args.get("External id"))
            if operation is not None:
                key = (operation["pid"], operation["tid"])
                position = bisect.bisect_right(starts.get(key, []), operation["ts"]) - 1
                candidates = []
                # Adapter scopes are shallow; the inner ZipServ op must be
                # attributed to KDA when it sits inside the KDA custom op.
                for i in range(position, max(-1, position - 8), -1):
                    scope = scopes[key][i]
                    if (scope["ts"] <= operation["ts"]
                            and operation["ts"] + operation["dur"]
                            <= scope["ts"] + scope["dur"] + .01):
                        candidates.append(CUSTOM[scope["name"]])
                if candidates:
                    component = "kda" if "kda" in candidates else candidates[0]
                    evidence["cpu_external_id_and_containment"] += 1
            if component is None and "_row_gather_kernel" in lower:
                component = "embedding"
                evidence["distinct_embedding_kernel"] += 1
            if component is None and "mxwave_grouped_" in name:
                component = "expert_scales"
                evidence["distinct_expert_kernel"] += 1
            if (component is None and model_only_graphs and args.get("graph id")
                    and "bf16triplebitmap_" in lower):
                component = "kda"
                evidence["model_only_graph_assumption"] += 1
            if component is None:
                component = "unattributed"
            group = component + "/" + kernel_group(name)
            row = groups.setdefault(group, {"component": component,
                "kernel_group": kernel_group(name), "calls": 0, "gpu_duration_sum_us": 0.})
            row["calls"] += 1
            row["gpu_duration_sum_us"] += float(event["dur"])
    return {"groups": list(groups.values()), "attribution_evidence": dict(evidence),
        "cpu_custom_ops": {name: {"calls": len(values), "median_us": statistics.median(values),
            "maximum_us": max(values), "sum_us": sum(values)} for name, values in durations.items()},
        "model_only_graphs_assumed": model_only_graphs,
        "limitations": ["CPU scopes are instrumented, nested and must not be added together.",
            "GPU duration sums can overlap and are not end-to-end request latency.",
            "Unattributed work is retained, including ordinary stock head kernels.",
            ("Graph attribution assumes the pinned runner captures model forward only; "
             "the vocabulary head is computed separately by the sampler.")]}


def main() -> None:
    """Review saved both-rank traces with per-component and matched-forward evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-only-graphs", action="store_true")
    args = parser.parse_args()
    profiles, paired = {}, []
    for rank, location in ((0, "traces"), (1, "traces-worker")):
        arms = {}
        for mode in ("stock", "kda"):
            paths = list((args.root / mode / location).glob("dp*.trace.json.gz"))
            if len(paths) != 1:
                raise ValueError("Require one worker trace per arm and rank")
            path = paths[0]
            steps = gpu_steps(path)
            arms[mode] = steps.pop("steps")
            profiles[f"{mode}_rank{rank}"] = {**steps,
                "shape_samples": shape_samples(arms[mode]),
                "adapter_attribution": attribute(path, model_only_graphs=args.model_only_graphs),
                "path": str(path.relative_to(args.root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        paired.extend({"rank": rank, **row} for row in matched_steps(arms["stock"], arms["kda"]))
    report = {"status": "complete", "profiles": profiles, "matched_steps": paired,
        "analysis_code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()
            ).hexdigest() for name in ("analyze_combined_profiles.py", "analyze_full_profiles.py",
                                      "analyze_kda_profiles.py", "compare_full_model.py")}}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "profiles": len(profiles), "matched_shapes": len(paired)}))


if __name__ == "__main__":
    main()
