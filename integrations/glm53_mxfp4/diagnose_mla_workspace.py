"""Isolate the installed sparse MLA call with fixed cache, plan and private workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from decode_state_replay import tensor_digest
from decode_state_values import save_tensor, value_metrics

BACKEND_SHA = "4449ea25921dcae1ec7988581ff26a6ff5a9e5d6d4a0ac4a68a0017ea108f136"
WRAPPER_SHA = "58d06e61aa753f938bf5682640767541dbd4cb6989340abcdc8f6f90792db6e7"
ROWS, HEADS, WIDTH, LENGTH = 4, 32, 512, 259
TOPK_WIDTH, MAX_TOKENS = 2176, 2048


def make_inputs(device: torch.device, fp8: bool) -> dict[str, torch.Tensor]:
    """Use identical logical cache values in both formats, with valid sparse slot indices."""
    generator = torch.Generator().manual_seed(6149)
    q = (torch.randn((ROWS, HEADS, WIDTH), generator=generator) * .125).to(torch.bfloat16)
    codes = (torch.randn((ROWS * LENGTH, 1, WIDTH), generator=generator) * .25).to(
        torch.float8_e4m3fn)
    cache = codes if fp8 else codes.to(torch.bfloat16) * .5
    indices = torch.zeros((ROWS, TOPK_WIDTH), dtype=torch.int32)
    indices[:, :LENGTH] = torch.arange(ROWS * LENGTH).reshape(ROWS, LENGTH)
    return {"q": q.to(device), "q_pe": q[..., :0].to(device), "cache": cache.to(device),
            "indices": indices.to(device)}


def workspace_changes(current: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Compare opaque buffers as bytes without interpreting temporary floats or NaNs."""
    a = current.reshape(-1).view(torch.uint8)
    b = reference.reshape(-1).view(torch.uint8)
    changed = 0
    for start in range(0, a.numel(), 8 * 1024**2):
        changed += int(torch.count_nonzero(a[start:start + 8 * 1024**2]
                                         != b[start:start + 8 * 1024**2]))
    return {"bytes": a.numel(), "changed_bytes": changed, "bytes_exact": changed == 0}


def main() -> None:
    """Retain all five graph results; this component does not measure serving speed or KL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    case = plan["cases"][args.case]
    if (plan["version"] != 1 or plan["iterations"] != 5 or plan["model_loads"] != 0
            or plan["qualification_attempted"] or plan["shape"] != {
                "rows": ROWS, "heads": HEADS, "latent_width": WIDTH, "kv_length": LENGTH,
                "topk_width": TOPK_WIDTH, "maximum_tokens": MAX_TOKENS}):
        raise ValueError("Frozen MLA component plan changed")
    args.output.mkdir(exist_ok=False)
    report_path = args.output / "report.json"
    report: dict[str, Any] = {"status": "running", "case": case,
        "case_index": args.case, "started_epoch": time.time(),
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_loads": 0, "qualification_attempted": False, "replays": [],
        "limits": ["Synthetic latent queries/cache at the observed model dimensions.",
            "Isolates FA2 attention, excluding latent/value projections and KV writes.",
            "Both private workspace images exist in both policies; only restoration differs.",
            "Five fixed replays, all retained; no selected successful retry.",
            "No serving KL, task quality, throughput or release qualification."]}

    def save() -> None:
        temporary = report_path.with_suffix(".incomplete")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(report_path)

    save()
    try:
        import flashinfer.mla._core as wrapper_module
        import vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 as backend

        if (hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest() != BACKEND_SHA
                or hashlib.sha256(Path(wrapper_module.__file__).read_bytes()).hexdigest()
                != WRAPPER_SHA):
            raise ValueError("Installed attention backend or FlashInfer wrapper changed")
        torch.set_num_threads(2)
        torch.cuda.set_device(0)
        if torch.cuda.get_device_capability() != (12, 1):
            raise ValueError("Require the pinned SM121 platform")
        properties = torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(2.5 * 1024**3 / properties.total_memory)
        device = torch.device("cuda", 0)
        fp8 = case["kv_dtype"] == "fp8_e4m3"
        values = make_inputs(device, fp8)
        state = backend._SM90State(device, HEADS,
            torch.float8_e4m3fn if fp8 else torch.bfloat16, MAX_TOKENS, TOPK_WIDTH,
            WIDTH, 0, .0625)
        state.kv_indices[:ROWS * TOPK_WIDTH].copy_(values["indices"].reshape(-1))
        captured_length = TOPK_WIDTH if case["plan_history"] == "long_then_replan" else LENGTH
        state.plan(ROWS, torch.full((ROWS,), captured_length, dtype=torch.int32))
        captured_plan = list(state.wrapper._plan_info)

        def operation() -> torch.Tensor:
            return state.wrapper.run(values["q"], values["q_pe"], values["cache"],
                values["cache"][..., WIDTH:],
                **({"ckv_scale": .5, "kpe_scale": 1.0} if fp8 else {}))

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                operation()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = operation()
        torch.cuda.synchronize()
        # Refresh the same stable buffers outside capture, as the model's builder does.
        state.plan(ROWS, torch.full((ROWS,), LENGTH, dtype=torch.int32))
        current_plan = list(state.wrapper._plan_info)
        eager_reference = operation().detach().cpu().clone()
        report["eager_reference"] = save_tensor(args.output, "eager.bin.gz", eager_reference)
        workspaces = {"float_workspace": state.wrapper._float_workspace_buffer,
                      "int_workspace": state.wrapper._int_workspace_buffer}
        torch.cuda.synchronize()
        images = {name: tensor.clone() for name, tensor in workspaces.items()}
        torch.cuda.synchronize()
        fixed = {**values, "qo_indptr": state.wrapper._qo_indptr_buf,
                 "kv_indptr": state.wrapper._kv_indptr_buf,
                 "kv_indices": state.wrapper._kv_indices_buf,
                 "kv_len": state.wrapper._kv_len_arr_buf}
        before = {name: tensor_digest(tensor) for name, tensor in fixed.items()}
        report.update(backend_sha256=BACKEND_SHA, wrapper_sha256=WRAPPER_SHA,
            backend=state.wrapper._backend, captured_plan_info=captured_plan,
            current_plan_info=current_plan, plan_info_equal=captured_plan == current_plan,
            fixed_inputs_before=before,
            snapshot_bytes=sum(t.numel() * t.element_size() for t in images.values()),
            workspace_images={name: tensor_digest(t) for name, t in images.items()})
        reference = None
        for iteration in range(5):
            if case["workspace_policy"] == "restore_private":
                for name, tensor in workspaces.items():
                    tensor.copy_(images[name])
            torch.cuda.synchronize()
            initial = {name: workspace_changes(t, images[name]) for name, t in workspaces.items()}
            if case["workspace_policy"] == "restore_private" and any(
                    not row["bytes_exact"] for row in initial.values()):
                raise RuntimeError("Private workspace restore did not preserve every byte")
            graph.replay()
            torch.cuda.synchronize()
            current = output.detach().cpu().clone()
            if reference is None:
                reference = current
            post = {name: workspace_changes(t, images[name]) for name, t in workspaces.items()}
            if not bool(torch.isfinite(current).all()):
                raise RuntimeError("Graph output contains nonfinite values")
            report["replays"].append({"index": iteration,
                "output": save_tensor(args.output, f"graph-{iteration}.bin.gz", current),
                "metrics_vs_first": value_metrics(current, reference),
                "metrics_vs_eager": value_metrics(current, eager_reference),
                "initial_workspace": initial, "resulting_workspace": post})
            save()
        after = {name: tensor_digest(tensor) for name, tensor in fixed.items()}
        if before != after:
            raise RuntimeError("An explicit input or device plan buffer was mutated")
        report.update(status="complete", fixed_inputs_after=after,
            all_replay_outputs_exact=all(r["metrics_vs_first"]["bytes_exact"]
                                        for r in report["replays"]),
            all_graph_outputs_match_current_eager=all(r["metrics_vs_eager"]["bytes_exact"]
                                                     for r in report["replays"]),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), finished_epoch=time.time())
        save()
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        save()
        raise


if __name__ == "__main__":
    main()
