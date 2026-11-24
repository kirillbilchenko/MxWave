"""Compare small pinned GLM weight slices; never load a complete model or GPU.

Prepare uses bounded HTTP ranges. Compare uses frozen MxWave math on the CPU.
Weight reconstruction is a screening metric, not downstream model quality.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import runpy
import struct
import time
from pathlib import Path
from typing import Any

from audit_checkpoint import DEFAULT_REPOSITORY, DEFAULT_REVISION, digest, fetch, write_json

BF16_REPOSITORY = "zai-org/GLM-5.3-Flash-BF16"


def prepare(root: Path) -> None:
    """Fetch only selected rows and their corresponding scales from pinned sources."""
    destination = root / "row-probe-v1"
    source_path = destination / "sources.json"
    if source_path.exists():
        sources = json.loads(source_path.read_bytes())
    else:
        api = json.loads(fetch(f"https://huggingface.co/api/models/{BF16_REPOSITORY}"))
        sources = {
            "bf16_repository": BF16_REPOSITORY,
            "bf16_revision": api["sha"],
            "upstream_repository": DEFAULT_REPOSITORY,
            "upstream_revision": DEFAULT_REVISION,
        }
        write_json(source_path, sources)
    bf16_base = f"https://huggingface.co/{BF16_REPOSITORY}/resolve/{sources['bf16_revision']}"
    upstream_base = f"https://huggingface.co/{DEFAULT_REPOSITORY}/resolve/{DEFAULT_REVISION}"
    bf16_index_path = destination / "bf16-index.json"
    if not bf16_index_path.exists():
        bf16_index_path.write_bytes(fetch(bf16_base + "/model.safetensors.index.json"))
    bf16_index = json.loads(bf16_index_path.read_bytes())["weight_map"]
    upstream_index_path = root / "metadata/model.safetensors.index.json"
    upstream_index = json.loads(upstream_index_path.read_bytes())["weight_map"]
    sources["bf16_index_sha256"] = digest(bf16_index_path.read_bytes())
    sources["upstream_index_sha256"] = digest(upstream_index_path.read_bytes())
    write_json(source_path, sources)
    records: list[dict[str, Any]] = []

    def sample(
        base: str, index: dict[str, str], name: str, first: int, rows: int, tag: str
    ) -> dict[str, Any]:
        shard = index[name]
        header_path = (
            root / "headers" / (shard + ".json")
            if tag == "upstream"
            else destination / "bf16-headers" / (shard + ".json")
        )
        if not header_path.exists():
            url = base + "/" + shard
            header_size = struct.unpack("<Q", fetch(url, 0, 8))[0]
            if not 1 <= header_size <= 32 * 1024**2:
                raise ValueError("Invalid remote header size")
            raw = fetch(url, 8, 8 + header_size)
            write_json(
                header_path,
                {
                    "header_size": header_size,
                    "header": json.loads(raw),
                    "raw_header_sha256": digest(raw),
                },
            )
        saved = json.loads(header_path.read_bytes())
        spec = saved["header"][name]
        if len(spec["shape"]) != 2 or not 0 <= first < first + rows <= spec["shape"][0]:
            raise ValueError("Invalid sampled matrix row range")
        begin, end = spec["data_offsets"]
        stride, remainder = divmod(end - begin, spec["shape"][0])
        if remainder:
            raise ValueError("Tensor payload cannot be divided into complete rows")
        offset = 8 + saved["header_size"] + begin + first * stride
        path = destination / "samples" / f"{tag}-{name}-{first}-{rows}.bin"
        sidecar = path.with_suffix(".json")
        if path.exists() and sidecar.exists():
            cached = json.loads(sidecar.read_bytes())
            if digest(path.read_bytes()) != cached["sha256"]:
                raise ValueError("Cached sample hash has changed")
            return cached
        payload = fetch(base + "/" + shard, offset, offset + rows * stride)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        result = {
            "name": name,
            "dtype": spec["dtype"],
            "shape": [rows, spec["shape"][1]],
            "full_shape": spec["shape"],
            "first_row": first,
            "rows": rows,
            "bytes": len(payload),
            "sha256": digest(payload),
            "shard": shard,
            "path": str(path.relative_to(destination)),
        }
        write_json(sidecar, result)
        return result

    cases: list[tuple[str, str, int]] = []
    for layer, expert, fraction in ((3, 0, 0), (23, 143, 1), (44, 287, 2), (45, 0, 0)):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            module = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
            cases.append((module, "MXFP4", fraction))
    for stem in (
        "model.language_model.layers.0.mlp",
        "model.language_model.layers.23.mlp.shared_experts",
        "model.language_model.layers.45.mlp.shared_experts",
    ):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            cases.append((stem + "." + projection, "MXFP8", 1))
    for module in (
        "lm_head",
        "model.language_model.embed_tokens",
        "model.language_model.layers.23.self_attn.q_a_proj",
    ):
        cases.append((module, "BF16", 1))
    for module, kind, fraction in cases:
        name = module + (".weight_packed" if kind == "MXFP4" else ".weight")
        shard = upstream_index[name]
        saved = json.loads((root / "headers" / (shard + ".json")).read_bytes())
        total_rows = saved["header"][name]["shape"][0]
        first = (total_rows - 32) * fraction // 2
        upstream = sample(upstream_base, upstream_index, name, first, 32, "upstream")
        bf16 = sample(bf16_base, bf16_index, module + ".weight", first, 32, "bf16")
        record = {"module": module, "format": kind, "bf16": bf16, "upstream": upstream}
        if kind != "BF16":
            record["scales"] = sample(
                upstream_base, upstream_index, module + ".weight_scale", first, 32, "upstream"
            )
        records.append(record)
        write_json(destination / "selection.json", {"sources": sources, "cases": records})
        print(f"Slices {len(records)}/{len(cases)}: {module}", flush=True)
    write_json(
        destination / "prepare-state.json",
        {
            "status": "complete",
            "cases": len(records),
            "sampled_payload_bytes": sum(
                v["bytes"]
                for r in records
                for v in r.values()
                if isinstance(v, dict) and "bytes" in v
            ),
        },
    )


def compare(root: Path, core_path: Path) -> None:
    """Measure raw-code/scale agreement and paired CPU reconstruction errors."""
    import resource

    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    destination = root / "row-probe-v1"
    selection = json.loads((destination / "selection.json").read_bytes())
    core = runpy.run_path(str(core_path))
    quantize = core["quantize_mxfp4"]
    dequant = core["dequant_mxfp4"]

    def load(record: dict[str, Any], dtype: torch.dtype) -> torch.Tensor:
        raw = (destination / record["path"]).read_bytes()
        if digest(raw) != record["sha256"]:
            raise ValueError("Sampled tensor bytes have changed")
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(record["shape"])

    report: dict[str, Any] = {
        "status": "running",
        "device": "cpu",
        "sources": selection["sources"],
        "core_sha256": digest(core_path.read_bytes()),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "quantization_revision": core["QUANTIZATION_REVISION"],
        "torch": torch.__version__,
        "cases": [],
        "limitations": [
            "Only 32 selected rows per matrix, with a deterministic non-random selection.",
            "Unweighted reconstruction is not activation replay or whole-model quality.",
            "The upstream card does not pin its original BF16 source revision.",
            "No inference, GPU memory or throughput test was performed.",
        ],
    }
    for record in selection["cases"]:
        source = load(record["bf16"], torch.bfloat16).float()
        kind = record["format"]
        signal = source.square().sum(dtype=torch.float64).item()
        case: dict[str, Any] = {
            "module": record["module"],
            "format": kind,
            "first_row": record["bf16"]["first_row"],
            "shape": list(source.shape),
            "signal_squared": signal,
        }
        if kind == "BF16":
            restored = load(record["upstream"], torch.bfloat16).float()
            case["bf16_bytes_exact"] = record["upstream"]["sha256"] == record["bf16"]["sha256"]
        else:
            scales = load(record["scales"], torch.uint8)
            if (scales == 255).any():
                raise ValueError("Reserved E8M0 NaN scale found in selected source rows")
            if kind == "MXFP4":
                packed = load(record["upstream"], torch.uint8)
                restored = dequant(packed, scales, tuple(source.shape))
                comparisons = {}
                for method in ("rtn", "mse"):
                    codes, selected = quantize(source, method=method)
                    reconstructed = dequant(codes, selected, tuple(source.shape))
                    error = (source - reconstructed).square().sum(dtype=torch.float64).item()
                    comparisons[method] = {
                        "packed_byte_agreement": (codes == packed).double().mean().item(),
                        "scale_byte_agreement": (selected == scales).double().mean().item(),
                        "error_squared": error,
                        "sqnr_db": 10 * math.log10(signal / error),
                    }
                case["mxwave"] = comparisons
            else:
                values = load(record["upstream"], torch.float8_e4m3fn).float()
                restored = values * torch.pow(2.0, scales.float() - 127).repeat_interleave(32, -1)
            histogram = torch.bincount(scales.flatten().long(), minlength=256)
            case["scale_histogram"] = {str(i): int(v) for i, v in enumerate(histogram) if v}
        if not torch.isfinite(source).all() or not torch.isfinite(restored).all():
            raise ValueError("Nonfinite original or reconstructed weight values")
        error = (source - restored).square().sum(dtype=torch.float64).item()
        case["upstream_error_squared"] = error
        case["upstream_sqnr_db"] = 10 * math.log10(signal / error) if error else None
        case["upstream_relative_l2"] = math.sqrt(error / signal)
        report["cases"].append(case)
        write_json(destination / "comparison.json", report)
        print(json.dumps({k: v for k, v in case.items() if k != "scale_histogram"}), flush=True)
    base_experts = [
        r for r in report["cases"] if r["format"] == "MXFP4" and ".layers.45." not in r["module"]
    ]
    signal = sum(r["signal_squared"] for r in base_experts)
    errors = {
        method: sum(r["mxwave"][method]["error_squared"] for r in base_experts)
        for method in ("rtn", "mse")
    }
    errors["upstream"] = sum(r["upstream_error_squared"] for r in base_experts)
    report["base_expert_pooled_sqnr_db"] = {
        method: 10 * math.log10(signal / error) for method, error in errors.items()
    }
    report["base_expert_mse_error_reduction_vs_upstream"] = 1 - errors["mse"] / errors["upstream"]
    report["status"] = "complete"
    report["seconds"] = time.monotonic() - started
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    write_json(destination / "comparison.json", report)


def main() -> None:
    """Prepare or compare a small immutable set of GLM weight rows."""
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "compare"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--core", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root)
    else:
        if args.core is None:
            parser.error("compare requires --core")
        compare(args.root, args.core)


if __name__ == "__main__":
    main()
