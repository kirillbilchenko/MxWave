"""Package and publish the independently measured standalone softer-RMS checkpoint."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import statistics
from pathlib import Path

from publish_kolibri_release import REPOSITORY, _require_qualified, publish, stage


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _link(view: Path, name: str, target: Path) -> None:
    link = view / name
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise ValueError(f"Unexpected existing release link: {link}")
    elif link.exists():
        raise ValueError(f"Release link would replace an existing path: {link}")
    else:
        link.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def prepare(root: Path, archive: Path, provenance: Path) -> None:
    """Bind stock-loader qualification, exact payloads, calibration, and release source."""
    output = root / "checkpoint-soft-rms"
    manifest_path = output / "mxwave-manifest.json"
    manifest = _read(manifest_path)
    digest = _sha(manifest_path)
    export = _read(root / "checkpoint-soft-rms-export.json")
    result = _read(root / "soft-rms-release-result.json")
    directory = root / "soft-rms-standalone-check/mixed-evaluation"
    quality = _read(directory / "quality-comparison.json")
    measurement = _read(directory / "measurements/mse.json")
    source = _read(provenance)
    if (
        export["status"] != "complete"
        or export["shards"] != 32
        or export["verified_payloads"] != 116303
        or export["manifest_sha256"] != digest
        or result["checkpoint_manifest_sha256"] != digest
        or quality["checkpoint_manifest_sha256"] != digest
        or measurement["checkpoint_manifest_sha256"] != digest
        or manifest["config_coverage_gaps"]
        or len(manifest["payload_sha256"]) != 116303
        or len(manifest["shard_integrity"]["per_shard"]) != 32
    ):
        raise ValueError("Standalone qualification does not identify the complete exported payloads")
    expected_source = {
        "scripts/export_kolibri_soft_rms.py": export["export_source_sha256"],
        "scripts/kolibri_soft_rms_trial.py": export["patch_loader_source_sha256"],
        "scripts/evaluate_kolibri_mixed.py": measurement["runtime_wrapper_sha256"],
        "scripts/evaluate_kolibri_release.py": measurement["executed_source_sha256"],
    }
    if _sha(archive) != source["archive_sha256"] or any(
        source["source_files_sha256"][name] != expected
        for name, expected in expected_source.items()
    ):
        raise ValueError("Archived release source differs from executed export or qualification")
    specification = root / "soft-rms-trial.json.gz"
    with gzip.open(specification, "rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != export["trial_specification_sha256"]:
            raise ValueError("Archived softer-RMS specification differs from the measured recipe")
    calibration = root / "checkpoint-activation-rms/calibration"
    for name, key in (("routed-rms.safetensors", "calibration_file_sha256"),
                      ("routed-rms.json", "counts_sha256")):
        if _sha(calibration / name) != manifest["activation_calibration"][key]:
            raise ValueError(f"Calibration artifact changed: {name}")

    view = root / "soft-rms-publication"
    view.mkdir(exist_ok=True)
    links = {
        "checkpoint-mse": output,
        "source-bf16": root / "source-bf16",
        "validation": root / "validation",
        "conversion-result.json": root / "checkpoint-soft-rms-export.json",
        "qualification-result.json": root / "soft-rms-release-result.json",
        "quality-comparison.json": directory / "quality-comparison.json",
    }
    for label in ("fp8", "mse"):
        base = root if label == "fp8" else directory
        for suffix in ("json", "safetensors"):
            links[f"measurements/{label}.{suffix}"] = base / f"measurements/{label}.{suffix}"
    for name, target in links.items():
        _link(view, name, target)
    proof = {
        "status": "passed", "manifest_sha256": digest, "verified_payloads": 116303,
        "payload_counts": manifest["payload_counts"], "config_coverage_gaps": [],
        "trial_specification_sha256": export["trial_specification_sha256"],
        "scope": "All exported raw payloads match the measured compact trial; every shard "
        "has an independently recorded file SHA256. Soft-RMS SQNR was not measured.",
    }
    (view / "payload-verification.json").write_text(json.dumps(proof, indent=2) + "\n")
    _require_qualified(view, experimental=True)
    evidence = output / "validation"
    evidence.mkdir(exist_ok=True)
    for name in ("quality-comparison.json", "qualification-result.json", "conversion-result.json",
                 "payload-verification.json", "validation/protocol.json"):
        shutil.copyfile(view / name, evidence / Path(name).name)
    for label in ("fp8", "mse"):
        for suffix in ("json", "safetensors"):
            shutil.copyfile(view / f"measurements/{label}.{suffix}", evidence / f"{label}.{suffix}")
    for name in ("soft-rms-result.json", "soft-rms-weight-audit-summary.json",
                 "soft-rms-trial.json.gz", "soft-rms-retune-state.json.gz",
                 "cleanup-strong-rms-for-soft-release.json", "measurements/fp8-decode-graphs.json"):
        shutil.copyfile(root / name, evidence / Path(name).name)
    trial = root / "soft-rms-trials/strength025"
    for name in ("conversion-result.json", "load-audit.json"):
        shutil.copyfile(trial / name, evidence / ("compact-trial-" + name))
    destination = output / "calibration"
    destination.mkdir(exist_ok=True)
    for name in ("routed-rms.safetensors", "routed-rms.json", "protocol.json", "verified-licenses.json"):
        shutil.copyfile(calibration / name, destination / name)
    shutil.copyfile(archive, output / "mxwave-source.tar.gz")
    shutil.copyfile(provenance, evidence / "release-source.json")
    shutil.copyfile(root / "source-bf16/LICENSE", output / "LICENSE")
    notice = root / "source-bf16/NOTICE"
    if notice.is_file():
        shutil.copyfile(notice, output / "NOTICE")
    else:
        (output / "NOTICE").write_text(
            "Kolibri-1 weights provided by Aleph Alpha GmbH and developed by Aleph Alpha "
            "Research GmbH. Original model weights and configurations are Apache-2.0 licensed.\n"
            "Expert source: Aleph-Alpha/Kolibri-1-BF16 at "
            "7a8f290e7858825c3cf5e4c447ba68345de9f1d3.\n"
            "Backbone source: Aleph-Alpha/Kolibri-1 at "
            "e52eb4627d11516b0c01de49210ab5a4e4061444.\n"
            "MxWave modifications: routed experts converted to MXFP4 with softer "
            "activation-RMS scale selection; official FP8 backbone payloads preserved.\n"
            "Calibration, source provenance, and experimental qualification are included.\n"
        )

    rows = []
    for domain in ("en", "de", "code", "all"):
        row = quality["domains"][domain]
        rows.append(f"| {domain} | {row['scored_tokens']:,} | {row['reference_ppl']:.6f} | "
                    f"{row['candidate_ppl']:.6f} | {100 * (row['ppl_ratio'] - 1):+.2f}% |")
    speed = []
    for title, report in (("Official FP8", _read(root / "measurements/fp8-decode-graphs.json")),
                          ("Softer RMS MXFP4", measurement)):
        rates = [statistics.median(row["aggregate_output_tokens_per_second"]
                                  for row in report["throughput"] if row["concurrency"] == c)
                 for c in (1, 4)]
        speed.append(f"| {title} | {rates[0]:.2f} | {rates[1]:.2f} |")
    size = export["weight_files_bytes"]
    card = f"""---
license: apache-2.0
base_model:
- Aleph-Alpha/Kolibri-1-BF16
- Aleph-Alpha/Kolibri-1
library_name: vllm
pipeline_tag: text-generation
language:
- en
- de
tags:
- kolibri1
- moe
- mxfp4
- compressed-tensors
- mxwave
- experimental
---

# Kolibri-1 MXFP4 — MxWave softer RMS

**Experimental release.** The unchanged strict KL target was missed:
**{quality['mean_next_token_kl']:.6f} nats versus {quality['gate']['max_mean_next_token_kl']:.3f}**.
The likelihood bounds and all six behavior checks passed. The original failed
screen is included in `validation/`; its threshold was not changed.

The 32 safetensors weight files total **{size / 1024**3:.2f} GiB**
({size:,} bytes), 44.9% smaller than the official FP8 files. All 57,600 routed
expert matrices use OCP MXFP4 E2M1 with one E8M0 scale per 32 weights. The official
block-FP8 attention and shared-expert backbone, scales, routers, embeddings,
norms, and head retain their original payload bytes. Expert inference uses
Marlin W4A16; backbone activation quantization remains dynamic FP8.

## Recipe and provenance

Expert source: [`Aleph-Alpha/Kolibri-1-BF16`](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16)
at `7a8f290e7858825c3cf5e4c447ba68345de9f1d3`.
Backbone and behavioral reference: [`Aleph-Alpha/Kolibri-1`](https://huggingface.co/Aleph-Alpha/Kolibri-1)
at `e52eb4627d11516b0c01de49210ab5a4e4061444`.

Scale selection blends uniform MSE with routed activation variance:
`alpha = 0.25 * n / (n + 128)`;
`weight = (1 - alpha) + alpha * clip(RMS² / mean(RMS²), 0.25, 4)`.
Target-wide gamma normalization preserves this objective. MSE clip depth is 4,
percentile 99.5, with round-to-nearest-even `mxfp4-rne-v2`. No rotation is used.
The teacher collected 32,768 tokens in 64 training-split windows from English
WikiText-2, GermanQuAD, and MBPP. Evaluation passages were excluded.
53,646 projections were observed; 3,954 unobserved projections retain their
unweighted payloads. 53,591 projections changed, affecting 0.3033% of expert blocks.

Every one of the **116,303 exported payloads** matches the measured compact trial.
This standalone checkpoint was then measured with the stock mixed-format vLLM
loader. Serving requires no private trial worker or patch files. The converter
manifest is immutable export-time provenance; subsequent qualification is recorded
in `validation/qualification-result.json`. Soft-RMS SQNR was not independently measured.

## Run on DGX Spark

Tested on NVIDIA GB10 / SM121 with vLLM 0.29.0 and
`aleph-alpha-inference==1.0.0`. The plugin implements Kolibri's routing, sandwich
norms, and mixed sliding/full attention; install it in every runtime process.
It declares `vllm>=0.29.0,<0.30.0`. vLLM 0.30 was not qualified for this release.

```bash
python -m pip install 'vllm==0.29.0' 'aleph-alpha-inference==1.0.0'
VLLM_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \\
vllm serve {REPOSITORY} \\
  --load-format safetensors --dtype bfloat16 --moe-backend marlin \\
  --max-model-len 9216 --max-num-batched-tokens 1024 --max-num-seqs 4 \\
  --kv-cache-dtype bfloat16 --kv-cache-memory-bytes 2147483648 \\
  --gpu-memory-utilization 0.85 --no-enable-prefix-caching \\
  --compilation-config '{{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}}' \\
  --enable-auto-tool-choice --tool-call-parser kolibri1 --reasoning-parser kolibri1
```

Startup must select `MarlinExperts` for routed experts and automatic block-FP8
dense kernels for the backbone. Inference memory includes weights, KV cache,
and kernel workspaces in addition to the 40.45 GiB weight files.

## Measured quality and speed

The frozen screen scores 48 passages: 16 English WikiText-2 validation windows,
16 unique GermanQuAD test passages, and 16 HumanEval prompts with canonical
solutions. Code PPL is a likelihood screen, not HumanEval pass@1.

| Domain | Scored tokens | Official FP8 PPL | Softer RMS PPL | Relative PPL |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Mean full-vocabulary forward KL uses one fixed prefix per passage. The six
behavior checks cover English/German arithmetic, German retrieval, JSON,
a tool call, and retrieval near 8k tokens, with reasoning disabled.
The screen does not validate the advertised full context or broad task accuracy.
The original compact trial measured PPL 24.482698 and mean KL 0.053467,
8.7% lower mean KL than the unweighted mixed candidate's 0.058545.
Its paired passage-bootstrap interval includes zero, and some prefixes worsen.
This bounded result does not establish a broad quality improvement.

| Model / decode graphs | Output tok/s, c1 | Aggregate output tok/s, c4 |
|---|---:|---:|
{chr(10).join(speed)}

Throughput uses 512 input tokens, 128 forced output tokens, BF16 KV, no prefix
caching, and the median of three repetitions. Both use graph captures `[1,2,4]`
with Torch compilation disabled. The standalone run overlapped a CPU/network
weight upload; its timing describes that run and may include host I/O contention.
The earlier compact trial measured 48.86 / 135.26 tok/s at c1/c4.

The frozen protocol, raw full-vocabulary distributions, hashes, export proof,
and original failed qualification are included in `validation/` and
`mxwave-manifest.json`. Exact routed statistics, training tokens, source pins,
and dataset licenses are included in `calibration/`. WikiText uses CC BY-SA 3.0,
GermanQuAD CC BY 4.0, HumanEval MIT, and MBPP CC BY 4.0 as recorded in the protocols.
The licensed converter and tests are in `mxwave-source.tar.gz`; archive and Git
provenance are in `validation/release-source.json`. Extract and run
`pip install -e '.[dev]'`. Vendor plugin source is not bundled.
[MxWave source](https://github.com/kirillbilchenko/MxWave).

## GGUF

These files use compressed-tensors for vLLM. A Kolibri GGUF adapter and compatible
llama.cpp runtime require separate implementation and validation. Stock llama.cpp
and Ollama support is not claimed for this checkpoint.
"""
    (output / "README.md").write_text(card)
    prepared = {
        "repository": REPOSITORY, "manifest_sha256": digest,
        "quality_sha256": _sha(view / "quality-comparison.json"), "prepared": True,
        "experimental": True, "strict_screen_status": quality["status"],
    }
    (view / "publication-prepared.json").write_text(json.dumps(prepared, indent=2) + "\n")
    print(json.dumps(prepared), flush=True)


def stage_preuploaded(root: Path) -> None:
    """Adopt only this verified private preupload, then stage all metadata and evidence."""
    from huggingface_hub import HfApi

    view = root / "soft-rms-publication"
    prepared = _read(view / "publication-prepared.json")
    upload = _read(view / "private-weight-upload.json")
    export = _read(root / "checkpoint-soft-rms-export.json")
    api = HfApi()
    if (
        upload["status"] != "weights-verified-private"
        or upload["verified_shards"] != 32
        or upload["repository"] != REPOSITORY
        or upload["manifest_sha256"] != prepared["manifest_sha256"]
        or upload["trial_specification_sha256"] != export["trial_specification_sha256"]
        or api.whoami()["name"] != REPOSITORY.split("/")[0]
        or not api.model_info(REPOSITORY).private
    ):
        raise ValueError("Preupload is not the verified private softer-RMS checkpoint")
    marker = view / "upload-repository.json"
    if marker.exists() and _read(marker) != prepared:
        raise ValueError("Existing staging intent belongs to another checkpoint")
    marker.write_text(json.dumps(prepared, indent=2) + "\n")
    stage(view)


def main() -> None:
    """Prepare, privately stage, or publish the user-selected experimental checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "stage", "publish"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--source-record", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.source_archive is None or args.source_record is None:
            parser.error("prepare requires --source-archive and --source-record")
        prepare(args.root.resolve(), args.source_archive, args.source_record)
    elif args.phase == "stage":
        stage_preuploaded(args.root.resolve())
    else:
        publish(args.root.resolve() / "soft-rms-publication")


if __name__ == "__main__":
    main()
