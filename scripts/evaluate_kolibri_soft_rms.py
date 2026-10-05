"""Measure a compact softened-RMS ablation using the unchanged Kolibri evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path


def main() -> None:
    """Probe loader compatibility or collect all frozen quality and throughput measurements."""
    import evaluate_kolibri_release as evaluator
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "collect"))
    parser.add_argument("--root", type=Path, default=Path("/work"))
    parser.add_argument("--specification", type=Path, required=True)
    args = parser.parse_args()
    specification = args.specification.resolve()
    spec_raw = specification.read_bytes()
    spec = json.loads(spec_raw)
    os.environ["KOLIBRI_SOFT_RMS_SPEC"] = str(specification)
    compilation = CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=[1, 2, 4],
    )
    runtime = {
        "moe_backend": "marlin",
        "enforce_eager": False,
        "compilation_config": compilation,
        "worker_extension_cls": "kolibri_soft_rms_worker.SoftRMSTrialWorker",
    }
    if args.phase == "smoke":
        started = time.monotonic()
        llm = LLM(
            model=spec["baseline"],
            load_format="safetensors",
            dtype="bfloat16",
            skip_tokenizer_init=True,
            seed=20261004,
            max_model_len=128,
            max_num_batched_tokens=128,
            max_num_seqs=2,
            gpu_memory_utilization=0.04,
            kv_cache_memory_bytes=64 * 2**20,
            **runtime,
        )
        result = llm.generate(
            {"prompt_token_ids": [1, 2, 3, 4]},
            SamplingParams(
                max_tokens=8, temperature=0, ignore_eos=True, detokenize=False, logprobs=5
            ),
            use_tqdm=False,
        )[0].outputs[0]
        if (
            len(result.token_ids) != 8
            or result.logprobs is None
            or len(result.logprobs) != 8
            or not all(
                math.isfinite(value.logprob) for row in result.logprobs for value in row.values()
            )
        ):
            raise ValueError("Soft RMS probe produced missing or non-finite outputs")
        audit = json.loads(specification.with_name("load-audit.json").read_text())
        if audit["status"] != "passed" or audit["verified_payloads"] != len(
            spec["effective_payload_sha256"]
        ):
            raise ValueError("Worker did not verify the complete effective checkpoint")
        record = {
            "status": "passed",
            "seconds": time.monotonic() - started,
            "output_tokens": list(result.token_ids),
            "specification_sha256": hashlib.sha256(spec_raw).hexdigest(),
            "scope": "Compact softened-RMS reconstruction loads the two-layer/eight-expert "
            "checkpoint; every effective payload verified before kernel processing",
        }
        specification.with_name("smoke-result.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        return
    root = args.root
    probe = json.loads((root / "soft-rms-trials/smoke/smoke-result.json").read_text())
    if probe["status"] != "passed":
        raise ValueError("Loader compatibility has not passed")
    separate = specification.parent / "evaluation"
    (separate / "validation").mkdir(parents=True, exist_ok=True)
    (separate / "measurements").mkdir(exist_ok=True)
    if (separate / "measurements/mse.json").exists():
        raise FileExistsError("Refusing to overwrite a completed scale trial")
    for name, target in (
        ("checkpoint-mse", Path(spec["baseline"])),
        ("reference-fp8", root / "reference-fp8"),
    ):
        link = separate / name
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
    shutil.copyfile(root / "validation/protocol.json", separate / "validation/protocol.json")
    for suffix in ("json", "safetensors"):
        link = separate / f"measurements/fp8.{suffix}"
        if not link.exists():
            link.symlink_to(root / f"measurements/fp8.{suffix}")

    class TrialLLM(LLM):
        """Use the qualified mixed kernels and explicit payload selector."""

        def __init__(self, **kwargs):
            kwargs.pop("linear_backend", None)
            kwargs["model"] = spec["baseline"]
            kwargs.update(runtime)
            super().__init__(**kwargs)

    vllm.LLM = TrialLLM
    namespace = argparse.Namespace(root=str(separate), label="mse", wait_conversion=False)
    evaluator.collect(namespace)
    if specification.read_bytes() != spec_raw:
        raise ValueError("Trial selection changed during evaluation")
    audit = json.loads(specification.with_name("load-audit.json").read_text())
    if audit["status"] != "passed" or audit["verified_payloads"] != len(
        spec["effective_payload_sha256"]
    ):
        raise ValueError("Incomplete effective payload verification")
    provenance = {
        "trial_specification_sha256": hashlib.sha256(spec_raw).hexdigest(),
        "baseline_manifest_sha256": spec["baseline_manifest_sha256"],
        "calibration_file_sha256": spec["calibration_file_sha256"],
        "route_counts_sha256": spec["counts_sha256"],
        "recipe": spec["recipe"],
        "retune_source_sha256": spec["retune_source_sha256"],
        "gamma_source_sha256": spec["gamma_source_sha256"],
        "runtime_wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection_source_sha256": hashlib.sha256(
            Path(__file__).with_name("kolibri_soft_rms_trial.py").read_bytes()
        ).hexdigest(),
        "worker_extension_sha256": hashlib.sha256(
            Path(__file__).with_name("kolibri_soft_rms_worker.py").read_bytes()
        ).hexdigest(),
        "load_audit": audit,
    }
    path = separate / "measurements/mse.json"
    measurement = json.loads(path.read_text())
    measurement["engine_kwargs"].pop("linear_backend", None)
    measurement["engine_kwargs"].update(
        model=spec["baseline"],
        moe_backend="marlin",
        enforce_eager=False,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4],
        },
        worker_extension_cls=runtime["worker_extension_cls"],
    )
    measurement.update(provenance)
    path.write_text(json.dumps(measurement, indent=2) + "\n")
    evaluator.compare(namespace)
    path = separate / "quality-comparison.json"
    quality = json.loads(path.read_text())
    quality["candidate"] = (
        "Frozen confidence-shrunk softened-RMS scale-selection ablation, official FP8 backbone"
    )
    quality.update(provenance)
    path.write_text(json.dumps(quality, indent=2) + "\n")


if __name__ == "__main__":
    main()
