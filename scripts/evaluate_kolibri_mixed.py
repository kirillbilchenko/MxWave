"""Qualify MXFP4 experts plus the official FP8 backbone on the frozen release screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path


def main() -> None:
    """Check mixed runtime compatibility or collect separate full-model measurements."""
    import evaluate_kolibri_release as evaluator
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "collect"))
    parser.add_argument("--root", type=Path, default=Path("/work"))
    parser.add_argument("--moe-backend", choices=("marlin", "b12x"), default="marlin")
    args = parser.parse_args()
    root = args.root
    wrapper_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.phase == "smoke":
        started = time.monotonic()
        llm = LLM(
            model=str(root / "smoke-mixed"), load_format="safetensors", dtype="bfloat16",
            skip_tokenizer_init=True, enforce_eager=False, seed=20261004,
            max_model_len=128, max_num_batched_tokens=128, max_num_seqs=2,
            gpu_memory_utilization=0.04, kv_cache_memory_bytes=64 * 2**20,
            moe_backend=args.moe_backend,
            compilation_config=CompilationConfig(
                mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                cudagraph_capture_sizes=[1, 2],
            ),
        )
        result = llm.generate(
            {"prompt_token_ids": [1, 2, 3, 4]},
            SamplingParams(max_tokens=8, temperature=0, ignore_eos=True,
                           detokenize=False, logprobs=5), use_tqdm=False,
        )[0].outputs[0]
        if (len(result.token_ids) != 8 or result.logprobs is None or len(result.logprobs) != 8
                or not all(math.isfinite(value.logprob)
                           for row in result.logprobs for value in row.values())):
            raise ValueError("Mixed-format probe did not produce complete finite outputs")
        record = {
            "status": "passed", "seconds": time.monotonic() - started,
            "output_tokens": list(result.token_ids), "wrapper_sha256": wrapper_sha,
            "moe_backend": args.moe_backend, "linear_backend": "auto",
            "scope": "Two layers/eight experts with production geometry; compatibility only",
        }
        (root / "mixed-smoke-result.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        return

    if json.loads((root / "mixed-smoke-result.json").read_text())["status"] != "passed":
        raise RuntimeError("The mixed-format compatibility probe has not passed")
    separate = root / "mixed-evaluation"
    (separate / "validation").mkdir(parents=True, exist_ok=True)
    (separate / "measurements").mkdir(exist_ok=True)
    if (separate / "measurements/mse.json").exists():
        raise FileExistsError("Refusing to overwrite a completed mixed-format measurement")
    for name, target in (("checkpoint-mse", "checkpoint-mixed"),
                         ("source-bf16", "source-bf16"), ("reference-fp8", "reference-fp8")):
        link = separate / name
        if not link.exists():
            link.symlink_to(root / target, target_is_directory=True)
    shutil.copyfile(root / "validation/protocol.json", separate / "validation/protocol.json")
    for suffix in ("json", "safetensors"):
        link = separate / f"measurements/fp8.{suffix}"
        if not link.exists():
            link.symlink_to(root / f"measurements/fp8.{suffix}")

    class MixedLLM(LLM):
        """Override execution while preserving the historical evaluator's exact bytes."""

        def __init__(self, **kwargs):
            kwargs.pop("linear_backend", None)
            kwargs.update(
                moe_backend=args.moe_backend, enforce_eager=False,
                compilation_config=CompilationConfig(
                    mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                    cudagraph_capture_sizes=[1, 2, 4],
                ),
            )
            super().__init__(**kwargs)

    vllm.LLM = MixedLLM
    namespace = argparse.Namespace(root=str(separate), label="mse", wait_conversion=False)
    evaluator.collect(namespace)
    path = separate / "measurements/mse.json"
    report = json.loads(path.read_text())
    report["engine_kwargs"].pop("linear_backend", None)
    report["engine_kwargs"].update(
        moe_backend=args.moe_backend, enforce_eager=False,
        compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                            "cudagraph_capture_sizes": [1, 2, 4]},
    )
    report["runtime_wrapper_sha256"] = wrapper_sha
    report["checkpoint_manifest_sha256"] = hashlib.sha256(
        (root / "checkpoint-mixed/mxwave-manifest.json").read_bytes()).hexdigest()
    path.write_text(json.dumps(report, indent=2) + "\n")
    evaluator.compare(namespace)
    path = separate / "quality-comparison.json"
    quality = json.loads(path.read_text())
    quality["candidate"] = "BF16-master MXFP4 MSE experts, official FP8 backbone, full decode graphs"
    quality["runtime_wrapper_sha256"] = wrapper_sha
    quality["checkpoint_manifest_sha256"] = report["checkpoint_manifest_sha256"]
    path.write_text(json.dumps(quality, indent=2) + "\n")


if __name__ == "__main__":
    main()
