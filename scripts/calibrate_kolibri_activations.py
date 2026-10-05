"""Freeze separate calibration prompts and collect routing-aware Kolibri activation RMS."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import time
from pathlib import Path


def prepare(root: Path) -> None:
    """Freeze training-split calibration windows separately from the held-out release screen."""
    from aleph_alpha_inference import register
    from evaluate_kolibri_release import (
        GERMANQUAD_REVISION,
        WIKITEXT_REVISION,
        _parquet,
        _token_sha,
    )
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    from mxwave.calibration_cli import _tokenize_sequences

    output = root / "activation-calibration/protocol.json"
    if output.exists():
        raise FileExistsError("Refusing to replace the frozen activation calibration prompts")
    register()
    tokenizer = AutoTokenizer.from_pretrained(root / "source-bf16", local_files_only=True)
    english, en_source = _parquet(root, "Salesforce/wikitext", WIKITEXT_REVISION,
                                "wikitext-2-raw-v1/train-00000-of-00001.parquet")
    german, de_source = _parquet(root, "deepset/germanquad", GERMANQUAD_REVISION,
                               "plain_text/train-00000-of-00001.parquet")
    mbpp = HfApi().dataset_info("google-research-datasets/mbpp")
    filename = next(file.rfilename for file in mbpp.siblings
                    if file.rfilename.startswith("full/train-") and file.rfilename.endswith(".parquet"))
    code, code_source = _parquet(root, "google-research-datasets/mbpp", mbpp.sha, filename)
    german_texts = list(dict.fromkeys(row["context"] for row in german))
    domains = (
        ("en", [row["text"] for row in english], 24, en_source, "CC BY-SA 3.0"),
        ("de", german_texts, 24, de_source, "CC BY 4.0"),
        ("code", [row["text"] + "\n\n" + row["code"] for row in code], 16,
         code_source, "CC BY 4.0"),
    )
    windows, sources = [], []
    frozen = json.loads((root / "validation/protocol.json").read_text())
    frozen_hashes = {window["token_ids_sha256"] for window in frozen["windows"]}
    for domain, texts, count, provenance, license_name in domains:
        sequences = _tokenize_sequences(tokenizer, texts, num_sequences=count, sequence_length=512)
        sources.append({"domain": domain, **provenance, "license": license_name})
        for index, sequence in enumerate(sequences):
            digest = _token_sha(sequence)
            if digest in frozen_hashes:
                raise ValueError("Calibration window duplicates a frozen evaluation window")
            windows.append({"id": f"{domain}-{index:02d}", "domain": domain,
                            "token_ids": sequence, "token_ids_sha256": digest})
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({
        "sources": sources, "windows": windows, "sequence_length": 512,
        "num_tokens": sum(len(window["token_ids"]) for window in windows),
        "teacher_repository": "Aleph-Alpha/Kolibri-1",
        "teacher_revision": "e52eb4627d11516b0c01de49210ab5a4e4061444",
        "weight_source_repository": "Aleph-Alpha/Kolibri-1-BF16",
        "weight_source_revision": "7a8f290e7858825c3cf5e4c447ba68345de9f1d3",
        "selection": "Deterministic training-split packing; held-out release screen unchanged",
    }, indent=2) + "\n")
    print(json.dumps({"status": "prepared", "windows": len(windows), "tokens": 32768}), flush=True)


def collect(root: Path, *, smoke: bool = False) -> None:
    """Collect actual gate/up and down input statistics from the FP8 teacher with eager execution."""
    from kolibri_activation_capture import finish_capture, install_capture
    from vllm import LLM, SamplingParams

    started = time.monotonic()
    directory = root / "activation-calibration"
    directory.mkdir(exist_ok=True)
    path = directory / ("smoke-rms.safetensors" if smoke else "routed-rms.safetensors")
    if path.exists():
        raise FileExistsError("Refusing to replace a complete calibration artifact")
    protocol_raw = b"" if smoke else (directory / "protocol.json").read_bytes()
    protocol = {"windows": [{"token_ids": [1, 2, 3, 4]}], "sequence_length": 4,
                "num_tokens": 4} if smoke else json.loads(protocol_raw)
    llm = LLM(
        model=str(root / ("smoke-reference-fp8" if smoke else "reference-fp8")),
        load_format="safetensors", dtype="bfloat16", seed=20261004, enforce_eager=True,
        skip_tokenizer_init=smoke, max_model_len=128 if smoke else 1024,
        max_num_batched_tokens=128 if smoke else 1024, max_num_seqs=1,
        kv_cache_memory_bytes=(64 if smoke else 256) * 2**20,
        gpu_memory_utilization=0.04 if smoke else 0.85, enable_prefix_caching=False,
        moe_backend="triton",
    )
    installed = llm.apply_model(install_capture)
    print(json.dumps({"capture": installed}), flush=True)
    smoke_tokens = None
    for index, window in enumerate(protocol["windows"], 1):
        result = llm.generate(
            {"prompt_token_ids": window["token_ids"]},
            SamplingParams(max_tokens=8 if smoke else 1, temperature=0, ignore_eos=True,
                           detokenize=False), use_tqdm=False,
        )[0].outputs[0]
        if smoke:
            smoke_tokens = list(result.token_ids)
            if smoke_tokens != json.loads((root / "fp8-smoke-result.json").read_text())["output_tokens"]:
                raise ValueError("Installing observers changed the native FP8 smoke token outputs")
        print(f"Calibration window {index}/{len(protocol['windows'])}", flush=True)
    metadata = {
        "source_repository": "Aleph-Alpha/Kolibri-1-BF16",
        "source_revision": "7a8f290e7858825c3cf5e4c447ba68345de9f1d3",
        "teacher_repository": "Aleph-Alpha/Kolibri-1",
        "teacher_revision": "e52eb4627d11516b0c01de49210ab5a4e4061444",
        "num_sequences": str(len(protocol["windows"])),
        "sequence_length": str(protocol["sequence_length"]), "num_tokens": str(protocol["num_tokens"]),
        "token_ids_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "corpus_sha256": hashlib.sha256(protocol_raw).hexdigest(),
    }
    reports = llm.apply_model(functools.partial(finish_capture, path=str(path), metadata=metadata))
    record = {"status": "passed", "seconds": time.monotonic() - started, "reports": reports,
              "smoke": smoke, "output_tokens": smoke_tokens,
              "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (directory / ("smoke-result.json" if smoke else "result.json")).write_text(
        json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


def main() -> None:
    """Freeze calibration windows, verify observer compatibility, or collect full statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "smoke", "collect"))
    parser.add_argument("--root", type=Path, default=Path("/work"))
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.root)
    else:
        collect(args.root, smoke=args.phase == "smoke")


if __name__ == "__main__":
    main()
