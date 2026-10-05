"""Freeze and measure a bounded Kolibri release screen against its official FP8 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import time
from pathlib import Path
from typing import Any

SEED = 20261004
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
HUMANEVAL_REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
GERMANQUAD_REVISION = "16aa458eef26a021a3ad072f0acce63b129d355a"
IMAGE = "sha256:785fd756b4ae1cacda7612cb859dbf227330e8ced9a6fdea42e9379128bebce0"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _token_sha(tokens: list[int]) -> str:
    return _sha(b"".join(struct.pack("<q", token) for token in tokens))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _parquet(root: Path, repository: str, revision: str, filename: str) -> tuple[Any, dict]:
    from huggingface_hub import hf_hub_download
    from pyarrow import parquet

    path = Path(hf_hub_download(
        repository, filename, repo_type="dataset", revision=revision,
        local_dir=root / "dataset-source" / repository.replace("/", "--"), token=False,
    ))
    return parquet.read_table(path).to_pylist(), {
        "repository": repository, "revision": revision, "filename": filename,
        "file_sha256": _sha(path.read_bytes()),
    }


def _behavior_cases() -> list[dict[str, Any]]:
    return [
        {"id": "math-en", "content": "Compute 17 * 23. Answer with only the integer.",
         "expected": "391", "validator": "exact"},
        {"id": "math-de", "content": "Berechne 125 * 8. Antworte nur mit der ganzen Zahl.",
         "expected": "1000", "validator": "exact"},
        {"id": "retrieval-de", "content":
         "Im Inventar steht: Die Seriennummer lautet KOLIBRI-7429. "
         "Wie lautet die Seriennummer? Antworte ausschließlich mit der Seriennummer.",
         "expected": "KOLIBRI-7429", "validator": "exact"},
        {"id": "json", "content":
         'Return only a JSON object with keys "model" and "count", '
         'values "kolibri" and the integer 3. No markdown.',
         "expected": {"model": "kolibri", "count": 3}, "validator": "json"},
        {"id": "tool", "content":
         "Call get_weather for Berlin. Do not answer without calling the tool.",
         "tools": [{"type": "function", "function": {
             "name": "get_weather", "description": "Retrieve current weather for a city.",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                            "required": ["city"]},
         }}], "expected": {"name": "get_weather", "arguments": {"city": "Berlin"}},
         "validator": "tool"},
    ]


def prepare(args: argparse.Namespace) -> None:
    """Freeze dataset bytes, deterministic tokens, and small behavioral checks."""
    from aleph_alpha_inference import register
    from transformers import AutoTokenizer

    root = Path(args.root)
    output = root / "validation/protocol.json"
    if output.exists():
        raise FileExistsError(f"Protocol is already frozen: {output}")
    register()
    tokenizer = AutoTokenizer.from_pretrained(root / "source-bf16", local_files_only=True)
    english, english_source = _parquet(
        root, "Salesforce/wikitext", WIKITEXT_REVISION,
        "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
    )
    corpus = "\n\n".join(row["text"] for row in english)
    english_texts = [corpus[start:start + 6000] for start in range(0, len(corpus), 6000)]
    code, code_source = _parquet(
        root, "openai/openai_humaneval", HUMANEVAL_REVISION,
        "openai_humaneval/test-00000-of-00001.parquet",
    )
    german, german_source = _parquet(
        root, "deepset/germanquad", GERMANQUAD_REVISION,
        "plain_text/test-00000-of-00001.parquet",
    )
    german_texts = list(dict.fromkeys(row["context"] for row in german))
    sources = {
        "en": {**english_source, "split": "validation", "license": "cc-by-sa-3.0"},
        "de": {**german_source, "split": "test", "license": "cc-by-4.0"},
        "code": {**code_source, "split": "test", "license": "mit"},
    }
    windows = []
    for domain, texts in (
        ("en", english_texts), ("de", german_texts),
        ("code", [row["prompt"] + row["canonical_solution"] for row in code]),
    ):
        eligible = []
        for index, text in enumerate(texts):
            tokens = tokenizer.encode(text, add_special_tokens=False)[:1024]
            if len(tokens) >= (256 if domain != "code" else 32):
                eligible.append((index, text, tokens))
        if len(eligible) < 16:
            raise ValueError(f"Insufficient {domain} evaluation passages")
        for ordinal in range(16):
            index, text, tokens = eligible[ordinal * (len(eligible) - 1) // 15]
            windows.append({
                "id": f"{domain}-{ordinal}", "domain": domain, "source_index": index,
                "text_sha256": _sha(text.encode()), "token_ids": tokens,
                "token_ids_sha256": _token_sha(tokens),
                "kl_prefix_tokens": min(256, len(tokens) - 1),
            })
    cases = _behavior_cases()
    needle = "KOLIBRI-NEEDLE-86371"
    records = [f"Datensatz {index}: Stadt Birkenau, Produkt Papier, Status archiviert."
               for index in range(2000)]
    records[260] = f"Gesuchter Archivschlüssel: {needle}."
    long_text = "\n".join(records)
    long_ids = tokenizer.encode(long_text, add_special_tokens=False)[:7800]
    long_context = tokenizer.decode(long_ids)
    if needle not in long_context:
        raise ValueError("Long-context fixture lost its needle")
    cases.append({
        "id": "long-context-de", "content": long_context +
        "\nNenne nur den gesuchten Archivschlüssel.",
        "expected": needle, "validator": "exact", "max_tokens": 64,
    })
    for case in cases:
        case["token_ids"] = tokenizer.apply_chat_template(
            [{"role": "user", "content": case["content"]}],
            tools=case.get("tools"), tokenize=True, add_generation_prompt=True,
            reasoning_effort="none", return_dict=False,
        )
        case["token_ids_sha256"] = _token_sha(case["token_ids"])
    _write(output, {
        "format": "mxwave-kolibri-release-protocol-v1", "seed": SEED,
        "sources": sources, "tokenizer_source_revision":
        "7a8f290e7858825c3cf5e4c447ba68345de9f1d3",
        "windows": windows, "behavior_cases": cases,
        "quality_gate": {"max_mean_delta_nll": 0.05, "max_domain_delta_nll": 0.10,
                         "max_mean_next_token_kl": 0.03, "require_behavior_checks": True},
        "scope": "48 frozen passages and six behavioral checks; bounded release screen, "
        "not a full benchmark suite or a claim of validated 262k context",
        "activation_calibration": "none; these passages are used only for evaluation",
    })
    print(f"Frozen {len(windows)} passages and {len(cases)} checks", flush=True)


def _check(case: dict[str, Any], text: str) -> bool:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if case["validator"] == "exact":
        return text == case["expected"]
    try:
        if case["validator"] == "tool":
            match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
            return bool(match and json.loads(match[1]) == case["expected"])
        return json.loads(text) == case["expected"]
    except json.JSONDecodeError:
        return False


def collect(args: argparse.Namespace) -> None:
    """Measure paired prompt NLL, complete next-token distributions, and behavior."""
    import torch
    from evaluate_next_token_kl import _extract_full_logprobs
    from safetensors.torch import save_file
    from vllm import LLM, SamplingParams
    from vllm.logprobs import FlatLogprobs

    root = Path(args.root)
    protocol_raw = (root / "validation/protocol.json").read_bytes()
    protocol = json.loads(protocol_raw)
    output = root / "measurements" / f"{args.label}.json"
    if output.exists():
        raise FileExistsError(f"Refusing to replace a completed measurement: {output}")
    if args.wait_conversion:
        while not (root / "conversion-result.json").exists():
            print("Waiting for conversion to finish before isolated measurements", flush=True)
            time.sleep(30)
    kwargs = {
        "model": str(root / ("reference-fp8" if args.label == "fp8" else "checkpoint-mse")),
        "load_format": "safetensors", "dtype": "bfloat16", "seed": SEED,
        "enforce_eager": True, "max_model_len": 9216, "max_num_seqs": 4,
        "max_num_batched_tokens": 1024, "enable_chunked_prefill": True,
        "enable_prefix_caching": False, "kv_cache_dtype": "bfloat16",
        "kv_cache_memory_bytes": 2 * 1024**3, "gpu_memory_utilization": 0.85,
        "max_logprobs": -1, "disable_log_stats": True,
    }
    if args.label != "fp8":
        kwargs.update(linear_backend="marlin", moe_backend="marlin")
    started = time.monotonic()
    llm = LLM(**kwargs)
    report = {
        "label": args.label, "protocol_sha256": _sha(protocol_raw),
        "runtime_image": IMAGE, "plugin_version": "1.0.0", "engine_kwargs": kwargs,
        "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "load_seconds": time.monotonic() - started,
        "executed_source_sha256": _sha(Path(__file__).read_bytes()),
        "windows": [], "behavior": [], "throughput": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix = torch.empty((len(protocol["windows"]), 128000), dtype=torch.float32)
    for index, window in enumerate(protocol["windows"]):
        tokens = window["token_ids"]
        if _token_sha(tokens) != window["token_ids_sha256"]:
            raise ValueError("Frozen window token hash mismatch")
        result = llm.generate(
            {"prompt_token_ids": tokens},
            SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0), use_tqdm=False,
        )[0]
        prompt = result.prompt_logprobs
        if prompt is None or len(prompt) != len(tokens):
            raise ValueError("Prompt logprobs do not match the frozen tokens")
        values = [prompt[position][token].logprob for position, token in enumerate(tokens)
                  if position > 0 and prompt[position] is not None]
        if len(values) != len(tokens) - 1 or not all(math.isfinite(value) for value in values):
            raise ValueError("Prompt likelihoods are incomplete or non-finite")
        report["windows"].append({
            "id": window["id"], "domain": window["domain"],
            "token_ids_sha256": window["token_ids_sha256"], "scored_tokens": len(values),
            "nll": -sum(values), "mean_nll": -sum(values) / len(values),
        })
        kl_result = llm.generate(
            {"prompt_token_ids": tokens[:window["kl_prefix_tokens"]]},
            SamplingParams(max_tokens=1, temperature=1, logprobs=-1, flat_logprobs=True,
                           seed=SEED, detokenize=False), use_tqdm=False,
        )[0].outputs[0]
        if not isinstance(kl_result.logprobs, FlatLogprobs):
            raise TypeError("Exact next-token comparison requires FlatLogprobs")
        matrix[index] = _extract_full_logprobs(kl_result.logprobs, 128000)
        _write(output.with_suffix(".partial.json"), report)
        print(f"{args.label}: passage {index + 1}/{len(matrix)}", flush=True)
    save_file({"logprobs": matrix}, str(output.with_suffix(".safetensors")),
              metadata={"protocol_sha256": _sha(protocol_raw), "label": args.label})
    for case in protocol["behavior_cases"]:
        tokens = case["token_ids"]
        if _token_sha(tokens) != case["token_ids_sha256"]:
            raise ValueError("Frozen behavior token hash mismatch")
        completion = llm.generate(
            {"prompt_token_ids": tokens},
            SamplingParams(max_tokens=case.get("max_tokens", 128), temperature=0, seed=SEED),
            use_tqdm=False,
        )[0].outputs[0]
        report["behavior"].append({
            "id": case["id"], "prompt_tokens": len(tokens), "text": completion.text,
            "token_ids": list(completion.token_ids), "passed": _check(case, completion.text),
        })
        _write(output.with_suffix(".partial.json"), report)
    # Both models use identical frozen prompts, forced output lengths, and eager execution.
    prompt = protocol["windows"][0]["token_ids"][:512]
    params = SamplingParams(max_tokens=128, temperature=0, ignore_eos=True, detokenize=False)
    llm.generate({"prompt_token_ids": prompt}, SamplingParams(max_tokens=16, temperature=0),
                 use_tqdm=False)
    for concurrency in (1, 4):
        for repetition in range(3):
            before = time.monotonic()
            outputs = llm.generate([{"prompt_token_ids": prompt}] * concurrency, params,
                                   use_tqdm=False)
            elapsed = time.monotonic() - before
            count = sum(len(item.outputs[0].token_ids) for item in outputs)
            if count != 128 * concurrency:
                raise ValueError("Throughput measurement did not produce the forced token count")
            report["throughput"].append({
                "concurrency": concurrency, "repetition": repetition,
                "prompt_tokens_per_request": len(prompt), "output_tokens_per_request": 128,
                "seconds": elapsed, "aggregate_output_tokens_per_second": count / elapsed,
            })
    report["seconds"] = time.monotonic() - started
    _write(output, report)
    print(f"Completed {args.label}: {output}", flush=True)


def compare(args: argparse.Namespace) -> None:
    """Compare frozen measurements and enforce the prespecified bounded quality gate."""
    import numpy as np
    from evaluate_next_token_kl import _distribution_metrics
    from safetensors.torch import load_file

    root = Path(args.root)
    raw = (root / "validation/protocol.json").read_bytes()
    protocol = json.loads(raw)
    ref = json.loads((root / "measurements/fp8.json").read_text())
    candidate = json.loads((root / "measurements/mse.json").read_text())
    if ref["protocol_sha256"] != _sha(raw) or candidate["protocol_sha256"] != _sha(raw):
        raise ValueError("Measurements used different protocols")
    for report in (ref, candidate):
        if len(report["windows"]) != len(protocol["windows"]):
            raise ValueError("Likelihood measurements do not cover every frozen passage")
        if [row["id"] for row in report["behavior"]] != [
            case["id"] for case in protocol["behavior_cases"]
        ]:
            raise ValueError("Behavior measurements do not cover the frozen checks")
    distributions = [load_file(str(root / f"measurements/{label}.safetensors"))["logprobs"]
                     for label in ("fp8", "mse")]
    rows = []
    for index, (left, right) in enumerate(zip(ref["windows"], candidate["windows"], strict=True)):
        if (left["id"], left["token_ids_sha256"], left["scored_tokens"]) != (
            right["id"], right["token_ids_sha256"], right["scored_tokens"]
        ):
            raise ValueError("Paired likelihood windows disagree")
        metrics = _distribution_metrics(distributions[0][index], distributions[1][index])
        rows.append({**left, "candidate_nll": right["nll"],
                     "delta_mean_nll": right["mean_nll"] - left["mean_nll"], **metrics})
    domains = {}
    for domain in ("all", "en", "de", "code"):
        subset = [row for row in rows if domain == "all" or row["domain"] == domain]
        count = sum(row["scored_tokens"] for row in subset)
        left_nll = sum(row["nll"] for row in subset) / count
        right_nll = sum(row["candidate_nll"] for row in subset) / count
        deltas = np.asarray([row["delta_mean_nll"] for row in subset])
        rng = np.random.default_rng(SEED)
        bootstrap = rng.choice(deltas, size=(10000, len(deltas)), replace=True).mean(axis=1)
        domains[domain] = {
            "passages": len(subset), "scored_tokens": count,
            "reference_ppl": math.exp(left_nll), "candidate_ppl": math.exp(right_nll),
            "delta_mean_nll": right_nll - left_nll,
            "ppl_ratio": math.exp(right_nll - left_nll),
            "passage_mean_delta_nll_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        }
    mean_kl = sum(row["forward_kl_nats"] for row in rows) / len(rows)
    gate = protocol["quality_gate"]
    passed = (
        domains["all"]["delta_mean_nll"] <= gate["max_mean_delta_nll"]
        and all(domains[name]["delta_mean_nll"] <= gate["max_domain_delta_nll"]
                for name in ("en", "de", "code"))
        and mean_kl <= gate["max_mean_next_token_kl"]
        and all(row["passed"] for row in candidate["behavior"])
    )
    _write(root / "quality-comparison.json", {
        "status": "passed" if passed else "failed", "protocol_sha256": _sha(raw),
        "reference": "official Aleph-Alpha/Kolibri-1 FP8", "candidate": "expert-only MXFP4 MSE",
        "gate": gate, "domains": domains, "mean_next_token_kl": mean_kl,
        "behavior": {label: report["behavior"] for label, report in
                     (("fp8", ref), ("mse", candidate))},
        "per_passage": rows, "scope": protocol["scope"],
    })
    print(json.dumps({"status": "passed" if passed else "failed", "domains": domains,
                      "mean_next_token_kl": mean_kl}), flush=True)


def main() -> None:
    """Run one phase of the bounded Kolibri release qualification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "compare"))
    parser.add_argument("--root", default="/work")
    parser.add_argument("--label", choices=("fp8", "mse"))
    parser.add_argument("--wait-conversion", action="store_true")
    args = parser.parse_args()
    if args.phase == "collect" and args.label is None:
        parser.error("collect requires --label")
    {"prepare": prepare, "collect": collect, "compare": compare}[args.phase](args)


if __name__ == "__main__":
    main()
