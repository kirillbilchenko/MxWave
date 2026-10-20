"""Frozen public test samples, strict practical scoring and token-weighted log loss."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
from benchmark_full_model import request, write
from measure_palette_model import final_text

BASE = "http://127.0.0.1:18784"


def numeric_answer(text: str) -> Decimal | None:
    """Accept an explicit final numeric answer, never a number inside unfinished reasoning."""
    value = text.rsplit("####", 1)[-1].strip() if "####" in text else text.strip()
    if not re.fullmatch(r"[-+]?\$?\d[\d,]*(?:\.\d+)?\.?", value):
        return None
    return Decimal(value.replace("$", "").replace(",", "").rstrip("."))


def practical_cases() -> list[dict[str, Any]]:
    """Specify exact output contracts before either model arm is evaluated."""
    cases = [
        {
            "name": "json",
            "kind": "json",
            "prompt": "Return only this JSON object: "
            '{"status":"ok","count":3,"items":["a","b","c"]}.',
            "expected": {"status": "ok", "count": 3, "items": ["a", "b", "c"]},
        },
        {
            "name": "json_types",
            "kind": "json",
            "prompt": "Return only JSON with keys "
            'enabled=false, retries=0, note=null, tags=["red","blue"].',
            "expected": {"enabled": False, "retries": 0, "note": None, "tags": ["red", "blue"]},
        },
        {
            "name": "constraints",
            "kind": "exact",
            "prompt": "Return exactly three lines: ALPHA, BETA, GAMMA, in that order. No numbering.",
            "expected": "ALPHA\nBETA\nGAMMA",
        },
        {
            "name": "unicode_copy",
            "kind": "exact",
            "prompt": "Copy exactly the following text, with no quotes or explanation: Київ — München — 42",
            "expected": "Київ — München — 42",
        },
        {
            "name": "slugify",
            "kind": "code",
            "prompt": "Write only Python code defining slugify(text: str) -> str. Normalize with "
            "unicodedata NFKD, remove non-ASCII characters, lowercase, replace each run of "
            "non-alphanumeric characters with a hyphen, strip leading/trailing hyphens. "
            "Return an empty string for empty input. No markdown or explanation.",
        },
        {
            "name": "merge_intervals",
            "kind": "code",
            "prompt": "Write only Python code defining merge_intervals(intervals). Input is a list of "
            "[start,end] integer lists. Return a list of two-element lists, sorted by start, "
            "representing merged closed intervals; touching endpoints merge. Do not mutate "
            "the input. Handle empty input. No markdown or explanation.",
        },
    ]
    for count, needle in ((120, 42), (400, 317)):
        records = [
            f"Record {i:04d}: access phrase TULIP_{(i * 7919 + 4197) % 100000:05d}."
            for i in range(count)
        ]
        cases.append(
            {
                "name": f"retrieval_{count}",
                "kind": "exact",
                "prompt": "Read these records:\n"
                + "\n".join(records)
                + f"\nReturn only the exact access phrase for Record {needle:04d}.",
                "expected": f"TULIP_{(needle * 7919 + 4197) % 100000:05d}",
            }
        )
    for city, unit in (("Berlin", "celsius"), ("Boston", "fahrenheit")):
        cases.append(
            {
                "name": "tool_" + city.lower(),
                "kind": "tool",
                "prompt": f"Call get_weather for {city} using {unit}. Do not invent weather data.",
                "expected": {"city": city, "unit": unit},
            }
        )
    return cases


def freeze(root: Path, data: Path) -> dict[str, Any]:
    """Freeze identical sampled token windows and tasks before any candidate inference."""
    path = root / "quality-protocol.json"
    if path.exists():
        return json.loads(path.read_bytes())
    manifest = json.loads((data / "dataset-manifest.json").read_bytes())
    if manifest["status"] != "prepared":
        raise ValueError("Public test corpus preparation is incomplete")
    raw = (data / "wikitext-test.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["text_sha256"]:
        raise ValueError("WikiText source text changed")
    ids = request(
        BASE,
        "/tokenize",
        {"model": "mxwave-glm53-pilot", "prompt": raw.decode(), "add_special_tokens": False},
        timeout=180,
    )["tokens"]
    if len(ids) < 32 * 1025:
        raise ValueError("Held-out token pool is too short")
    offsets = [i * (len(ids) - 1025) // 31 for i in range(32)]
    windows = [ids[first : first + 1025] for first in offsets]
    gsm_raw = (data / "gsm8k-test.jsonl").read_bytes()
    if hashlib.sha256(gsm_raw).hexdigest() != manifest["files"]["gsm8k-test.jsonl"]["sha256"]:
        raise ValueError("GSM8K test source changed")
    gsm = [json.loads(line) for line in gsm_raw.decode().splitlines()]
    indices = sorted(random.Random(20261017).sample(range(len(gsm)), 24))
    tasks = practical_cases()
    tasks.extend(
        {
            "name": f"gsm8k_{i:04d}",
            "kind": "math",
            "source_index": i,
            "prompt": gsm[i]["question"]
            + "\nSolve the problem. End your final answer with #### followed by the numeric answer.",
            "expected": gsm[i]["answer"].rsplit("####", 1)[1].strip(),
        }
        for i in indices
    )
    protocol = {
        "version": 1,
        "datasets": manifest,
        "wiki_windows": windows,
        "wiki_offsets": offsets,
        "wiki_total_tokens": len(ids),
        "wiki_scored_tokens": 32 * 1024,
        "tasks": tasks,
        "strict_control_stability_gate": {"mean_kl": 1e-5, "absolute_mean_nll": 1e-4},
        "heldout_gate": {
            "absolute_mean_nll": "max(1e-5, 5 * absolute reference-repeat delta)",
            "reference_repeat_max_absolute_mean_nll": 1e-4,
        },
        "practical_gate": "No task passed by the reference may fail in the candidate",
        "limits": [
            "Sampled WikiText test windows, not full-corpus standard PPL.",
            "24 fixed GSM8K test questions, not the full benchmark.",
            "Reference is already quantized MXFP4; pre-existing quantization error remains.",
        ],
    }
    write(path, protocol)
    return protocol


def heldout(root: Path, protocol: dict[str, Any], label: str) -> dict[str, Any]:
    """Retain every scored token logprob and compute token-weighted NLL and perplexity."""
    path = root / (label + "-heldout.json")
    if path.exists():
        raise FileExistsError("Preserve held-out evidence")
    report: dict[str, Any] = {"status": "running", "label": label, "windows": []}
    write(path, report)
    for index, ids in enumerate(protocol["wiki_windows"]):
        response = request(
            BASE,
            "/v1/completions",
            {
                "model": "mxwave-glm53-pilot",
                "prompt": ids,
                "max_tokens": 1,
                "temperature": 0,
                "seed": 42,
                "ignore_eos": True,
                "return_token_ids": True,
                "prompt_logprobs": 1,
            },
            timeout=300,
        )
        values = response["choices"][0]["prompt_logprobs"]
        if len(values) != len(ids):
            raise ValueError("Teacher-forced token/logprob alignment changed")
        logp = np.array(
            [
                float(row[str(token)]["logprob"])
                for token, row in zip(ids[1:], values[1:], strict=True)
            ],
            dtype="<f4",
        )
        if not np.isfinite(logp).all():
            raise ValueError("Non-finite held-out logprobs")
        file = root / f"{label}-wiki-{index:02d}.f32"
        file.write_bytes(logp.tobytes())
        report["windows"].append(
            {
                "index": index,
                "tokens": len(logp),
                "file": file.name,
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                "prompt_sha256": hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode()
                ).hexdigest(),
                "nll_sum": -float(logp.astype(np.float64).sum()),
            }
        )
        write(path, report)
    tokens = sum(row["tokens"] for row in report["windows"])
    nll = sum(row["nll_sum"] for row in report["windows"]) / tokens
    report.update(status="complete", scored_tokens=tokens, mean_nll=nll, perplexity=math.exp(nll))
    write(path, report)
    return report


def tasks(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Retain natural completions and strict task scores; code is executed separately."""
    report: dict[str, Any] = {"status": "running", "cases": []}
    path = root / "real-work.json"
    write(path, report)
    for case in protocol["tasks"]:
        kind, name = case["kind"], case["name"]
        body: dict[str, Any] = {
            "model": "mxwave-glm53-pilot",
            "temperature": 0,
            "seed": 42,
            "max_tokens": 768 if kind == "math" else 512 if kind == "code" else 256,
            "messages": [{"role": "user", "content": case["prompt"]}],
            "chat_template_kwargs": {"reasoning_effort": "low"},
        }
        if kind == "tool":
            body.update(
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a city.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string"},
                                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                                },
                                "required": ["city", "unit"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                tool_choice={"type": "function", "function": {"name": "get_weather"}},
            )
        row: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
        }
        try:
            response = request(BASE, "/v1/chat/completions", body, timeout=300)
            choice = response["choices"][0]
            message = choice["message"]
            text = final_text(message)
            row.update(response=response, final_answer=text, finish_reason=choice["finish_reason"])
            stopped = choice["finish_reason"] in ("stop", "tool_calls")
            if kind == "json":
                row["passed"] = stopped and json.dumps(
                    json.loads(text), sort_keys=True
                ) == json.dumps(case["expected"], sort_keys=True)
            elif kind == "exact":
                row["passed"] = stopped and text == case["expected"]
            elif kind == "math":
                row["passed"] = stopped and numeric_answer(text) == numeric_answer(case["expected"])
            elif kind == "tool":
                calls = message.get("tool_calls") or []
                row["passed"] = (
                    stopped
                    and len(calls) == 1
                    and calls[0]["function"]["name"] == "get_weather"
                    and json.loads(calls[0]["function"]["arguments"]) == case["expected"]
                )
            else:
                code, strict_format = text, not text.startswith("```")
                if text.startswith("```") and text.endswith("```"):
                    code = text.split("\n", 1)[1].rsplit("```", 1)[0]
                tree = ast.parse(code)
                if not any(
                    isinstance(node, ast.FunctionDef) and node.name == name for node in tree.body
                ):
                    raise ValueError("Expected function definition missing")
                file = root / (name + ".py")
                file.write_text(code)
                row.update(
                    code_file=file.name,
                    code_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                    format_passed=bool(strict_format and stopped),
                    passed=None,
                    sandbox_pending=True,
                )
        except Exception as exc:  # noqa: BLE001 -- retain a failed task and continue the frozen suite.
            row.update(passed=False, error=repr(exc))
        report["cases"].append(row)
        write(path, report)
    report["status"] = "collected_pending_code_sandbox"
    write(path, report)
    return report
