"""Collect actual full-model source, log-probability and practical-task evidence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import runpy
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark_full_model import request, write
from check_full_model import rpc
from decode_quality import collect

BASE = "http://127.0.0.1:18784"


def final_text(message: dict[str, Any]) -> str:
    """Retain the model's final text without treating unfinished reasoning as an answer."""
    text = message.get("content") or ""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    return "" if "<think>" in text else text.strip()


def real_work(root: Path) -> dict[str, Any]:
    """Run fixed structured-output, tool, code and context tasks; save code for sandbox tests."""
    passages = [
        f"Record {i:04d}: the access phrase is TULIP_{(i * 7919 + 4197) % 100000:05d}."
        for i in range(240)
    ]
    needle = f"TULIP_{(42 * 7919 + 4197) % 100000:05d}"
    cases = [
        (
            "json",
            'Return only this JSON object: {"status":"ok","count":3,"items":["a","b","c"]}.',
            256,
        ),
        (
            "constraints",
            "Return exactly three lines: ALPHA, BETA, GAMMA, in that order. No numbering.",
            256,
        ),
        (
            "slugify",
            (
                "Write only Python code defining slugify(text: str) -> str. Normalize with "
                "unicodedata NFKD, remove non-ASCII characters, lowercase, replace each run of "
                "non-alphanumeric characters with a hyphen, strip leading/trailing hyphens. "
                "Return an empty string for empty input. No markdown or explanation."
            ),
            512,
        ),
        (
            "merge_intervals",
            (
                "Write only Python code defining merge_intervals(intervals). Input "
                "is a list of [start,end] integer pairs. Return sorted merged closed intervals; "
                "touching endpoints merge. Do not mutate the input. Handle empty input. "
                "No markdown or explanation."
            ),
            512,
        ),
        (
            "retrieval",
            "Read these records:\n"
            + "\n".join(passages)
            + "\nReturn only the exact access phrase for Record 0042.",
            256,
        ),
        ("tool", "Call get_weather for Berlin using Celsius. Do not invent weather data.", 256),
    ]
    report: dict[str, Any] = {
        "status": "running",
        "cases": [],
        "note": "Authored practical tasks, not a standard accuracy benchmark; code execution is sandboxed separately.",
    }
    path = root / "real-work.json"
    write(path, report)
    for name, prompt, limit in cases:
        body: dict[str, Any] = {
            "model": "mxwave-glm53-pilot",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 42,
            "max_tokens": limit,
            "chat_template_kwargs": {"reasoning_effort": "low"},
        }
        if name == "tool":
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
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }
        try:
            response = request(BASE, "/v1/chat/completions", body, timeout=300)
            choice = response["choices"][0]
            message = choice["message"]
            text = final_text(message)
            row.update(response=response, final_answer=text)
            if name == "json":
                row["passed"] = json.loads(text) == {
                    "status": "ok",
                    "count": 3,
                    "items": ["a", "b", "c"],
                }
            elif name == "constraints":
                row["passed"] = text.splitlines() == ["ALPHA", "BETA", "GAMMA"]
            elif name == "retrieval":
                row["passed"] = text == needle
            elif name == "tool":
                calls = message.get("tool_calls") or []
                row["passed"] = (
                    len(calls) == 1
                    and calls[0]["function"]["name"] == "get_weather"
                    and (
                        json.loads(calls[0]["function"]["arguments"])
                        == {"city": "Berlin", "unit": "celsius"}
                    )
                )
            else:
                code = text
                if code.startswith("```") and code.endswith("```"):
                    code = code.split("\n", 1)[1].rsplit("```", 1)[0]
                tree = ast.parse(code)
                if not any(
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
                    for node in tree.body
                ):
                    raise ValueError("Expected function definition missing")
                file = root / (name + ".py")
                file.write_text(code)
                row.update(
                    code_file=file.name,
                    code_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                    passed=None,
                    sandbox_pending=True,
                )
        except Exception as exc:  # noqa: BLE001 -- a failed task must not erase the remaining evidence.
            row.update(passed=False, error=repr(exc))
        report["cases"].append(row)
        write(path, report)
    report["status"] = "collected_pending_code_sandbox"
    write(path, report)
    return report


def main() -> None:
    """Reuse the existing both-worker source/graph diagnostic, then collect actual distributions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", choices=("control", "candidate"), required=True)
    parser.add_argument("--vocabulary", type=int, required=True)
    parser.add_argument("--cpu-audit", type=Path, required=True)
    args = parser.parse_args()
    path = args.root / (args.label + "-measurement.json")
    if path.exists():
        raise FileExistsError("Preserve prior full-model measurements")
    report: dict[str, Any] = {"status": "running", "label": args.label}
    write(path, report)
    coverage = rpc(BASE, "mxwave_palette_report")
    if any(r["candidate"] != (args.label == "candidate") for r in coverage):
        raise RuntimeError("Full-model arm does not match its declared codec")
    cpu = json.loads(args.cpu_audit.read_bytes())
    if (
        cpu["status"] != "passed"
        or not cpu["all_main_expert_layers"]
        or len(cpu["cases"]) != 168
        or {r["tp_rank"] for r in coverage} != {0, 1}
    ):
        raise RuntimeError("Independent CPU source audit is incomplete")
    for rank in coverage:
        expected = Counter(
            (tuple(c["shape"]), c["raw_sha256"])
            for c in cpu["cases"]
            if c["rank"] == rank["tp_rank"]
        )
        actual = Counter((tuple(c["shape"]), c["sha256"]) for c in rank["source_scale_hashes"])
        if actual != expected:
            raise RuntimeError(
                "Actual GPU-loaded scale groups differ from the independent source audit"
            )
    report["complete_scale_gpu_source_checks"] = coverage
    write(path, report)
    diagnostic = args.root / (args.label + "-diagnostic")
    diagnostic.mkdir()
    sys.argv = [
        str(Path(__file__).with_name("check_full_model.py")),
        "--root",
        str(diagnostic),
        "--protocol",
        str(args.root / "protocol.json"),
        "--vocabulary",
        str(args.vocabulary),
    ]
    runpy.run_path(sys.argv[0], run_name="__main__")
    checked = json.loads((diagnostic / "model-diagnostic.json").read_bytes())
    report["model_diagnostic_status"] = checked["status"]
    report["all_source_checks_passed"] = all(
        r["status"] == "passed" for r in checked["loaded_weight_checks"]
    )
    report["graph_eager_graph_chats_passed"] = all(p["sanity_passed"] for p in checked["phases"])
    write(path, report)
    if not report["all_source_checks_passed"] or not report["graph_eager_graph_chats_passed"]:
        raise RuntimeError("Real loaded-model source/chat gate failed")
    proto = json.loads((args.root / "protocol.json").read_bytes())
    prompts = [proto.get("quality_prefix_token_ids", []) + ids for ids in proto["quality"]]
    collect(BASE, prompts, args.root, args.label, args.vocabulary)
    collect(BASE, prompts, args.root, args.label + "-repeat", args.vocabulary)
    tasks = args.root / args.label
    tasks.mkdir()
    report["real_work"] = real_work(tasks)
    report["gpu_source_checks_after_inference"] = rpc(BASE, "mxwave_palette_report")
    report["status"] = "collected_pending_code_sandbox"
    write(path, report)


if __name__ == "__main__":
    main()
