"""One-load graph/eager/graph diagnostic with source checks before any speed experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from benchmark_full_model import quality, request, write

SANITY = (
    ("What is 2 + 2? Reply with only the number.", "4"),
    ("What is the capital of France? Reply with only the city name.", "Paris"),
    ("Reply with exactly the word BLUE and nothing else.", "BLUE"),
)


def assess_chat(choice: dict[str, Any], expected: str) -> dict[str, Any]:
    """Score a naturally finished final answer, retaining raw reasoning for review."""
    message = choice["message"]
    output = message.get("content") or ""
    if not isinstance(output, str):
        raise TypeError("Expected a text chat completion")
    if "</think>" in output:
        answer: str | None = output.rsplit("</think>", 1)[1].strip()
    else:
        answer = None if "<think>" in output else output.strip()
    normalized = answer.strip(".!").strip().casefold() if answer is not None else None
    return {"expected": expected, "output": output, "raw_message": message,
            "final_answer": answer, "finish_reason": choice["finish_reason"],
            "passed": choice["finish_reason"] == "stop" and normalized == expected.casefold()}


def rpc(base: str, method: str, args: list[Any] | None = None) -> list[dict[str, Any]]:
    """Require both TP workers to acknowledge the diagnostic operation."""
    value = request(base, "/collective_rpc", {"method": method, "args": args or [],
                     "kwargs": {}, "timeout": 300}, timeout=360)
    if len(value.get("results", [])) != 2:
        raise ValueError("Missing TP2 diagnostic acknowledgement")
    return value["results"]


def main() -> None:
    """Stop at correctness qualification; no throughput sweep on a broken control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18784")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--vocabulary", type=int, required=True)
    args = parser.parse_args()
    args.root.mkdir(exist_ok=True)
    path = args.root / "model-diagnostic.json"
    if path.exists():
        raise FileExistsError("Preserve previous one-load diagnostic evidence")
    state: dict[str, Any] = {"status": "running", "started_epoch": time.time(), "phases": [],
        "purpose": "Source/runtime correctness, not production performance qualification",
        "chat_protocol": {"reasoning_effort": "low", "max_tokens": 256,
                          "require_natural_stop": True, "score": "Final answer only"},
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        state["loaded_weight_checks"] = rpc(args.base, "mxwave_checkpoint_check")
        write(path, state)
        if any(row["status"] != "passed" for row in state["loaded_weight_checks"]):
            raise RuntimeError("Selected loaded weights disagree with the exact source")
        state["mhc_warmup"] = rpc(args.base, "mxwave_mhc_warmup", [2048])
        write(path, state)
        protocol = json.loads(args.protocol.read_bytes())
        contexts = [[154822, 154824, *ids] for ids in protocol["quality"][:2]]
        for label, eager in (("graphs_before", False), ("eager", True), ("graphs_after", False)):
            phase: dict[str, Any] = {"label": label, "chat_checks": [], "dispatch": []}
            state["phases"].append(phase)
            if label != "graphs_before":
                phase["dispatch"] = rpc(args.base, "mxwave_set_eager", [eager])
                write(path, state)
                if any(row["status"] != "complete" for row in phase["dispatch"]):
                    raise RuntimeError("Dispatch switch did not acknowledge the requested mode")
            for prompt, expected in SANITY:
                result = request(args.base, "/v1/chat/completions", {
                    "model": "mxwave-glm53-pilot", "messages": [{"role": "user", "content": prompt}],
                    "chat_template_kwargs": {"reasoning_effort": "low"}, "temperature": 0,
                    "seed": 42, "max_tokens": 256})
                choice = result["choices"][0]
                phase["chat_checks"].append({"prompt": prompt, **assess_chat(choice, expected)})
                write(path, state)
            phase["quality"] = quality(args.base, contexts, args.root, label, args.vocabulary)
            phase["sanity_passed"] = all(row["passed"] for row in phase["chat_checks"])
            write(path, state)
            if label == "graphs_before":
                # Retain default-mode inference before any switch can fail.
                phase["dispatch"] = rpc(args.base, "mxwave_set_eager", [False])
                write(path, state)
                if any(row["status"] != "complete" for row in phase["dispatch"]):
                    raise RuntimeError("Initial graph dispatch could not be verified")
        state.update(status="complete", finished_epoch=time.time(),
            any_phase_sanity_passed=any(phase["sanity_passed"] for phase in state["phases"]),
            throughput_benchmark_run=False,
            limitation="Two diagnostic next-token distributions and three simple chats per mode.")
        write(path, state)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), finished_epoch=time.time())
        write(path, state)
        raise


if __name__ == "__main__":
    main()
