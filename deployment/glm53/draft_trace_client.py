"""Collect a few native and draft-eager proposals; no performance claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from profile_client import call, counters


def main() -> None:
    """Run two development prompts under the bounded observational worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads((args.root / "profile-plan.json").read_text())
    result: dict[str, Any] = {"scope": "zero-acceptance diagnostic, timings invalid",
                              "responses": []}

    def rpc(method: str, *values: Any) -> Any:
        return call(args.port, "collective_rpc", {
            "method": method, "timeout": 120, "args": list(values), "kwargs": {}})

    for index, case in enumerate(plan["cases"][:2]):
        for mode in ("native", "draft_eager"):
            label = f"case{index}-" + mode.replace("_", "-")
            start = rpc("mxwave_draft_trace_start", label, mode, 8)
            before = counters(call(args.port, "metrics"))
            try:
                response = call(args.port, "v1/completions", {
                    "model": "glm-5.3-flash", "prompt": case["prompt_ids"],
                    "max_tokens": 32, "ignore_eos": True, "temperature": 0, "seed": 42,
                    "add_special_tokens": False, "return_token_ids": True})
            finally:
                stop = rpc("mxwave_draft_trace_stop")
            after = counters(call(args.port, "metrics"))
            result["responses"].append({"label": label, "mode": mode, "start": start,
                                        "stop": stop, "response": response,
                                        "counter_delta": {k: after[k] - before.get(k, 0)
                                                          for k in after}})
            (args.root / "trace-result.json").write_text(json.dumps(result, indent=2) + "\n")
    result["status"] = "collected_not_qualified"
    (args.root / "trace-result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
