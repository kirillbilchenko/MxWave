"""Run matched sanity prompts through local-spark's documented authenticated client."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


def write(path: Path, state: dict[str, Any]) -> None:
    """Retain the matched serving check atomically without storing credentials."""
    temporary = path.with_suffix(".incomplete")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    """Use the existing client Settings; perform no Docker credential inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-project", type=Path, required=True)
    parser.add_argument("--base", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve the original matched serving check")
    sys.path.insert(0, str(args.client_project / "src"))
    from spark_eval.api import OpenAIChatClient
    from spark_eval.config import Settings

    settings = Settings(_env_file=args.client_project / ".env", api_base=args.base)
    source = args.protocol.read_bytes()
    tree = ast.parse(source)
    prompts = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "SANITY" for t in node.targets))
    state: dict[str, Any] = {
        "status": "running", "started_epoch": time.time(), "model": args.model,
        "chat_checks": [], "protocol_sha256": hashlib.sha256(source).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "authentication": "existing documented local-spark client Settings",
        "transport": "streaming",
        "limits": ["Different NVFP4 source and serving settings, including DFlash.",
                   "Serving sanity only, not a paired quantization or speed evaluation."],
    }
    try:
        write(args.report, state)
        with OpenAIChatClient(api_base=settings.api_base, api_key=settings.llm_api_key) as client:
            for prompt, expected in prompts:
                result = client.chat({
                    "model": args.model, "messages": [{"role": "user", "content": prompt}],
                    "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0,
                    "seed": 42, "max_tokens": 32})
                passed = (result.success and result.content.strip().strip(".!").strip().casefold()
                          == expected.casefold())
                state["chat_checks"].append({"prompt": prompt, "expected": expected,
                    "output": result.content, "reasoning": result.reasoning_content,
                    "passed": passed, "status_code": result.status_code,
                    "finish_reason": result.finish_reason, "error_type": result.error_type})
                write(args.report, state)
        state.update(status="complete", finished_epoch=time.time(),
                     sanity_passed=all(row["passed"] for row in state["chat_checks"]))
        write(args.report, state)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), finished_epoch=time.time())
        write(args.report, state)
        raise


if __name__ == "__main__":
    main()
