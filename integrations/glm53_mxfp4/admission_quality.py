"""Retain full-vocabulary live decode distributions from ordinary concurrent HTTP requests."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from benchmark_full_model import request, write


def collect_concurrent(base: str, prompts: list[list[int]], root: Path, label: str,
                       vocabulary: int, *, concurrency: int = 4) -> dict[str, Any]:
    """Keep sixteen contexts, four real generated tokens and actual concurrent admission."""
    path = root / (label + "-decode-quality.json")
    if path.exists() or len(prompts) != 16 or concurrency not in (1, 2, 4, 8, 16):
        raise ValueError("Require sixteen original contexts and a new quality report")
    report: dict[str, Any] = {"status": "running", "label": label, "vocabulary": vocabulary,
        "generated_tokens": 4, "contexts": [], "concurrency": concurrency,
        "scope": "Independent concurrent HTTP requests; live KV decode; no controlled scheduler pause"}
    write(path, report)

    def one(index: int, ids: list[int]) -> dict[str, Any]:
        result = request(base, "/v1/completions", {"model": "mxwave-glm53-pilot", "prompt": ids,
            "max_tokens": 4, "min_tokens": 4, "temperature": 0, "seed": 42, "ignore_eos": True,
            "return_token_ids": True, "return_tokens_as_token_ids": True, "logprobs": vocabulary},
            timeout=300)
        choices = result["choices"]
        if len(choices) != 1:
            raise ValueError("One quality request must return exactly one completion")
        tokens = choices[0]["token_ids"]
        tops = choices[0]["logprobs"]["top_logprobs"]
        if len(tokens) != 4 or len(tops) != 4:
            raise ValueError("Incomplete concurrent actual-decode response")
        context: dict[str, Any] = {"index": index, "prompt_sha256": hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
            "token_ids": tokens, "steps": []}
        for position, top in enumerate(tops):
            row = np.full(vocabulary, -np.inf, dtype="<f4")
            for key, value in top.items():
                if not key.startswith("token_id:") or math.isnan(value):
                    raise ValueError("Concurrent quality requires finite indexed vocabulary rows")
                token = int(key.removeprefix("token_id:"))
                if not 0 <= token < vocabulary:
                    raise ValueError("Invalid concurrent vocabulary index")
                row[token] = value
            if len(top) != vocabulary or not np.isfinite(row).any():
                raise ValueError("Missing full concurrent decode distribution")
            file = root / f"{label}-ctx{index:02d}-step{position}.f32"
            if file.exists():
                raise FileExistsError("Preserve earlier concurrent distribution bytes")
            file.write_bytes(row.tobytes())
            context["steps"].append({"position": position,
                "kind": "prefill" if position == 0 else "decode",
                "conditioning_token_ids": tokens[:position], "file": file.name,
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
        return context

    try:
        for offset in range(0, 16, concurrency):
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                jobs = [executor.submit(one, i, prompts[i])
                        for i in range(offset, offset + concurrency)]
                report["contexts"].extend(job.result() for job in jobs)
            write(path, report)
        report["status"] = "complete"
        write(path, report)
        return report
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        write(path, report)
        raise
