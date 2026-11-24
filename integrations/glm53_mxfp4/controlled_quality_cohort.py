"""Profile C8 quality repeats and retain exact row membership before testing a fixed cohort."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from benchmark_full_model import request, write
from quality_suite import BASE


def controlled_request(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Queue one fixed cohort of original prompts on this disposable service and resume in all cases."""
    if BASE != "http://127.0.0.1:18784" or len(body["prompt"]) not in (8, 16):
        raise ValueError(
            "Keep-pause is restricted to this isolated eight/sixteen-request diagnostic"
        )
    receipt: dict[str, Any] = {"started_epoch": time.time(), "grace_seconds": 1.0}
    paused = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            receipt["pause"] = request(BASE, "/pause?mode=keep&clear_cache=false", {})
            paused = True
            assert request(BASE, "/is_paused")["is_paused"]
            future = pool.submit(request, BASE, "/v1/completions", body, timeout=300)
            time.sleep(1.0)
            if future.done():
                future.result()
                raise RuntimeError("A controlled completion returned while scheduler was paused")
            assert request(BASE, "/is_paused")["is_paused"]
        finally:
            if paused:
                receipt["resume"] = request(BASE, "/resume", {})
        result = future.result()
    receipt["finished_epoch"] = time.time()
    return result, receipt


def collect_controlled_count(
    contexts: list[list[int]], root: Path, label: str, vocabulary: int, concurrency: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Keep all sixteen original contexts; prove actual grouping from separate worker records."""
    path = root / f"{label}-decode-quality.json"
    assert not path.exists() and len(contexts) == 16 and concurrency in (8, 16)
    report: dict[str, Any] = {
        "status": "running",
        "label": label,
        "vocabulary": vocabulary,
        "generated_tokens": 4,
        "concurrency": concurrency,
        "contexts": [],
        "scope": f"{concurrency} prompts queued under keep-pause; native 2048-token scheduler unchanged",
    }
    receipts: list[dict[str, Any]] = []
    write(path, report)
    for offset in range(0, len(contexts), concurrency):
        prompts = contexts[offset : offset + concurrency]
        response, receipt = controlled_request(
            {
                "model": "mxwave-glm53-pilot",
                "prompt": prompts,
                "max_tokens": 4,
                "min_tokens": 4,
                "temperature": 0,
                "seed": 42,
                "ignore_eos": True,
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
                "logprobs": vocabulary,
            }
        )
        receipts.append(receipt)
        write(root / f"{label}-admission-proofs.json", receipts)
        choices = sorted(response["choices"], key=lambda row: row["index"])
        assert [c["index"] for c in choices] == list(range(concurrency))
        for i, choice in enumerate(choices):
            tokens, tops = choice["token_ids"], choice["logprobs"]["top_logprobs"]
            assert len(tokens) == len(tops) == 4
            context = {
                "index": offset + i,
                "prompt_sha256": hashlib.sha256(
                    json.dumps(prompts[i], separators=(",", ":")).encode()
                ).hexdigest(),
                "token_ids": tokens,
                "steps": [],
            }
            for position, top in enumerate(tops):
                row = np.full(vocabulary, -np.inf, dtype="<f4")
                for key, value in top.items():
                    assert key.startswith("token_id:") and not math.isnan(value)
                    token = int(key.removeprefix("token_id:"))
                    assert 0 <= token < vocabulary
                    row[token] = value
                assert len(top) == vocabulary and np.isfinite(row).any()
                file = root / f"{label}-ctx{offset + i:02d}-step{position}.f32"
                assert not file.exists()
                file.write_bytes(row.tobytes())
                context["steps"].append(
                    {
                        "position": position,
                        "kind": "prefill" if position == 0 else "decode",
                        "conditioning_token_ids": tokens[:position],
                        "file": file.name,
                        "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                    }
                )
            report["contexts"].append(context)
        write(path, report)
    report["status"] = "complete"
    write(path, report)
    return report, receipts
