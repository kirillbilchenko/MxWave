"""Measure actual autoregressive decode distributions, retaining conditioning and raw bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from benchmark_full_model import request, write


def collect(base: str, prompts: list[list[int]], root: Path, label: str,
            vocabulary: int, *, generated_tokens: int = 4) -> dict[str, Any]:
    """Retain prefill plus three real decode steps without reconstructing them as prefills."""
    path = root / (label + "-decode-quality.json")
    if path.exists():
        raise FileExistsError("Preserve previous decode quality evidence")
    report: dict[str, Any] = {"status": "running", "label": label, "vocabulary": vocabulary,
        "generated_tokens": generated_tokens, "contexts": [],
        "scope": "Step zero is prefill; subsequent distributions come from live KV decode"}
    write(path, report)
    for index, ids in enumerate(prompts):
        result = request(base, "/v1/completions", {"model": "mxwave-glm53-pilot",
            "prompt": ids, "max_tokens": generated_tokens, "min_tokens": generated_tokens,
            "temperature": 0, "seed": 42, "ignore_eos": True, "return_token_ids": True,
            "return_tokens_as_token_ids": True, "logprobs": vocabulary}, timeout=300)
        choice = result["choices"][0]
        tokens, tops = choice["token_ids"], choice["logprobs"]["top_logprobs"]
        if len(tokens) != generated_tokens or len(tops) != generated_tokens:
            raise ValueError("Incomplete actual decode quality response")
        context = {"index": index, "prompt_sha256": hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
            "token_ids": tokens, "steps": []}
        for position, top in enumerate(tops):
            row = np.full(vocabulary, -np.inf, dtype="<f4")
            for key, value in top.items():
                if not key.startswith("token_id:"):
                    raise ValueError("Require complete distributions indexed by token id")
                token = int(key.removeprefix("token_id:"))
                if not 0 <= token < vocabulary or math.isnan(value):
                    raise ValueError("Invalid vocabulary index or NaN logprob")
                row[token] = value
            if len(top) != vocabulary or not np.isfinite(row).any():
                raise ValueError("Missing full decode distribution")
            file = root / f"{label}-ctx{index:02d}-step{position}.f32"
            file.write_bytes(row.tobytes())
            context["steps"].append({"position": position,
                "kind": "prefill" if position == 0 else "decode",
                "conditioning_token_ids": tokens[:position],
                "file": file.name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
        report["contexts"].append(context)
        write(path, report)
    report["status"] = "complete"
    write(path, report)
    return report


def compare(reference: dict[str, Any], candidate: dict[str, Any], root: Path) -> dict[str, Any]:
    """Compare only distributions conditioned on identical prompts and preceding generated ids."""
    if (reference["status"] != "complete" or candidate["status"] != "complete"
            or reference["vocabulary"] != candidate["vocabulary"]
            or reference["generated_tokens"] != candidate["generated_tokens"]):
        raise ValueError("Decode quality reports are incompatible")
    rows = []
    greedy_equal = True
    for left, right in zip(reference["contexts"], candidate["contexts"], strict=True):
        if left["index"] != right["index"] or left["prompt_sha256"] != right["prompt_sha256"]:
            raise ValueError("Decode quality prompts differ")
        greedy_equal &= left["token_ids"] == right["token_ids"]
        for a, b in zip(left["steps"], right["steps"], strict=True):
            if a["position"] != b["position"] or a["kind"] != b["kind"]:
                raise ValueError("Decode quality step positions differ")
            row: dict[str, Any] = {"index": left["index"], "position": a["position"],
                                  "kind": a["kind"], "matching_conditioning":
                                  a["conditioning_token_ids"] == b["conditioning_token_ids"]}
            if row["matching_conditioning"]:
                values = []
                for step in (a, b):
                    raw = (root / step["file"]).read_bytes()
                    if hashlib.sha256(raw).hexdigest() != step["sha256"]:
                        raise ValueError("Retained decode distribution changed")
                    value = np.frombuffer(raw, dtype="<f4").astype(np.float64)
                    if len(value) != reference["vocabulary"] or np.isnan(value).any():
                        raise ValueError("Invalid retained decode distribution")
                    maximum = value.max()
                    values.append(value - (maximum + np.log(np.exp(value - maximum).sum())))
                logp, logq = values
                supported = np.isfinite(logp)
                if not np.isfinite(logq[supported]).all():
                    raise ValueError("Candidate lost reference distribution support")
                token = left["token_ids"][a["position"]]
                row.update(kl=max(0., float(np.sum(np.exp(logp[supported]) *
                                                  (logp[supported] - logq[supported])))),
                           nll_difference=float(logp[token] - logq[token]))
            rows.append(row)
    decoded = [r for r in rows if r["kind"] == "decode" and r["matching_conditioning"]]
    expected = sum(r["kind"] == "decode" for r in rows)
    return {"steps": rows, "decode_steps": len(decoded), "expected_decode_steps": expected,
            "all_decode_conditioning_matches": len(decoded) == expected,
            "all_generated_ids_match": greedy_equal,
            "mean_decode_kl": float(np.mean([r["kl"] for r in decoded])) if decoded else None,
            "mean_decode_nll_difference": float(np.mean([r["nll_difference"] for r in decoded]))
                if decoded else None}


def main() -> None:
    """Collect the same frozen sixteen windows used for the earlier prefill comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18784")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--vocabulary", required=True, type=int)
    args = parser.parse_args()
    proto = json.loads((args.root / "protocol.json").read_bytes())
    prompts = [proto.get("quality_prefix_token_ids", []) + ids for ids in proto["quality"]]
    collect(args.base, prompts, args.root, args.label, args.vocabulary)


if __name__ == "__main__":
    main()
