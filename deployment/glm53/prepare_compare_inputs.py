"""Freeze realistic nonsealed prompts and retrieval probes before either timing arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def prepare(root: Path, model: Path) -> None:
    """Select by length alone, recording every source and all exact rendered tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)

    def render(text: str) -> list[int]:
        value = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
            reasoning_effort="low", return_dict=False)
        if not isinstance(value, list) or not all(isinstance(token, int) for token in value):
            raise TypeError("Require a flat token-ID list, not a tokenizer result dictionary")
        return value

    source = root / "training-prompts.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    cached = {row["id"]: row for row in map(
        json.loads, (root / "training-token-lengths.jsonl").read_text().splitlines())}
    if len(rows) != 4000 or any("gsm8k" in r["source"].lower() for r in rows):
        raise ValueError("Require the nonsealed, benchmark-excluded training prompt source")
    banks: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for domain in ("code", "chat", "explanation"):
        available = [r for r in rows if r["domain"] == domain]
        if any(cached[r["id"]]["prompt_sha256"] != r["prompt_sha256"] for r in available):
            raise ValueError("Cached source lengths do not bind the frozen prompt text")
        encoded = [(r, cached[r["id"]]["prompt_ids"]) for r in available]
        selected = sorted(encoded, key=lambda x: (abs(len(x[1]) - 256), x[0]["id"]))[:16]
        short = [{"source_ids": [r["id"]], "prompt": r["messages"][0]["content"],
                  "prompt_ids": render(r["messages"][0]["content"]),
                  "composite": False} for r, _ in selected]
        for prompt in short:
            if len(prompt["prompt_ids"]) >= 220:
                continue
            parts = [prompt["prompt"]]
            sources = list(prompt["source_ids"])
            prefix = "Give a concise explanation of each question below.\n\n"
            for _ in range(10):
                text = prefix + "\n\n".join(parts)
                ids = render(text)
                if 220 <= len(ids) <= 300:
                    break
                choices = sorted(
                    (item for item in encoded if item[0]["id"] not in sources),
                    key=lambda item: (abs(len(item[1]) - (256 - len(ids))), item[0]["id"]))[:64]
                measured = [(item[0], render(
                    prefix + "\n\n".join([*parts, item[0]["messages"][0]["content"]])))
                    for item in choices]
                fitting = [item for item in measured if len(ids) < len(item[1]) <= 300]
                if not fitting:
                    break
                added, _ = min(fitting, key=lambda item: (abs(len(item[1]) - 256), item[0]["id"]))
                parts.append(added["messages"][0]["content"])
                sources.append(added["id"])
            text = prefix + "\n\n".join(parts)
            prompt.update(prompt=text, prompt_ids=render(text), source_ids=sources, composite=True)
        if any(not 220 <= len(p["prompt_ids"]) <= 300 for p in short):
            raise ValueError(f"Short inputs outside 256-token band: {domain}, "
                             f"{[len(p['prompt_ids']) for p in short]}")
        # Long code prompts already contain real code. Other domains need longer supplied
        # reference material: concatenate distinct real requests, without repetitive padding.
        long = []
        ordered = sorted(encoded, key=lambda x: (abs(len(x[1]) - 2048), x[0]["id"]))
        for index in range(16):
            r, _ = ordered[index]
            ids = render(r["messages"][0]["content"])
            if 1700 <= len(ids) <= 2300:
                text, sources, composite = r["messages"][0]["content"], [r["id"]], False
                ids = render(text)
            else:
                instruction = {
                    "code": "Review these engineering requirements and source excerpts. "
                            "Explain a concrete implementation plan and the main edge cases.",
                    "chat": "I need help preparing responses to these requests. "
                            "Draft concise, useful replies in the same order.",
                    "explanation": "Prepare a briefing explaining the following material "
                                   "for a new colleague. Explain key ideas and connections.",
                }[domain]
                parts, sources = [], []
                current = render(instruction)
                for offset in range(20):
                    remaining = 2048 - len(current)
                    choices = sorted(
                        (item for item in ordered if item[0]["id"] not in sources
                         and len(item[1]) < remaining + 150),
                        key=lambda item: (abs(len(item[1]) - remaining), item[0]["id"]))[:16]
                    if not choices:
                        break
                    candidate = choices[(index + offset) % len(choices)][0]
                    proposed = [*parts, candidate["messages"][0]["content"]]
                    rendered = render(instruction + "\n\n" + "\n\n---\n\n".join(proposed))
                    if len(rendered) > 2300:
                        continue
                    parts, sources = proposed, [*sources, candidate["id"]]
                    current = rendered
                    if len(rendered) >= 1950:
                        break
                text = instruction + "\n\n" + "\n\n---\n\n".join(parts)
                ids, composite = render(text), True
            if not 1700 <= len(ids) <= 2300:
                raise ValueError(f"Long source selection outside window: {domain}/{index}, "
                                 f"tokens={len(ids)}, sources={len(sources)}")
            long.append({"source_ids": sources, "prompt": text, "prompt_ids": ids,
                         "composite": composite})
        banks[domain] = {"short": short, "long": long}
        for bank in (short, long):
            if len({p["prompt"] for p in bank}) != 16:
                raise ValueError("Every concurrency cohort requires 16 distinct real inputs")
    cells = [{"name": f"{domain}-{length}-c{concurrency}", "domain": domain,
              "length": length, "concurrency": concurrency,
              "prompt_counts": [len(p["prompt_ids"]) for p in bank[:concurrency]],
              "prompts": [p["prompt_ids"] for p in bank[:concurrency]]}
             for domain, lengths in banks.items() for length, bank in lengths.items()
             for concurrency in (1, 4, 8, 16)]
    probes = []
    material = [r["messages"][0]["content"] for r in sorted(rows, key=lambda r: r["id"])]
    for count in (32768, 131072):
        pieces, length = [], 0
        for index, text in enumerate(material):
            piece = f"Document {index}:\n{text}\n\n"
            pieces.append(piece)
            length += len(tokenizer.encode(piece, add_special_tokens=False))
            if length > count:
                break
        tokens = tokenizer.encode("".join(pieces), add_special_tokens=False)[:count - 300]
        thirds = [tokenizer.decode(tokens[a:b]) for a, b in (
            (0, len(tokens) // 3), (len(tokens) // 3, 2 * len(tokens) // 3),
            (2 * len(tokens) // 3, len(tokens)))]
        expected = ["ORCHID-48291", "MAPLE-73604", "COBALT-15928"]
        text = "Read this reference packet. Three project access phrases appear in it.\n\n"
        for i, part in enumerate(thirds):
            text += part + f"\nProject {i + 1} access phrase: {expected[i]}.\n\n"
        text += "Return only the three project access phrases, in project-number order."
        ids = render(text)
        probes.append({"name": f"retrieval-{count}", "prompt_ids": ids,
                       "actual_prompt_tokens": len(ids), "expected": expected, "prompt": text})
    provenance = {r["id"]: {k: r[k] for k in
                  ("source", "source_row", "license", "prompt_sha256", "group_sha256")} for r in rows}
    used = {source_id for lengths in banks.values() for bank in lengths.values()
            for prompt in bank for source_id in prompt["source_ids"]}
    result = {"version": 1, "salt": "glm53-stock-comparison-v1",
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "selection": "Length-only from training split; sealed test never accessed",
              "banks": banks, "cells": cells, "long_context": probes,
              "cache_probe_ids": probes[0]["prompt_ids"][:12288],
              "provenance": {key: provenance[key] for key in sorted(used)},
              "long_context_source": "Same source file, sorted IDs; nonrepetitive reference packet"}
    target = root / "workloads.json"
    with target.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                      "cells": [{k: c[k] for k in ("name", "prompt_counts")} for c in cells],
                      "long_context": [p["actual_prompt_tokens"] for p in probes]}))


def main() -> None:
    """Freeze tokenizer-rendered inputs in a CPU-only container."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.model)


if __name__ == "__main__":
    main()
