"""Recompute retained raw-logit, layer-boundary and API repeat evidence on CPU."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    return v - v.max() - np.log(np.exp(v - v.max()).sum())


def kl(a, b):
    p, q = normalize(a), normalize(b)
    supported = np.isfinite(p)
    return max(0.0, float(np.sum(np.exp(p[supported]) * (p[supported] - q[supported]))))


def values(root, event):
    path = root / event["file"]
    assert digest(path) == event["sha256"]
    assert path.stat().st_size == event["bytes"]
    if event["dtype"] == "torch.bfloat16":
        raw = np.fromfile(path, dtype="<u2").astype("<u4") << 16
        return raw.view("<f4").reshape(event["shape"])
    dtype = {
        "torch.float32": "<f4",
        "torch.float16": "<f2",
        "torch.float64": "<f8",
        "torch.int64": "<i8",
        "torch.int32": "<i4",
        "torch.uint8": "u1",
        "torch.bool": "?",
    }[event["dtype"]]
    return np.fromfile(path, dtype=dtype).reshape(event["shape"])


def compare(a, b):
    assert a.shape == b.shape
    delta = a.astype(np.float64) - b.astype(np.float64)
    return {
        "elements": a.size,
        "different": int(np.count_nonzero(a != b)),
        "max_abs": float(np.max(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(delta**2))),
        "all_finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
    }


def signature(event):
    p = event["prepared"]
    return dict(
        total=event["total_scheduled_tokens"],
        **{
            k: p[k]
            for k in (
                "num_reqs",
                "num_tokens",
                "num_tokens_after_padding",
                "has_prefill",
                "prefill_lengths",
                "descriptor",
            )
        },
        row_order=[
            {k: r[k] for k in ("prompt_sha256", "prompt_tokens", "scheduled_tokens")}
            for r in p["row_order"]
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    report = json.loads((root / "stock-measurement.json").read_bytes())
    records = {r["label"]: r for r in report["records"]}
    result = {
        "status": report["status"],
        "controls": [],
        "snapshots": [],
        "API_vs_raw_logits": [],
        "limitations": [
            "Captured requests are intrusive diagnostic observations, not performance trials.",
            "First differing recorded boundary does not by itself identify the internal kernel.",
        ],
    }
    api = {}
    tokens = {}
    for label, row in records.items():
        path = root / row["file"]
        assert digest(path) == row["sha256"]
        with gzip.open(path) as source:
            response = json.load(source)
        c = response["choices"][0]
        tokens[label] = c["token_ids"]
        dist = []
        for top in c["logprobs"]["top_logprobs"]:
            v = np.full(154880, np.nan)
            for name, value in top.items():
                v[int(name.split(":")[1])] = value
            assert not np.isnan(v).any()
            dist.append(v)
        api[label] = dist
    for prefix in (
        "control-before-t1",
        "control-before-t4",
        "control-after-t1",
        "control-after-t4",
        "coarse",
        "coarse-four",
        "history",
    ):
        labels = [l for l in records if l.startswith(prefix + "-r")]
        if len(labels) < 2:
            continue
        a = labels[0]
        for b in labels[1:]:
            rows = []
            for i, (p, q) in enumerate(zip(api[a], api[b])):
                match = tokens[a][:i] == tokens[b][:i]
                rows.append(
                    {
                        "position": i,
                        "same_conditioning": match,
                        "kl": kl(p, q) if match else None,
                        "raw_reported_logprobs_exact": bool(np.array_equal(p, q)),
                    }
                )
            traces = {
                l: [[signature(e) for e in rank["events"]] for rank in records[l]["trace"]]
                for l in (a, b)
            }
            assert traces[a][0] == traces[a][1] and traces[b][0] == traces[b][1]
            result["controls"].append(
                {
                    "left": a,
                    "right": b,
                    "rows": rows,
                    "layouts_match": traces[a] == traces[b],
                    "generated_tokens_match": tokens[a] == tokens[b],
                }
            )
    for label, row in records.items():
        if not row["capture"]:
            continue
        for snapshot in row["capture"]:
            rank = snapshot["rank"]
            directory = root / f"stock-r{rank}" / label
            if not directory.exists():
                continue
            for event in snapshot["events"]:
                path = directory / event["file"]
                assert path.stat().st_size == event["bytes"] and digest(path) == event["sha256"]
            event = next(e for e in snapshot["events"] if e["label"] == "logits/output")
            raw = values(directory, event).reshape(-1, 154880)[0]
            result["API_vs_raw_logits"].append(
                {
                    "label": label,
                    "rank": rank,
                    "kl": kl(raw, api[label][0]),
                    "max_normalized_logprob_difference": float(
                        np.max(np.abs(normalize(raw) - normalize(api[label][0])))
                    ),
                    "raw_top1": int(raw.argmax()),
                    "api_top1": int(api[label][0].argmax()),
                }
            )
    pairs = []
    for label in records:
        if label.endswith("-r0") and records[label]["capture"]:
            other = label[:-1] + "1"
            if other in records:
                pairs.append((label, other))
    for left, right in pairs:
        for rank in (0, 1):
            a = root / f"stock-r{rank}" / left
            b = root / f"stock-r{rank}" / right
            if not a.exists() or not b.exists():
                continue
            la = next(s for s in records[left]["capture"] if s["rank"] == rank)
            lb = next(s for s in records[right]["capture"] if s["rank"] == rank)
            assert len(la["events"]) == len(lb["events"])
            diff = []
            equal = 0
            for x, y in zip(la["events"], lb["events"]):
                assert all(x[k] == y[k] for k in ("label", "shape", "dtype"))
                if x["sha256"] == y["sha256"]:
                    equal += 1
                    continue
                detail = {
                    "label": x["label"],
                    "shape": x["shape"],
                    "dtype": x["dtype"],
                    "left_sha256": x["sha256"],
                    "right_sha256": y["sha256"],
                }
                if len(diff) < 12 or x["label"] == "logits/output":
                    vx, vy = values(a, x), values(b, y)
                    detail.update(compare(vx, vy))
                    if x["label"] == "logits/output":
                        detail["kl"] = kl(vx.ravel(), vy.ravel())
                diff.append(detail)
            result["snapshots"].append(
                {
                    "left": left,
                    "right": right,
                    "rank": rank,
                    "equal_tensors": equal,
                    "differing_tensors": len(diff),
                    "first_difference": diff[0] if diff else None,
                    "differences": diff,
                }
            )
    (root / "repeat-review.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "controls": result["controls"],
                "first_differences": [
                    {k: r[k] for k in ("left", "rank", "first_difference", "differing_tensors")}
                    for r in result["snapshots"]
                ],
                "API_vs_raw_logits": result["API_vs_raw_logits"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
