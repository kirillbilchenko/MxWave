"""Check likelihood extraction and window-level uncertainty in control evaluation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file


def _load_script(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent.parent / "scripts"))
    return importlib.import_module("evaluate_quantization_controls")


def _protocol_windows(script):
    windows = [
        {
            "index": index,
            "character_start": index * 4096,
            "character_end": (index + 1) * 4096,
            "text_sha256": str(index),
            "token_ids_sha256": script._token_digest(tokens),
            "token_ids": tokens,
        }
        for index, tokens in enumerate(([1, 2], [1, 2], list(range(10)), list(range(10, 20))))
    ]
    return {
        "ppl_windows": windows[:2],
        "kl_windows": windows[2:],
        "kl_contexts": [
            {
                "window_index": window["index"],
                "prefix_tokens": length,
                "token_ids": window["token_ids"][:length],
                "token_ids_sha256": script._token_digest(window["token_ids"][:length]),
            }
            for window in windows[2:]
            for length in range(1, 9)
        ],
    }


def test_prompt_likelihood_uses_actual_tokens_and_skips_only_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script(monkeypatch)
    output = SimpleNamespace(
        prompt_logprobs=[
            None,
            {999: SimpleNamespace(logprob=-0.1), 20: SimpleNamespace(logprob=-2.0)},
            {30: SimpleNamespace(logprob=-3.0)},
        ]
    )
    assert script._prompt_nll(output, [10, 20, 30]) == 5.0
    output.prompt_logprobs[1] = None
    with pytest.raises(ValueError, match="no likelihood"):
        script._prompt_nll(output, [10, 20, 30])


@pytest.mark.parametrize("storage", ["json", "gzip", "compact"])
def test_control_comparison_pairs_tokens_and_resamples_whole_kl_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    storage: str,
) -> None:
    script = _load_script(monkeypatch)
    labels = ["rtn", "mse", "norm", "h64", "bf16", "amd"]
    protocol = {
        "format": "mxwave-corrected-control-protocol-v1",
        "controls": labels,
        "executed_source_sha256": "test",
        "quantization_revision": "test",
        "dataset": {},
        "runtime": {},
        "runtime_image": "test",
        "remaining_windows": {},
        "historical_exclusion": {},
        **_protocol_windows(script),
    }
    protocol_path = tmp_path / "protocol.json"
    if storage == "compact":
        script._write_protocol(protocol_path, protocol)
        raw = protocol_path.read_bytes()
    else:
        raw = json.dumps(protocol).encode()
        if storage == "gzip":
            protocol_path = tmp_path / "protocol.json.gz"
            protocol_path.write_bytes(gzip.compress(raw, mtime=0))
        else:
            protocol_path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    teacher = torch.tensor([[0.6, 0.3, 0.1]]).log().repeat(16, 1)
    candidate = torch.tensor([[0.5, 0.25, 0.25]]).log().repeat(16, 1)
    candidate[8:] = torch.tensor([0.3, 0.3, 0.4]).log()
    for label in labels:
        distribution = tmp_path / f"{label}-logprobs.safetensors"
        save_file(
            {"logprobs": teacher if label == "bf16" else candidate},
            str(distribution),
            metadata={"protocol_sha256": digest},
        )
        nll = 1.0 if label == "bf16" else 1.0 + math.log(2)
        report = {
            "complete": True,
            "protocol_sha256": digest,
            "executed_source_sha256": "test",
            "ppl_windows": [
                {
                    "index": index,
                    "text_sha256": str(index),
                    "token_ids_sha256": protocol["ppl_windows"][index]["token_ids_sha256"],
                    "scored_tokens": 1,
                    "negative_log_likelihood": nll,
                }
                for index in range(2)
            ],
            "ppl": {},
            "decode": [],
            "checkpoint_sha256": label,
            "distributions": {"sha256": hashlib.sha256(distribution.read_bytes()).hexdigest()},
        }
        (tmp_path / f"{label}.json").write_text(json.dumps(report))
    output = tmp_path / "comparison.json"
    args = argparse.Namespace(
        output_dir=str(tmp_path),
        protocol=str(protocol_path),
        output=str(output),
        bootstrap_iterations=1000,
    )
    script.compare(args)
    result = json.loads(output.read_text())
    paired = result["paired_comparisons"]["rtn_vs_bf16"]
    assert paired["relative_perplexity_percent"] == pytest.approx(100)
    assert paired["relative_perplexity_ci95_percent"] == pytest.approx([100, 100])
    per_window = result["results"]["rtn"]["kl_per_window"]
    assert paired["forward_kl_difference_ci95"] == pytest.approx([min(per_window), max(per_window)])
    report_path = tmp_path / "rtn.json"
    report = json.loads(report_path.read_text())
    report["ppl_windows"][0]["token_ids_sha256"] = "mismatched"
    report_path.write_text(json.dumps(report))
    args.output = str(tmp_path / "invalid.json")
    with pytest.raises(ValueError, match="PPL pairing mismatch"):
        script.compare(args)


def test_compact_protocol_stores_each_window_once_and_restores_exact_prefixes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script(monkeypatch)
    original = _protocol_windows(script)
    path = tmp_path / "protocol.json"
    script._write_protocol(path, original)
    manifest = json.loads(path.read_bytes())
    assert '"token_ids":' not in path.read_text()
    assert manifest["token_data"]["count"] == 24
    assert manifest["token_data"]["sha256"] == hashlib.sha256(
        (tmp_path / manifest["token_data"]["filename"]).read_bytes()
    ).hexdigest()
    loaded, digest = script._load_protocol(path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    for role in ("ppl_windows", "kl_windows", "kl_contexts"):
        assert [item["token_ids"] for item in loaded[role]] == [
            item["token_ids"] for item in original[role]
        ]
    with pytest.raises(FileExistsError, match="overwrite frozen protocol"):
        script._write_protocol(path, original)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("sidecar", "sidecar hash mismatch"),
        ("window_hash", "window token hash mismatch"),
        ("prefix_hash", "prefix token hash mismatch"),
        ("offset", "invalid token ranges"),
        ("window_index", "invalid window or length"),
    ],
)
def test_compact_protocol_rejects_corrupted_or_mispaired_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
    message: str,
) -> None:
    script = _load_script(monkeypatch)
    path = tmp_path / "protocol.json"
    script._write_protocol(path, _protocol_windows(script))
    manifest = json.loads(path.read_bytes())
    if damage == "sidecar":
        sidecar = tmp_path / manifest["token_data"]["filename"]
        with sidecar.open("ab") as stream:
            stream.write(b"corrupted")
    elif damage == "window_hash":
        manifest["ppl_windows"][0]["token_ids_sha256"] = "mismatched"
    elif damage == "prefix_hash":
        manifest["kl_contexts"][0]["token_ids_sha256"] = "mismatched"
    elif damage == "offset":
        manifest["kl_windows"][0]["token_offset"] = 0
    elif damage == "window_index":
        manifest["kl_contexts"][0]["window_index"] = 0
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        script._load_protocol(path)


def test_archived_gzip_protocol_preserves_original_hash_chain(monkeypatch: pytest.MonkeyPatch):
    script = _load_script(monkeypatch)
    root = (
        Path(__file__).parent.parent / "benchmarks/qwen3.8-27b-corrected-controls-2026-10-03"
    )
    protocol, digest = script._load_protocol(root / "validation/protocol.json.gz")
    assert digest == "f15823f74400a6a95059d54eb9d3f81456c5d29e01ba5ae219a20c80c6a84289"
    assert len(protocol["ppl_windows"]) == 128
    assert len(protocol["kl_windows"]) == 64
    assert len(protocol["kl_contexts"]) == 512
    comparison = json.loads((root / "quality-comparison.json").read_bytes())
    assert comparison["protocol_sha256"] == digest
    for label in protocol["controls"]:
        report = json.loads((root / "measurements" / f"{label}.json").read_bytes())
        assert report["protocol_sha256"] == digest
        assert hashlib.sha256((root / "measurements" / f"{label}.json").read_bytes()).hexdigest() == (
            comparison["results"][label]["report_sha256"]
        )
