"""Tests for exact next-token divergence artifact compatibility."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_script() -> ModuleType:
    path = Path("scripts/evaluate_next_token_kl.py")
    spec = importlib.util.spec_from_file_location("evaluate_next_token_kl", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "format_name",
    ["mxwave-next-token-contexts-v1", "mxstream-next-token-contexts-v1"],
)
def test_context_loader_accepts_current_and_frozen_pre_rename_formats(
    tmp_path: Path,
    format_name: str,
) -> None:
    path = tmp_path / "contexts.json"
    document = {
        "format": format_name,
        "num_contexts": 2,
        "contexts": [{"token_ids": [1]}, {"token_ids": [2]}],
    }
    path.write_text(json.dumps(document))

    loaded, digest = _load_script()._load_context_manifest(path, 1)

    assert loaded["num_contexts"] == 1
    assert loaded["contexts"] == [{"token_ids": [1]}]
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_context_loader_rejects_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "contexts.json"
    path.write_text(json.dumps({"format": "unknown", "contexts": [{"token_ids": [1]}]}))

    with pytest.raises(ValueError, match="Unsupported context manifest"):
        _load_script()._load_context_manifest(path, None)


def test_fixed_chunk_indices_preserve_explicit_order() -> None:
    module = _load_script()

    assert module._fixed_chunk_indices("7, 2,5", 8) == [7, 2, 5]


@pytest.mark.parametrize(
    ("value", "population", "message"),
    [
        ("", 4, "non-empty comma-separated"),
        ("1,,2", 4, "non-empty comma-separated"),
        ("one,2", 4, "only integers"),
        ("-1,2", 4, "only nonnegative"),
        ("0,4", 4, "out of range"),
        ("2,0,2", 4, "duplicate"),
    ],
)
def test_fixed_chunk_indices_reject_invalid_values(
    value: str,
    population: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _load_script()._fixed_chunk_indices(value, population)


def test_fixed_chunk_indices_conflict_with_non_default_context_count() -> None:
    module = _load_script()
    args = SimpleNamespace(chunk_indices="0,2", num_contexts=2)

    with pytest.raises(ValueError, match="cannot be combined"):
        module._select_chunk_indices(args, 4)


def test_context_selection_keeps_legacy_even_spacing_by_default() -> None:
    module = _load_script()
    args = SimpleNamespace(chunk_indices=None, num_contexts=3)

    indices, selection = module._select_chunk_indices(args, 5)

    assert indices == [0, 2, 4]
    assert selection == "evenly-spaced-inclusive-endpoints"


def test_prepare_records_explicit_fixed_chunk_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert not add_special_tokens
            return [ord(text[0])]

        def __len__(self) -> int:
            return 256

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(
            model_path: Path,
            *,
            local_files_only: bool,
            trust_remote_code: bool,
        ) -> FakeTokenizer:
            assert model_path == tmp_path / "model"
            assert local_files_only
            assert not trust_remote_code
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("aaaabbbbccccdddd")
    output = tmp_path / "contexts.json"
    args = SimpleNamespace(
        model=str(tmp_path / "model"),
        corpus=str(corpus),
        output=str(output),
        num_contexts=128,
        chunk_indices="3,1",
        context_tokens=512,
        chunk_characters=4,
    )

    _load_script().prepare_contexts(args)

    manifest = json.loads(output.read_text())
    assert manifest["selection"] == "explicit-chunk-indices"
    assert manifest["selected_chunk_indices"] == [3, 1]
    assert manifest["num_contexts"] == 2
    assert [item["chunk_index"] for item in manifest["contexts"]] == [3, 1]


def _collect_args(module: ModuleType, *backend_args: str) -> object:
    return module.build_parser().parse_args(
        [
            "collect",
            "--model",
            "/models/candidate",
            "--model-label",
            "candidate",
            "--contexts",
            "/run/contexts.json",
            "--output",
            "/run/logprobs.safetensors",
            "--runtime-image",
            "vllm/vllm-openai:v0.29.0",
            *backend_args,
        ]
    )


def test_engine_kwargs_leave_backends_on_runtime_defaults() -> None:
    module = _load_script()
    args = _collect_args(module)

    engine_kwargs, max_model_len = module._build_engine_kwargs(
        args,
        Path("/models/candidate"),
        512,
    )

    assert max_model_len == 1024
    assert "linear_backend" not in engine_kwargs
    assert "moe_backend" not in engine_kwargs
    assert module._backend_metadata(args) == {
        "linear_backend": "auto",
        "moe_backend": "auto",
    }


def test_engine_kwargs_include_explicit_linear_and_moe_backends() -> None:
    module = _load_script()
    args = _collect_args(
        module,
        "--linear-backend",
        "marlin",
        "--moe-backend",
        "marlin",
    )

    engine_kwargs, _ = module._build_engine_kwargs(
        args,
        Path("/models/candidate"),
        512,
    )

    assert engine_kwargs["linear_backend"] == "marlin"
    assert engine_kwargs["moe_backend"] == "marlin"
    assert module._backend_metadata(args) == {
        "linear_backend": "marlin",
        "moe_backend": "marlin",
    }
