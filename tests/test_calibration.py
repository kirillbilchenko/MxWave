"""Tests for module-input calibration capture and artifact validation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from mxwave.calibration import (
    ActivationCollector,
    attach_activation_hooks,
    load_calibration_data,
    save_calibration_data,
)
from mxwave.calibration_cli import _load_corpus, _parse_objectives, _tokenize_sequences
from mxwave.calibration_stream import calibrate_decoder_sequentially


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(32, 4, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.proj(values)


class _ToyRotary(torch.nn.Module):
    def __init__(self, _config: object, device: torch.device | None = None) -> None:
        super().__init__()
        self.device = device

    def forward(
        self, values: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (1, position_ids.shape[-1], values.shape[-1])
        zeros = torch.zeros(shape, dtype=values.dtype, device=values.device)
        return zeros, zeros


class _ToyDecoderLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(32, 32, bias=False)

    def forward(self, hidden_states: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        return hidden_states + self.proj(hidden_states)


class _ToyDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3_5_text",
            layer_types=["full_attention", "full_attention"],
        )
        self.embed_tokens = torch.nn.Embedding(16, 32)
        self.layers = torch.nn.ModuleList([_ToyDecoderLayer(), _ToyDecoderLayer()])
        self.rotary_emb = _ToyRotary(self.config)


class _ToyCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ToyDecoder()


def _metadata() -> dict[str, str]:
    return {
        "policy": "all-linear",
        "source_repository": "example/model",
        "source_revision": "abc123",
        "num_sequences": "2",
        "sequence_length": "3",
        "num_tokens": "6",
        "corpus_sha256": "corpus",
        "token_ids_sha256": "tokens",
        "hessian_damp": "0.0",
    }


def test_collector_captures_exact_mean_rms_and_block_hessian() -> None:
    model = _TinyModel()
    collector = ActivationCollector(
        {"proj.weight": 32},
        ("mean-abs", "rms", "block-hessian"),
        hessian_damp=0.0,
    )
    handles = attach_activation_hooks(model, collector)
    values = torch.arange(-32, 32, dtype=torch.float32).reshape(2, 32)
    model(values)
    for handle in handles:
        handle.remove()

    statistics = collector.finalize()
    assert torch.equal(statistics["mean-abs"]["proj.weight"], values.abs().mean(dim=0))
    assert torch.equal(
        statistics["rms"]["proj.weight"],
        values.square().mean(dim=0).sqrt(),
    )
    expected_hessian = torch.einsum("ni,nj->ij", values, values).unsqueeze(0) / 2
    assert torch.equal(statistics["block-hessian"]["proj.weight"], expected_hessian)
    assert collector.observation_counts == {"proj.weight": 2}


def test_collector_requires_every_target_to_be_observed() -> None:
    collector = ActivationCollector({"proj.weight": 32})
    with pytest.raises(ValueError, match="did not observe"):
        collector.finalize()


def test_hook_attachment_rejects_missing_module() -> None:
    with pytest.raises(ValueError, match="missing 1 calibration module"):
        attach_activation_hooks(_TinyModel(), ActivationCollector({"other.weight": 32}))


def test_sequential_calibration_loads_and_propagates_one_layer_at_a_time(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    source = _ToyCausalLM()
    checkpoint = tmp_path / "model.safetensors"
    tensors = {name: value.detach().clone() for name, value in source.state_dict().items()}
    save_file(tensors, checkpoint)
    checkpoint_files = {name: checkpoint for name in tensors}
    sequences = [[1, 2], [3, 4]]
    progress: list[tuple[int, int]] = []

    result = calibrate_decoder_sequentially(
        _ToyCausalLM(),
        {
            "model.layers.0.proj.weight": 32,
            "model.layers.1.proj.weight": 32,
        },
        ("mean-abs",),
        sequences,
        checkpoint_files,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=1,
        hessian_damp=0.0,
        progress=lambda completed, total: progress.append((completed, total)),
    )

    token_ids = torch.tensor(sequences)
    embedded = torch.nn.functional.embedding(token_ids, tensors["model.embed_tokens.weight"])
    after_first = embedded + torch.nn.functional.linear(
        embedded, tensors["model.layers.0.proj.weight"]
    )
    assert torch.allclose(
        result.statistics["mean-abs"]["model.layers.0.proj.weight"],
        embedded.abs().reshape(-1, 32).mean(dim=0),
    )
    assert torch.allclose(
        result.statistics["mean-abs"]["model.layers.1.proj.weight"],
        after_first.abs().reshape(-1, 32).mean(dim=0),
    )
    assert result.observation_counts == {
        "model.layers.0.proj.weight": 4,
        "model.layers.1.proj.weight": 4,
    }
    assert progress == [(1, 2), (2, 2)]


def test_calibration_safetensors_round_trip_and_identity(tmp_path: Path) -> None:
    path = tmp_path / "stats.safetensors"
    mean_abs = torch.linspace(0.1, 1.0, 32)
    rms = torch.linspace(0.2, 1.1, 32)
    save_calibration_data(
        path,
        {
            "mean-abs": {"proj.weight": mean_abs},
            "rms": {"proj.weight": rms},
        },
        _metadata(),
    )

    loaded = load_calibration_data(
        path,
        "rms",
        {"proj.weight": 32},
        expected_policy="all-linear",
        expected_source_repository="example/model",
        expected_source_revision="abc123",
    )
    assert torch.equal(loaded.tensors["proj.weight"], rms)
    assert loaded.metadata["format"] == "mxwave-activation-stats"
    assert loaded.metadata["num_tokens"] == "6"
    assert len(loaded.file_sha256) == 64


def test_calibration_loader_accepts_pre_rename_artifact(tmp_path: Path) -> None:
    path = tmp_path / "legacy-stats.safetensors"
    metadata = _metadata()
    metadata.update(
        {
            "format": "mxstream-activation-stats",
            "format_version": "1",
            "objectives": '["mean-abs"]',
            "target_count": "1",
        }
    )
    expected = torch.linspace(0.1, 1.0, 32)
    save_file({"mean-abs::proj.weight": expected}, path, metadata=metadata)

    loaded = load_calibration_data(
        path,
        "mean-abs",
        {"proj.weight": 32},
        expected_policy="all-linear",
    )

    assert torch.equal(loaded.tensors["proj.weight"], expected)
    assert loaded.metadata["format"] == "mxstream-activation-stats"


def test_calibration_loader_rejects_partial_coverage(tmp_path: Path) -> None:
    path = tmp_path / "stats.safetensors"
    save_calibration_data(
        path,
        {"mean-abs": {"proj.weight": torch.ones(32)}},
        _metadata(),
    )
    with pytest.raises(ValueError, match="coverage mismatch"):
        load_calibration_data(
            path,
            "mean-abs",
            {"proj.weight": 32, "other.weight": 32},
            expected_policy="all-linear",
        )


def test_calibration_loader_rejects_wrong_source_revision(tmp_path: Path) -> None:
    path = tmp_path / "stats.safetensors"
    save_calibration_data(
        path,
        {"mean-abs": {"proj.weight": torch.ones(32)}},
        _metadata(),
    )
    with pytest.raises(ValueError, match="source_revision"):
        load_calibration_data(
            path,
            "mean-abs",
            {"proj.weight": 32},
            expected_policy="all-linear",
            expected_source_revision="different",
        )


class _Tokenizer:
    eos_token_id = 99

    def __call__(self, text: str, **_kwargs: object) -> dict[str, list[int]]:
        return {"input_ids": [int(value) for value in text.split()]}


def test_jsonl_corpus_is_deterministically_packed(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"text":"1 2 3"}\n{"text":"4 5 6 7 8"}\n')
    texts = _load_corpus(path, "text")
    sequences = _tokenize_sequences(
        _Tokenizer(),
        texts,
        num_sequences=2,
        sequence_length=4,
    )
    assert sequences == [[1, 2, 3, 99], [4, 5, 6, 7]]
    assert _parse_objectives("mean-abs,rms,mean-abs") == ("mean-abs", "rms")


def test_tokenize_sequences_supports_disjoint_offsets() -> None:
    sequences = _tokenize_sequences(
        _Tokenizer(),
        ["1 2 3", "4 5 6", "7 8 9", "10 11 12"],
        num_sequences=2,
        sequence_length=4,
        sequence_offset=1,
    )

    assert sequences == [[4, 5, 6, 99], [7, 8, 9, 99]]


def test_tokenize_sequences_rejects_negative_offset() -> None:
    with pytest.raises(ValueError, match="sequence_offset"):
        _tokenize_sequences(
            _Tokenizer(),
            ["1 2 3"],
            num_sequences=1,
            sequence_length=4,
            sequence_offset=-1,
        )
