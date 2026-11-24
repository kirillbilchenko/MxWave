"""Configuration-bound selection of the imported GLM checkpoint's BF16 KDA projections."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

CONFIG_SHA = "194d290ebb0f19aedc60d24ea9e236e33c86616e460f4a108a842f6fd5b8b198"
INDEX_SHA = "6d1364a27eef78675390422b1b5230601817fb7460ce816951b2cd515934c85d"
KDA_PARTS = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")


@dataclass(frozen=True)
class Target:
    """One exact runtime path with a TP2 logical weight shape and loading contract."""

    prefix: str
    layer: int
    kind: Literal["input", "output"]
    rows: int
    width: int

    @property
    def native_rows(self) -> int:
        """Return the 64-aligned decoding rows, including explicit zero padding."""
        return (self.rows + 63) // 64 * 64

    @property
    def expected_shards(self) -> frozenset[int | None]:
        """Require all six fused components or one complete row-parallel source."""
        return frozenset(range(6)) if self.kind == "input" else frozenset({None})


def targets_from_config(config: dict[str, Any]) -> dict[str, Target]:
    """Derive full runtime names from explicit layer types and validate redundant metadata."""
    text = config["text_config"]
    kinds = text["layer_types"]
    count = text["num_hidden_layers"]
    linear = text["linear_attn_config"]
    if (len(kinds) != count or any(k not in ("linear_attention", "deepseek_sparse_attention")
                                  for k in kinds)):
        raise ValueError("Incomplete or unknown GLM attention layer types")
    kda = [i for i, kind in enumerate(kinds) if kind == "linear_attention"]
    mla = [i for i, kind in enumerate(kinds) if kind == "deepseek_sparse_attention"]
    if linear["kda_layers"] != kda or linear["full_attn_layers"] != mla:
        raise ValueError("Conflicting explicit GLM KDA/MLA layer metadata")
    hidden, heads, dim = text["hidden_size"], linear["num_heads"], linear["head_dim"]
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (hidden, heads, dim)):
        raise ValueError("GLM dimensions must be positive integers")
    if heads % 2 or hidden % 64 or (heads * dim) % 128:
        raise ValueError("The adapter requires 64-aligned TP2 KDA dimensions")
    result = {}
    for layer in kda:
        prefix = f"language_model.model.layers.{layer}.self_attn."
        for suffix, kind, rows, width in (
            ("in_proj_qkvbfg_a", "input", 3 * heads * dim // 2 + heads // 2 + 2 * dim, hidden),
            ("o_proj", "output", hidden, heads * dim // 2),
        ):
            name = prefix + suffix
            result[name] = Target(name, layer, kind, rows, width)  # type: ignore[arg-type]
    return result


def load_policy(source: Path) -> dict[str, Target]:
    """Fail closed for other checkpoints instead of guessing from tensor-name suffixes."""
    for name, expected in (("config.json", CONFIG_SHA),
                           ("model.safetensors.index.json", INDEX_SHA)):
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Selective KDA adapter requires the audited imported checkpoint")
    result = targets_from_config(json.loads((source / "config.json").read_bytes()))
    if len(result) != 68:
        raise ValueError("Expected exactly 34 base KDA input/output pairs")
    return result
