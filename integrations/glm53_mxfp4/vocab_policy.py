"""Configuration-bound TP2 vocabulary targets for the imported GLM checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kda_policy import load_policy


@dataclass(frozen=True)
class VocabTarget:
    """One untied, unpadded BF16 vocabulary matrix with a stock row partition."""

    kind: str
    prefix: str
    source_key: str
    rows: int = 77440
    width: int = 4096
    vocabulary: int = 154880


def load_targets(source: Path) -> dict[str, VocabTarget]:
    """Bind the audited source metadata and reject tying or changed vocabulary dimensions."""
    load_policy(source)
    config = json.loads((source / "config.json").read_bytes())
    text = config["text_config"]
    if (text.get("tie_word_embeddings") is not False
            or config.get("tie_word_embeddings", False)
            or (text["vocab_size"], text["hidden_size"]) != (154880, 4096)):
        raise ValueError("Vocabulary compression requires the audited untied BF16 GLM TP2 model")
    targets = (
        VocabTarget("head", "language_model.lm_head", "lm_head.weight"),
        VocabTarget("embedding", "language_model.model.embed_tokens",
                    "model.language_model.embed_tokens.weight"),
    )
    return {target.prefix: target for target in targets}


def validate_layer(layer: Any, target: VocabTarget) -> None:
    """Keep stock masking, partitioning and gather behavior with no added or padded tokens."""
    if (layer.tp_size != 2 or layer.tp_rank not in (0, 1) or layer.disable_tp
            or layer.num_embeddings != target.vocabulary
            or layer.org_vocab_size != target.vocabulary
            or layer.num_embeddings_padded != target.vocabulary
            or layer.embedding_dim != target.width
            or layer.num_embeddings_per_partition != target.rows
            or layer.shard_indices.org_vocab_start_index != layer.tp_rank * target.rows
            or layer.shard_indices.org_vocab_end_index != (layer.tp_rank + 1) * target.rows):
        raise ValueError("Vocabulary compression requires the audited unpadded TP2 row partition")
