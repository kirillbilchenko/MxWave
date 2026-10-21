"""Validate and assemble accepted CT feature histories from native speculative frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from .drafter_fp8_roundtrip import digest

LAYERS = (5, 14, 24, 33, 42)


def load_frame(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Verify a captured payload and recover the original BF16 feature interpretation."""
    metadata = json.loads(path.with_suffix(".json").read_bytes())
    if digest(path) != metadata["payload_sha256"]:
        raise ValueError(f"Native frame integrity failure: {path.name}")
    tensors = load_file(str(path))
    for name, dtype in metadata["tensor_dtypes"].items():
        if dtype == "torch.bfloat16":
            tensors[name] = tensors[name].view(torch.bfloat16)
    return tensors, metadata


class AcceptedHistory:
    """Accumulate contiguous, verified context without retaining rejected draft rows."""

    def __init__(self, layers: tuple[int, ...] = LAYERS) -> None:
        if not layers or len(set(layers)) != len(layers) or any(
            type(layer) is not int or not 0 <= layer < 45 for layer in layers
        ):
            raise ValueError("Expected distinct target decoder layer IDs")
        self.layers = layers
        self.tokens = torch.empty(0, dtype=torch.int64)
        self.features: list[torch.Tensor] = []
        self.generated: list[int] = []
        self.frame_index = 0
        self.prompt_length = 0
        self.rejected_rows = 0

    def append(self, tensors: dict[str, torch.Tensor], metadata: dict[str, Any]) -> None:
        """Append the accepted input prefix, checking its token and position alignment."""
        if metadata["index"] != self.frame_index:
            raise ValueError("Missing, duplicated or unordered frame")
        ids = tensors["target_input_ids"].long()
        positions = tensors["target_positions"]
        produced = int(tensors["target_num_sampled"].item())
        sampled = tensors["target_sampled_tokens"].flatten()[:produced].long()
        if not 1 <= produced <= 8 or sampled.numel() != produced or (sampled < 0).any():
            raise ValueError("Invalid native sampled-token count")
        if self.frame_index == 0:
            if metadata["target_num_draft_tokens"] != 0 or produced != 1:
                raise ValueError("Initial frame must contain the full prompt prefill")
            valid = ids.numel()
            self.prompt_length = valid
        else:
            if metadata["target_num_draft_tokens"] != 7 or ids.numel() != 8:
                raise ValueError("Expected the fixed seven-draft verification policy")
            valid = produced
            if int(ids[0]) != self.generated[-1]:
                raise ValueError("Next verification anchor differs from previous bonus token")
            if not torch.equal(ids[1:valid], sampled[:valid - 1]):
                raise ValueError("Accepted draft inputs disagree with sampled CT tokens")
            rejected = int(tensors["target_num_rejected"].item())
            if rejected != 8 - valid:
                raise ValueError("Rejected count disagrees with accepted prefix")
            self.rejected_rows += rejected
        start = self.tokens.numel()
        if not torch.equal(positions[:valid], torch.arange(start, start + valid)):
            raise ValueError("Feature positions are not a contiguous accepted history")
        query_positions = tensors["draft_query_positions"]
        if not torch.equal(query_positions, torch.arange(start + valid, start + valid + 8)):
            raise ValueError("Next draft query does not follow accepted context")
        anchor = int(tensors["anchor_before_propose"].item())
        if anchor != int(sampled[-1]) or int(tensors["draft_query_ids"][0]) != anchor:
            raise ValueError("Next draft anchor disagrees with the final sampled token")
        features = []
        for layer in self.layers:
            value = tensors[f"aux_layer_{layer}"]
            if value.shape != (ids.numel(), 4096) or value.dtype != torch.bfloat16:
                raise ValueError(f"Unexpected target feature geometry/dtype: layer {layer}")
            features.append(value[:valid])
        self.features.append(torch.cat(features, dim=-1))
        self.tokens = torch.cat([self.tokens, ids[:valid]])
        self.generated.extend(sampled.tolist())
        self.frame_index += 1

    def context(self) -> torch.Tensor:
        """Return the configured concatenated streams for accepted inputs only."""
        return torch.cat(self.features, dim=0)

    def validate_response(self, prompt: list[int], response: list[int]) -> dict[str, int]:
        """Bind accepted features to the actual capped API response before exporting data."""
        if self.prompt_length != len(prompt) or self.tokens[:len(prompt)].tolist() != prompt:
            raise ValueError("Captured prefill differs from the frozen prompt")
        if self.generated[:len(response)] != response or len(self.generated) < len(response):
            raise ValueError("Captured sampling differs from the API response")
        needed = len(prompt) + len(response) - 1
        if self.tokens.numel() < needed:
            raise ValueError("Missing accepted target features for the capped response")
        expected = prompt + response[:-1]
        if self.tokens[:needed].tolist() != expected:
            raise ValueError("Accepted feature inputs differ from prompt/response tokens")
        return {
            "prompt_tokens": len(prompt), "response_tokens": len(response),
            "feature_rows": needed, "rejected_rows_excluded": self.rejected_rows,
            "post_cap_sampled_tokens_excluded": len(self.generated) - len(response),
            "post_cap_feature_rows_excluded": self.tokens.numel() - needed,
            "frames": self.frame_index,
        }
