"""Architecture adapters for runtime-operation and expert-layout discovery."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ..expert_ir import ExpertQuantizationLayout
from ..shard import TensorInfo
from . import qwen3_5_moe

__all__ = ["resolve_expert_layout"]


def resolve_expert_layout(
    config: Mapping[str, object],
    tensor_infos: Iterable[TensorInfo],
) -> ExpertQuantizationLayout:
    """Resolve exactly one validated fused-expert architecture adapter.

    The converter is deliberately fail-closed: adding another model family
    requires an explicit adapter with config and header validation. Tensor-name
    suffixes alone never select an architecture.
    """
    if qwen3_5_moe.matches(config):
        return qwen3_5_moe.build_expert_layout(config, tensor_infos)
    raise ValueError(
        "No routed-expert layout adapter matches this checkpoint config; "
        "supported policies: qwen3.5-moe-routed-experts"
    )
