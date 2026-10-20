"""A tiny original model exercising the same diagnostic method and argument interfaces."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch


def make_model(device: torch.device, drift: bool) -> tuple[Any, Any, Any, Any]:
    """Create deterministic arithmetic plus optional deliberately mutable GEMM semaphore state."""
    caches = [torch.zeros(4, 8, device=device) for _ in range(68)]
    locks = [torch.zeros(2, dtype=torch.int32, device=device) for _ in range(2)]

    def gemm(*args: Any, **kwargs: Any) -> torch.Tensor:
        args[10].add_(1)
        args[1].copy_(args[0] + args[10][0] if drift else args[0])
        return args[1]

    ops = SimpleNamespace(moe_wna16_marlin_gemm=gemm)

    class Attention(torch.nn.Module):
        """Write selected active state rows and return an independently overwritten buffer."""

        def __init__(self, layer: int) -> None:
            super().__init__()
            self.kv_cache = caches[layer * 2:layer * 2 + 2]

        def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            """Write one logical step without an accumulating cache advance."""
            for value in self.kv_cache:
                value.copy_(hidden_states)
            return hidden_states.clone()

    class Experts(torch.nn.Module):
        """Expose routing, two native-style GEMMs and final reduction interfaces."""

        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index
            self.router = SimpleNamespace(select_experts=lambda x: (
                torch.ones(4, 2, device=device), torch.zeros(4, 2, dtype=torch.int32, device=device)))

        def _apply_quant_method(self, x: torch.Tensor, weights: torch.Tensor,
                                ids: torch.Tensor) -> torch.Tensor:
            out1, out2 = torch.empty_like(x), torch.empty_like(x)
            for source, target in ((x, out1), (out1, out2)):
                args = [source, target, None, None, None, None, None, None, None, None,
                        locks[self.index], ids.flatten(), ids.flatten(), ids.flatten()[:1], weights]
                ops.moe_wna16_marlin_gemm(*args, size_m=4, size_n=8, size_k=8,
                                         use_atomic_add=False, use_fp32_reduce=True)
            return out2

        def _maybe_reduce_final_output(self, value: torch.Tensor) -> torch.Tensor:
            return value

        def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
            """Route then evaluate the two projections with the same stub-native signature."""
            weights, ids = self.router.select_experts(router_logits)
            return self._maybe_reduce_final_output(self._apply_quant_method(hidden_states, weights, ids))

    class MoE(torch.nn.Module):
        """Match the gate/expert call nesting of the inspected model."""

        def __init__(self, index: int) -> None:
            super().__init__()
            self.gate = torch.nn.Identity()
            self.experts = Experts(index)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Keep inputs stable while the deliberately untracked semaphore state can vary."""
            return self.experts(value, self.gate(value))

    class Layer(torch.nn.Module):
        """Expose the layer and mHC method signatures required by the adapter."""

        def __init__(self, layer: int) -> None:
            super().__init__()
            self.layer_idx = layer
            self.self_attn = Attention(layer)
            self.mlp = torch.nn.Identity() if layer == 2 else MoE(layer - 3)

        def hc_pre(self, value: torch.Tensor, *args: Any) -> torch.Tensor:
            """Leave this fixture's pre-mix arithmetic unchanged."""
            return value

        def hc_fused_post_pre(self, value: torch.Tensor, *args: Any) -> torch.Tensor:
            """Keep a small in-place-overwrite boundary for copy-order verification."""
            return value

        def forward(self, positions: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            """Evaluate one attention/MLP pair through the wrapped interfaces."""
            value = self.hc_fused_post_pre(value)
            value = self.self_attn(value, positions)
            value = self.hc_fused_post_pre(value)
            return self.mlp(value)

    class Model(torch.nn.Module):
        """Own the original miniature decoder stack."""

        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity(),
                                              Layer(2), Layer(3), Layer(4)])

        def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            """Produce a four-row result with optional reproducible semaphore-induced drift."""
            value = input_ids[:, None].expand(4, 8).float()
            for index in (2, 3, 4):
                value = self.layers[index](positions, value)
            return value

    return Model(), ops, caches, locks
