"""Functional, non-mutating replay of one bounded-residency PyTorch module."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = ["ModuleReplaySample", "make_module_replay"]


@dataclass(frozen=True)
class ModuleReplaySample:
    """Positional and keyword inputs for one module replay invocation."""

    args: tuple[Any, ...]
    kwargs: Mapping[str, Any] = field(default_factory=dict)


def make_module_replay(
    module: torch.nn.Module,
    output_adapter: Callable[[Any], Mapping[str, torch.Tensor]],
) -> Callable[[Mapping[str, torch.Tensor], ModuleReplaySample], Mapping[str, torch.Tensor]]:
    """Build a replay callback using temporary parameter or buffer overrides.

    ``torch.func.functional_call`` applies overrides without copying or mutating
    the resident module. Candidate mappings contain module-local state names.
    """
    resident_state = module.state_dict()

    def replay(
        overrides: Mapping[str, torch.Tensor],
        sample: ModuleReplaySample,
    ) -> Mapping[str, torch.Tensor]:
        replacement: dict[str, torch.Tensor] = {}
        for name, value in overrides.items():
            expected = resident_state.get(name)
            if expected is None:
                raise ValueError(f"Module replay override names unknown state {name!r}")
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Module replay override {name!r} is not a tensor")
            if value.shape != expected.shape:
                raise ValueError(
                    f"Module replay override {name!r} has shape {tuple(value.shape)}, "
                    f"expected {tuple(expected.shape)}"
                )
            if value.device != expected.device:
                raise ValueError(
                    f"Module replay override {name!r} is on {value.device}, "
                    f"expected {expected.device}"
                )
            replacement[name] = value

        raw_output = torch.func.functional_call(
            module,
            replacement,
            sample.args,
            dict(sample.kwargs),
            strict=False,
        )
        outputs = output_adapter(raw_output)
        if not isinstance(outputs, Mapping):
            raise TypeError("Module replay output adapter must return a mapping")
        return outputs

    return replay
