"""Registry and dispatch for runtime-operation architecture adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from .adapters import llama, qwen3_5
from .runtime_ir import RuntimeGraph

__all__ = [
    "RuntimeAdapter",
    "registered_runtime_adapters",
    "resolve_runtime_graph",
]

RuntimeMatcher = Callable[[Mapping[str, object]], bool]
RuntimeGraphBuilder = Callable[[Mapping[str, object], Iterable[str]], RuntimeGraph]


@dataclass(frozen=True)
class RuntimeAdapter:
    """Named architecture adapter registered for runtime graph discovery."""

    name: str
    matches: RuntimeMatcher
    build: RuntimeGraphBuilder

    def __post_init__(self) -> None:
        """Validate the stable adapter identifier."""
        if not self.name or any(character.isspace() for character in self.name):
            raise ValueError("Runtime adapter name must be non-empty and contain no whitespace")


_BUILTIN_ADAPTERS = (
    RuntimeAdapter("qwen3_5_text", qwen3_5.matches, qwen3_5.build),
    RuntimeAdapter("llama", llama.matches, llama.build),
)


def registered_runtime_adapters() -> tuple[RuntimeAdapter, ...]:
    """Return built-in adapters in deterministic dispatch order."""
    return _BUILTIN_ADAPTERS


def resolve_runtime_graph(
    config: Mapping[str, object],
    tensor_names: Iterable[str],
) -> RuntimeGraph:
    """Resolve a tested architecture adapter and build its runtime graph."""
    names = tuple(tensor_names)
    if not names:
        raise ValueError("Runtime graph discovery requires checkpoint tensor names")
    matching = tuple(adapter for adapter in _BUILTIN_ADAPTERS if adapter.matches(config))
    if len(matching) > 1:
        raise ValueError(
            "Multiple runtime-operation adapters match this model config: "
            f"{[adapter.name for adapter in matching]}"
        )
    if matching:
        return matching[0].build(config, names)
    raise ValueError(
        "No runtime-operation adapter matches this model config; add a tested architecture "
        f"adapter before using runtime-aware features. Registered adapters: "
        f"{[adapter.name for adapter in _BUILTIN_ADAPTERS]}"
    )
