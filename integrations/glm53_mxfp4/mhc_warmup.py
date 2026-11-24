"""Select warmup sizes by the installed mHC split policy, including mixed steps."""

from __future__ import annotations

from collections.abc import Callable


def token_sizes(max_tokens: int, split_for_tokens: Callable[[int], int]) -> list[int]:
    """Cover each split count plus small-token branches and common tile boundaries.

    The callable is the installed runtime's policy. No GPU-specific split
    formula is duplicated here. A token count of 257 can select a specialization
    that warming 256 and 512 misses.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    representatives: dict[int, int] = {}
    for tokens in range(1, max_tokens + 1):
        split = split_for_tokens(tokens)
        if split < 1:
            raise ValueError("The installed split policy returned a non-positive count")
        representatives.setdefault(split, tokens)
    common = {1, 2, 4, 7, 8, 16, 17, max_tokens}
    common.update(2 ** power for power in range(max_tokens.bit_length()))
    return sorted({size for size in common if size <= max_tokens} | set(representatives.values()))
