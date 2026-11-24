"""Reject metadata that can overflow the unchanged qualified ZipServ reconstruction."""

from __future__ import annotations


def require_qualified_exponent_base(base: int, *, where: str) -> None:
    """Require base + every three-bit code to fit the qualified kernel's exponent byte."""
    if isinstance(base, bool) or not isinstance(base, int) or not 0 <= base <= 248:
        raise ValueError(f"{where}: exponent base {base!r} is unsafe for the qualified decoder; "
                         "require 0..248 before loading or emitting packed weights")
