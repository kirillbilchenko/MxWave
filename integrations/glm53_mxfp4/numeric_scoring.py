"""Dependency-free final numeric-answer contract shared by quality clients."""

from __future__ import annotations

import re
from decimal import Decimal


def numeric_answer(text: str) -> Decimal | None:
    """Accept an explicit final number, never a number inside unfinished reasoning."""
    value = text.rsplit("####", 1)[-1].strip() if "####" in text else text.strip()
    if not re.fullmatch(r"[-+]?\$?\d[\d,]*(?:\.\d+)?\.?", value):
        return None
    return Decimal(value.replace("$", "").replace(",", "").rstrip("."))
