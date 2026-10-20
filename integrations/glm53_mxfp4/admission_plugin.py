"""Compose the unchanged model adapters and the isolated admission experiment hook."""

from __future__ import annotations

import os


def install() -> None:
    """Enable the queue wrapper only under the separately frozen experiment opt-in."""
    if os.environ.get("MXWAVE_PREFILL_ADMISSION") == "1":
        from prefill_admission import install as install_admission

        install_admission()
