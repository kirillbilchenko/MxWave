"""Install the compact softened-RMS patch loader before vLLM worker model loading."""

from __future__ import annotations

import json
import os
from pathlib import Path

from kolibri_soft_rms_trial import composed_weights
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


def install() -> None:
    """Wrap only the explicitly specified local parent model's weight iterator."""
    specification = Path(os.environ["KOLIBRI_SOFT_RMS_SPEC"])
    baseline = Path(json.loads(specification.read_bytes())["baseline"]).resolve()
    original = DefaultModelLoader._get_weights_iterator
    if getattr(original, "_mxwave_soft_rms_selector", False):
        return

    def selected(loader, source):
        weights = original(loader, source)
        if source.prefix or Path(source.model_or_path).resolve() != baseline:
            return weights
        return composed_weights(weights, specification)

    selected._mxwave_soft_rms_selector = True
    DefaultModelLoader._get_weights_iterator = selected


class SoftRMSTrialWorker:
    """Worker extension whose import installs verified compact patch reconstruction."""


install()
