"""Apply one sparse conversion policy before native model graph capture."""

from __future__ import annotations

from typing import Any

from high_concurrency_worker import HighConcurrencyWorker
from stable_sparse_conversion import install_stable_conversion

INSTALLATION = install_stable_conversion()


class StableSparseQualityWorker(HighConcurrencyWorker):
    """Expose source-bound policy metadata without adding model replay hooks."""

    def mxwave_stable_sparse_policy(self) -> dict[str, Any]:
        """Report every installed alias and the unchanged DCP-one scope."""
        return {
            **INSTALLATION,
            "actual_decode_context_parallel_size": self.model_runner.vllm_config.parallel_config.decode_context_parallel_size,
        }
