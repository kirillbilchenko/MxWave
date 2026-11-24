"""Install native-source-bound MLA observation only when the disposable worker imports it."""

from __future__ import annotations

from mla_internal_worker import MLAInternalWorker, install

install()


class MLAInternalExtension(MLAInternalWorker):
    """Retain original reference arithmetic and the bounded actual-operation observer."""
