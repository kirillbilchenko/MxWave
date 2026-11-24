"""Register the explicit test adapter and save startup ownership before serving."""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any


def install() -> None:
    """Keep all original production hooks and attach only the tested NVFP4 consumer."""
    import nv_adapter

    nv_adapter.install()
    from vllm.v1.worker.gpu_worker import Worker

    original = Worker.compile_or_warm_up_model
    if getattr(original, '_mxwave_nvfp4_test', False):
        return

    @functools.wraps(original)
    def prepared(worker: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(worker, *args, **kwargs)
        report = worker.nv_snapshot(True)
        target = Path('/traces') / f"nv-startup-r{report['rank']}.json"
        target.write_text(json.dumps(report, indent=2) + '\n')
        print('MxWave NVFP4 startup coverage complete', flush=True)
        return result

    prepared._mxwave_nvfp4_test = True
    Worker.compile_or_warm_up_model = prepared
