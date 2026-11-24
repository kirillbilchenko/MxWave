"""Protect rollback and benchmark evidence during the explicit CT serving handover."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest


def test_source_identity_normalizes_order_but_rejects_mount_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept the benchmark preparer's ordering convention without dropping host checks."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "deployment/glm53"))
    live = importlib.import_module("control_ct_live")
    source = {"Id": "original", "Image": "frozen", "Config": {}, "Mounts": [],
              "HostConfig": {"Binds": ["b:/b:ro", "a:/a:ro"], "Dns": None,
                             "OomKillDisable": None}}
    expected = copy.deepcopy(source)
    expected["HostConfig"] = {"Binds": ["a:/a:ro", "b:/b:ro"], "Dns": [],
                              "OomKillDisable": False}
    assert live.tested_identity(source) == live.c.identity(expected)
    source["HostConfig"]["Binds"][0] = "other:/b:ro"
    assert live.tested_identity(source) != live.c.identity(expected)


def test_live_clone_keeps_settings_and_isolates_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Only the listener, dev endpoint flag and writable report destination may change."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "deployment/glm53"))
    live = importlib.import_module("control_ct_live")
    source = {"Image": "sha256:frozen", "Config": {
        "Hostname": "source", "Image": "tag", "Entrypoint": ["secret-aware-launcher"],
        "Env": ["PRIVATE_SETTING=preserved", "VLLM_SERVER_DEV_MODE=1", "MXWAVE_PALETTE_MODEL=1"],
        "Cmd": ["serve", "/model", "--port", "8051", "--max-model-len", "500000",
                "--kv-cache-memory-bytes", "6442450944", "--speculative-config", "unchanged"]},
        "HostConfig": {"NetworkMode": "host", "Binds": [
            "/frozen:/trial:ro", "/evidence/reports:/reports:rw", "/checkpoint:/model:ro"]}}
    before = copy.deepcopy(source)
    result = live.live_config(source, tmp_path)
    assert source == before
    assert result["HostConfig"]["Binds"] == [
        "/frozen:/trial:ro", f"{tmp_path}/reports:/reports:rw", "/checkpoint:/model:ro"]
    assert result["Env"] == [
        "PRIVATE_SETTING=preserved", "MXWAVE_PALETTE_MODEL=1", "VLLM_SERVER_DEV_MODE=0"]
    result["Cmd"][result["Cmd"].index("--port") + 1] = "8051"
    assert result["Cmd"] == source["Config"]["Cmd"]
    assert result["Entrypoint"] == source["Config"]["Entrypoint"]
    source["HostConfig"]["Binds"].pop(1)
    with pytest.raises(ValueError, match="report mount"):
        live.live_config(source, tmp_path)


@pytest.mark.parametrize("committed", [False, True])
def test_watchdog_restores_abandoned_startup_but_keeps_requested_live_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, committed: bool,
) -> None:
    """A dead controller requires restoration only before the verified handover."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "deployment/glm53"))
    live = importlib.import_module("control_ct_live")
    (tmp_path / "controller.json").write_text(json.dumps({
        "phase": "complete_live" if committed else "loading_ct",
        "service_paused": not committed}))
    restored = []
    monkeypatch.setattr(live.c, "ticks", lambda pid: None)
    monkeypatch.setattr(live.c, "restore", restored.append)
    live.watch(tmp_path, 12345, "old-start-time")
    assert restored == ([] if committed else [tmp_path])
