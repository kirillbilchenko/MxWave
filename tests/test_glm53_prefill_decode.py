"""Guard the component's restoration ownership and order-sensitive performance gates."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
import full_model_control as control
from benchmark_prefill_decode import ratio_passes
from prefill_pack_oracle import unpack_words
from run_prefill_decode import command
from zipserv_contract import require_qualified_exponent_base


def test_restore_recognizes_every_gpu_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    revision = "prefilldecodev2"
    commands = [command(tmp_path, revision, stage)
                for stage in ("smoke", "timing", "confirmation", "routing")]
    wrapped = command(tmp_path, revision, "smoke", build_directory="prefill-decode-v4/build",
                      correctness_only=True)
    assert wrapped[wrapped.index("--name") + 1] in {
        argv[argv.index("--name") + 1] for argv in commands}
    assert "/work/smoke_prefill_wrap.py" in wrapped
    assert "/audit/prefill-decode-v4/build" in wrapped
    with pytest.raises(ValueError, match="cannot launch timing"):
        command(tmp_path, revision, "timing", correctness_only=True)
    owned = {argv[argv.index("--name") + 1] for argv in commands}
    stopped = set()

    def fake_run(rank: int, argv: list[str], timeout: float = 90) -> str:
        if "--filter" in argv:
            name = argv[-1].removeprefix("name=^/").removesuffix("$")
            return "container-id" if rank == 1 and name in owned else ""
        if argv[:2] == ["docker", "stop"]:
            stopped.add(argv[-1])
        return ""

    monkeypatch.setattr(control, "run", fake_run)
    control.stop_owned(revision)
    assert stopped == owned
    for argv in commands:
        assert argv[argv.index("--network") + 1] == "none"
        assert "/usr/bin/timeout" in argv and "180" in argv
        assert f"{control.ROOT}:/audit:ro" in argv
        assert f"{tmp_path / 'src'}:/work:ro" in argv


def test_favorable_median_cannot_hide_an_order_regression() -> None:
    measured = {"median_paired_ratio": {"candidate": .90}, "order_ratios": {
        "forward": {"candidate": .85}, "reverse": {"candidate": 1.03}}}
    assert not ratio_passes(measured, "candidate", 0, 1.02)
    measured["order_ratios"]["reverse"]["candidate"] = .94
    assert ratio_passes(measured, "candidate", 0, .95)


def test_wrapped_exponent_base_explains_positive_zero_sign_flip() -> None:
    # A literal format fixture: every code is 1, every sign/mantissa byte is 0.
    # Base 255 encodes -1; recovering exponent zero requires eight-bit wrapping.
    medium = struct.pack("<8i", 0, 0, 1024, 0, 2048, 0, 3072, 0)
    buffers = (bytes(4096), b"", b"\xff" * 512, bytes(512), bytes(512), medium,
               struct.pack("<4i", 0, 0, 4096, 0))
    assert unpack_words(buffers, 64, 64, 255) == b"\x00\x80" * 4096
    assert unpack_words(buffers, 64, 64, 255, wrap_exponent=True) == bytes(8192)


@pytest.mark.parametrize("base", [249, 250, 251, 252, 253, 254, 255, -1, 256, True])
def test_qualified_loader_refuses_every_unsafe_exponent_base(base: int) -> None:
    with pytest.raises(ValueError, match="unsafe for the qualified decoder"):
        require_qualified_exponent_base(base, where="fixture")


@pytest.mark.parametrize("base", [0, 1, 115, 248])
def test_qualified_loader_accepts_safe_boundary_bases(base: int) -> None:
    require_qualified_exponent_base(base, where="fixture")
