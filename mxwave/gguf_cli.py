"""Command-line bridge from MxWave checkpoints to llama.cpp's GGUF exporter."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from typing import Any, cast

from .gguf import install_llama_cpp_exporter

PROGRAM_NAME = "mxwave-export-gguf"


def main(argv: Sequence[str] | None = None) -> int:
    """Install MxWave support and delegate all arguments to llama.cpp."""
    install_llama_cpp_exporter()
    upstream_main = cast(Any, importlib.import_module("convert_hf_to_gguf")).main
    arguments = list(sys.argv[1:] if argv is None else argv)
    previous_argv = sys.argv
    sys.argv = [PROGRAM_NAME, *arguments]
    try:
        result = upstream_main()
    finally:
        sys.argv = previous_argv
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
