"""Install only the metadata entry point in a disposable image, then exec its vLLM CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    """Make the opt-in adapter discoverable without touching host or production packages."""
    package = Path(__file__).parent / "engine_plugin"
    # Setuptools writes egg-info while building even a metadata-only package.
    # Keep the qualified source mounted read-only and build a temporary copy.
    with tempfile.TemporaryDirectory(prefix="mxwave-pilot-entrypoint-") as temporary:
        writable = Path(temporary) / "package"
        shutil.copytree(package, writable)
        subprocess.run(["uv", "pip", "install", "--python", "/usr/bin/python3",
                        "--break-system-packages", "--no-deps", "--no-build-isolation",
                        str(writable)], check=True)
    if os.environ.get("MXWAVE_VALIDATE_ONLY") == "1":
        from vllm import platforms
        from vllm.platforms.cpu import CpuPlatform

        if platforms.current_platform.is_unspecified():
            platforms.current_platform = CpuPlatform()
        from vllm.entrypoints.cli.serve import cmd_init
        from vllm.plugins import load_general_plugins
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        load_general_plugins()
        parser = FlexibleArgumentParser()
        subparsers = parser.add_subparsers(dest="subparser")
        commands = cmd_init()
        for command in commands:
            command.subparser_init(subparsers)
        args = parser.parse_args(sys.argv[1:])
        next(command for command in commands if command.name == args.subparser).validate(args)
        import model_plugin

        if not model_plugin._INSTALLED:
            raise RuntimeError("The pilot entry point was not discovered by vLLM")
        if os.environ.get("MXWAVE_GLM53_ATTENTION_MAPPING") == "1":
            from attention_diagnostic import AttentionDiagnosticWorker, DiagnosticWorker
            from attention_mapping import FUSED_ATTENTION_COMPONENTS
            from vllm.models.glm5next.nvidia.model import Glm5NextForConditionalGeneration

            mapping = Glm5NextForConditionalGeneration.packed_modules_mapping
            if any(mapping.get(name) != parts
                   for name, parts in FUSED_ATTENTION_COMPONENTS.items()):
                raise RuntimeError("Attention aliases were not registered before model setup")
            if (AttentionDiagnosticWorker.mxwave_set_eager
                    is not DiagnosticWorker.mxwave_set_eager):
                raise RuntimeError("The qualified graph-switch method changed")
        print(json.dumps({"status": "validated", "plugin": "discovered",
                          "model": args.model_tag, "node_rank": args.node_rank,
                          "max_model_len": args.max_model_len}))
        return
    os.execv("/usr/bin/python3", ["/usr/bin/python3", "-c",
             "from vllm.entrypoints.cli.main import main; main()", *sys.argv[1:]])


if __name__ == "__main__":
    main()
