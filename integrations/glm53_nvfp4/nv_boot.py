"""Install experiment entry-point metadata only inside a disposable container."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    """Validate the exact source checkpoint and launch original production CLI settings."""
    expected = {'config.json': 'f14dc13ce3bef88a5539c9e61b3e4f3dbef6958ebc3acda4b7f05f418642c3b0',
                'model.safetensors.index.json':
                '8ac9d03981d2029223a66bdbda51fa224cb36de8f522d8c695f0724e4f34697e'}
    for name, digest in expected.items():
        if hashlib.sha256((Path('/model') / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Original ModelOpt checkpoint metadata changed: ' + name)
    with tempfile.TemporaryDirectory(prefix='mxwave-nvfp4-entry-') as tmp:
        package = Path(tmp)
        (package / 'pyproject.toml').write_text('''[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
[project]
name = "mxwave-nvfp4-model-test"
version = "0.0.1"
requires-python = ">=3.12"
[project.entry-points."vllm.general_plugins"]
mxwave_nvfp4_test = "nv_plugin:install"
[tool.setuptools]
py-modules = []
''')
        subprocess.run(['uv', 'pip', 'install', '--python', '/usr/bin/python3',
                        '--break-system-packages', '--no-deps', '--no-build-isolation',
                        str(package)], check=True)
    if os.environ.get('MXWAVE_VALIDATE_ONLY') == '1':
        from vllm import platforms
        from vllm.entrypoints.cli.serve import cmd_init
        from vllm.platforms.cpu import CpuPlatform
        from vllm.plugins import load_general_plugins
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        platforms.current_platform = CpuPlatform()
        load_general_plugins()
        parser = FlexibleArgumentParser()
        sub = parser.add_subparsers(dest='subparser')
        commands = cmd_init()
        for command in commands:
            command.subparser_init(sub)
        args = parser.parse_args(sys.argv[1:])
        next(c for c in commands if c.name == args.subparser).validate(args)
        assert args.max_model_len == 500000 and args.max_num_batched_tokens == 6919
        assert args.max_num_seqs == 32 and args.speculative_config['method'] == 'dflash'
        assert args.enable_prefix_caching and args.kv_cache_memory_bytes == 6442450944
        assert os.environ['GLM_CERT_HEAD'] == '1' and os.environ['GLM_LV_DRAFT_FP8_KV'] == '1'
        print(json.dumps({'status': 'validated', 'mode': os.environ['MXWAVE_NVFP4_MODE']}))
        return
    secret = Path('/run/secrets/llm_api_key').read_text().strip()
    assert secret and '\n' not in secret
    os.environ['VLLM_API_KEY'] = secret
    os.execv('/usr/bin/python3', ['/usr/bin/python3', '-c',
             'from vllm.entrypoints.cli.main import main; main()', *sys.argv[1:]])


if __name__ == '__main__':
    main()
