# Native release inputs

The release distributes the three libraries pinned in `mxwave_glm53/bundle.py`,
compiled for Linux ARM64 / SM121. The wheel checks SHA256 before loading them.
Source builds are available for review/reproduction; different compiler/linker
versions need not produce byte-identical binaries. Do not weaken runtime hashes
to use a new build without qualifying it.

The libraries use a C ABI and CUDA, not the Torch C++ ABI. The qualified build
uses CUDA 13. Build on ARM64 with nvcc, CMake and Ninja. Run from the MxWave source
root with clean output directories.

## Marlin MoE

```bash
git clone https://github.com/vllm-project/vllm.git external-vllm
git -C external-vllm checkout db9527a46873454610df6dbedf79a36d6bf1a7f6
TORCH_INCLUDE=$(python3 -c 'from torch.utils.cpp_extension import include_paths; print(include_paths()[0])')
python3 integrations/glm53_mxfp4/build_moe.py --source external-vllm \
  --output build-moe-raw --torch-include "$TORCH_INCLUDE" --variants raw
python3 integrations/glm53_mxfp4/build_moe.py --source external-vllm \
  --output build-moe-sparse --torch-include "$TORCH_INCLUDE" --variants sparse \
  --affine-only --shared-stage
```

`moe_sources.json` pins adapted source hashes. `build_moe.py` retains notices,
adds MxWave's decoder/C ABI bridge, and writes provenance. The qualified sparse
library uses affine palettes and shared staging, without the later experimental
full-K or nibble-stage options.

## ZipServ vocabulary codec

```bash
git clone https://github.com/xxyux/ZipServ.git external-zipserv
git -C external-zipserv checkout f69cea959005222bed64fc847bc3fcdcca32a904
python3 deployment/glm53/native/zipserv/build_native.py \
  --source external-zipserv --arch 121 \
  --output build-zipserv/libmxwave_kda_zipserv.so
```

The historical filename includes KDA; the stock recipe uses it for vocabulary
assets. MxWave's bridge comes from its original `lossless-mxfp4` worktree. The
builder retains ZipServ's license and hashes, fixes per-call shared-memory sizing
and removes an unused legacy cuSPARSE include. No optional `--n128-warps` override
is selected.

Both upstream projects provide Apache-2.0 grants. Included license copies,
`THIRD_PARTY_NOTICES.txt` and `CONTRIBUTING.md` record attribution/modifications.
The native archive includes these MxWave build sources; upstream source remains
at the pinned repositories. No DeepInfer implementation is included.
