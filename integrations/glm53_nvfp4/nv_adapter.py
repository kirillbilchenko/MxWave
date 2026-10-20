"""Opt-in ModelOpt NVFP4 scale storage, preserving installed routing and arithmetic."""
from __future__ import annotations

import functools
import hashlib
import inspect
import os
import weakref
from pathlib import Path
from typing import Any

import torch
from benchmark import Native, pack

_INSTALLED = False
_NATIVE: Native | None = None
_STORAGE: dict[int, dict[str, Any]] = {}
_LAYERS: list[dict[str, Any]] = []
_RAW_REFS: list[weakref.ReferenceType[torch.Tensor]] = []
_SCRATCH: dict[tuple[int, int], torch.Tensor] = {}
_CALLS: dict[tuple[int, ...], int] = {}
_LIBRARY_SHA = '5fe6c21821b8f683a4db18a80523231e2dff8c99d7ef15fd4f196fade48c17da'
_PINNED = {
    'modelopt': 'edc285a5df75b61dc935c5caeb8b69a1fd1bc487d846cc4bc65376985dd4bb21',
    'marlin': 'f00460310b033bcb84644677e16af83773464f026c0dad15bcfe25bd28b39745',
    'ops': '5ca3b8955ff2dca446813057e6678065a44ec283d73db4a9d8fd24328cb4ed07',
}


def native() -> Native:
    """Load the exact standard-math component library in a device-zero worker."""
    global _NATIVE
    if _NATIVE is None:
        path = Path(os.environ['MXWAVE_NVFP4_LIBRARY'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != _LIBRARY_SHA:
            raise RuntimeError('NVFP4 library differs from qualified component')
        _NATIVE = Native(path)
    return _NATIVE


def replace_references(method: Any, old: torch.Tensor, replacement: torch.Tensor) -> int:
    """Replace scale owners in the bounded installed MoE quant-descriptor graph."""
    queue = [method.moe_quant_config, method.moe_kernel]
    seen: set[int] = set()
    count = 0
    while queue:
        value = queue.pop()
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        if len(seen) > 512:
            raise RuntimeError('Unexpected MoE descriptor graph')
        if isinstance(value, (torch.Tensor, str, bytes, int, float, bool, type)):
            continue
        if isinstance(value, dict):
            queue.extend(value.values())
        elif isinstance(value, (tuple, list)):
            queue.extend(value)
        elif type(value).__module__.startswith('vllm.model_executor.layers.fused_moe'):
            for key, child in vars(value).items():
                if key == 'scale' and isinstance(child, torch.Tensor):
                    if child.data_ptr() == old.data_ptr():
                        setattr(value, key, replacement)
                        count += 1
                elif not isinstance(child, torch.Tensor):
                    queue.append(child)
    return count


def prepare_layer(method: Any, layer: torch.nn.Module) -> None:
    """Pack after stock conversion, independently verify bytes, then release dense scales."""
    if not method.use_a16 or type(method.moe_kernel.fused_experts).__name__ != 'MarlinExperts':
        raise ValueError('Require original W4A16 NVFP4 Marlin experts')
    mode = os.environ['MXWAVE_NVFP4_MODE']
    label = str(getattr(layer, 'prefix', getattr(method.moe, 'layer_name', 'unknown')))
    entry: dict[str, Any] = {'layer': label, 'mode': mode, 'scales': []}
    for name, parts in [('w13_weight_scale', 2), ('w2_weight_scale', 1)]:
        raw = getattr(layer, name)
        expected = (288, 256, 2048) if parts == 2 else (288, 64, 4096)
        if (tuple(raw.shape) != expected or raw.dtype != torch.float8_e4m3fn
                or not raw.is_contiguous() or raw.device.index != 0):
            raise ValueError('Outside qualified TP2 NVFP4 scale geometry/device')
        record = {'name': name, 'shape': list(raw.shape), 'raw_bytes': raw.numel(),
                  'source_sha256': hashlib.sha256(
                      raw.detach().view(torch.uint8).cpu().numpy()).hexdigest()}
        if mode == 'packed':
            packed = pack(raw.detach(), parts, native(), affine=True)
            # Dtype is retained for the original production tuning wrapper. The
            # 24-byte object is an opaque descriptor, never passed to stock GEMM.
            replacement = torch.nn.Parameter(
                packed['descriptor'].view(torch.float8_e4m3fn), requires_grad=False)
            refs = replace_references(method, raw, replacement)
            if refs < 1:
                raise RuntimeError('No MoE quant descriptor referenced source scales')
            setattr(layer, name, replacement)
            _RAW_REFS.append(weakref.ref(raw))
            _STORAGE[id(replacement)] = {'shape': expected, 'packed': packed}
            record.update(retained_bytes=packed['retained_bytes'], gpu_bytes_exact=True,
                          decoder_graph_replays=packed['decoder_graph_replays'],
                          quant_descriptor_references=refs, parts=packed['records'])
        else:
            record.update(retained_bytes=raw.numel(), gpu_bytes_exact=None)
        entry['scales'].append(record)
    _LAYERS.append(entry)
    print(f'MxWave NVFP4 scales prepared: {label} ({mode}); layers={len(_LAYERS)}', flush=True)


def _launch(
    x: torch.Tensor, weight: torch.Tensor, descriptor: torch.Tensor, global_scale: torch.Tensor,
    output: torch.Tensor, temporary: torch.Tensor, locks: torch.Tensor,
    sorted_ids: torch.Tensor, expert_ids: torch.Tensor, padded: torch.Tensor,
    routing_weights: torch.Tensor, m: int, n: int, k: int, experts: int, topk: int,
    block: int, multiply: bool, tile_k: int, tile_n: int, blocks_per_sm: int,
) -> None:
    """Use the validated native ABI with dynamic inputs and caller-owned stream buffers."""
    tensors = (x, weight, descriptor, global_scale, output, temporary, locks,
               sorted_ids, expert_ids, padded, routing_weights)
    native().launch([*(t.data_ptr() for t in tensors), m, n, k, experts, topk, block,
                     int(multiply), tile_k, tile_n, blocks_per_sm])


_MUTATIONS = ('output', 'temporary', 'locks')
_LIBRARY = torch.library.Library('mxwave_nvfp4', 'FRAGMENT')
_LIBRARY.define('gemm' + torch.library.infer_schema(_launch, mutates_args=_MUTATIONS))
_LIBRARY.impl('gemm', _launch, 'CUDA')


@torch.library.register_fake('mxwave_nvfp4::gemm')
def _fake(*args: Any, **kwargs: Any) -> None:
    return None


@functools.lru_cache(maxsize=2048)
def plan(m: int, n: int, k: int, block: int, topk: int, experts: int) -> tuple[int, ...]:
    """Select the same untuned tile configuration as installed Marlin."""
    return tuple(native().plan(m, n, k, block, topk, experts))


def scratch(x: torch.Tensor) -> torch.Tensor:
    """Retain bounded FP32 reduction scratch separately for each CUDA stream."""
    key = (x.device.index, torch.cuda.current_stream(x.device).cuda_stream)
    if key not in _SCRATCH:
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        _SCRATCH[key] = torch.empty(sms * 4 * 64 * 256 * 2,
                                    dtype=torch.float32, device=x.device)
    return _SCRATCH[key]


def report() -> dict[str, Any]:
    """Return coverage, source bindings and actual allocation without retaining raw scales."""
    return {'mode': os.environ['MXWAVE_NVFP4_MODE'], 'layers': _LAYERS,
            'layers_seen': len(_LAYERS), 'source_bytes': sum(
                s['raw_bytes'] for e in _LAYERS for s in e['scales']),
            'retained_bytes': sum(s['retained_bytes'] for e in _LAYERS for s in e['scales']),
            'raw_scale_owners_alive': sum(ref() is not None for ref in _RAW_REFS),
            'scratch_bytes': sum(t.numel() * t.element_size() for t in _SCRATCH.values()),
            'calls': [{'geometry': list(k), 'host_calls': v} for k, v in _CALLS.items()],
            'cuda_allocated_bytes': torch.cuda.memory_allocated(),
            'cuda_reserved_bytes': torch.cuda.memory_reserved(),
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated()}


def install() -> None:
    """Install explicitly in disposable engines; retain original production tuning hooks."""
    global _INSTALLED
    if _INSTALLED:
        return
    mode = os.environ.get('MXWAVE_NVFP4_MODE', 'stock')
    if mode not in ('stock', 'packed'):
        raise ValueError('MXWAVE_NVFP4_MODE must be stock or packed')
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.experts import marlin_moe
    from vllm.model_executor.layers.quantization import modelopt

    for key, module in [('ops', ops), ('marlin', marlin_moe), ('modelopt', modelopt)]:
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != _PINNED[key]:
            raise RuntimeError('Installed NVFP4 source changed: ' + key)
    original_prepare = modelopt.ModelOptNvFp4FusedMoE.process_weights_after_loading

    @functools.wraps(original_prepare)
    def prepare(method: Any, layer: torch.nn.Module) -> None:
        original_prepare(method, layer)
        prepare_layer(method, layer)

    modelopt.ModelOptNvFp4FusedMoE.process_weights_after_loading = prepare
    if mode == 'packed':
        original = ops.moe_wna16_marlin_gemm
        signature = inspect.signature(original)

        def dispatch(*args: Any, **kwargs: Any) -> torch.Tensor:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            a = bound.arguments
            scales = a['b_scales']
            meta = _STORAGE.get(id(scales))
            if meta is None:
                raise RuntimeError('Unknown compressed scale owner')
            x, output, weight, gs = (a[k] for k in ('input', 'output', 'b_qweight', 'global_scale'))
            absent = ('b_bias', 'a_scales', 'b_qzeros', 'g_idx', 'perm')
            if (any(a[k] is not None and a[k].numel() for k in absent)
                    or a['use_atomic_add'] or not a['use_fp32_reduce']
                    or not a['is_k_full'] or a['is_zp_float'] or gs is None
                    or x.dtype != torch.bfloat16 or x.device.index != 0):
                raise ValueError('NVFP4 GEMM outside qualified arithmetic contract')
            m, n, k, block, topk = (int(a[key]) for key in
                                  ('size_m', 'size_n', 'size_k', 'moe_block_size', 'top_k'))
            e, groups, cols = meta['shape']
            if (n != cols or k != groups * 16 or tuple(weight.shape) != (e, k // 16, n * 2)
                    or a['b_q_type'].id != __import__('vllm.scalar_type', fromlist=[
                        'scalar_types']).scalar_types.float4_e2m1f.id):
                raise ValueError('Compressed scale descriptor does not match weight geometry')
            selected = plan(m, n, k, block, topk, e)
            if a.get('thread_k', -1) > 0 or a.get('thread_n', -1) > 0:
                selected = tuple(int(a[key]) for key in ('thread_k', 'thread_n', 'blocks_per_sm'))
            if output is None:
                output = torch.empty((m * topk, n), dtype=x.dtype, device=x.device)
            geometry = (m, n, k, block, topk, *selected)
            _CALLS[geometry] = _CALLS.get(geometry, 0) + 1
            torch.ops.mxwave_nvfp4.gemm.default(
                x, weight, scales, gs, output, scratch(x), a['workspace'],
                a['sorted_token_ids'], a['expert_ids'], a['num_tokens_past_padded'],
                a['topk_weights'], m, n, k, e, topk, block, a['mul_topk_weights'], *selected)
            return output

        # Apply the existing production policy to the new consumer. This API
        # chooses tile sizes; it is not copied or altered. JIT entries require
        # separate compressed-source support and are rejected before loading.
        import glm_marlin_tune
        table = glm_marlin_tune.S['table']
        if table is None or any('jit' in entry for entries in table.moe.values()
                                for entry in entries.values()):
            raise ValueError('Require the pinned non-JIT production MoE tuning table')
        tuned = glm_marlin_tune.make_moe_gemm_wrapper(dispatch)

        @functools.wraps(original)
        def select(*args: Any, **kwargs: Any) -> torch.Tensor:
            scales = args[4] if len(args) > 4 else kwargs['b_scales']
            return (tuned if id(scales) in _STORAGE else original)(*args, **kwargs)

        ops.moe_wna16_marlin_gemm = select
    _INSTALLED = True
