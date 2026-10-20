"""Isolated real-weight NVFP4 component: exact bytes, stock parity and paired timing."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import statistics
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from codec import decode, encode
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    _nvfp4_compute_scale_factor,
    nvfp4_marlin_process_global_scale,
    nvfp4_marlin_process_scales,
)
from vllm.scalar_type import scalar_types


def save(path: Path, value: Any) -> None:
    """Write progress atomically, keeping partial evidence after a failed gate."""
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha(raw: bytes) -> str:
    """Bind exact data bytes."""
    return hashlib.sha256(raw).hexdigest()


def equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Compare raw bytes including signed zero."""
    return bool(torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                            b.contiguous().reshape(-1).view(torch.uint8)))


def telemetry() -> str:
    """Retain available clock, thermal and power readings without changing clocks."""
    return subprocess.check_output([
        'nvidia-smi', '--query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem',
        '--format=csv,noheader'], text=True).strip()


def difference(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    """Retain numerical and bitwise failure detail without weakening the gate."""
    changed = a.view(torch.int16) != b.view(torch.int16)
    return {'changed_bits': int(changed.sum()), 'values': a.numel(),
            'changed_numeric': int((a != b).sum()),
            'max_abs': float((a.float() - b.float()).abs().max()),
            'examples': list(zip(a[changed][:8].float().tolist(),
                                 b[changed][:8].float().tolist(), strict=True))}


class Native:
    """Narrow device-zero component interface; all buffers live through graph replay."""

    def __init__(self, path: Path) -> None:
        self.lib = ctypes.CDLL(str(path))
        self.lib.nv_last_error.restype = ctypes.c_char_p
        self.lib.nv_plan.argtypes = [ctypes.c_int] * 6 + [ctypes.POINTER(ctypes.c_int)]
        self.lib.nv_run.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int] * 10 + [ctypes.c_size_t]
        self.lib.nv_decode.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 2 + [ctypes.c_size_t]

    def plan(self, m: int, n: int, k: int, block: int, topk: int, e: int) -> list[int]:
        """Use the same tile plan in raw, packed and installed stock calls."""
        plan = (ctypes.c_int * 3)()
        if self.lib.nv_plan(m, n, k, block, topk, e, plan):
            raise RuntimeError('No supported NVFP4 component plan')
        return list(plan)

    def launch(self, args: list[Any]) -> None:
        """Launch on the active CUDA stream."""
        if self.lib.nv_run(*args, torch.cuda.current_stream().cuda_stream):
            raise RuntimeError(self.lib.nv_last_error().decode())


def pack(scales: torch.Tensor, parts: int, native: Native, *, affine: bool = False
         ) -> dict[str, Any]:
    """Pack each original expert/projection separately and verify GPU reconstruction."""
    if (scales.ndim != 3 or scales.dtype != torch.float8_e4m3fn
            or scales.device.type != 'cuda' or not scales.is_contiguous()
            or parts not in (1, 2)):
        raise ValueError('Expected contiguous CUDA post-Marlin FP8 storage and one/two parts')
    e, groups, n = scales.shape
    if e < 1 or groups < 1 or n < 64 or n & (n - 1) or n % (parts * 32):
        raise ValueError('Component requires power-of-two columns and complete high-plane words')
    raw = scales.view(torch.uint8).cpu().numpy()
    retained, pointers, records = [], [], []
    for expert in range(e):
        for part in range(parts):
            source = raw[expert, :, part * (n // parts):(part + 1) * (n // parts)]
            encoded = encode(source, affine=affine)
            assert np.array_equal(decode(encoded), source.reshape(-1))
            arrays = (encoded.low, encoded.high, encoded.palette, encoded.raw)
            buffers = [torch.from_numpy(a.copy()).cuda() if a.size else None for a in arrays]
            pointers.append([b.data_ptr() if b is not None else 0 for b in buffers])
            retained.extend(b for b in buffers if b is not None)
            records.append({'expert_slot': expert, 'part': part, 'source_bytes': source.size,
                            'source_sha256': sha(source.tobytes()), 'encoded_bytes': encoded.nbytes,
                            'raw_fallback': bool(encoded.raw.size),
                            'distinct': int(np.unique(source).size),
                            'minimum_byte': int(source.min()), 'maximum_byte': int(source.max()),
                            'affine': affine})
    table = torch.tensor(pointers, dtype=torch.int64, device='cuda')
    row_shift = int(math.log2(n // 16))
    part_shift = int(math.log2(n // parts // 16))
    assert 1 << row_shift == n // 16 and 1 << part_shift == n // parts // 16
    descriptor = torch.frombuffer(bytearray(struct.pack(
        '<Qiiii', table.data_ptr(), row_shift, part_shift, parts, 0)), dtype=torch.uint8).cuda()
    retained.extend([table, descriptor])
    output = torch.empty_like(scales).view(torch.uint8)
    assert native.lib.nv_decode(descriptor.data_ptr(), output.data_ptr(), groups * n // 16,
                                scales.numel() // 16, torch.cuda.current_stream().cuda_stream) == 0
    torch.cuda.synchronize()
    assert equal(output, scales.view(torch.uint8)), 'GPU scale reconstruction mismatch'
    # Decoder graph/side-stream coverage, independent of the GEMM equality checks below.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        assert native.lib.nv_decode(descriptor.data_ptr(), output.data_ptr(), groups * n // 16,
                                    scales.numel() // 16, stream.cuda_stream) == 0
    for _ in range(8):
        graph.replay()
    torch.cuda.synchronize()
    assert equal(output, scales.view(torch.uint8))
    return {'descriptor': descriptor, 'buffers': retained, 'records': records,
            'source_bytes': scales.numel(),
            'retained_bytes': sum(t.numel() * t.element_size() for t in retained),
            'source_sha256': sha(raw.tobytes()), 'gpu_bytes_exact': True,
            'decoder_graph_replays': 8}


def load(model: Path, rank: int, experts: list[int], layer: int) -> tuple[list[Any], list[Any]]:
    """Read active source slices and use installed repacking/global-scale preparation."""
    config_raw = (model / 'config.json').read_bytes()
    assert sha(config_raw) == 'f14dc13ce3bef88a5539c9e61b3e4f3dbef6958ebc3acda4b7f05f418642c3b0'
    config = json.loads(config_raw)
    index_raw = (model / 'model.safetensors.index.json').read_bytes()
    assert sha(index_raw) == '8ac9d03981d2029223a66bdbda51fa224cb36de8f522d8c695f0724e4f34697e'
    index = json.loads(index_raw)['weight_map']
    parent = f'model.language_model.layers.{layer}.mlp.experts'
    assert config['quantization_config']['quantized_layers'][parent]['quant_algo'] == 'W4A16_NVFP4'
    headers, records = {}, []

    def read(expert: int, proj: str, role: str) -> torch.Tensor:
        key = f'{parent}.{expert}.{proj}.{role}'
        path = model / index[key]
        with path.open('rb') as file:
            if path.name not in headers:
                length = struct.unpack('<Q', file.read(8))[0]
                headers[path.name] = (length, json.loads(file.read(length)))
            length, header = headers[path.name]
            item = header[key]
            lo, hi = item['data_offsets']
            file.seek(8 + length + lo)
            raw = file.read(hi - lo)
        assert len(raw) == hi - lo
        records.append({'name': key, 'file': path.name, 'sha256': sha(raw),
                        'bytes': len(raw), 'dtype': item['dtype'], 'shape': item['shape']})
        dtype = {'U8': torch.uint8, 'F8_E4M3': torch.float8_e4m3fn, 'F32': torch.float32}[item['dtype']]
        value = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(item['shape'])
        if role == 'weight_scale_2':
            return value
        if proj == 'down_proj':
            divisor = 2 if role == 'weight' else 16
            return value[:, rank * 1024 // divisor:(rank + 1) * 1024 // divisor].contiguous()
        return value[rank * 1024:(rank + 1) * 1024].contiguous()

    families = []
    for family in ('w13', 'w2'):
        n, k = (2048, 4096) if family == 'w13' else (4096, 1024)
        projections = ('gate_proj', 'up_proj') if family == 'w13' else ('down_proj',)
        raw_scales, global_scales = [], []
        weights = torch.empty((len(experts), k // 16, n * 2), dtype=torch.int32, device='cuda')
        for slot, expert in enumerate(experts):
            raw_weight = torch.cat([read(expert, p, 'weight') for p in projections], dim=0).cuda()
            perm = torch.empty(0, dtype=torch.int32, device='cuda')
            weights[slot].copy_(ops.gptq_marlin_repack(
                raw_weight.view(torch.int32).T.contiguous(), perm, k, n, 4, False))
            raw_scales.append(torch.cat([read(expert, p, 'weight_scale').view(torch.uint8)
                                         for p in projections], dim=0))
            gs = [read(expert, p, 'weight_scale_2') for p in projections]
            assert all(equal(gs[0], g) for g in gs)
            global_scales.append(gs[0].reshape(()))
        # The installed factor depends only on the maximum positive scale. Its
        # full-tensor boolean indexing allocates a large nonzero-index temporary.
        # Positive finite E4M3 bytes are monotonic: reduce on CPU, then invoke the
        # installed factor oracle on the exact maximum. Stage one expert at a time.
        maximum_byte = max(int(s.max()) for s in raw_scales)
        assert maximum_byte < 127, 'Expected finite nonnegative source E4M3 scales'
        maximum = torch.tensor([maximum_byte], dtype=torch.uint8, device='cuda')
        factor = _nvfp4_compute_scale_factor(
            maximum.view(torch.float8_e4m3fn).to(torch.bfloat16), torch.bfloat16)
        processed = torch.empty((len(experts), k // 16, n), dtype=torch.float8_e4m3fn, device='cuda')
        for slot in range(len(experts)):
            source = raw_scales[slot].cuda().view(torch.float8_e4m3fn).to(torch.bfloat16)
            permuted = marlin_permute_scales(source.T, k, n, 16)
            value, _ = nvfp4_marlin_process_scales(permuted, factor, torch.bfloat16)
            processed[slot].copy_(value)
        gs = nvfp4_marlin_process_global_scale(torch.stack(global_scales).cuda(), torch.bfloat16)
        gs = (gs / factor).contiguous()
        families.append((weights, processed, gs, factor))
    return families, records


def capture(fn: Any) -> torch.cuda.CUDAGraph:
    """Capture a warmed component on a private stream."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def measure(graphs: dict[str, Any], trials: int = 15) -> dict[str, Any]:
    """Interleave both arms with two separately captured raw repeat controls."""
    samples: dict[str, list[float]] = {name: [] for name in graphs}
    thermal = []
    orders = [('raw', 'candidate', 'repeat'), ('repeat', 'candidate', 'raw')]
    if 'installed' in graphs:
        orders = [('installed', 'raw', 'candidate', 'repeat'),
                  ('repeat', 'candidate', 'raw', 'installed')]
    for trial in range(trials):
        thermal.append({'trial': trial, 'before': telemetry(), 'arms': []})
        for name in orders[trial % 2]:
            graph = graphs[name]
            arm_telemetry = {'arm': name, 'before': telemetry()}
            for _ in range(20):
                graph.replay()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(60):
                graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / 60)
            arm_telemetry['after'] = telemetry()
            thermal[-1]['arms'].append(arm_telemetry)
        thermal[-1]['after'] = telemetry()
    medians = {name: statistics.median(values) for name, values in samples.items()}
    control_delta = abs(medians['repeat'] / medians['raw'] - 1)
    return {'samples_us': samples, 'median_us': medians, 'telemetry': thermal,
            'control_relative_difference': control_delta, 'control_stable_2pct': control_delta <= .02,
            'candidate_time_ratio': medians['candidate'] / medians['raw'],
            'timed_replays_per_arm': trials * 60}


def case(m: int, fixture: Any, packed: list[Any], libraries: dict[str, Native],
         directory: Path, tag: str, *, trials: int = 15,
         include_installed: bool = False, shared_buffers: bool = False) -> dict[str, Any]:
    """Compare each GEMM and the complete routed layer with stock and raw controls."""
    w13, s13, g13, _ = fixture[0]
    w2, s2, g2, _ = fixture[1]
    e, topk, block = w13.shape[0], 8, 8
    torch.manual_seed(1700 + m)
    x = torch.randn(m, 4096, dtype=torch.bfloat16, device='cuda') * .125
    ids = torch.randn(m, e, device='cuda').topk(topk, dim=-1).indices.to(torch.int32)
    weights = torch.softmax(torch.randn(m, topk, device='cuda'), dim=-1)
    capacity = min(m * topk + e * (block - 1), m * topk * block)
    route = [torch.empty(capacity, dtype=torch.int32, device='cuda'),
             torch.empty((capacity + block - 1) // block, dtype=torch.int32, device='cuda'),
             torch.empty(1, dtype=torch.int32, device='cuda')]
    ops.moe_align_block_size(ids, e, block, *route, None)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    plans = [libraries['raw'].plan(m, 2048, 4096, block, topk, e),
             libraries['raw'].plan(m * topk, 4096, 1024, block, 1, e)]
    arms = {}
    for name in ('raw', 'candidate', 'repeat'):
        native = libraries['candidate' if name == 'candidate' else 'raw']
        up = torch.empty(m * topk, 2048, dtype=torch.bfloat16, device='cuda')
        act = torch.empty(m * topk, 1024, dtype=torch.bfloat16, device='cuda')
        down = torch.empty(m * topk, 4096, dtype=torch.bfloat16, device='cuda')
        result = torch.empty_like(x)
        if shared_buffers and name != 'raw':
            up, act, down, result = (arms['raw'][k] for k in ('up', 'act', 'down', 'result'))
        owned, launches = [], []
        for i, (a, w, s, g, y, mm, nn, kk, tk) in enumerate([
            (x, w13, s13, g13, up, m, 2048, 4096, topk),
            (act, w2, s2, g2, down, m * topk, 4096, 1024, 1)]):
            tmp = torch.empty(min(nn * capacity, sms * 4 * block * 256) * 2,
                              dtype=torch.float32, device='cuda')
            locks = torch.zeros(sms * 4, dtype=torch.int32, device='cuda')
            if shared_buffers and name != 'raw':
                tmp, locks = arms['raw']['owned'][2 * i:2 * i + 2]
            owned.extend([tmp, locks])
            ss = packed[i]['descriptor'] if name == 'candidate' else s
            tensors = [a, w, ss, g, y, tmp, locks, *route, weights]
            launches.append([t.data_ptr() for t in tensors] +
                            [mm, nn, kk, e, tk, block, i, *plans[i]])

        def fn(native: Native = native, launches: Any = launches, up: Any = up,
               act: Any = act, down: Any = down, result: Any = result) -> None:
            native.launch(launches[0])
            torch.ops._C.silu_and_mul(act, up)
            native.launch(launches[1])
            torch.sum(down.view(m, topk, 4096), dim=1, out=result)

        fn()
        snapshots = {'up': up.clone(), 'down': down.clone(), 'result': result.clone()}
        arms[name] = {'fn': fn, 'up': up, 'down': down, 'result': result, 'owned': owned,
                      'launches': launches, 'native': native, 'act': act,
                      'snapshots': snapshots}
    torch.cuda.synchronize()
    stock_lock = torch.zeros(sms * 4, dtype=torch.int32, device='cuda')
    common = {'b_bias': None, 'a_scales': None, 'b_qzeros': None, 'g_idx': None, 'perm': None,
              'workspace': stock_lock, 'sorted_token_ids': route[0], 'expert_ids': route[1],
              'num_tokens_past_padded': route[2], 'topk_weights': weights, 'moe_block_size': block,
              'b_q_type': scalar_types.float4_e2m1f, 'is_k_full': True, 'use_atomic_add': False,
              'use_fp32_reduce': True, 'is_zp_float': False}
    ref_up = ops.moe_wna16_marlin_gemm(input=x, output=None, b_qweight=w13, b_scales=s13,
        global_scale=g13, top_k=topk, mul_topk_weights=False, size_m=m, size_n=2048, size_k=4096,
        thread_k=plans[0][0], thread_n=plans[0][1], blocks_per_sm=plans[0][2], **common)
    ref_act = torch.empty_like(arms['raw']['down'][:, :1024]).contiguous()
    torch.ops._C.silu_and_mul(ref_act, ref_up)
    ref_down = ops.moe_wna16_marlin_gemm(input=ref_act, output=None, b_qweight=w2, b_scales=s2,
        global_scale=g2, top_k=1, mul_topk_weights=True, size_m=m * topk, size_n=4096, size_k=1024,
        thread_k=plans[1][0], thread_n=plans[1][1], blocks_per_sm=plans[1][2], **common)
    reference = {'up': ref_up, 'down': ref_down, 'result': ref_down.view(m, topk, 4096).sum(1)}
    checks = {name: {key: equal(arm['snapshots'][key], value) for key, value in reference.items()}
              for name, arm in arms.items()}
    result = {'rows': m, 'experts': e, 'topk': topk, 'block': block, 'plans': plans,
              'exact_vs_installed': checks, 'finite': bool(torch.isfinite(reference['result']).all()),
              'shared_output_and_scratch_buffers': shared_buffers,
              'differences': {name: {key: difference(arm['snapshots'][key], value)
                                     for key, value in reference.items()}
                              for name, arm in arms.items()},
              'candidate_vs_raw': {key: equal(arms['candidate']['snapshots'][key],
                                             arms['raw']['snapshots'][key])
                                   for key in reference},
              'reference_sha256': {key: sha(value.contiguous().view(torch.uint8).cpu().numpy()
                                            .tobytes()) for key, value in reference.items()},
              'output_sha256': {name: {key: sha(arm['snapshots'][key].contiguous().view(torch.uint8)
                                                   .cpu().numpy().tobytes())
                                        for key in reference} for name, arm in arms.items()}}
    save(directory / (tag + '-correctness.json'), result)
    assert result['finite'] and all(all(v.values()) for v in checks.values()), result
    graphs = {name: capture(arm['fn']) for name, arm in arms.items()}
    if include_installed:
        stock_up, stock_act, stock_down = (torch.empty_like(ref_up), torch.empty_like(ref_act),
                                          torch.empty_like(ref_down))
        stock_result = torch.empty_like(x)
        if shared_buffers:
            stock_up, stock_act, stock_down, stock_result = (
                arms['raw'][k] for k in ('up', 'act', 'down', 'result'))

        def installed() -> None:
            ops.moe_wna16_marlin_gemm(input=x, output=stock_up, b_qweight=w13, b_scales=s13,
                global_scale=g13, top_k=topk, mul_topk_weights=False,
                size_m=m, size_n=2048, size_k=4096,
                thread_k=plans[0][0], thread_n=plans[0][1], blocks_per_sm=plans[0][2], **common)
            torch.ops._C.silu_and_mul(stock_act, stock_up)
            ops.moe_wna16_marlin_gemm(input=stock_act, output=stock_down, b_qweight=w2,
                b_scales=s2, global_scale=g2, top_k=1, mul_topk_weights=True,
                size_m=m * topk, size_n=4096, size_k=1024,
                thread_k=plans[1][0], thread_n=plans[1][1], blocks_per_sm=plans[1][2], **common)
            torch.sum(stock_down.view(m, topk, 4096), dim=1, out=stock_result)

        graphs['installed'] = capture(installed)
        graphs['installed'].replay()
        torch.cuda.synchronize()
        assert equal(stock_result, reference['result'])
    for _ in range(20):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    for arm in arms.values():
        arm['fn']()
        torch.cuda.synchronize()
        assert all(equal(arm[key], reference[key]) for key in reference)
    result['whole_layer'] = measure(graphs, trials)
    if include_installed:
        med = result['whole_layer']['median_us']
        result['whole_layer']['candidate_over_installed'] = med['candidate'] / med['installed']
        result['whole_layer']['raw_over_installed'] = med['raw'] / med['installed']
        assert equal(stock_result, reference['result'])
    result['gemms'] = []
    for i in range(2):
        individual = {name: capture(lambda a=arm, i=i: a['native'].launch(a['launches'][i]))
                      for name, arm in arms.items()}
        result['gemms'].append(measure(individual, trials))
    if m == 8:
        trace = directory / (tag + '-trace.json')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            for name in ('raw', 'candidate'):
                with torch.profiler.record_function('nvfp4-' + name):
                    for _ in range(4):
                        graphs[name].replay()
                    torch.cuda.synchronize()
        prof.export_chrome_trace(str(trace))
        result['trace'] = {'file': trace.name, 'sha256': sha(trace.read_bytes())}
    torch.cuda.synchronize()
    for arm in arms.values():
        arm['fn']()
        torch.cuda.synchronize()
        assert all(equal(arm[key], reference[key]) for key in reference)
    result['graph_outputs_exact_after_timing'] = True
    return result


def main() -> None:
    """Execute the finite rank/row matrix; leave the live model untouched."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-library', default='libnv_direct5.so')
    parser.add_argument('--affine', action='store_true')
    parser.add_argument('--experts', type=int, choices=(16, 288), default=16)
    parser.add_argument('--ranks', type=int, nargs='+', choices=(0, 1), default=(0, 1))
    parser.add_argument('--rows', type=int, nargs='+', choices=(1, 8, 16, 64, 256),
                        default=(1, 8, 16, 64, 256))
    parser.add_argument('--trials', type=int, default=15)
    parser.add_argument('--include-installed', action='store_true')
    parser.add_argument('--shared-buffers', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    report: dict[str, Any] = {'status': 'running', 'started': time.time(), 'cases': [], 'fixtures': [],
        'scope': (f'{args.experts} real experts from layer23, both TP2 partitions; '
                  'synthetic BF16 activations/routing; fixed eight-row routing blocks'),
        'serving_modified': False, 'full_model_qualified': False}
    path = args.output / 'report.json'
    try:
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.03 if args.experts == 288 else .02)
        libraries = {'raw': Native(args.native / 'libnv_raw.so'),
                     'candidate': Native(args.native / args.candidate_library)}
        report['synthetic_scale_checks'] = []
        for parts in (1, 2):
            for fallback in (False, True):
                codes = np.arange(2 * 8 * 128).reshape(2, 8, 128)
                raw = ((codes % 32) + 200 if args.affine else (codes % 32) * 7).astype(np.uint8)
                if fallback:
                    raw[1] = (codes[1] % 256).astype(np.uint8)
                tensor = torch.from_numpy(raw).cuda().view(torch.float8_e4m3fn)
                check = pack(tensor, parts, libraries['candidate'], affine=args.affine)
                report['synthetic_scale_checks'].append(
                    {'parts': parts, 'mixed_raw_fallback': fallback, 'bytes_exact': True,
                     'graph_replays': check['decoder_graph_replays']})
                del check, tensor
        experts = list(range(288)) if args.experts == 288 else [round(i * 287 / 15) for i in range(16)]
        for rank in args.ranks:
            fixture, sources = load(args.model, rank, experts, 23)
            packed = [pack(fixture[0][1], 2, libraries['candidate'], affine=args.affine),
                      pack(fixture[1][1], 1, libraries['candidate'], affine=args.affine)]
            report['fixtures'].append({'rank': rank, 'experts': experts, 'sources': sources,
                'scales': [{k: v for k, v in p.items() if k not in ('buffers', 'descriptor')}
                           for p in packed], 'scale_factors': [f[3] for f in fixture]})
            save(path, report)
            for m in args.rows:
                result = case(m, fixture, packed, libraries, args.output, f'rank{rank}-m{m}',
                              trials=args.trials, include_installed=args.include_installed,
                              shared_buffers=args.shared_buffers)
                report['cases'].append({'rank': rank, **result})
                save(path, report)
                print(json.dumps({'rank': rank, 'rows': m, 'exact': True,
                                  'timing': result['whole_layer']['median_us']}), flush=True)
            del packed, fixture
            torch.cuda.empty_cache()
        report.update(status='complete', finished=time.time(),
                      maximum_allocated_bytes=torch.cuda.max_memory_allocated(),
                      maximum_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        report.update(status='failed', error=repr(error), finished=time.time())
        raise
    finally:
        save(path, report)


if __name__ == '__main__':
    main()
