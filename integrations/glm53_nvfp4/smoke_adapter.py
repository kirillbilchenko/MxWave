"""Validate the serving hook against installed tuned MoE across production row counts."""
from __future__ import annotations

import argparse
import gc
import json
import time
from functools import partial
from pathlib import Path
from typing import Any

import nv_adapter
import torch
from benchmark import capture, equal, load, pack, save


def main() -> None:
    """Run real-weight layer parity in eager and captured execution, without timing claims."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(.07)
    nv_adapter.install()
    from vllm.model_executor.layers.fused_moe.experts import marlin_moe
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.scalar_type import scalar_types

    # Isolate the scale consumer from stock routing-order repeat noise. Retain
    # one actual installed alignment for both arms; full serving keeps stock
    # alignment untouched and receives a separate full-model repeat gate.
    original_align = marlin_moe.moe_align_block_size
    route_cache: list[Any] = []

    def fixed_align(*a: Any, **kw: Any) -> Any:
        if not route_cache:
            route_cache.append(original_align(*a, **kw))
        return route_cache[0]

    marlin_moe.moe_align_block_size = fixed_align
    args.output.mkdir(exist_ok=True)
    state = {'status': 'running', 'started': time.time(), 'cases': [], 'fixture_sources': [],
             'routing': 'one installed alignment retained and shared between arms'}
    path = args.output / 'smoke.json'
    save(path, state)
    for rank in (0, 1):
        families, sources = load(args.model, rank, list(range(288)), 23)
        state['fixture_sources'].append({'rank': rank, 'sources': sources})
        packed = [pack(row[1], parts, nv_adapter.native(), affine=True)
                  for row, parts in zip(families, (2, 1), strict=True)]
        desc = [p['descriptor'].view(torch.float8_e4m3fn) for p in packed]
        for row, p, d in zip(families, packed, desc, strict=True):
            nv_adapter._STORAGE[id(d)] = {'shape': tuple(row[1].shape), 'packed': p}
        w1, s1, g1, _ = families[0]
        w2, s2, g2, _ = families[1]
        for rows in (1, 4, 8, 12, 16, 32, 256, 1024, 6919):
            route_cache.clear()
            torch.manual_seed(974 + rows)
            x = torch.randn(rows, 4096, dtype=torch.bfloat16, device='cuda') * .125
            ids = torch.randn(rows, 288, device='cuda').topk(8, dim=-1).indices.to(torch.int32)
            weights = torch.softmax(torch.randn(rows, 8, device='cuda'), dim=-1)

            run = partial(fused_marlin_moe, hidden_states=x, w1=w1, w2=w2,
                          bias1=None, bias2=None, topk_weights=weights, topk_ids=ids,
                          quant_type_id=scalar_types.float4_e2m1f.id,
                          global_scale1=g1, global_scale2=g2)
            ref = run(w1_scale=s1, w2_scale=s2).clone()
            got = run(w1_scale=desc[0], w2_scale=desc[1]).clone()
            torch.cuda.synchronize()
            case = {'rank': rank, 'rows': rows, 'eager_exact': equal(ref, got),
                    'shape': list(got.shape)}
            state['cases'].append(case)
            save(path, state)
            assert case['eager_exact'], case
            # Capture small decode/verification and 256-row paths; the large
            # prefill cases exercise eager allocation without redundant graphs.
            if rows <= 256:
                outputs = {}

                replay_fn = partial(run, w1_scale=desc[0], w2_scale=desc[1])

                def replay_body(target: dict[str, Any] = outputs, fn: Any = replay_fn) -> None:
                    target['value'] = fn()

                graph = capture(replay_body)
                for _ in range(16):
                    graph.replay()
                torch.cuda.synchronize()
                case['graph_exact'] = equal(ref, outputs['value'])
                case['graph_replays'] = 16
                save(path, state)
                assert case['graph_exact'], case
                del graph, outputs, replay_body, replay_fn
            print(json.dumps(case), flush=True)
            save(path, state)
            del x, ids, weights, ref, got, run
            gc.collect()
            torch.cuda.empty_cache()
        for d in desc:
            del nv_adapter._STORAGE[id(d)]
        del families, packed, desc, w1, w2, s1, s2, g1, g2
        gc.collect()
        torch.cuda.empty_cache()
    state.update(status='complete', finished=time.time(), adapter=nv_adapter.report())
    save(path, state)


if __name__ == '__main__':
    main()
