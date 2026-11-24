"""Disposable-engine controls for a reversible KDA weight-rounding diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from kda_mxfp8_roundtrip import numpy_reference, roundtrip
from kda_policy import load_policy
from warmed_model_worker import WarmedModelWorker


def digest(value: torch.Tensor) -> str:
    """Hash BF16 bytes without numerical conversion."""
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


class KdaRoundingWorker(WarmedModelWorker):
    """Keep native BF16 operations and only replace the selected weight values."""

    @torch.inference_mode()
    def mxwave_rounding_initialize(self) -> dict[str, Any]:
        """Back up exactly the 68 configuration-bound BF16 TP2 projections in RAM."""
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        if hasattr(self, "_rounding_originals"):
            raise RuntimeError("Preserve the single initialization")
        config = self.model_runner.vllm_config
        assert config.speculative_config is None
        assert not config.cache_config.enable_prefix_caching
        targets = load_policy(Path("/model"))
        modules = dict(self.model_runner.model.named_modules())
        originals, records, pointers = {}, [], set()
        available = next(
            int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
        backup_bytes = sum(t.rows * t.width * 2 for t in targets.values())
        assert available > backup_bytes + 3 * 1024**3, (available, backup_bytes)
        for name, target in targets.items():
            module = modules[name]
            value = module.weight
            assert type(module.quant_method) is UnquantizedLinearMethod
            assert value.dtype == torch.bfloat16 and tuple(value.shape) == (
                target.rows, target.width
            )
            assert value.is_contiguous() and value.data_ptr() not in pointers
            pointers.add(value.data_ptr())
            original = value.detach().cpu().clone()
            originals[name] = original
            records.append({
                "name": name, "kind": target.kind, "layer": target.layer,
                "shape": list(value.shape), "pointer": value.data_ptr(),
                "source_sha256": digest(original),
            })
        assert len(originals) == 68
        self._rounding_originals = originals
        self._rounding_records = records
        self._rounding_mode = "original"
        self._rounding_capture = None
        self._rounding_rank = get_tensor_model_parallel_rank()
        self._install_rounding_observer()
        return {
            "status": "complete", "rank": self._rounding_rank,
            "mode": self._rounding_mode, "backup_bytes": backup_bytes, "layers": records,
            "speculation": False, "native_mhc": self.mxwave_native_mhc_policy(),
            "scope": "Weight-only MXFP8 roundtrip with unchanged native BF16 GEMMs",
        }

    @torch.inference_mode()
    def mxwave_rounding_set(self, mode: str) -> dict[str, Any]:
        """Mutate weights in place between requests, retaining all captured addresses."""
        assert mode in ("original", "mxfp8_roundtrip")
        assert self._rounding_capture is None
        modules = dict(self.model_runner.model.named_modules())
        self._rounding_mode = "changing"
        records = []
        for source in self._rounding_records:
            name = source["name"]
            value = modules[name].weight
            original = self._rounding_originals[name]
            assert value.data_ptr() == source["pointer"] and value.shape == original.shape
            error2 = signal2 = maximum = 0.0
            changed = zeros = 0
            for first in range(0, value.shape[0], 512):
                source_chunk = original[first:first + 512].to(value.device)
                if mode == "original":
                    value[first:first + 512].copy_(source_chunk)
                    continue
                decoded, _, _ = roundtrip(source_chunk)
                replacement = decoded.bfloat16()
                assert torch.equal(replacement.float(), decoded)
                difference = decoded - source_chunk.float()
                error2 += float(difference.double().square().sum())
                signal2 += float(source_chunk.double().square().sum())
                maximum = max(maximum, float(difference.abs().max()))
                changed += int(torch.count_nonzero(difference))
                zeros += int(torch.count_nonzero((decoded == 0) & (source_chunk != 0)))
                value[first:first + 512].copy_(replacement)
            actual_sha = digest(value)
            entry = {
                **source, "result_sha256": actual_sha, "elements": value.numel(),
                "changed": changed, "new_zeros": zeros, "error_squared_sum": error2,
                "signal_squared_sum": signal2, "max_absolute_error": maximum,
                "sqnr_db": 10 * math.log10(signal2 / error2) if error2 else None,
            }
            if mode == "original":
                assert actual_sha == source["source_sha256"]
                entry["restored_bits_exact"] = True
            else:
                # Independent analytic-codebook check on both ends of every real TP tensor.
                sample = torch.cat((original[:2], original[-2:])).float().numpy()
                expected, codes, scales = numpy_reference(sample)
                actual = torch.cat((value[:2], value[-2:])).float().cpu().numpy()
                np.testing.assert_array_equal(actual, expected)
                entry["independent_sample_elements"] = int(sample.size)
                file = Path("/traces") / f"roundtrip-layer{source['layer']}-{source['kind']}.npz"
                assert not file.exists(), "Preserve the one declared quantization attempt"
                np.savez_compressed(file, original=sample, decoded=actual, codes=codes, scales=scales)
                entry["sample_file"] = file.name
                entry["sample_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
            records.append(entry)
        torch.cuda.synchronize()
        self._rounding_mode = mode
        result = {
            "status": "complete", "rank": self._rounding_rank, "mode": mode, "layers": records,
            "same_weight_addresses": True, "same_BF16_GEMMs": True,
            "checkpoint_files_changed": False,
        }
        path = Path("/traces") / f"weights-{mode}-r{self._rounding_rank}.json"
        assert not path.exists()
        path.write_text(json.dumps(result, indent=2) + "\n")
        return result

    def _install_rounding_observer(self) -> None:
        """Capture unmodified logits, then impose the same continuation on the sampler."""
        original = self.model_runner.model.compute_logits
        original_sample = self.model_runner.sample
        self._rounding_sample_batch = None

        def sample(hidden: Any, batch: Any, grammar: Any) -> Any:
            self._rounding_sample_batch = batch
            try:
                return original_sample(hidden, batch, grammar)
            finally:
                self._rounding_sample_batch = None

        self.model_runner.sample = sample

        def observed(*args: Any, **kwargs: Any) -> Any:
            logits = original(*args, **kwargs)
            capture = self._rounding_capture
            if capture is None or logits is None:
                return logits
            assert self._rounding_mode in ("original", "mxfp8_roundtrip")
            assert tuple(logits.shape) == (1, 154880), tuple(logits.shape)
            batch = self._rounding_sample_batch
            assert batch is not None and batch.num_reqs == 1 and batch.num_draft_tokens == 0
            active = ~batch.is_padding
            positions = batch.positions[active].cpu().tolist()
            ids = batch.input_ids[active].cpu().tolist()
            assert positions and len(positions) == len(ids)
            proof = {
                "positions": positions, "input_ids": ids,
                "num_tokens": batch.num_tokens,
                "num_tokens_after_padding": batch.num_tokens_after_padding,
                "has_prefill": bool(batch.has_prefill),
            }
            expected = capture["prompt"] + (capture["forced"] or [])
            for at, token in zip(positions, ids, strict=True):
                if at < len(expected):
                    assert expected[at] == token, ("Changed actual conditioning", at)
            capture["batches"].append(proof)
            if max(positions) < len(capture["prompt"]) - 1:
                # A non-final chunk is sampled internally but its output is discarded.
                return logits
            position = len(capture["steps"])
            assert position < capture["count"]
            assert max(positions) == len(capture["prompt"]) - 1 + position
            raw = logits.detach().float().cpu().numpy().astype("<f4").tobytes()
            assert np.isfinite(np.frombuffer(raw, dtype="<f4")).all()
            name = f"{capture['name']}-step{position:02d}.f32"
            if self._rounding_rank == 0:
                path = Path("/traces") / name
                assert not path.exists(), "Preserve raw unmodified logits"
                path.write_bytes(raw)
            capture["steps"].append({
                "position": position, "file": name, "sha256": hashlib.sha256(raw).hexdigest(),
                "argmax": int(logits.argmax()), "raw_before_forcing": True,
            })
            if capture["forced"] is not None:
                token = capture["forced"][position]
                logits.fill_(-torch.inf)
                logits[0, token] = 0
            return logits

        self.model_runner.model.compute_logits = observed

    def mxwave_rounding_arm(
        self, name: str, count: int, forced: list[int] | None, prompt: list[int]
    ) -> dict[str, Any]:
        """Arm a single serial request; API token IDs must confirm the forced history."""
        assert self._rounding_capture is None
        assert re.fullmatch(r"[a-z0-9_]+", name)
        assert 1 <= count <= 32
        assert prompt and all(type(t) is int and 0 <= t < 154880 for t in prompt)
        assert forced is None or (
            len(forced) == count and all(type(t) is int and 0 <= t < 154880 for t in forced)
        )
        self._rounding_capture = {
            "name": name, "count": count, "forced": forced, "prompt": prompt,
            "steps": [], "batches": [],
            "rank": self._rounding_rank, "mode": self._rounding_mode,
        }
        return {"status": "armed", "rank": self._rounding_rank, "name": name}

    def mxwave_rounding_disarm(self) -> dict[str, Any]:
        """Retain capture completion and require every planned rank-zero observation."""
        capture = self._rounding_capture
        assert capture is not None
        self._rounding_capture = None
        if self._rounding_rank == 0 or capture["steps"]:
            assert len(capture["steps"]) == capture["count"]
        capture["status"] = "complete"
        path = Path("/traces") / f"{capture['name']}-r{self._rounding_rank}.json"
        assert not path.exists()
        path.write_text(json.dumps(capture, indent=2) + "\n")
        return capture
