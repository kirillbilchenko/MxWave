"""Opt-in startup timing and on-disk cache for packed sparse-palette MoE expert scales.

Both features are off unless enabled by env or by a marker file in SWITCH_DIR (default
/cache/mxwave-scale-cache): MXWAVE_STARTUP_TIMING=1 or file TIMING; MXWAVE_SCALE_CACHE_DIR=<path>
or file ENABLE (cache root is then SWITCH_DIR itself). A set env var always wins over the file.
The cache only stores CPU-side pack output; every start still proves the uploaded
tensors on the GPU against the live raw bytes (model_plugin.prepare_layer).

Besides the palette buffers, an entry carries what the staged consumer needs: the source
value range, the staging policy integer and the aligned nibble base (PackedScales.descriptor
word 2 and PackedScales.aligned_nibble_base). On load the range is recomputed from the cached
buffers themselves and the policy and base from that range, and all must match what was stored.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

FORMAT_VERSION = 2
SWITCH_DIR = Path(os.environ.get("MXWAVE_SCALE_SWITCH_DIR", "/cache/mxwave-scale-cache"))
EXPECTED_LAYERS = 42
_HEX = frozenset("0123456789abcdef")
_PHASES = ("repack", "load", "d2h", "sha", "pack", "roundtrip", "upload", "store", "gpu_check")


@dataclass
class Entry:
    """One cached scale group: exact CPU pack output plus the source hash."""

    selectors: np.ndarray
    exceptions: np.ndarray
    descriptor: int
    shape: tuple[int, ...]
    sha256: str
    raw_min: int
    raw_max: int
    policy: int  # descriptor word the staged consumer reads (palette plus stage bits)
    base: int | None  # aligned nibble base, None when the exact U8 fallback is required


@dataclass
class _State:
    hits: dict[int, int] = field(default_factory=dict)  # id(packed) -> group index
    groups: int = 0
    hit_groups: int = 0
    phases: dict[str, float] = field(default_factory=dict)
    totals: dict[str, float] = field(default_factory=dict)
    repack: float = 0.0
    retry: int | None = None
    stale_id: int | None = None
    warned: bool = False
    last: str = "off"
    disabled: bool = False  # a cache operation failed: stay uncached for the process
    bypass: bool = False  # a cached entry failed its GPU proof: stop reading the cache
    initialized: bool = False
    switches: tuple[bool, Path | None] | None = None  # resolved once per process
    inactive_logged: bool = False
    pending: dict[int, tuple[int, Any]] = field(default_factory=dict)  # id(packed) -> (index, Entry)


_STATE = _State()


def _marker(name: str) -> bool:
    """Whether SWITCH_DIR/<name> exists; any error (permissions, odd FS) means off."""
    try:
        return os.path.exists(SWITCH_DIR / name)
    except Exception:  # noqa: BLE001 -- any cache failure must degrade to the uncached path.
        return False


def _switches() -> tuple[bool, Path | None]:
    """(timing, cache root), resolved once per process so startup behavior cannot change."""
    if _STATE.switches is not None:
        return _STATE.switches
    timing_env = os.environ.get("MXWAVE_STARTUP_TIMING")
    if timing_env is not None:
        timing = timing_env == "1"
    else:
        timing = _marker("TIMING")
        if timing:
            print(f"MxWave timing: enabled by {SWITCH_DIR / 'TIMING'}", flush=True)
    cache_env = os.environ.get("MXWAVE_SCALE_CACHE_DIR")
    root: Path | None
    if cache_env is not None:
        root = None if cache_env.strip().lower() in ("", "0", "off") else Path(cache_env)
    elif _marker("ENABLE"):
        root = SWITCH_DIR
        print(f"MxWave scale cache: enabled by {SWITCH_DIR / 'ENABLE'}", flush=True)
    else:
        root = None
    _STATE.switches = (timing, root)
    return _STATE.switches


def timing_enabled() -> bool:
    """MXWAVE_STARTUP_TIMING=1, or (env unset) a TIMING file in SWITCH_DIR."""
    return _switches()[0]


def cache_dir() -> Path | None:
    """Cache root: MXWAVE_SCALE_CACHE_DIR (""/0/off disable), else SWITCH_DIR if ENABLE exists."""
    return _switches()[1]


def active() -> bool:
    """Whether the instrumented sparse path must replace the original packer call."""
    return timing_enabled() or cache_dir() is not None


def sync(device: Any = None) -> None:
    """Synchronize CUDA only for honest timings; callers gate on timing_enabled()."""
    if torch.cuda.is_available():
        torch.cuda.synchronize(device if getattr(device, "type", None) == "cuda" else None)


def _add(phase: str, seconds: float) -> None:
    _STATE.phases[phase] = _STATE.phases.get(phase, 0.0) + seconds
    _STATE.totals[phase] = _STATE.totals.get(phase, 0.0) + seconds


def run_repack(fn: Callable[[], None]) -> None:
    """Run stock repacking, timing it (with syncs) only when timing is on."""
    if not timing_enabled():
        fn()
        return
    sync()
    start = time.perf_counter()
    fn()
    sync()
    _STATE.repack = time.perf_counter() - start
    _STATE.totals["repack"] = _STATE.totals.get("repack", 0.0) + _STATE.repack


def _file_sha(path: str | os.PathLike[str] | None) -> str:
    if not path or not Path(path).is_file():
        return "missing"
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_file(name: str) -> str | None:
    """Locate a module's source file without requiring it to be importable."""
    try:
        module = sys.modules.get(name)
        if module is not None:
            return inspect.getfile(module)
        spec = importlib.util.find_spec(name)
        return spec.origin if spec else None
    except (ImportError, ValueError, TypeError, AttributeError):
        return None


def _source_paths() -> dict[str, str | None]:
    """Files whose content defines the encoding, the repack producer or the consumer."""
    names = ("scale_tiles", "moe_runtime", "moe_scale_policy", "palette_model_worker",
             "model_plugin", "scale_cache")
    paths = {name: _module_file(name) for name in names}
    prefix = "vllm.model_executor.layers.quantization."
    paths["marlin_utils_fp4"] = _module_file(prefix + "utils.marlin_utils_fp4")
    paths["w4a4_mxfp4_moe"] = _module_file(
        prefix + "compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4")
    return paths


def _checkpoint_root() -> Path:
    return Path(os.environ.get("MXWAVE_SCALE_CACHE_SOURCE")
                or os.environ.get("MXWAVE_GLM53_KDA_SOURCE")
                or os.environ.get("ZIPSERV_CT_SOURCE") or "/model")


def _checkpoint_files() -> list[Path]:
    root = _checkpoint_root()
    if not root.is_dir():
        return []
    return sorted(root.glob("*.safetensors.index.json")) + [root / "config.json"]


def _shard_sizes() -> dict[str, Any]:
    """File name -> byte size of every shard the index references (no mtimes)."""
    root, sizes = _checkpoint_root(), {}
    for index in sorted(root.glob("*.safetensors.index.json")) if root.is_dir() else []:
        try:
            names = set(json.loads(index.read_text())["weight_map"].values())
        except (OSError, ValueError, KeyError, AttributeError):
            sizes[index.name] = "unreadable"
            continue
        for name in sorted(names):
            try:
                sizes[name] = (root / name).stat().st_size
            except OSError:
                sizes[name] = "missing"
    return sizes


def _tp() -> tuple[int, int]:
    """Tensor-parallel (rank, world size) with a safe fallback outside vLLM."""
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        return int(get_tensor_model_parallel_rank()), int(get_tensor_model_parallel_world_size())
    except Exception:  # noqa: BLE001 -- any cache failure must degrade to the uncached path.
        return 0, 1


def _version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # noqa: BLE001 -- any cache failure must degrade to the uncached path.
        return "unknown"


def key_payload() -> dict[str, Any]:
    """Everything the packed bytes depend on, as a JSON-serializable mapping."""
    rank, world = _tp()
    return {
        "format": FORMAT_VERSION,
        "tp_rank": rank,
        "tp_world_size": world,
        "checkpoint": {p.name: _file_sha(p) for p in _checkpoint_files()},
        "shards": _shard_sizes(),
        "sources": {name: _file_sha(path) for name, path in sorted(_source_paths().items())},
        "aligned_library": _file_sha(os.environ.get("MXWAVE_MOE_ALIGNED_LIBRARY")),
        "vllm": _version("vllm"),
        "torch": str(torch.__version__),
    }


def _key_gaps(payload: dict[str, Any]) -> list[str]:
    """Names of key inputs that could not be resolved (weakening the key)."""
    gaps = [k for k in ("checkpoint", "shards") if not payload[k]]
    for group in ("checkpoint", "shards", "sources"):
        gaps += [f"{group}.{k}" for k, v in payload[group].items()
                 if v in ("missing", "unreadable")]
    gaps += [k for k in ("aligned_library", "vllm") if payload[k] in ("missing", "unknown")]
    return gaps


@functools.lru_cache(maxsize=1)
def cache_key() -> str:
    """Hex sha256 of the canonical key payload, computed once per process."""
    payload = key_payload()
    gaps = _key_gaps(payload)
    if gaps:
        print(f"MxWave scale cache: key inputs unresolved: {', '.join(gaps)}", flush=True)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _entry_path(root: Path, index: int) -> Path:
    return root / cache_key() / f"{index:03d}.npz"


def _warn_disabled(what: str, exc: BaseException) -> None:
    """Degrade to the uncached path for the rest of the process, with one warning."""
    _STATE.disabled = True
    if not _STATE.warned:
        _STATE.warned = True
        print(f"MxWave scale cache: {what} failed ({type(exc).__name__}: {exc}); "
              "continuing without caching", flush=True)


def _root() -> Path | None:
    """The usable cache root; runs the one-time key computation and directory hygiene."""
    root = cache_dir()
    if root is None or _STATE.disabled:
        return None
    if not _STATE.initialized:
        _STATE.initialized = True
        try:
            _init_dir(root)
        except Exception as exc:  # noqa: BLE001 -- any cache failure must degrade to the uncached path.
            _warn_disabled("initialization", exc)
            return None
    return root


def _init_dir(root: Path) -> None:
    """Remove stale tmp files from this key's directory and report other key directories."""
    key_dir = root / cache_key()
    if key_dir.is_dir():
        for stale in key_dir.glob("*.tmp*"):
            try:
                stale.unlink()
            except OSError:
                pass
    others = [d for d in root.iterdir() if d.is_dir() and d != key_dir
              and len(d.name) == 64 and set(d.name) <= _HEX] if root.is_dir() else []
    size = sum(f.stat().st_size for d in others for f in d.rglob("*") if f.is_file())
    print(f"MxWave scale cache: {len(others)} other key dirs, {size / 1e9:.2f} GB "
          f"(not deleted) in {root}", flush=True)


def _atomic_write(path: Path, writer: Callable[[Any], None]) -> None:
    """Write via a same-directory temp file, fsync it, then rename over the target."""
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def sparse_range(
    palette: tuple[int, int, int], selectors: np.ndarray, exceptions: np.ndarray,
) -> tuple[int, int]:
    """Exact (min, max) of the source bytes, recomputed from valid sparse-palette buffers.

    Every source byte is either a palette value reached through a used 2-bit code of a
    non-escaped selector word, or a byte of an exception vector. The caller must already have
    checked scale_tiles.check_sparse_structure (no code 3 outside escaped words). A palette
    value that no code selects is not a source byte (palettes of near-constant data pad it).
    """
    low_bits = np.uint32(0x55555555)
    used = [False, False, False]
    for start in range(0, selectors.size, 1 << 20):
        words = selectors[start:start + (1 << 20)]
        words = words[words >> 30 != 3]
        if not words.size:
            continue
        lower, upper = words & low_bits, words >> 1 & low_bits
        used[0] |= bool(np.any(~(lower | upper) & low_bits))
        used[1] |= bool(np.any(lower & ~upper))
        used[2] |= bool(np.any(upper & ~lower))
    values = [palette[i] for i in range(3) if used[i]]
    if exceptions.size:
        values += [int(exceptions.min()), int(exceptions.max())]
    if not values:
        raise ValueError("Sparse scale buffers hold no source bytes")
    return min(values), max(values)


def store_entry(root: Path, index: int, entry: Entry) -> None:
    """Persist one entry atomically; any failure warns once and disables the cache."""
    path = _entry_path(root, index)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = np.array([FORMAT_VERSION, index, entry.descriptor, entry.selectors.size,
                         entry.exceptions.shape[0], entry.raw_min, entry.raw_max, entry.policy,
                         -1 if entry.base is None else entry.base], dtype=np.int64)
        _atomic_write(path, lambda f: np.savez(
            f, selectors=entry.selectors, exceptions=entry.exceptions, meta=meta,
            shape=np.array(entry.shape, dtype=np.int64), sha256=np.array(entry.sha256)))
    except Exception as exc:  # noqa: BLE001 -- any cache failure must degrade to the uncached path.
        _warn_disabled("write", exc)


def load_entry(root: Path, index: int, shape: tuple[int, ...]) -> Entry | None:
    """Read and fully validate one entry; None means missing, corrupt or mismatched.

    The descriptor is recomputed from its palette with scale_tiles.sparse_descriptor and must
    match exactly (no unknown or flipped flag bits); the selectors must satisfy the escape
    invariants so a corrupt entry cannot index outside the exception buffer. The source value
    range is recomputed from the buffers, and the staging policy and nibble base from that range
    with the live moe_scale_policy; every stored value must equal its recomputation.
    """
    import moe_runtime
    import scale_tiles

    path = _entry_path(root, index)
    try:
        with np.load(path, allow_pickle=False) as data:
            selectors, exceptions = data["selectors"], data["exceptions"]
            meta, stored_shape, sha = data["meta"], data["shape"], str(data["sha256"])
        (version, stored_index, descriptor, n_sel, n_exc, raw_min, raw_max, policy,
         stored_base) = (int(v) for v in meta)
        count = int(np.prod(shape, dtype=np.int64))
        palette = (descriptor & 255, descriptor >> 8 & 255, descriptor >> 16 & 255)
        if (version != FORMAT_VERSION or stored_index != index
                or tuple(int(v) for v in stored_shape) != tuple(shape)
                or selectors.dtype != np.uint32 or selectors.ndim != 1
                or selectors.size != n_sel or count % 16 or n_sel != count // 16
                or exceptions.dtype != np.uint8 or exceptions.ndim != 2
                or exceptions.shape != (n_exc, 16) or len(set(palette)) != 3
                or not 0 <= descriptor < 2**26
                or descriptor != scale_tiles.sparse_descriptor(palette)
                or len(sha) != 64 or not set(sha) <= _HEX
                or not selectors.flags.c_contiguous or not exceptions.flags.c_contiguous
                or not scale_tiles.check_sparse_structure(selectors, n_exc)):
            return None
        if sparse_range(palette, selectors, exceptions) != (raw_min, raw_max):
            return None
        want_policy, want_base = moe_runtime.sparse_stage(raw_min, raw_max, descriptor)
        base = None if stored_base == -1 else stored_base
        if policy != want_policy or base != want_base or stored_base < -1:
            return None
    except Exception:  # missing file, BadZipFile, absent key, bad dtype: all mean a miss  # noqa: BLE001 -- any cache failure must degrade to the uncached path.
        return None
    return Entry(selectors, exceptions, descriptor, tuple(shape), sha, raw_min, raw_max, policy,
                 base)


def drop_entry(index: int) -> None:
    """Delete one entry file (best effort)."""
    root = cache_dir()
    if root is not None and not _STATE.disabled:
        try:
            _entry_path(root, index).unlink()
        except Exception:  # noqa: BLE001, S110 -- best effort; load revalidates entries.
            pass


def invalidate(packed: Any) -> None:
    """Drop a cache-hit tensor that failed its GPU check and stop reading the cache."""
    index = _STATE.hits.pop(id(packed), None)
    if index is None:
        return
    _STATE.hit_groups -= 1
    _STATE.stale_id = id(packed)
    drop_entry(index)
    first = not _STATE.bypass
    _STATE.bypass = True
    print(f"MxWave scale cache: GPU check failed for group {index}; entry invalidated, "
          "repacking without cache" + ("; cache reads bypassed for this process" if first else ""),
          flush=True)


def forget(packed: Any) -> None:
    """Drop per-tensor bookkeeping (ids get reused) without writing anything."""
    _STATE.hits.pop(id(packed), None)
    _STATE.pending.pop(id(packed), None)


def commit(packed: Any) -> None:
    """After the GPU proof passed, persist a freshly packed group (no-op for hits)."""
    _STATE.hits.pop(id(packed), None)
    pending = _STATE.pending.pop(id(packed), None)
    root = _root() if pending is not None else None
    if pending is None or root is None:
        return
    start = time.perf_counter()
    store_entry(root, *pending)
    _add("store", time.perf_counter() - start)


def hit_index(packed: Any) -> int | None:
    """Group index when this tensor came from the cache, else None."""
    return _STATE.hits.get(id(packed))


@contextmanager
def retry_uncached(index: int) -> Iterator[None]:
    """Make the next sparse pack for this group skip the cache read and not recount it."""
    _STATE.retry = index
    try:
        yield
    finally:
        _STATE.retry = None


def retry_index() -> int | None:
    """Group index being redone without the cache, if any."""
    return _STATE.retry


def pop_stale_id() -> int | None:
    """Return (once) the id of the invalidated tensor so callers can drop its bookkeeping."""
    value, _STATE.stale_id = _STATE.stale_id, None
    return value


def take_phases() -> dict[str, float]:
    """Return and reset the per-group phase timings."""
    value, _STATE.phases = _STATE.phases, {}
    return value


def note_inactive() -> None:
    """Say once that MXWAVE_SCALE_CACHE_DIR is ignored because the palette model is off."""
    if not _STATE.inactive_logged:
        _STATE.inactive_logged = True
        print("MxWave scale cache: inactive (MXWAVE_SCALE_CACHE_DIR is set but "
              "MXWAVE_PALETTE_MODEL != 1)", flush=True)


def pack_group(
    scales: torch.Tensor, index: int, *, pack: Callable[[Any], Any],
    verify: Callable[[Any, Any], None], upload: Callable[..., Any],
    on_sha: Callable[[str], None],
) -> tuple[Any, str]:
    """Pack one sparse-palette group, via the cache when allowed.

    on_sha receives the source sha256 before anything that can raise (a hit supplies the
    sha stored at write time; the caller's GPU compare proves the bytes). Writes are deferred
    to commit() so only GPU-proven groups are cached. Returns (PackedScales, sha256).
    """
    timing = timing_enabled()
    root = _root()
    shape = tuple(scales.shape)
    device = scales.device
    retrying = retry_index() is not None
    if not retrying:
        _STATE.groups += 1
    if root is not None and not retrying and not _STATE.bypass:
        start = time.perf_counter()
        try:
            entry = load_entry(root, index, shape)
        except Exception as exc:  # e.g. key computation  # noqa: BLE001 -- any cache failure must degrade to the uncached path.
            _warn_disabled("read", exc)
            entry = None
        _add("load", time.perf_counter() - start)
        if entry is not None:
            on_sha(entry.sha256)
            start = time.perf_counter()
            packed = upload(entry.selectors, entry.exceptions, entry.descriptor, shape, device,
                            entry.raw_min, entry.raw_max)
            if timing:
                sync(device)
            _add("upload", time.perf_counter() - start)
            _STATE.hits[id(packed)] = index
            _STATE.hit_groups += 1
            _STATE.last = "hit"
            return packed, entry.sha256
    if timing:
        sync(device)
    start = time.perf_counter()
    raw_t = scales.view(torch.uint8).cpu()
    _add("d2h", time.perf_counter() - start)
    start = time.perf_counter()
    raw = raw_t.numpy().reshape(-1)
    sha = hashlib.sha256(raw).hexdigest()
    _add("sha", time.perf_counter() - start)
    on_sha(sha)
    start = time.perf_counter()
    sparse = pack(raw)
    _add("pack", time.perf_counter() - start)
    start = time.perf_counter()
    verify(sparse, raw)
    _add("roundtrip", time.perf_counter() - start)
    start = time.perf_counter()
    raw_min, raw_max = int(raw.min()), int(raw.max())
    packed = upload(sparse.selectors, sparse.exceptions, sparse.descriptor, shape, device,
                    raw_min, raw_max)
    if timing:
        sync(device)
    _add("upload", time.perf_counter() - start)
    if root is not None:
        # Store exactly what the consumer was given, read back from the built tensors.
        policy = int(packed.descriptor[2].item())
        _STATE.pending[id(packed)] = (index, Entry(
            sparse.selectors, sparse.exceptions, int(sparse.descriptor), shape, sha, raw_min,
            raw_max, policy, packed.aligned_nibble_base))
    _STATE.last = "miss" if root is not None else "off"
    return packed, sha


def log_group(layer: int, name: str, gpu_check: float) -> None:
    """Print one timing line for a scale group (layer is the per-process counter)."""
    _add("gpu_check", gpu_check)
    phases = take_phases()
    phases["repack"] = _STATE.repack if name.startswith("w13") else 0.0
    fields = " ".join(f"{p}={phases.get(p, 0.0):.3f}" for p in _PHASES)
    cache, _STATE.last = _STATE.last, "off"
    print(f"MxWave timing: layer={layer} name={name} {fields} cache={cache}", flush=True)


def log_summary() -> None:
    """Print the cache line (if it was used) and cumulative timing (if timing) once."""
    if cache_dir() is not None and _STATE.groups:
        total, hit = _STATE.groups, _STATE.hit_groups
        state = "hit" if hit == total else "miss" if hit == 0 else "mixed"
        try:
            key = cache_key()[:12]
        except Exception:  # noqa: BLE001 -- any cache failure must degrade to the uncached path.
            key = "unavailable"
        print(f"MxWave scale cache: {state} key={key} dir={cache_dir()} "
              f"groups={hit}/{total}", flush=True)
    if timing_enabled():
        fields = " ".join(f"{p}={_STATE.totals.get(p, 0.0):.3f}" for p in _PHASES)
        print(f"MxWave timing: total layers={EXPECTED_LAYERS} {fields}", flush=True)
