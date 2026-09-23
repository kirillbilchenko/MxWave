"""Activation-statistics capture and safe calibration artifact I/O.

The collector operates on the actual input of every selected weight module.  It
can retain inexpensive per-channel moments or a block-diagonal second moment
whose block size matches MXFP4.  Keeping collection separate from quantization
lets one calibration pass drive controlled objective ablations.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .core import BLOCK_SIZE

CalibrationObjective = Literal["mean-abs", "rms", "block-hessian"]

_FORMAT = "mxwave-activation-stats"
_FORMAT_VERSION = "1"
_SUPPORTED_FORMATS: frozenset[str] = frozenset(
    {_FORMAT, "mxstream-activation-stats"}
)
_OBJECTIVES: frozenset[str] = frozenset({"mean-abs", "rms", "block-hessian"})
_KEY_SEPARATOR = "::"

__all__ = [
    "ActivationCollector",
    "CalibrationArtifact",
    "CalibrationData",
    "CalibrationObjective",
    "attach_activation_hooks",
    "load_calibration_data",
    "open_calibration_data",
    "save_calibration_data",
]


@dataclass(frozen=True)
class CalibrationData:
    """One validated objective loaded from a calibration artifact."""

    objective: CalibrationObjective
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, str]
    file_sha256: str


@dataclass(frozen=True)
class _CalibrationFileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int


def _calibration_file_identity(path: Path) -> _CalibrationFileIdentity:
    stat = path.stat()
    return _CalibrationFileIdentity(
        device=stat.st_dev,
        inode=stat.st_ino,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def _require_calibration_file_identity(
    path: Path,
    expected: _CalibrationFileIdentity,
) -> None:
    if _calibration_file_identity(path) != expected:
        raise ValueError(f"Calibration artifact changed during quantization: {path}")


class _LazyCalibrationTensors(Mapping[str, torch.Tensor]):
    """Non-caching, on-demand view of one calibration objective."""

    def __init__(
        self,
        path: Path,
        objective: CalibrationObjective,
        target_widths: Mapping[str, int],
        file_identity: _CalibrationFileIdentity,
    ) -> None:
        self._path = path
        self._objective = objective
        self._target_widths = dict(target_widths)
        self._target_names = tuple(sorted(target_widths))
        self._file_identity = file_identity

    def __getitem__(self, target_name: str) -> torch.Tensor:
        try:
            width = self._target_widths[target_name]
        except KeyError:
            raise KeyError(target_name) from None
        _require_calibration_file_identity(self._path, self._file_identity)
        key = _tensor_key(self._objective, target_name)
        with safe_open(str(self._path), framework="pt", device="cpu") as artifact:
            tensor = cast(torch.Tensor, artifact.get_tensor(key))
        _validate_statistic(target_name, self._objective, tensor, width)
        _require_calibration_file_identity(self._path, self._file_identity)
        return tensor

    def __iter__(self) -> Iterator[str]:
        return iter(self._target_names)

    def __len__(self) -> int:
        return len(self._target_names)


@dataclass(frozen=True)
class CalibrationArtifact:
    """Validated calibration artifact whose statistic tensors load on demand.

    The ``tensors`` mapping never caches values. Each lookup opens the
    safetensors artifact, loads only the requested statistic, validates its
    numerical contents, and returns it to the caller. Iterating the mapping or
    taking its length only inspects the validated target-name plan.
    """

    objective: CalibrationObjective
    tensors: Mapping[str, torch.Tensor]
    metadata: dict[str, str]
    file_sha256: str
    path: Path
    _file_identity: _CalibrationFileIdentity

    def load_tensor(self, target_name: str) -> torch.Tensor:
        """Load and fully validate one target statistic without caching it."""
        return self.tensors[target_name]

    def validate_all(self) -> None:
        """Numerically validate every target while retaining only one at a time."""
        for target_name in self.tensors:
            tensor = self.tensors[target_name]
            del tensor

    def verify_unchanged(self, *, verify_sha256: bool = False) -> None:
        """Fail if the artifact changed since it was opened."""
        _require_calibration_file_identity(self.path, self._file_identity)
        if verify_sha256 and _sha256_file(self.path) != self.file_sha256:
            raise ValueError(f"Calibration artifact content changed during quantization: {self.path}")
        _require_calibration_file_identity(self.path, self._file_identity)


class ActivationCollector:
    """Accumulate real module-input statistics without retaining activations.

    Sums stay on the activation device during collection.  This avoids a GPU
    synchronization for every hooked module and forward pass; only finalized
    statistics are copied to CPU.
    """

    def __init__(
        self,
        target_widths: Mapping[str, int],
        objectives: Iterable[CalibrationObjective] = ("mean-abs",),
        *,
        hessian_damp: float = 1e-6,
    ) -> None:
        """Create a collector for exact target weight names and input widths."""
        if not target_widths:
            raise ValueError("At least one calibration target is required")
        if hessian_damp < 0.0:
            raise ValueError("hessian_damp must be non-negative")
        normalized_objectives = frozenset(objectives)
        if not normalized_objectives:
            raise ValueError("At least one calibration objective is required")
        invalid = sorted(set(normalized_objectives).difference(_OBJECTIVES))
        if invalid:
            raise ValueError(f"Unsupported calibration objectives: {invalid}")
        for name, width in target_widths.items():
            if not name.endswith(".weight"):
                raise ValueError(f"Calibration target must be a weight name: {name!r}")
            if width <= 0:
                raise ValueError(f"Calibration target {name!r} has invalid width {width}")
            if "block-hessian" in normalized_objectives and width % BLOCK_SIZE != 0:
                raise ValueError(
                    f"Calibration target {name!r} width {width} is not divisible by {BLOCK_SIZE}"
                )

        self.target_widths = dict(target_widths)
        self.objectives = normalized_objectives
        self.hessian_damp = hessian_damp
        self._counts: dict[str, int] = {}
        self._absolute_sums: dict[str, torch.Tensor] = {}
        self._square_sums: dict[str, torch.Tensor] = {}
        self._gram_sums: dict[str, torch.Tensor] = {}

    def update(self, target_name: str, activation: torch.Tensor) -> None:
        """Accumulate one invocation of a target module's input tensor."""
        width = self.target_widths.get(target_name)
        if width is None:
            raise KeyError(f"Unknown calibration target: {target_name}")
        if activation.ndim == 0 or activation.shape[-1] != width:
            raise ValueError(
                f"Input for {target_name!r} has shape {tuple(activation.shape)}, "
                f"expected final dimension {width}"
            )
        values = activation.detach().to(torch.float32).reshape(-1, width)
        if values.shape[0] == 0:
            return

        self._counts[target_name] = self._counts.get(target_name, 0) + values.shape[0]
        if "mean-abs" in self.objectives:
            absolute_sum = values.abs().sum(dim=0)
            self._accumulate(self._absolute_sums, target_name, absolute_sum)
        if "rms" in self.objectives:
            square_sum = values.square().sum(dim=0)
            self._accumulate(self._square_sums, target_name, square_sum)
        if "block-hessian" in self.objectives:
            blocked = values.reshape(values.shape[0], width // BLOCK_SIZE, BLOCK_SIZE)
            gram_sum = torch.einsum("nbi,nbj->bij", blocked, blocked)
            self._accumulate(self._gram_sums, target_name, gram_sum)

    @staticmethod
    def _accumulate(
        destination: dict[str, torch.Tensor],
        target_name: str,
        value: torch.Tensor,
    ) -> None:
        previous = destination.get(target_name)
        if previous is None:
            destination[target_name] = value
        else:
            previous.add_(value)

    def finalize(self) -> dict[CalibrationObjective, dict[str, torch.Tensor]]:
        """Return complete CPU float32 statistics, failing on every missed hook."""
        missing = sorted(set(self.target_widths).difference(self._counts))
        if missing:
            raise ValueError(
                f"Calibration hooks did not observe {len(missing)} target(s): {missing[:5]}"
            )

        result: dict[CalibrationObjective, dict[str, torch.Tensor]] = {}
        for objective in sorted(self.objectives):
            objective_values: dict[str, torch.Tensor] = {}
            for name in sorted(self.target_widths):
                count = self._counts[name]
                if objective == "mean-abs":
                    value = self._absolute_sums[name] / count
                elif objective == "rms":
                    value = torch.sqrt(self._square_sums[name] / count)
                else:
                    value = self._gram_sums[name] / count
                    if self.hessian_damp:
                        value.diagonal(dim1=-2, dim2=-1).add_(self.hessian_damp)
                value = value.cpu().to(torch.float32).contiguous()
                _validate_statistic(name, objective, value, self.target_widths[name])
                objective_values[name] = value
            result[objective] = objective_values
        return result

    @property
    def observation_counts(self) -> dict[str, int]:
        """Return the flattened-token count observed by each target hook."""
        return dict(self._counts)


def attach_activation_hooks(
    model: torch.nn.Module,
    collector: ActivationCollector,
    *,
    module_prefix: str = "",
) -> list[Any]:
    """Attach strict forward-pre-hooks to exact target modules.

    ``module_prefix`` lets a layer-local model capture globally keyed target
    names during sequential calibration.

    The caller owns the returned handles and must remove them after collection.
    """
    modules = dict(model.named_modules())
    handles: list[Any] = []
    missing: list[str] = []
    for target_name, width in sorted(collector.target_widths.items()):
        module_name = target_name.removesuffix(".weight")
        local_module_name = module_name
        if module_prefix:
            prefix = f"{module_prefix}."
            if not module_name.startswith(prefix):
                raise ValueError(
                    f"Calibration target {target_name!r} is outside module prefix "
                    f"{module_prefix!r}"
                )
            local_module_name = module_name.removeprefix(prefix)
        module = modules.get(local_module_name)
        if module is None:
            missing.append(local_module_name)
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[-1] != width:
            actual_shape = tuple(weight.shape) if isinstance(weight, torch.Tensor) else None
            raise ValueError(
                f"Calibration module {module_name!r} has weight shape {actual_shape}, "
                f"expected (*, {width})"
            )

        def capture(
            _module: torch.nn.Module,
            inputs: tuple[Any, ...],
            *,
            name: str = target_name,
        ) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise ValueError(f"Calibration module {name!r} received no tensor input")
            collector.update(name, inputs[0])

        handles.append(module.register_forward_pre_hook(capture))

    if missing:
        for handle in handles:
            handle.remove()
        raise ValueError(f"Model is missing {len(missing)} calibration module(s): {missing[:5]}")
    return handles


def _tensor_key(objective: CalibrationObjective, target_name: str) -> str:
    return f"{objective}{_KEY_SEPARATOR}{target_name}"


def _parse_objectives(metadata: Mapping[str, str]) -> frozenset[str]:
    raw = metadata.get("objectives")
    if raw is None:
        return frozenset()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Calibration metadata has invalid objectives JSON") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("Calibration metadata objectives must be a JSON string list")
    objectives = frozenset(parsed)
    invalid = sorted(set(objectives).difference(_OBJECTIVES))
    if invalid:
        raise ValueError(f"Calibration artifact contains unsupported objectives: {invalid}")
    return objectives


def _expected_statistic_shape(
    objective: CalibrationObjective,
    width: int,
) -> tuple[int, ...]:
    if width <= 0:
        raise ValueError(f"Calibration target width must be positive, got {width}")
    if objective == "block-hessian" and width % BLOCK_SIZE != 0:
        raise ValueError(
            f"Block-Hessian calibration target width {width} is not divisible by {BLOCK_SIZE}"
        )
    if objective == "block-hessian":
        return (width // BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    return (width,)


def _validate_statistic_header(
    target_name: str,
    objective: CalibrationObjective,
    shape: tuple[int, ...],
    dtype: str,
    width: int,
) -> None:
    expected_shape = _expected_statistic_shape(objective, width)
    if shape != expected_shape:
        raise ValueError(
            f"Calibration statistic {target_name!r}/{objective} has shape "
            f"{shape}, expected {expected_shape}"
        )
    if dtype != "F32":
        raise ValueError(
            f"Calibration statistic {target_name!r}/{objective} must be float32, "
            f"got safetensors dtype {dtype}"
        )


def _validate_statistic(
    target_name: str,
    objective: CalibrationObjective,
    tensor: torch.Tensor,
    width: int,
) -> None:
    expected_shape = _expected_statistic_shape(objective, width)
    if tensor.shape != expected_shape:
        raise ValueError(
            f"Calibration statistic {target_name!r}/{objective} has shape "
            f"{tuple(tensor.shape)}, expected {expected_shape}"
        )
    if tensor.dtype != torch.float32:
        raise ValueError(
            f"Calibration statistic {target_name!r}/{objective} must be float32, "
            f"got {tensor.dtype}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(
            f"Calibration statistic contains NaN or infinity: {target_name}/{objective}"
        )
    if objective != "block-hessian":
        if (tensor < 0).any() or not (tensor > 0).any():
            raise ValueError(
                f"Calibration magnitudes must be non-negative and nonzero: {target_name}"
            )
        return
    if (tensor.diagonal(dim1=-2, dim2=-1) < 0).any():
        raise ValueError(f"Calibration Hessian has a negative diagonal: {target_name}")
    if not torch.allclose(tensor, tensor.transpose(-1, -2), rtol=1e-4, atol=1e-5):
        raise ValueError(f"Calibration Hessian is not symmetric: {target_name}")


def save_calibration_data(
    path: str | Path,
    statistics: Mapping[CalibrationObjective, Mapping[str, torch.Tensor]],
    metadata: Mapping[str, str],
) -> None:
    """Atomically save one or more activation objectives as safetensors."""
    destination = Path(path)
    if not statistics:
        raise ValueError("No calibration statistics were provided")
    tensors: dict[str, torch.Tensor] = {}
    target_names: set[str] | None = None
    for objective, objective_values in statistics.items():
        if objective not in _OBJECTIVES:
            raise ValueError(f"Unsupported calibration objective: {objective}")
        current_names = set(objective_values)
        if target_names is None:
            target_names = current_names
        elif current_names != target_names:
            raise ValueError("Every calibration objective must cover the same target names")
        for target_name, value in objective_values.items():
            tensors[_tensor_key(objective, target_name)] = value.cpu().to(torch.float32).contiguous()
    if not target_names:
        raise ValueError("Calibration statistics contain no targets")

    artifact_metadata = dict(metadata)
    artifact_metadata.update(
        {
            "format": _FORMAT,
            "format_version": _FORMAT_VERSION,
            "objectives": json.dumps(sorted(statistics), separators=(",", ":")),
            "target_count": str(len(target_names)),
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.incomplete")
    temporary.unlink(missing_ok=True)
    try:
        save_file(tensors, str(temporary), metadata=artifact_metadata)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_calibration_data(
    path: str | Path,
    objective: CalibrationObjective,
    expected_widths: Mapping[str, int],
    *,
    expected_policy: str,
    expected_source_repository: str | None = None,
    expected_source_revision: str | None = None,
) -> CalibrationArtifact:
    """Open a validated calibration objective without retaining its tensors.

    Metadata, target coverage, tensor shapes, and tensor dtypes are validated
    eagerly from the safetensors header. Numerical contents are validated when
    each target is requested from the returned non-caching ``tensors`` mapping.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Calibration artifact is missing: {source}")
    if objective not in _OBJECTIVES:
        raise ValueError(f"Unsupported calibration objective: {objective}")

    file_identity = _calibration_file_identity(source)
    target_widths = dict(expected_widths)
    if not target_widths:
        raise ValueError("At least one expected calibration target is required")
    prefix = f"{objective}{_KEY_SEPARATOR}"
    with safe_open(str(source), framework="pt", device="cpu") as artifact:
        raw_metadata = artifact.metadata()
        metadata = dict(raw_metadata) if raw_metadata is not None else {}
        artifact_format = metadata.get("format")
        if (
            artifact_format not in _SUPPORTED_FORMATS
            or metadata.get("format_version") != _FORMAT_VERSION
        ):
            raise ValueError("File is not a supported MxWave activation-stats v1 artifact")
        if objective not in _parse_objectives(metadata):
            raise ValueError(f"Calibration artifact does not contain objective {objective!r}")
        if metadata.get("policy") != expected_policy:
            raise ValueError(
                f"Calibration policy {metadata.get('policy')!r} does not match "
                f"quantization policy {expected_policy!r}"
            )
        for key in ("num_sequences", "sequence_length", "num_tokens", "target_count"):
            raw_value = metadata.get(key)
            try:
                parsed_value = int(raw_value) if raw_value is not None else 0
            except ValueError as exc:
                raise ValueError(f"Calibration metadata {key!r} must be an integer") from exc
            if parsed_value <= 0:
                raise ValueError(f"Calibration metadata {key!r} must be positive")
        raw_offset = metadata.get("sequence_offset")
        if raw_offset is not None:
            try:
                sequence_offset = int(raw_offset)
            except ValueError as exc:
                raise ValueError(
                    "Calibration metadata 'sequence_offset' must be an integer"
                ) from exc
            if sequence_offset < 0:
                raise ValueError("Calibration metadata 'sequence_offset' must be non-negative")
        identity_checks = (
            ("source_repository", expected_source_repository),
            ("source_revision", expected_source_revision),
        )
        for key, expected in identity_checks:
            if expected is not None and metadata.get(key) != expected:
                raise ValueError(
                    f"Calibration {key} {metadata.get(key)!r} does not match {expected!r}"
                )
        keys = list(artifact.keys())
        actual_names = {
            key.removeprefix(prefix) for key in keys if key.startswith(prefix)
        }
        expected_names = set(target_widths)
        if actual_names != expected_names:
            missing = sorted(expected_names.difference(actual_names))
            extra = sorted(actual_names.difference(expected_names))
            raise ValueError(
                "Calibration target coverage mismatch: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        if int(metadata["target_count"]) != len(target_widths):
            raise ValueError("Calibration metadata target_count does not match the target plan")
        for target_name, width in target_widths.items():
            statistic = artifact.get_slice(_tensor_key(objective, target_name))
            _validate_statistic_header(
                target_name,
                objective,
                tuple(statistic.get_shape()),
                statistic.get_dtype(),
                width,
            )

    file_sha256 = _sha256_file(source)
    _require_calibration_file_identity(source, file_identity)
    return CalibrationArtifact(
        objective=objective,
        tensors=_LazyCalibrationTensors(
            source,
            objective,
            target_widths,
            file_identity,
        ),
        metadata=metadata,
        file_sha256=file_sha256,
        path=source,
        _file_identity=file_identity,
    )


def load_calibration_data(
    path: str | Path,
    objective: CalibrationObjective,
    expected_widths: Mapping[str, int],
    *,
    expected_policy: str,
    expected_source_repository: str | None = None,
    expected_source_revision: str | None = None,
) -> CalibrationData:
    """Eagerly load a calibration objective for API compatibility.

    New streaming callers should use :func:`open_calibration_data` and release
    each value from its non-caching ``tensors`` mapping after processing the
    corresponding target or output shard.
    """
    artifact = open_calibration_data(
        path,
        objective,
        expected_widths,
        expected_policy=expected_policy,
        expected_source_repository=expected_source_repository,
        expected_source_revision=expected_source_revision,
    )
    tensors = {target_name: artifact.load_tensor(target_name) for target_name in artifact.tensors}

    return CalibrationData(
        objective=artifact.objective,
        tensors=tensors,
        metadata=artifact.metadata,
        file_sha256=artifact.file_sha256,
    )
