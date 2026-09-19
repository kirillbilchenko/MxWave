"""Run the fixed measured-precision screen inside one detached Spark container."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

_DEFAULT_RUNTIME_IMAGE = "vllm/vllm-openai:v0.29.0"


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded screen runner parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--reference-logprobs", required=True)
    parser.add_argument("--baseline-logprobs", required=True)
    parser.add_argument("--evaluation-script", required=True)
    parser.add_argument("--selection-script", required=True)
    parser.add_argument("--runtime-image", default=_DEFAULT_RUNTIME_IMAGE)
    parser.add_argument("--linear-backend", default="marlin")
    parser.add_argument("--screen-contexts", type=int, default=32)
    parser.add_argument("--split-size", type=int, default=16)
    parser.add_argument("--max-selected-buckets", type=int, default=4)
    parser.add_argument("--max-premium-bytes", type=int, default=1024**3)
    return parser


def _read_json_object(path: Path) -> dict[str, Any]:
    value: Any = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return cast(dict[str, Any], value)


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run(command: list[str]) -> None:
    print(f"[precision-budget] run: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def _remove_temporary_model(path: Path, run_root: Path) -> None:
    if path.name != "candidate-model" or path.parent.resolve() != run_root.resolve():
        raise ValueError(f"Refusing to remove unexpected temporary path: {path}")
    if path.is_symlink():
        raise ValueError(f"Refusing to remove symlinked temporary model: {path}")
    if path.exists():
        shutil.rmtree(path)


def _state(
    state_path: Path,
    *,
    status: str,
    started: float,
    completed_candidates: list[str],
    current_candidate: str | None = None,
    detail: str | None = None,
) -> None:
    _write_json_atomic(
        state_path,
        {
            "format": "mxwave-precision-budget-screen-state-v1",
            "status": status,
            "started_at_utc": dt.datetime.fromtimestamp(started, tz=dt.UTC).isoformat(),
            "updated_at_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
            "elapsed_seconds": time.time() - started,
            "completed_candidates": completed_candidates,
            "current_candidate": current_candidate,
            "detail": detail,
        },
    )


def _compose_command(plan: Path, output: Path, buckets: list[str]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "mxwave.precision_budget_cli",
        "compose",
        "--plan",
        str(plan),
        "--output",
        str(output),
        "--quant-device",
        "cuda",
    ]
    for bucket in buckets:
        command.extend(("--bucket", bucket))
    return command


def _collect_command(
    args: argparse.Namespace,
    *,
    model: Path,
    label: str,
    output: Path,
    limit: int | None,
) -> list[str]:
    command = [
        sys.executable,
        args.evaluation_script,
        "collect",
        "--model",
        str(model),
        "--model-label",
        label,
        "--contexts",
        str(Path(args.contexts)),
        "--output",
        str(output),
        "--runtime-image",
        args.runtime_image,
        "--linear-backend",
        args.linear_backend,
        "--max-model-len",
        "1024",
        "--gpu-memory-utilization",
        "0.65",
    ]
    if limit is not None:
        command.extend(("--limit", str(limit)))
    return command


def run(args: argparse.Namespace) -> None:
    """Execute the pre-registered screen, selection, and optional final gate."""
    if args.screen_contexts != args.split_size * 2:
        raise ValueError("screen contexts must equal two complete selection splits")
    plan_path = Path(args.plan)
    run_root = Path(args.run_root)
    contexts_path = Path(args.contexts)
    reference_logprobs = Path(args.reference_logprobs)
    baseline_logprobs = Path(args.baseline_logprobs)
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run root is missing: {run_root}")
    plan = _read_json_object(plan_path)
    if plan.get("format") != "mxwave-precision-budget-plan-v1":
        raise ValueError("Unsupported precision-budget plan")
    raw_buckets = plan.get("buckets")
    if not isinstance(raw_buckets, list) or not raw_buckets:
        raise ValueError("Precision-budget plan contains no candidates")
    bucket_names: list[str] = []
    for raw_bucket in raw_buckets:
        if not isinstance(raw_bucket, dict) or not isinstance(raw_bucket.get("name"), str):
            raise TypeError("Precision-budget plan contains an invalid candidate")
        bucket_names.append(cast(str, raw_bucket["name"]))

    candidate_model = run_root / "candidate-model"
    if candidate_model.exists():
        raise FileExistsError(f"Temporary candidate already exists: {candidate_model}")
    candidates_root = run_root / "candidates"
    candidates_root.mkdir(exist_ok=True)
    state_path = run_root / "state.json"
    selection_path = run_root / "selection.json"
    final_path = run_root / "final.json"
    started = time.time()
    completed: list[str] = []
    _state(state_path, status="running", started=started, completed_candidates=completed)

    try:
        for bucket_name in bucket_names:
            _state(
                state_path,
                status="running",
                started=started,
                completed_candidates=completed,
                current_candidate=bucket_name,
            )
            _run(_compose_command(plan_path, candidate_model, [bucket_name]))
            output = candidates_root / f"{bucket_name}-logprobs.safetensors"
            _run(
                _collect_command(
                    args,
                    model=candidate_model,
                    label=bucket_name,
                    output=output,
                    limit=args.screen_contexts,
                )
            )
            if not output.is_file():
                raise FileNotFoundError(f"Collector did not write {output}")
            _remove_temporary_model(candidate_model, run_root)
            completed.append(bucket_name)

        _run(
            [
                sys.executable,
                args.selection_script,
                "select",
                "--reference",
                str(reference_logprobs),
                "--baseline",
                str(baseline_logprobs),
                "--contexts",
                str(contexts_path),
                "--plan",
                str(plan_path),
                "--candidate-dir",
                str(candidates_root),
                "--output",
                str(selection_path),
                "--split-size",
                str(args.split_size),
                "--max-selected-buckets",
                str(args.max_selected_buckets),
                "--max-premium-bytes",
                str(args.max_premium_bytes),
            ]
        )
        selection = _read_json_object(selection_path)
        raw_selected = selection.get("selected_buckets")
        if not isinstance(raw_selected, list) or not all(
            isinstance(bucket, str) for bucket in raw_selected
        ):
            raise TypeError("Selection report has an invalid selected-bucket list")
        selected = cast(list[str], raw_selected)
        if not selected:
            _state(
                state_path,
                status="complete-no-selection",
                started=started,
                completed_candidates=completed,
                detail="No bucket improved both fixed selection splits",
            )
            return

        combined_model = run_root / "combined-model"
        if combined_model.exists():
            raise FileExistsError(f"Combined candidate already exists: {combined_model}")
        _run(_compose_command(plan_path, combined_model, selected))
        combined_logprobs = run_root / "combined-logprobs.safetensors"
        _run(
            _collect_command(
                args,
                model=combined_model,
                label="combined",
                output=combined_logprobs,
                limit=None,
            )
        )
        _run(
            [
                sys.executable,
                args.selection_script,
                "final",
                "--reference",
                str(reference_logprobs),
                "--baseline",
                str(baseline_logprobs),
                "--candidate",
                str(combined_logprobs),
                "--contexts",
                str(contexts_path),
                "--selection",
                str(selection_path),
                "--output",
                str(final_path),
                "--selection-contexts",
                str(args.screen_contexts),
            ]
        )
        final = _read_json_object(final_path)
        passed = final.get("promotion_passed") is True
        if not passed:
            failed_model = run_root / "candidate-model"
            combined_model.rename(failed_model)
            _remove_temporary_model(failed_model, run_root)
        _state(
            state_path,
            status="complete-passed" if passed else "complete-rejected",
            started=started,
            completed_candidates=completed,
            detail="Combined holdout gate passed" if passed else "Combined holdout gate failed",
        )
    except BaseException as exc:
        _state(
            state_path,
            status="failed",
            started=started,
            completed_candidates=completed,
            detail=f"{type(exc).__name__}: {exc}",
        )
        raise


def main() -> int:
    """Run the detached-screen orchestration."""
    try:
        run(build_parser().parse_args())
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"precision-budget screen error: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
