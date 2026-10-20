"""Run exact decoder smoke/timing fixtures in one watched idle pause; never load a model."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import full_model_control as control


def command(root: Path, revision: str, stage: str, *,
            build_directory: str = "prefill-decode-v1/build",
            correctness_only: bool = False) -> list[str]:
    """Bound each GPU process and leave qualified native libraries read-only."""
    if correctness_only and stage != "smoke":
        raise ValueError("The wrapped-base qualification cannot launch timing or confirmation")
    mode = {"smoke": "raw", "timing": "aligned", "confirmation": "kda", "routing": "aligned"}[stage]
    script = ("smoke_prefill_wrap.py" if correctness_only else "smoke_prefill_decode.py"
              ) if stage == "smoke" else (
              "benchmark_prefill_routing.py" if stage == "routing" else "benchmark_prefill_decode.py")
    argv = ["docker", "run", "-d", "--name",
        f"mxwave-glm53-full-{revision}-{mode}-r1", "--gpus", "all", "--network", "none",
        "--cpus", "2", "--memory", "6g", "--memory-swap", "6g", "--pids-limit", "256",
        "-v", f"{root / 'src'}:/work:ro", "-v", f"{root / 'results'}:/result",
        "-v", f"{control.ROOT}:/audit:ro", "-v", f"{control.OLD}:/experiment:ro",
        "-e", "PYTHONPATH=/work:/experiment/triple-v2/src/integrations/zipserv",
        "-e", "MXWAVE_ZIPSERV_LIBRARY=/audit/native/libmxwave_kda_zipserv.so",
        "-e", "CUBLAS_WORKSPACE_CONFIG=:4096:8", "-e", "OMP_NUM_THREADS=1",
        "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1",
        "--entrypoint", "/usr/bin/timeout", control.IMAGE,
        "--signal=TERM", "--kill-after=10", "180", "/usr/bin/python3", "-u",
        f"/work/{script}", "--build",
        f"/audit/{build_directory}",
        "--report", f"/result/{stage}.json"]
    if stage != "smoke":
        argv.extend(["--root", "/audit", "--smoke", "/result/smoke.json",
                     "--plan", f"/audit/{root.name}/component-plan.json"])
    if stage == "confirmation":
        argv.append("--confirmation")
    return argv


def main() -> None:
    """Prepare/verify before pausing; restore exact original containers on every exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="prefilldecodev3")
    parser.add_argument("--watch", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--correctness-only", action="store_true")
    mode.add_argument("--routing", action="store_true")
    parser.add_argument("--build-directory", default="prefill-decode-v1/build")
    args = parser.parse_args()
    version = re.fullmatch(r"prefilldecodev([1-9][0-9]*)", args.revision)
    if version is None:
        raise ValueError("Require a numbered prefilldecodev evidence revision")
    root = control.ROOT / f"prefill-decode-v{version.group(1)}"
    allowed_builds = {"prefill-decode-v1/build", f"{root.name}/build"}
    if args.routing:
        allowed_builds.add("prefill-decode-v4/build")
    if args.build_directory not in allowed_builds:
        raise ValueError("Use a source-bound component build")
    path = control.ROOT / f"prefill-decode-controller-{args.revision}.json"
    if args.watch is not None:
        control.watch(args.revision, args.watch, path)
        return
    if path.exists():
        raise FileExistsError("Preserve the first paused component attempt")
    lock = (control.ROOT / "diagnostics-v1/component.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state: dict[str, Any] = {"phase": "preparing", "started_epoch": time.time(),
        "service_paused": False, "full_model_loaded": False, "stages": [],
        "scope": "Wrapped-base correctness only" if args.correctness_only else (
        "Fused split-K=1 routing sweep" if args.routing else
        "Three synthetic cases, two real KDA matrices, decoder and same-stock-GEMM timing")}
    success = False

    def save(phase: str) -> None:
        state.update(phase=phase, updated_epoch=time.time())
        control.write(path, state)

    def interrupted(signum: int, frame: Any) -> None:
        raise SystemExit(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        save("preparing")
        originals = [control.inspect(rank, control.ORIGINAL_IDS[rank]) for rank in (0, 1)]
        if (not control.health(8000) or any(not row["State"]["Running"] for row in originals)):
            raise RuntimeError("Require healthy originals before preparation")
        state["originals"] = [control.identity(row) for row in originals]
        state["controller_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        # CPU-only bindings and all source/fixture checks precede the idle pause.
        check = """import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]);name=sys.argv[2];s=r/name/'src';p=r/name/'prepared.json'
prepared=json.loads(p.read_bytes())
code={str(f.relative_to(s)):hashlib.sha256(f.read_bytes()).hexdigest()
      for f in s.rglob('*')if f.is_file()and f.suffix in('.py','.cuh')}
assert code==prepared['source_sha256']
plan=r/name/'component-plan.json'
assert hashlib.sha256(plan.read_bytes()).hexdigest()==prepared['plan_sha256']
cpu=json.loads((r/name/'results/pack-oracle.json').read_bytes())
assert cpu['status']=='complete' and not cpu['cuda_initialized']
assert cpu['supported_fixtures_exact'] and cpu['all_wrapped_source_words_exact']
assert cpu['code_sha256']==code['prefill_pack_oracle.py']
if sys.argv[4]=='wrapped':
 wrapped=json.loads((r/name/'results/wrapped-oracle.json').read_bytes())
 assert wrapped['status']=='cpu_verified' and wrapped['cpu_preflight_without_cuda']
 assert wrapped['code_sha256']==code['smoke_prefill_wrap.py']
 assert len(wrapped['packing_preflight'])==5
 assert all(x['base']==255 and x['unmasked_mismatches']>0
            for x in wrapped['packing_preflight'][3:])
b=r/sys.argv[3];native=json.loads((b/'native-build.json').read_bytes())
assert native['status']=='built_pending_gpu_qualification'
assert hashlib.sha256((b/'native-build.json').read_bytes()).hexdigest()==prepared['native_manifest_sha256']
assert native['library_sha256']==prepared['native_library_sha256']
assert hashlib.sha256((b/native['library']).read_bytes()).hexdigest()==native['library_sha256']
assert hashlib.sha256((s/'prefill_decode.cuh').read_bytes()).hexdigest()==native['header_sha256']
assert hashlib.sha256((r/'native/libmxwave_kda_zipserv.so').read_bytes()).hexdigest()==(
 '6d51630cd075a8c0dcbb957594341fb21e42d4b4634bb1c92735275d81a70f62')
f=r/'bf16-attention-v1/artifacts-v2';cases=json.loads((f/'prepared.json').read_bytes())['cases']
artifacts={}
for row in cases:
 if row['name'].startswith('kda-'):
  a=f/row['artifact'];assert a.name==row['artifact']
  actual=hashlib.sha256(a.read_bytes()).hexdigest();assert actual==row['artifact_sha256']
  artifacts[a.name]=actual
assert len(artifacts)==2
print(json.dumps({'source_files':len(code),'prepared_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'native_library_sha256':native['library_sha256'],'artifacts':artifacts}))
"""
        state["preflight"] = json.loads(control.run(1, ["python3", "-S", "-c", check,
            str(control.ROOT), root.name, args.build_directory,
            "wrapped" if args.correctness_only else "original"], timeout=120))
        prepared = json.loads((root / "prepared.json").read_bytes())
        actual = {str(p.relative_to(root / "src")): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (root / "src").rglob("*")
                  if p.is_file() and p.suffix in (".py", ".cuh")}
        if actual != prepared["source_sha256"]:
            raise RuntimeError("Head source differs from the prepared worker")
        spec = importlib.util.spec_from_file_location("quiet_guard", control.OLD /
            "triple-v2/src/integrations/marlin_scales/quiet_window.py")
        assert spec and spec.loader
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        previous, quiet = None, 0
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            current = guard.activity()
            quiet = quiet + 1 if not current["busy"] and current["counters"] == previous else 0
            previous = current["counters"]
            state["last_activity"] = current
            save("waiting_for_idle")
            if quiet >= 3:
                break
            time.sleep(5)
        else:
            raise TimeoutError("No idle window within thirty minutes")
        watchdog = subprocess.Popen(["python3", "-S", "-u", str(Path(__file__).resolve()),
            "--revision", args.revision, "--watch", str(os.getpid())],
            stdout=(root / "watchdog.log").open("xb"), stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
        state["watchdog_pid"] = watchdog.pid
        for _ in range(30):
            ack = control.ROOT / f"watchdog-{args.revision}.json"
            if ack.exists() and json.loads(ack.read_text()).get("ready"):
                break
            time.sleep(1)
        else:
            raise RuntimeError("Independent restore watchdog did not arm")
        final = guard.activity()
        if final["busy"] or final["counters"] != previous:
            raise RuntimeError("Production became active; leave originals running")
        state.update(service_paused=True, restore_deadline_epoch=time.time() + 900)
        save("pausing_glm")
        for rank in (0, 1):
            control.run(rank, ["docker", "stop", "--timeout", "30", control.ORIGINAL_IDS[rank]],
                        timeout=90)
        memory_code = ("from pathlib import Path;print(next(int(x.split()[1])*1024 "
                       "for x in Path('/proc/meminfo').read_text().splitlines() "
                       "if x.startswith('MemAvailable:')))")
        for _ in range(30):
            available = int(control.run(1, ["python3", "-S", "-c", memory_code]))
            if available >= 30 * 1024**3:
                break
            time.sleep(2)
        else:
            raise RuntimeError("Original GPU contexts did not release their memory")
        stages = ["smoke"] if args.correctness_only else ["smoke", "routing" if args.routing else "timing"]
        reports = {}
        for stage in stages:
            argv = command(root, args.revision, stage, build_directory=args.build_directory,
                           correctness_only=args.correctness_only)
            name = argv[argv.index("--name") + 1]
            state["stages"].append({"stage": stage, "container": name,
                                    "started_epoch": time.time()})
            save("starting_" + stage)
            control.run(1, argv)
            deadline = time.monotonic() + 205
            while time.monotonic() < deadline:
                info = control.inspect(1, name)
                if not info["State"]["Running"]:
                    break
                time.sleep(1)
            else:
                raise TimeoutError(f"{stage} exceeded its process limit")
            log_code = ("import subprocess,sys;p=subprocess.run(['docker','logs',sys.argv[1]],"
                        "stdout=subprocess.PIPE,stderr=subprocess.STDOUT);"
                        "sys.stdout.buffer.write(p.stdout)")
            (root / (stage + ".log")).write_text(control.run(
                1, ["python3", "-S", "-c", log_code, name]))
            result = control.remote_json(1, root / f"results/{stage}.json")
            control.write(root / (stage + ".json"), result)
            reports[stage] = result
            state["stages"][-1].update(exit_code=info["State"]["ExitCode"],
                result_status=result["status"], finished_epoch=time.time())
            if info["State"]["ExitCode"] or (stage == "smoke" and result["status"] != "passed"):
                raise RuntimeError(f"{stage} failed; no further GPU stage is allowed")
            if args.correctness_only and (result.get("wrapped_bases_passed") != 2
                                          or not result.get("full_bf16_word_space_tested")):
                raise RuntimeError("Wrapped-base coverage is incomplete")
            if stage != "smoke" and (result["status"] != "complete"
                                         or not result["correctness_passed"]):
                raise RuntimeError("Real fixture correctness failed")
            if stage == "timing" and result["performance_passed"]:
                stages.append("confirmation")
            save("finished_" + stage)
        winners = []
        if "confirmation" in reports:
            winners = [name for name, value in reports["timing"]["candidates"].items()
                       if value["passed"] and reports["confirmation"]["candidates"][name]["passed"]]
        state.update(confirmed_component_winners=winners, quality_qualified=False,
                     serving_adapter_changed=False)
        success = True
        save("measured_pending_restoration")
    except BaseException as exc:
        state["error"] = repr(exc)
        save("failed_pending_restoration" if state["service_paused"] else "preparation_failed")
        raise
    finally:
        if state["service_paused"]:
            save("restoring")
            state["restoration"] = control.restore(args.revision, path)
            state["service_paused"] = False
            save("complete" if success else "failed_restored")
        lock.close()


if __name__ == "__main__":
    main()
