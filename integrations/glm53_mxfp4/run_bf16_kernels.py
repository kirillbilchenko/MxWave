"""Run one bounded BF16 timing/NCU component session and restore both original GLM ranks."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import full_model_control as control

NCU = Path("/opt/nvidia/nsight-compute/2025.3.1")
SECTIONS = ("SpeedOfLight", "LaunchStats", "Occupancy", "SchedulerStats", "WarpStateStats",
            "MemoryWorkloadAnalysis", "InstructionStats")


def memory_available(rank: int) -> int:
    """Read free unified memory without importing a GPU-dependent helper."""
    script = ("from pathlib import Path; print(next(int(x.split()[1])*1024 "
              "for x in Path('/proc/meminfo').read_text().splitlines() "
              "if x.startswith('MemAvailable:')))")
    return int(control.run(rank, ["python3", "-S", "-c", script]).strip())


def command(revision: str, root: Path, *, profile: bool, cache: bool = False,
            n8: bool = False, confirmation: bool = False,
            candidates: tuple[str, ...] = (), sparse: bool = False,
            sparse_diagnostic: bool = False, sparse_ncu: bool = False) -> list[str]:
    """Use the pinned image and read-only source; enable counter access only in the NCU job."""
    if cache and profile:
        raise ValueError("Cache-pressure timings must not run under a profiler")
    if (n8 and (profile or cache)) or (confirmation and not (n8 or sparse)):
        raise ValueError("Native N8 sessions require separate unprofiled timing/confirmation arms")
    if sparse and (profile or cache or n8 or candidates):
        raise ValueError("Sparse scales require their own unprofiled component session")
    if sparse_diagnostic and (not sparse or confirmation):
        raise ValueError("Sparse diagnostics require their own single-process session")
    if sparse_ncu and (not sparse or sparse_diagnostic):
        raise ValueError("Sparse NCU requires a separate counter session")
    if candidates and (not n8 or len(set(candidates)) != len(candidates)
                       or any(c not in ("compact", "narrow", "compact_x4", "tailfirst",
                                        "compact_tailfirst") for c in candidates)):
        raise ValueError("Require distinct recognized N8 mechanism candidates")
    stage = ("n8-confirm" if confirmation else "n8") if n8 else (
        "cache" if cache else ("profile" if profile else "timing"))
    if sparse:
        stage = "sparse-confirm" if confirmation else "sparse"
    name = f"mxwave-glm53-full-{revision}-{'aligned' if profile or confirmation else 'raw'}-r1"
    argv = ["docker", "run", "-d", "--name", name, "--gpus", "all", "--network", "none",
        "--cpus", "4", "--pids-limit", "256", "--memory", "8g" if sparse else "6g",
        "--memory-swap", "8g" if sparse else "6g",
        "-v", f"{control.ROOT}:/audit:ro", "-v", f"{root / 'results'}:/result",
        "-v", f"{control.OLD}:/experiment:ro", "-v", f"{root / 'src'}:/work:ro",
        "-e", "PYTHONPATH=/work:/experiment/triple-v2/src/integrations/zipserv",
        "-e", "MXWAVE_ZIPSERV_LIBRARY=/audit/native/libmxwave_kda_zipserv.so",
        "-e", "OMP_NUM_THREADS=2", "-e", "MKL_NUM_THREADS=2", "-e", "OPENBLAS_NUM_THREADS=2",
        "-e", "XDG_CACHE_HOME=/result/cache"]
    if sparse:
        argv.extend(["-v", str(control.ROOT.parent / "grouped-moe-full-layer-v1/fixture")
                     + ":/fixtures:ro"])
    if sparse_diagnostic:
        argv.extend(["-e", "CUDA_LAUNCH_BLOCKING=1"])
    if profile or sparse_ncu:
        # Counter access is restricted by the host driver. No global driver setting changes.
        argv.extend(["--cap-add", "SYS_ADMIN", "-v", f"{NCU}:/ncu:ro"])
    argv.extend(["--entrypoint", "/usr/bin/timeout", control.IMAGE,
                 "--signal=TERM", "--kill-after=10", "180" if cache else "600"])
    if profile or sparse_ncu:
        prefix = stage if sparse_ncu else "kernels"
        argv.extend(["/ncu/ncu", "--profile-from-start", "off", "--nvtx",
            "--clock-control", "none", "--cache-control", "all", "--launch-count", "80",
            "--export", f"/result/{prefix}", "--csv", "--page", "raw",
            "--log-file", f"/result/{stage}-ncu.csv" if sparse_ncu else "/result/ncu.csv"])
        for section in SECTIONS:
            argv.extend(["--section", section])
    script = ("benchmark_bf16_n8_matrix.py" if candidates else "benchmark_bf16_n8.py") if n8 else (
        "diagnose_bf16_cache.py" if cache else "benchmark_bf16_kernels.py")
    if sparse:
        script = "profile_sparse_scales.py" if sparse_ncu else "benchmark_sparse_scales.py"
    argv.extend(["/usr/bin/python3", "-u", "/work/" + script,
                 "--report", f"/result/{stage}.json"])
    if cache:
        argv.extend(["--source", "/audit/source", "--glm-paused"])
    if profile:
        argv.append("--profile")
    if n8:
        argv.extend(["--build", "/audit/" + root.name + "/build"])
        if candidates:
            argv.extend(["--candidates", *candidates])
        if confirmation:
            argv.append("--confirmation")
    if sparse:
        argv.extend(["--build", "/audit/" + root.name + "/build", "--fixture", "/fixtures",
                     "--prior", "/audit/scale-sparse-cpu-v2/results/screen.json"])
        if confirmation:
            argv.append("--confirmation")
        if sparse_diagnostic:
            argv.append("--diagnostic")
    return argv


def main() -> None:
    """Prepare before the idle pause and arm restoration independently of the laptop/session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--pause-glm", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--n8-only", action="store_true")
    parser.add_argument("--n8-candidates", nargs="+", default=[])
    parser.add_argument("--scale-sparse-only", action="store_true")
    parser.add_argument("--scale-sparse-diagnostic", action="store_true")
    parser.add_argument("--scale-sparse-ncu", action="store_true")
    parser.add_argument("--watch", type=int)
    args = parser.parse_args()
    if not args.revision.isalnum():
        raise ValueError("Require a new alphanumeric evidence revision")
    root = control.ROOT / f"bf16-kernels-{args.revision}"
    path = control.ROOT / f"bf16-kernels-controller-{args.revision}.json"
    if args.watch is not None:
        control.watch(args.revision, args.watch, path)
        return
    if not args.pause_glm:
        raise ValueError("GPU session needs the previously authorized idle GLM pause")
    if sum((args.cache_only, args.n8_only, args.scale_sparse_only)) > 1:
        raise ValueError("Select one component protocol")
    if args.scale_sparse_diagnostic and not args.scale_sparse_only:
        raise ValueError("Sparse diagnostics require --scale-sparse-only")
    if args.scale_sparse_ncu and (not args.scale_sparse_only or args.scale_sparse_diagnostic):
        raise ValueError("Sparse NCU requires --scale-sparse-only without diagnostics")
    candidates = tuple(args.n8_candidates)
    if candidates:
        # Validate argv before any production pause or watchdog launch.
        command(args.revision, root, profile=False, n8=args.n8_only, candidates=candidates)
    if path.exists():
        raise FileExistsError("Preserve prior controller evidence")
    lock = (control.ROOT / "diagnostics-v1/component.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state: dict[str, Any] = {"phase": "preparing", "service_paused": False,
        "started_epoch": time.time(), "model_weights_loaded": False,
        "scope": ("NCU counters for four selected scale GEMMs per arm; no model load" if args.scale_sparse_ncu else
                  "M1 scale launch diagnostic; no performance qualification" if args.scale_sparse_diagnostic else
                  "Complete 288-expert scale comparison with fresh confirmation" if args.scale_sparse_only else
                  "Fourteen N8 fixtures plus a fresh confirmation process" if args.n8_only else
                  "One BF16 output cache-pressure fixture" if args.cache_only else
                  "Three BF16 component fixtures; no full-model reload or adapter change")}
    success = False

    def save(phase: str) -> None:
        state.update(phase=phase, updated_epoch=time.time())
        control.write(path, state)

    def terminated(signum: int, frame: Any) -> None:
        raise SystemExit(f"Signal {signum}")

    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    try:
        save("preparing")
        originals = [control.inspect(rank, control.ORIGINAL_IDS[rank]) for rank in (0, 1)]
        if any(not row["State"]["Running"] or row["Image"] != control.IMAGE for row in originals):
            raise RuntimeError("Require both original GLM ranks running before the component")
        state["originals"] = [control.identity(row) for row in originals]
        state["preflight"] = control.preflight(1, root / "src")
        state["controller_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        # The large artifacts are hash-checked before stopping the production service.
        check = """import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]); out={}
for directory in ('bf16-attention-v1/artifacts-v2','vocab-v2'):
 p=r/directory/'prepared.json'; data=json.loads(p.read_bytes())
 assert data['status']=='packed_gpu_verification_pending'
 out[directory]={'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'artifacts':{}}
 for row in data['cases']:
  if (row.get('name','').startswith('kda-') or (row.get('kind'),row.get('rank'))==('head',0)):
   file=p.parent/row['artifact']; h=hashlib.sha256()
   assert file.name==row['artifact']
   with file.open('rb') as f:
    for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
   assert h.hexdigest()==row['artifact_sha256']
   out[directory]['artifacts'][file.name]=h.hexdigest()
assert Path('/opt/nvidia/nsight-compute/2025.3.1/ncu').is_file()
print(json.dumps(out))
"""
        if not args.scale_sparse_only:
            state["fixtures_preflight"] = json.loads(control.run(
                1, ["python3", "-S", "-c", check, str(control.ROOT)]))
        else:
            if args.scale_sparse_ncu:
                control.run(1, ["test", "-x", str(NCU / "ncu")])
            check_sparse = """import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]); root=Path(sys.argv[2]); code=r/'src'; b=r/'build'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
manifest=json.loads((b/'manifest.json').read_text())
assert set(manifest['variants'])=={'aligned','sparse'}
for name,value in manifest['libraries_sha256'].items():
 assert Path(name).name==name and digest(b/'build'/name)==value
prior=root/'scale-sparse-cpu-v2/results/screen.json'; cpu=json.loads(prior.read_text())
assert cpu['status']=='passed' and len(cpu['cases'])==4
assert cpu['format']=='mxwave-palette16-v2'
assert cpu['codec_sha256']==digest(code/'scale_tiles.py')
assert all(c['cpu_roundtrip_exact'] and c['incremental_saving']>=.20 for c in cpu['cases'])
f=root.parent/'grouped-moe-full-layer-v1/fixture'; p=f/'manifest.json'
fixture=json.loads(p.read_text()); assert fixture['expert_ids']==list(range(288))
for item in fixture['tensors']:
 assert Path(item['path']).name==item['path'] and digest(f/item['path'])==item['sha256']
print(json.dumps({'build_manifest_sha256':digest(b/'manifest.json'),
 'fixture_manifest_sha256':digest(p),'prior_sha256':digest(prior),
 'libraries':manifest['libraries_sha256'],'fixture_tensors':len(fixture['tensors'])}))
"""
            state["sparse_preflight"] = json.loads(control.run(
                1, ["python3", "-S", "-c", check_sparse, str(root), str(control.ROOT)], timeout=180))
        if args.n8_only:
            check_n8 = """import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]); code=r/'src'; b=r/'build'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
manifest=json.loads((b/'native-build.json').read_text())
assert manifest['status']=='built_pending_gpu_qualification'
assert manifest['builder_sha256']==digest(code/'build_bf16_n8.py')
arms={a['arm'] for a in manifest['arms']}
requested=set(sys.argv[3:]) or {'compact'}
assert len(arms)==len(manifest['arms']) and arms==requested|{'control'}
for a in manifest['arms']:
 assert Path(a['library']).name==a['library']
 assert digest(b/a['library'])==a['library_sha256']
fixtures=json.loads((b/'fixtures/prepared.json').read_text())
assert fixtures['status']=='prepared_pending_gpu_qualification' and len(fixtures['cases'])==14
assert fixtures['code_sha256']==digest(code/'prepare_bf16_n8.py')
for row in fixtures['cases']:
 p=Path(row['artifact_path']); assert str(p).startswith('/audit/') and '..' not in p.parts
 actual=Path(sys.argv[2])/str(p)[len('/audit/'):]
 assert digest(actual)==row['artifact_sha256']
print(json.dumps({'native_manifest_sha256':digest(b/'native-build.json'),
 'fixture_manifest_sha256':digest(b/'fixtures/prepared.json'),'fixture_count':14,
 'libraries':{a['arm']:a['library_sha256'] for a in manifest['arms']}}))
"""
            state["n8_preflight"] = json.loads(control.run(
                1, ["python3", "-S", "-c", check_n8, str(root), str(control.ROOT), *candidates],
                timeout=120))
        spec = importlib.util.spec_from_file_location("quiet_guard", control.OLD /
            "triple-v2/src/integrations/marlin_scales/quiet_window.py")
        assert spec and spec.loader
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        previous, quiet = None, 0
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            now = guard.activity()
            quiet = quiet + 1 if not now["busy"] and now["counters"] == previous else 0
            previous = now["counters"]
            state["last_activity"] = now
            save("waiting_for_idle")
            if quiet >= 3:
                final = guard.activity()
                if not final["busy"] and final["counters"] == previous:
                    break
            time.sleep(5)
        else:
            raise TimeoutError("No idle window within thirty minutes")
        watchdog = subprocess.Popen(["python3", "-u", str(Path(__file__).resolve()),
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
            raise RuntimeError("Independent restoration watchdog did not arm")
        state.update(service_paused=True, restore_deadline_epoch=time.time() + 1500)
        save("pausing_glm")
        for rank in (0, 1):
            control.run(rank, ["docker", "stop", "--timeout", "30", control.ORIGINAL_IDS[rank]],
                        timeout=90)
        for _ in range(30):
            if memory_available(1) >= 30 * 1024**3:
                break
            time.sleep(2)
        else:
            raise RuntimeError("GLM contexts did not release enough unified memory")
        stages = ([(False, False)] if args.scale_sparse_diagnostic else
                  [(False, False), (False, True)] if args.n8_only or args.scale_sparse_only else
                  [(False, False)] if args.cache_only else [(False, False), (True, False)])
        for profile, confirmation in stages:
            stage = ("n8-confirm" if confirmation else "n8") if args.n8_only else (
                "cache" if args.cache_only else ("profile" if profile else "timing"))
            if args.scale_sparse_only:
                stage = "sparse-confirm" if confirmation else "sparse"
            argv = command(args.revision, root, profile=profile, cache=args.cache_only,
                           n8=args.n8_only, confirmation=confirmation, candidates=candidates,
                           sparse=args.scale_sparse_only, sparse_diagnostic=args.scale_sparse_diagnostic,
                           sparse_ncu=args.scale_sparse_ncu)
            name = argv[argv.index("--name") + 1]
            control.run(1, ["mkdir", "-p", str(root / "results")])
            state[stage + "_container"] = name
            save("starting_" + stage)
            control.run(1, argv)
            deadline = time.monotonic() + 625
            while time.monotonic() < deadline:
                container = control.inspect(1, name)
                if not container["State"]["Running"]:
                    state[stage + "_state"] = container["State"]
                    capture_logs = ("import subprocess,sys; r=subprocess.run(['docker','logs',"
                                    "sys.argv[1]],stdout=subprocess.PIPE,stderr=subprocess.STDOUT); "
                                    "sys.stdout.buffer.write(r.stdout)")
                    logs = control.run(1, ["python3", "-S", "-c", capture_logs, name])
                    (root / f"{stage}.log").write_text(logs)
                    if container["State"]["ExitCode"]:
                        raise RuntimeError(f"{stage} component exited {container['State']['ExitCode']}")
                    result = control.remote_json(1, root / f"results/{stage}.json")
                    control.write(root / f"{stage}.json", result)
                    if result.get("status") != "passed":
                        raise RuntimeError(f"{stage} component did not pass its gates")
                    state[stage + "_completed"] = True
                    save("finished_" + stage)
                    break
                time.sleep(2)
            else:
                raise TimeoutError(f"{stage} component exceeded ten minutes")
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
