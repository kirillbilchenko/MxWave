"""One authorized idle pause, two matched model arms, independent automatic GLM restoration."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import full_model_control as c


def validate_decode_control(previous: dict[str, Any], current: dict[str, Any],
                            folder: Path, protocol: dict[str, Any]) -> dict[str, str]:
    """Reuse a completed stock load only when identity, prompts and retained bytes still match."""
    c.validate_transport_control(previous["launches"]["stock"],
        current["launches"].get("stock") or current["launches"]["kda"])
    if (previous["phase"] not in ("complete", "failed_restored")
            or previous.get("service_paused")
            or previous.get("restoration", {}).get("status") != "complete"
            or not any(arm["mode"] == "stock" and arm.get("status") == "complete"
                       for arm in previous["arms"])
            or previous["originals"] != current["originals"]
            or any(previous["preflight"][key] != current["preflight"][key]
                   for key in ("source_metadata", "libraries"))
            or json.loads((folder / "protocol.json").read_bytes()) != protocol):
        raise ValueError("Saved actual-decode control differs in identity, source or protocol")
    diagnostic = json.loads((folder / "combined-diagnostic/model-diagnostic.json").read_bytes())
    if (diagnostic["status"] != "complete" or len(diagnostic["phases"]) != 3
            or not all(phase["sanity_passed"] for phase in diagnostic["phases"])
            or len(diagnostic["loaded_weight_checks"]) != 2
            or not all(row["status"] == "passed" for row in diagnostic["loaded_weight_checks"])):
        raise ValueError("Saved actual-decode control did not pass the source/chat gate")
    hashes = {}
    for label in ("stock", "stock-repeat"):
        report_path = folder / (label + "-decode-quality.json")
        report = json.loads(report_path.read_bytes())
        if (report["status"] != "complete" or report["vocabulary"] != 154880
                or report["generated_tokens"] != 4 or len(report["contexts"]) != 16):
            raise ValueError("Saved actual-decode control is incomplete")
        for index, (context, ids) in enumerate(zip(report["contexts"], protocol["quality"],
                                                  strict=True)):
            prompt = protocol["quality_prefix_token_ids"] + ids
            digest = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
            if (context["index"] != index or context["prompt_sha256"] != digest
                    or len(context["steps"]) != 4 or len(context["token_ids"]) != 4):
                raise ValueError("Saved actual-decode conditioning differs")
            for position, step in enumerate(context["steps"]):
                if (Path(step["file"]).name != step["file"] or step["position"] != position
                        or step["kind"] != ("decode" if position else "prefill")
                        or step["conditioning_token_ids"] != context["token_ids"][:position]):
                    raise ValueError("Saved actual-decode step differs")
                raw = (folder / step["file"]).read_bytes()
                if len(raw) != 154880 * 4 or hashlib.sha256(raw).hexdigest() != step["sha256"]:
                    raise ValueError("Saved actual-decode distribution changed")
        hashes[label] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    return hashes


def validate_moe_dispatch_fixture(
    fixture: dict[str, Any], reports: list[dict[str, Any]], bound: dict[str, Any],
) -> None:
    """Require both real installed-MoE comparisons against the proposed code and libraries."""
    if (fixture.get("phase") != "complete_idle" or not fixture.get("dispatch_comparison")
            or len(reports) != 2 or {r.get("mode") for r in reports} != {"raw", "aligned"}):
        raise ValueError("Finish the raw/aligned paired dispatch fixture first")
    for proof in reports:
        if proof.get("status") != "complete":
            raise ValueError("The paired dispatch fixture did not complete")
        for name in ("model_plugin.py", "moe_runtime.py"):
            if proof["code_sha256"][name] != bound["code"][name]:
                raise ValueError("MoE execution changed after paired GPU qualification")
        for variant in ("RAW", "ALIGNED"):
            name = f"libmxwave_moe_{variant.lower()}.so"
            if proof["library_sha256"][f"MXWAVE_MOE_{variant}_LIBRARY"] != bound["libraries"][name]:
                raise ValueError("The paired dispatch fixture used different native libraries")
        cases = proof.get("cases", [])
        expected = {(m, concentrated) for m in (1, 4, 8, 64, 128)
                    for concentrated in (False, True)}
        if (len(cases) != len(expected)
                or {(r["tokens"], r["concentrated"]) for r in cases} != expected
                or not all(r.get("stock_eager_stream_graph_bits_exact")
                    and r.get("dispatch_modes") == ["custom", "dispatcher"] for r in cases)):
            raise ValueError("Require exact eager/stream/graph outputs in all twenty dispatch cases")


def main() -> None:
    """Qualify frozen sources/CPU dispatch before pausing; restore original IDs on every exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--fixture-revision", default="kdaadapterv3")
    parser.add_argument("--fused-max-tokens", type=int, choices=(1, 8), default=1)
    parser.add_argument("--reuse-stock-run")
    parser.add_argument("--reuse-decode-run")
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--vocab-fixture-revision", default="vocabfixturev2")
    parser.add_argument("--moe-dispatch", choices=("custom", "dispatcher"), default="custom")
    parser.add_argument("--moe-fixture-revision")
    args = parser.parse_args()
    rev = args.revision
    if args.combined and (not args.reuse_stock_run or args.fused_max_tokens != 8):
        raise ValueError("The combined pilot requires the saved stock control and KDA limit eight")
    if args.reuse_decode_run and not args.combined:
        raise ValueError("Actual decode control reuse requires the combined pilot")
    if args.moe_dispatch != "custom" and (not args.combined or not args.moe_fixture_revision):
        raise ValueError("A changed MoE dispatch needs the combined pilot and paired GPU fixture")
    if (not rev.isalnum() or not args.fixture_revision.isalnum()
            or args.reuse_stock_run is not None and not args.reuse_stock_run.isalnum()
            or args.reuse_decode_run is not None and not args.reuse_decode_run.isalnum()
            or not args.vocab_fixture_revision.isalnum()
            or args.moe_fixture_revision is not None and not args.moe_fixture_revision.isalnum()):
        raise ValueError("Use an alphanumeric immutable revision")
    code = c.ROOT / f"full-{rev}-src"
    path = c.ROOT / f"full-controller-{rev}.json"
    if path.exists():
        raise FileExistsError("Preserve prior full-model state")
    lock = (c.ROOT / "diagnostics-v1/component.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state: dict[str, Any] = {"phase": "preparing", "controller_pid": os.getpid(),
        "started_epoch": time.time(), "service_paused": False, "arms": [],
        "purpose": "Same imported checkpoint, corrected attention, stock versus selective KDA"}

    def save(phase: str) -> None:
        state.update(phase=phase, updated_epoch=time.time())
        c.write(path, state)

    def terminate(signum: int, frame: Any) -> None:
        raise SystemExit(f"Signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    save("preparing")
    success = False
    launches: dict[str, Any] = {}
    modes = ("kda",) if args.reuse_stock_run and (
        not args.combined or args.reuse_decode_run) else ("stock", "kda")
    try:
        fixture = c.remote_json(0, c.ROOT /
                                f"diagnostics-v1/controller-{args.fixture_revision}.json")
        proof = fixture["attempts"][-1]["report"]
        if fixture["phase"] != "complete_idle" or not proof["passed"] or len(proof["cases"]) != 12:
            raise ValueError("Finish all twelve real-vLLM TP-slice fixtures first")
        if proof.get("fused_max_tokens", 1) != args.fused_max_tokens:
            raise ValueError("The requested fused token limit differs from its GPU qualification")
        critical = ["kda_policy.py", "kda_runtime.py", "kda_adapter.py"]
        if not args.combined:
            critical.append("model_plugin.py")
        for name in critical:
            if hashlib.sha256((code / name).read_bytes()).hexdigest() != proof["code_sha256"][name]:
                raise ValueError("Adapter code differs from the passed GPU fixture")
        state["gpu_fixture"] = {name: proof[name] for name in ("code_sha256", "gates", "seconds")}
        if args.combined:
            vocabulary_fixture = c.remote_json(0, c.ROOT /
                f"diagnostics-v1/controller-{args.vocab_fixture_revision}.json")
            vocab_proof = vocabulary_fixture["attempts"][-1]["report"]
            if (vocabulary_fixture["phase"] != "complete_idle" or vocab_proof["status"] != "passed"
                    or len(vocab_proof["cases"]) != 4
                    or len(vocab_proof["combined_kda_selected_prefixes"]) != 68):
                raise ValueError("Finish all four full vocabulary-shard GPU fixtures first")
            for name, expected in vocab_proof["code_sha256"].items():
                if hashlib.sha256((code / name).read_bytes()).hexdigest() != expected:
                    raise ValueError("Combined adapter differs from its vocabulary GPU fixture")
            for name, expected in vocab_proof["core_sha256"].items():
                if hashlib.sha256((code / "mxwave" / name).read_bytes()).hexdigest() != expected:
                    raise ValueError("Row codec differs from its complete GPU source-hash fixture")
            manifest = c.ROOT / "vocab-v2/prepared.json"
            manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            if manifest_sha != vocab_proof["manifest_sha256"]:
                raise ValueError("Vocabulary artifacts changed after their GPU qualification")
            for rank in (0, 1):
                actual = c.run(rank, ["sha256sum", str(manifest)]).split()[0]
                if actual != manifest_sha:
                    raise ValueError("Vocabulary manifest differs between nodes")
            state.update(vocabulary_fixture=vocab_proof, combined=True,
                purpose="Combined expert scales, KDA, head and embedding; actual decode quality",
                stock_measurement="Reuse saved timing; fresh actual decode control only")
        bound = [c.preflight(rank, code) for rank in (0, 1)]
        if bound[0] != bound[1]:
            raise ValueError("Source, native library or code differs between Sparks")
        state["preflight"] = bound[0]
        if args.moe_fixture_revision:
            moe_fixture = c.remote_json(0, c.ROOT /
                f"plugin-fixture-controller-{args.moe_fixture_revision}.json")
            accepted = [r for r in moe_fixture["runs"] if r.get("accepted")]
            moe_proofs = [c.remote_json(1, Path(r["results"]) / "report.json") for r in accepted]
            validate_moe_dispatch_fixture(moe_fixture, moe_proofs, bound[0])
            state["moe_fixture"] = {"revision": args.moe_fixture_revision,
                "reports": moe_proofs, "selected_dispatch": args.moe_dispatch}
        originals = [c.inspect(rank, c.ORIGINAL_IDS[rank]) for rank in (0, 1)]
        if any(o["Image"] != c.IMAGE or not o["State"]["Running"] for o in originals):
            raise ValueError("The authorized original GLM identities or state changed")
        state["originals"] = [c.identity(o) for o in originals]
        for mode in modes:
            launches[mode] = [c.engine(rank, mode, rev, originals[rank], bound[0]["vocabulary"],
                                      attention_mapping=True) for rank in (0, 1)]
            for argv, meta in launches[mode]:
                if mode == "kda":
                    key = "MXWAVE_GLM53_KDA_FUSED_MAX_TOKENS"
                    meta["environment"][key] = str(args.fused_max_tokens)
                    argv[argv.index("--entrypoint"):argv.index("--entrypoint")] = [
                        "-e", f"{key}={args.fused_max_tokens}"]
                for command in (argv, meta["command"]):
                    command[command.index("--worker-extension-cls") + 1] = (
                        "combined_worker.CombinedWorker" if args.combined else "kda_worker.KdaWorker")
                    command[command.index("--compilation-config") + 1] = json.dumps({
                        "mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                        "cudagraph_capture_sizes": [1], "max_cudagraph_capture_size": 1})
                if args.combined and mode == "kda":
                    argv[argv.index("MXWAVE_MOE_MODE=stock")] = "MXWAVE_MOE_MODE=aligned"
                    meta["environment"]["MXWAVE_MOE_MODE"] = "aligned"
                    extra = {"MXWAVE_GLM53_VOCAB_MODE": "packed",
                        "MXWAVE_MOE_DISPATCH": args.moe_dispatch,
                        "MXWAVE_GLM53_VOCAB_ARTIFACTS": "/audit/full-model-v1/vocab-v2",
                        "MXWAVE_GLM53_VOCAB_MANIFEST_SHA256": manifest_sha}
                    meta["environment"].update(extra)
                    argv[argv.index("--entrypoint"):argv.index("--entrypoint")] = [
                        token for key, value in extra.items() for token in ("-e", key + "=" + value)]
        state["launches"] = {mode: [value[1] for value in values]
                             for mode, values in launches.items()}
        results = c.ROOT / f"run-{rev}/results"
        results.mkdir(exist_ok=False)
        proto_source = c.ROOT / "run-v4/results/protocol.json"
        proto = json.loads(proto_source.read_bytes())
        proto["timing"] = [row for row in proto["timing"]
                           if (row["prompt_tokens"], row["batch"]) in
                           ((256, 1), (2048, 1), (256, 4))]
        if len(proto["timing"]) != 3 or len(proto["quality"]) != 16:
            raise ValueError("Unexpected frozen protocol cases")
        proto.update(quality_prefix_token_ids=[154822, 154824], profile_case_indices=[0, 1, 2],
                     profile_generated_tokens=8,
                     strict_control_stability_gate={"mean_kl": 1e-5, "absolute_mean_nll": 1e-4},
                     graph_capture_sizes=[1])
        for mode in ("stock", "kda"):
            folder = results / mode
            if mode == "stock" and args.reuse_stock_run:
                previous = c.ROOT / f"run-{args.reuse_stock_run}"
                old = c.remote_json(0, c.ROOT / f"full-controller-{args.reuse_stock_run}.json")
                c.validate_transport_control(old["launches"]["stock"],
                    state["launches"].get("stock") or state["launches"]["kda"])
                baseline = c.remote_json(0, previous / "results/stock/stock-benchmark.json")
                old_command = list(old["launches"]["stock"][0]["command"])
                if args.combined:
                    old_command[old_command.index("--worker-extension-cls") + 1] = (
                        "combined_worker.CombinedWorker")
                if (old["phase"] != "complete" or baseline["status"] != "complete"
                        or old["originals"] != state["originals"]
                        or old["preflight"]["source_metadata"] != bound[0]["source_metadata"]
                        or old["preflight"]["libraries"] != bound[0]["libraries"]
                        or old_command !=
                        launches["kda"][0][1]["command"]):
                    raise ValueError("The saved stock arm differs in source, runtime or settings")
                if json.loads((previous / "results/stock/protocol.json").read_bytes()) != proto:
                    raise ValueError("The saved stock arm used a different protocol")
                state["reused_stock"] = {"run": args.reuse_stock_run,
                    "benchmark_sha256": hashlib.sha256(
                        (previous / "results/stock/stock-benchmark.json").read_bytes()).hexdigest(),
                    "started_epoch": baseline["started_epoch"],
                    "limitation": "Saved sequential control; not a contemporaneous interleaved arm"}
                shutil.copytree(previous / "results/stock", folder)
                if args.reuse_decode_run:
                    prior_root = c.ROOT / f"run-{args.reuse_decode_run}"
                    prior = c.remote_json(0, c.ROOT /
                                          f"full-controller-{args.reuse_decode_run}.json")
                    hashes = validate_decode_control(prior, state, prior_root / "results/stock", proto)
                    shutil.copytree(prior_root / "results/stock", folder, dirs_exist_ok=True)
                    state["reused_decode"] = {"run": args.reuse_decode_run,
                        "report_sha256": hashes,
                        "limitation": "Saved matched control, not a fresh or interleaved load"}
                    state["stock_measurement"] = "Reuse saved timing and completed actual-decode control"
                for name in ("traces", "traces-worker"):
                    shutil.copytree(previous / "stock" / name,
                                    results.parent / "stock" / name, dirs_exist_ok=True)
            else:
                folder.mkdir()
                c.write(folder / "protocol.json", proto)
        state["protocol_sha256"] = hashlib.sha256(
            (results / "stock/protocol.json").read_bytes()).hexdigest()
        state["protocol_source_sha256"] = hashlib.sha256(proto_source.read_bytes()).hexdigest()
        cpu_result = c.ROOT / f"run-{rev}/cpu"
        cpu_result.mkdir()
        mounts = []
        for source, target, access in launches["kda"][0][1]["mounts"]:
            if target == "/overlay" or target.startswith(
                    "/usr/local/lib/python3.12/dist-packages/vllm/"):
                mounts.extend(["-v", f"{source}:{target}:{access}"])
        cpu = ["docker", "run", "--rm", "--network", "none", "--cpus", "2",
            "--memory", "2g", "--memory-swap", "2g", "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{code}:/src:ro", "-v", f"{c.ROOT / 'source'}:/model:ro",
            "-v", f"{cpu_result}:/result", *mounts,
            "-e", "CUDA_VISIBLE_DEVICES=", "-e", "NVIDIA_VISIBLE_DEVICES=void",
            "-e", "PYTHONPATH=/src:/overlay/overlay",
            "-e", "MXWAVE_GLM53_ATTENTION_MAPPING=1", "--entrypoint", "/usr/bin/python3", c.IMAGE]
        state["cpu_header_dispatch"] = c.run(0, [
            *cpu, "-u", "/src/audit_kda_adapter.py", "--source", "/model",
            "--report", "/result/kda-dispatch.json"], timeout=180)
        audit = json.loads((cpu_result / "kda-dispatch.json").read_bytes())
        if not audit["passed"] or len(audit["targets"]) != 68:
            raise ValueError("CPU header and actual method-selection gate failed")
        cpu_boot = cpu[:cpu.index("--entrypoint")] + [
            "-e", "MXWAVE_VALIDATE_ONLY=1", "-e", "MXWAVE_MOE_MODE=stock",
            "-e", "MXWAVE_GLM53_KDA_MODE=packed", "-e", "MXWAVE_GLM53_KDA_SOURCE=/model",
            "-e", f"MXWAVE_GLM53_KDA_FUSED_MAX_TOKENS={args.fused_max_tokens}",
            *cpu[cpu.index("--entrypoint"):], "-u", "/src/engine_boot.py",
            *launches["kda"][0][1]["command"]]
        # The disposable entrypoint installs metadata into the container's system
        # Python. Keep the read-only audit unprivileged; match the GPU containers'
        # default user only for this isolated installation/CLI check.
        user = cpu_boot.index("--user")
        del cpu_boot[user:user + 2]
        cpu_boot[cpu_boot.index("--entrypoint"):cpu_boot.index("--entrypoint")] = [
            "-e", "UV_CACHE_DIR=/tmp/mxwave-uv-cache"]
        state["cpu_cli_plugin"] = c.run(0, cpu_boot, timeout=180)
        spec = importlib.util.spec_from_file_location(
            "guard", c.OLD / "triple-v2/src/integrations/marlin_scales/quiet_window.py")
        assert spec and spec.loader
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        save("waiting_for_idle")
        deadline, previous, quiet = time.monotonic() + 3600, None, 0
        while time.monotonic() < deadline:
            now = guard.activity()
            quiet = quiet + 1 if not now["busy"] and previous == now["counters"] else 0
            previous = now["counters"]
            if quiet >= 4:
                state["idle_before_pause"] = now
                break
            time.sleep(5)
        else:
            raise TimeoutError("No authorized idle window within one hour")
        watchdog = subprocess.Popen([
            "python3", "-u", str(code / "full_model_control.py"), "--revision", rev,
            "--watch", str(os.getpid())], stdout=(c.ROOT / f"watchdog-{rev}.log").open("xb"),
            stderr=subprocess.STDOUT, start_new_session=True)
        state["watchdog_pid"] = watchdog.pid
        for _ in range(20):
            ack = c.ROOT / f"watchdog-{rev}.json"
            if ack.exists() and json.loads(ack.read_bytes()).get("ready"):
                break
            time.sleep(.5)
        else:
            raise RuntimeError("Independent restoration watchdog did not arm")
        if guard.activity()["busy"]:
            raise RuntimeError("GLM became active before the pause; production remains running")
        state.update(service_paused=True, restore_deadline_epoch=time.time() + 3600)
        save("pausing_service")
        for rank in (0, 1):
            c.run(rank, ["docker", "stop", "--timeout", "30", c.ORIGINAL_IDS[rank]], 90)
        for mode in modes:
            arm: dict[str, Any] = {"mode": mode, "started_epoch": time.time()}
            state["arms"].append(arm)
            save("loading_" + mode)
            for rank in (1, 0):
                c.run(rank, launches[mode][rank][0], 120)
            deadline = time.monotonic() + 1500
            while time.monotonic() < deadline:
                if c.health(18784):
                    break
                for rank in (0, 1):
                    if not c.inspect(rank, launches[mode][rank][1]["name"])["State"]["Running"]:
                        raise RuntimeError(f"{mode} engine exited during startup")
                time.sleep(5)
            else:
                raise TimeoutError(f"{mode} engine startup exceeded 25 minutes")
            arm["ready_epoch"] = time.time()
            save("measuring_" + mode)
            name = f"mxwave-glm53-bench-{rev}-{mode}"
            script = "measure_combined_arm.py" if args.combined else "measure_kda_arm.py"
            command = ["docker", "run", "-d", "--name", name, "--network", "host",
                "--cpus", "2", "--memory", "1536m", "--memory-swap", "1536m",
                "--user", f"{os.getuid()}:{os.getgid()}", "-e", "CUDA_VISIBLE_DEVICES=",
                "-e", "NVIDIA_VISIBLE_DEVICES=void", "-v", f"{code}:/src:ro",
                "-v", f"{results / mode}:/results", "--entrypoint", "/usr/bin/python3",
                c.IMAGE, "-u", "/src/" + script, "--mode", mode,
                "--root", "/results", "--vocabulary", str(bound[0]["vocabulary"])]
            command.extend(["--fused-max-tokens", str(args.fused_max_tokens)])
            if args.combined and mode == "stock" and args.reuse_stock_run:
                command.append("--reuse-timing")
            c.run(0, command)
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                info = c.inspect(0, name)["State"]
                if not info["Running"]:
                    if info["ExitCode"]:
                        raise RuntimeError(f"{mode} correctness/benchmark client failed")
                    arm.update(status="complete", finished_epoch=time.time())
                    break
                time.sleep(5)
            else:
                raise TimeoutError(f"{mode} measurement exceeded twenty minutes")
            c.stop_owned(rev)
            save("measured_" + mode)
        success = True
    except BaseException as exc:
        state["error"] = repr(exc)
        save("failed_pending_restoration" if state["service_paused"] else "preparation_failed")
        raise
    finally:
        if state["service_paused"]:
            save("restoring")
            state["restoration"] = c.restore(rev, path)
            state["service_paused"] = False
            save("restored_analyzing" if success else "failed_restored")
        lock.close()
    if success:
        for mode in modes:
            if mode == "stock" and args.reuse_stock_run:
                continue
            local = c.ROOT / f"run-{rev}" / mode
            destination = local / "traces-worker"
            destination.mkdir()
            c.run(0, ["rsync", "-a", "--timeout=120", "--bwlimit=65536", "-e",
                     shlex.join(c.SSH[:-1]), c.SSH[-1] + ":" + str(local / "traces") + "/",
                     str(destination) + "/"], 600)
        script = "compare_combined_model.py" if args.combined else "compare_kda_model.py"
        command = ["docker", "run", "--rm", "--network", "none", "--cpus", "2",
            "--memory", "2g", "--memory-swap", "2g", "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "CUDA_VISIBLE_DEVICES=", "-e", "NVIDIA_VISIBLE_DEVICES=void",
            "-v", f"{code}:/src:ro", "-v", f"{c.ROOT / ('run-' + rev)}:/run",
            "--entrypoint", "/usr/bin/python3", c.IMAGE,
            "/src/" + script, "--root", "/run/results"]
        state["analysis"] = c.run(0, command, 600)
        save("complete")


if __name__ == "__main__":
    main()
