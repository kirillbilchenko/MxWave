"""Consolidate a restored two-rank capture with a hash-checked bulk transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import tarfile
import time
from pathlib import Path


def retain(root: Path, finalizer_pid: int) -> None:
    """Replace only the owned post-restoration file copier, preserving its original source."""
    state = json.loads((root / "controller.json").read_bytes())
    if state["phase"] != "complete_restored" or state["service_paused"]:
        raise ValueError("Production must already be restored")
    if (root / "archive-receipt.json").exists():
        raise FileExistsError("Evidence is already finalized")
    cmdline = (Path("/proc") / str(finalizer_pid) / "cmdline").read_bytes().split(b"\0")
    expected = str(root / "finalize.py").encode()
    if expected not in cmdline:
        raise ValueError("PID is not this capture's original finalizer")
    children = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            stat = (proc / "stat").read_text().split(") ", 1)[1].split()
            if int(stat[1]) == finalizer_pid:
                child_cmd = (proc / "cmdline").read_bytes()
                if b"scp" not in child_cmd or str(root).encode() not in child_cmd:
                    raise ValueError("Unexpected finalizer child")
                children.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    os.kill(finalizer_pid, signal.SIGTERM)
    for pid in children:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    while (Path("/proc") / str(finalizer_pid)).exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Owned finalizer did not exit")
        time.sleep(0.1)
    time.sleep(1)
    original_log = root / "finalize-queue.log"
    if original_log.exists():
        original_log.rename(root / "finalize-per-file-interrupted.log")
    ssh = ["ssh", "-o", "BatchMode=yes", "-i", "/home/kirya/.ssh/id_glm_cluster",
           "kirya@192.168.192.2"]
    archive = root.parent / (root.name + "-rank1-transfer.tar")
    remote_code = """
import pathlib,tarfile,json,hashlib,sys
root=pathlib.Path(sys.argv[1]);archive=pathlib.Path(sys.argv[2])
assert not archive.exists()
paths=sorted(p for p in root.iterdir() if p.is_file()
             and p.suffix in (".json",".gz",".txt",".safetensors"))
with tarfile.open(archive,"w") as t:
    for p in paths:t.add(p,arcname=p.name,recursive=False)
with archive.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
print(json.dumps({"files":len(paths),"bytes":archive.stat().st_size,"sha256":sha}))
"""
    remote = subprocess.check_output(
        [*ssh, shlex.join(["python3", "-S", "-c", remote_code,
                          str(root / "ct-r1"), str(archive)])],
        text=True, timeout=180,
    )
    receipt = json.loads(remote)
    subprocess.run(
        ["scp", "-q", "-i", "/home/kirya/.ssh/id_glm_cluster",
         f"kirya@192.168.192.2:{archive}", str(archive)],
        check=True, timeout=180,
    )
    with archive.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if sha != receipt["sha256"] or archive.stat().st_size != receipt["bytes"]:
        raise ValueError("Bulk transfer integrity failure")
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        if len(members) != receipt["files"]:
            raise ValueError("Bulk archive file count mismatch")
        for member in members:
            if not member.isfile() or Path(member.name).name != member.name:
                raise ValueError("Unexpected archive member")
        tar.extractall(root / "ct-r1", filter="data")
    source = (root / "finalize.py").read_text()
    if source.count("for name in names:") != 1:
        raise ValueError("Unexpected original finalizer")
    # All peer files were transferred above, so retain the existing identity,
    # manifest and archive checks while omitting redundant individual SCP calls.
    replacement = source.replace("for name in names:", "for name in []:")
    (root / "finalize_bulk.py").write_text(replacement)
    receipt.update(
        original_finalizer_pid=finalizer_pid,
        reason="1351 rank-one frames: individual SCP connections made retention unnecessarily slow.",
        scope="Post-restoration evidence transfer only; no model, inference or captured bytes changed.",
        original_finalizer_sha256=hashlib.sha256(source.encode()).hexdigest(),
        replacement_sha256=hashlib.sha256(replacement.encode()).hexdigest(),
        bulk_helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (root / "bulk-transfer-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (root / "finalize-queue.log").open("w") as log:
        subprocess.run(
            ["python3", "-S", "-B", "-u", str(root / "finalize_bulk.py")],
            stdout=log, stderr=subprocess.STDOUT, check=True,
        )


def main() -> None:
    """Consolidate an already-restored capture with explicit copier ownership."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--finalizer-pid", type=int, required=True)
    args = parser.parse_args()
    retain(args.root, args.finalizer_pid)


if __name__ == "__main__":
    main()
