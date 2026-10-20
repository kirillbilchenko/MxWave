"""Prepare a frozen wider component replay without importing CUDA or changing production."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from replay_drafter_expert import digest, weight_records


def prepare(root: Path) -> None:
    """Verify all 36 fixtures, pinned sources and original-service identities before launch."""
    base = root.parent
    assert not (root / "prepared.json").exists()
    plan = json.loads((root / "plan.json").read_bytes())
    prior = base / "drafter-expert-padding-v3"
    state = json.loads((prior / "controller.json").read_bytes())
    assert state["phase"] == "complete_restored" and not state["service_paused"]
    original = json.loads((prior / "prepared.json").read_bytes())
    for name, sha in plan["source_sha256"].items():
        assert digest(root / name) == sha, name
        compile((root / name).read_text(), str(root / name), "exec")
    for name, sha in plan["frozen_bindings"].items():
        assert digest(base / name) == sha, name
    assert digest(base / "drafter-components-v2/control.py") == plan["restore_controller_sha256"]
    fixtures = root / "fixtures"
    assert digest(fixtures / "manifest.json") == plan["fixture_manifest_sha256"]
    manifest = json.loads((fixtures / "manifest.json").read_bytes())
    expected = {(r, a, m, p) for r in (0, 1) for a in ("a1", "b1")
                for m in (8, 16, 32, 64, 128)
                for p in (["repeat"] if m == 8 else ["repeat", "spread"])}
    assert len(manifest["files"]) == len(expected) == 36
    assert {(f["rank"], f["arm"], f["rows"], f["pattern"])
            for f in manifest["files"]} == expected
    for item in manifest["files"]:
        path = fixtures / item["path"]
        assert path.parent == fixtures and path.stat().st_size == item["bytes"]
        assert digest(path) == item["sha256"]
    for item in manifest["sources"]:
        path = base / "drafter-components-v2" / f"{item['arm']}-r{item['rank']}" / Path(
            item["path"],
        ).name
        assert digest(path) == item["sha256"]
        assert digest(path.with_suffix(".json")) == item["metadata_sha256"]
    records = {family: len(weight_records(base / "source", family)) for family in ("w13", "w2")}
    assert records == {"w13": 1152, "w2": 576}
    entry = original["launches"]["replay"][0]
    name = "mxwave-" + root.name + "-r0"
    argv = [value.replace(str(prior), str(root)).replace(entry["name"], name)
            .replace("/trial/replay_expert_padding.py", "/trial/replay_expert_wide.py")
            for value in entry["argv"]]
    argv.extend(["--fixtures", "/trial/fixtures"])
    assert plan["image"] in argv and "/trial/replay_expert_wide.py" in argv
    assert plan["outage_seconds"] >= plan["replay_seconds"] + plan["restoration_seconds"]
    (root / "result").mkdir()
    (root / "result/plan.json").write_bytes((root / "plan.json").read_bytes())
    prepared = {
        "status": "passed", "live": original["live"], "originals": original["originals"],
        "launches": {"replay": [{"rank": 0, "name": name, "argv": argv}]},
        "cpu_checks": {"checkpoint_tensors": records, "frozen_bindings": 4,
                       "source_compiled": True, "fixture_payloads": 36, "source_frames": 4},
        "plan_sha256": digest(root / "plan.json"),
    }
    (root / "prepared.json").write_text(json.dumps(prepared, indent=2) + "\n")
    print(json.dumps({"status": "passed", "plan_sha256": prepared["plan_sha256"],
                      "cpu_checks": prepared["cpu_checks"]}))


def main() -> None:
    """Prepare a fresh directory while preserving all preceding experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    prepare(parser.parse_args().root)


if __name__ == "__main__":
    main()
