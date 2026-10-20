"""CPU-only preparation for the bounded expert-padding replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from replay_drafter_expert import digest, weight_records


def prepare(root: Path) -> None:
    """Verify immutable inputs and reuse the exact original-service restoration identities."""
    base = root.parent
    assert not (root / "prepared.json").exists()
    plan = json.loads((root / "plan.json").read_bytes())
    prior = base / "drafter-expert-v1"
    state = json.loads((prior / "controller.json").read_bytes())
    assert state["phase"] == "complete_restored" and not state["service_paused"]
    original = json.loads((prior / "prepared.json").read_bytes())
    for name, sha in plan["source_sha256"].items():
        assert digest(root / name) == sha, name
        compile((root / name).read_text(), str(root / name), "exec")
    for name, sha in plan["frozen_bindings"].items():
        assert digest(base / name) == sha, name
    assert digest(base / "drafter-components-v2/control.py") == plan["restore_controller_sha256"]
    records = {family: len(weight_records(base / "source", family)) for family in ("w13", "w2")}
    assert records == {"w13": 1152, "w2": 576}
    entry = original["launches"]["replay"][0]
    name = "mxwave-" + root.name + "-r0"
    argv = [
        value.replace(str(prior), str(root)).replace(entry["name"], name)
        .replace("/trial/replay_drafter_expert.py", "/trial/replay_expert_padding.py")
        .replace("PYTHONPATH=/base/", "PYTHONPATH=/trial:/base/")
        for value in entry["argv"]
    ]
    assert plan["image"] in argv
    (root / "result").mkdir()
    (root / "result/plan.json").write_bytes((root / "plan.json").read_bytes())
    prepared = {
        "status": "passed", "live": original["live"], "originals": original["originals"],
        "launches": {"replay": [{"rank": 0, "name": name, "argv": argv}]},
        "cpu_checks": {"checkpoint_tensors": records, "frozen_bindings": 4,
                       "source_compiled": True},
        "plan_sha256": digest(root / "plan.json"),
    }
    (root / "prepared.json").write_text(json.dumps(prepared, indent=2) + "\n")
    print(json.dumps({"status": "passed", "plan_sha256": prepared["plan_sha256"],
                      "cpu_checks": prepared["cpu_checks"]}))


def main() -> None:
    """Prepare the fresh directory without importing GPU libraries or stopping production."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    prepare(parser.parse_args().root)


if __name__ == "__main__":
    main()
