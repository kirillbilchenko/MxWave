"""Verify captured selector data and restoration independently of the training loop."""

from __future__ import annotations

import hashlib
import json
import re
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

_SOURCE_SELECTOR_DIGESTS = {
    "projection": "0d974ed37ad1faed0fab43a128459c2a87c2bc80c18e7465c90f6ac99508c1a7",
    "predecessor": "55e5599845ac0380451322097e842efb771bda20031b4ea0284a1485ae4fcc2e",
    "successor": "dd0a45392b5678ae6f3366872800d9606c028b5bc18a16370f7596b39cca1451",
}


def sha256(path: Path) -> str:
    """Hash a payload incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def analyze(root: Path) -> dict[str, Any]:
    """Fail closed on missing frames, changed controls, weights or archive payloads."""
    manifest = json.loads((root / "evidence-manifest.json").read_bytes())
    for row in manifest["files"]:
        path = root / row["path"]
        assert path.is_relative_to(root) and ".." not in Path(row["path"]).parts
        assert path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"]
    plan = json.loads((root / "plan.json").read_bytes())
    state = json.loads((root / "controller.json").read_bytes())
    service = json.loads((root / "final-service-review.json").read_bytes())
    assert state["phase"] == "complete_restored" and not state["service_paused"]
    assert service["health_200"] and service["originals_running"]
    assert any(x["max_model_len"] == 500000 for x in state["restoration"]["models"]["data"])
    measurement = json.loads((root / "ct-measurement.json").read_bytes())
    assert measurement["status"] == "complete" and len(measurement["cases"]) == 52
    assert all(x["output_equal"] and x["counters_equal"] for x in measurement["controls"])
    assert len(measurement["controls"]) == 2
    for case in plan["control_cases"]:
        controls = [x for x in measurement["cases"] if x["case"] == case]
        assert [x["policy"] for x in controls] == ["before", "capture", "after"]
        assert all(x["result"]["token_ids"] == controls[0]["result"]["token_ids"]
                   and x["counter_delta"] == controls[0]["counter_delta"] for x in controls)
    inventories = [
        json.loads((root / f"ct-r{rank}/selector-inventory-r{rank}.json").read_bytes())
        for rank in (0, 1)
    ]
    assert inventories[0] == inventories[1]
    assert {name: info["sha256"] for name, info in inventories[0].items()} \
        == _SOURCE_SELECTOR_DIGESTS
    raw = (root / "ct-r0/selector-original.safetensors").read_bytes()
    header_size = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + header_size])
    for name, info in inventories[0].items():
        spec = header[name]
        begin, end = spec["data_offsets"]
        assert spec["dtype"] == "BF16" and spec["shape"] == info["shape"]
        assert hashlib.sha256(raw[8 + header_size + begin:8 + header_size + end]).hexdigest() \
            == info["sha256"]
    assert len({x["text_sha256"] for x in plan["calibration_cases"]}) == 48
    results = []
    for case in plan["calibration_cases"]:
        requests = [x for x in measurement["cases"]
                    if x["case"] == case["id"] and x["policy"] == "capture"]
        assert len(requests) == 1
        request = requests[0]
        tokens = request["result"]["token_ids"]
        assert len(tokens) == 128 and request["result"]["usage"]["completion_tokens"] == 128
        prompt_length = len(case["prompt_ids"])
        assert request["result"]["usage"]["prompt_tokens"] == prompt_length
        paths = sorted((root / "ct-r0").glob(f"calibration-{case['id']}-*.json"))
        starts = []
        complete_blocks = 0
        for index, path in enumerate(paths):
            metadata = json.loads(path.read_bytes())
            assert metadata["index"] == index and metadata["case"] == case["id"]
            assert metadata["rank"] == 0 and metadata["native_full_graph"]
            tensor_path = path.with_suffix(".safetensors")
            assert sha256(tensor_path) == metadata["payload_sha256"]
            tensors = load_file(tensor_path)
            positions = tensors["sample_positions"].reshape(-1).tolist()
            starts.append(int(positions[0]))
            assert positions == list(range(starts[-1], starts[-1] + 7))
            ids = tensors["ids"].reshape(7, 16)
            assert all(len(set(row.tolist())) == 16 for row in ids)
            raw_scores = tensors["scores"]
            if metadata["tensor_dtypes"]["scores"] == "torch.bfloat16":
                scores = (raw_scores.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                scores = raw_scores.astype(np.float32)
            scores = scores.reshape(7, 16, 16)
            assert np.isfinite(scores).all()
            previous = 0
            proposal = []
            for step in range(7):
                previous = int(np.argmax(scores[step, previous]))
                proposal.append(int(ids[step, previous]))
            assert proposal == tensors["proposal"].reshape(-1).tolist()
            offset = starts[-1] - prompt_length
            assert offset >= 1
            if offset + 7 <= len(tokens):
                assert int(tensors["anchors"].item()) == tokens[offset - 1]
                complete_blocks += 1
        counters = request["counter_delta"]

        def total(metric: str, values: dict[str, float] = counters) -> int:
            return int(sum(value for key, value in values.items()
                           if key.split("{", 1)[0] == metric))

        rounds = total("vllm:spec_decode_num_drafts_total")
        accepted = total("vllm:spec_decode_num_accepted_tokens_total")
        assert starts and starts[0] == prompt_length + 1 and len(starts) > rounds
        advances = [b - a - 1 for a, b in pairwise(starts)]
        assert all(0 <= count <= 7 for count in advances)
        assert sum(advances[:rounds]) == accepted
        position_counts = [0] * 7
        for key, value in counters.items():
            if key.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total{"):
                match = re.search(r'position="(\d+)"', key)
                assert match is not None
                position_counts[int(match[1])] += int(value)
        assert position_counts == [
            sum(count > position for count in advances[:rounds]) for position in range(7)
        ]
        assert total("vllm:spec_decode_num_draft_tokens_total") == rounds * 7
        results.append({
            "case": case["id"], "split": case["split"], "frames": len(paths),
            "verified_rounds": rounds, "post_cap_replays": len(paths) - rounds,
            "complete_training_format_blocks": complete_blocks,
            "incomplete_tail_frames": len(paths) - complete_blocks,
            "counter_reconstruction_passed": True,
        })
    return {
        "schema": "mxwave-selector-capture-review-v1",
        "payloads_verified": len(manifest["files"]), "cases": results,
        "controls_passed": True, "weights_match_both_ranks": True,
        "weights": inventories[0], "source_manifest_sha256": sha256(root / "source-manifest.json"),
        "restoration_epoch": state["restoration"]["epoch"],
        "original_production_healthy_500000_context": True,
        "training_started": False, "timing_claim": False,
    }


if __name__ == "__main__":
    import sys

    review = analyze(Path(sys.argv[1]))
    output = Path(sys.argv[2])
    assert not output.exists()
    output.write_text(json.dumps(review, indent=2) + "\n")
    print(json.dumps({
        "payloads": review["payloads_verified"], "cases": len(review["cases"]),
        "controls_passed": review["controls_passed"],
    }))
