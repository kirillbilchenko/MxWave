"""Accepted CT features must exclude rejected tokens and post-cap work."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from integrations.glm53_mxfp4.drafter_capture_pairing import compare_ranks
from integrations.glm53_mxfp4.drafter_feature_sequence import LAYERS, AcceptedHistory
from integrations.glm53_mxfp4.drafter_fp8_roundtrip import digest
from integrations.glm53_mxfp4.drafter_incremental_export import (
    audit_rank_export,
    export_case,
)


def frame(index: int) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Make a prompt plus one partially accepted verification fixture."""
    ids = [10, 11, 12] if index == 0 else [20, 21, 99, 99, 99, 99, 99, 99]
    start, valid, sampled = (0, 3, [20]) if index == 0 else (3, 2, [21, 22])
    data = {
        "target_input_ids": torch.tensor(ids),
        "target_positions": torch.arange(start, start + len(ids)),
        "target_num_sampled": torch.tensor([len(sampled)]),
        "target_sampled_tokens": torch.tensor([sampled]),
        "target_num_rejected": torch.tensor([0 if index == 0 else 6]),
        "draft_query_positions": torch.arange(start + valid, start + valid + 8),
        "draft_query_ids": torch.tensor([sampled[-1]] + [154856] * 7),
        "anchor_before_propose": torch.tensor([[sampled[-1]]]),
    }
    data.update({
        f"aux_layer_{layer}": torch.arange(len(ids)).to(torch.bfloat16)[:, None]
        .expand(-1, 4096).clone()
        for layer in LAYERS
    })
    return data, {"index": index, "target_num_draft_tokens": 0 if index == 0 else 7}


def test_accepted_features_exclude_rejection_and_post_cap() -> None:
    history = AcceptedHistory()
    history.append(*frame(0))
    history.append(*frame(1))
    report = history.validate_response([10, 11, 12], [20, 21])
    assert history.tokens.tolist() == [10, 11, 12, 20, 21]
    assert history.context().shape == (5, 20480)
    assert report["feature_rows"] == 4
    assert report["rejected_rows_excluded"] == 6
    assert report["post_cap_sampled_tokens_excluded"] == 1
    assert report["post_cap_feature_rows_excluded"] == 1


@pytest.mark.parametrize("fault", ["position", "accepted_token", "anchor", "rejected_count"])
def test_corrupt_alignment_is_rejected(fault: str) -> None:
    history = AcceptedHistory()
    history.append(*frame(0))
    data, meta = copy.deepcopy(frame(1))
    key, index = {
        "position": ("target_positions", (0,)),
        "accepted_token": ("target_input_ids", (1,)),
        "anchor": ("anchor_before_propose", (0, 0)),
        "rejected_count": ("target_num_rejected", (0,)),
    }[fault]
    data[key][index] += 1
    with pytest.raises(ValueError):
        history.append(data, meta)


def test_missing_frame_and_response_disagreement_fail_closed() -> None:
    history = AcceptedHistory()
    data, meta = frame(0)
    with pytest.raises(ValueError, match="unordered"):
        history.append(data, {**meta, "index": 1})
    history.append(data, meta)
    with pytest.raises(ValueError, match="API response"):
        history.validate_response([10, 11, 12], [999])


def test_incremental_rank_exports_match_and_numpy_detects_corruption(tmp_path: Path) -> None:
    case = {"id": "case-one", "split": "train", "domain": "code", "prompt_ids": [10, 11, 12],
            "prompt_sha256": "prompt", "token_ids_sha256": "tokens"}
    reports = []
    for rank in (0, 1):
        directory = tmp_path / f"ct-r{rank}"
        directory.mkdir()
        for index in (0, 1):
            data, meta = frame(index)
            path = directory / f"context-case-one-{index:04d}-r{rank}.safetensors"
            save_file(data, str(path))
            meta.update(case="case-one", rank=rank, draft_graph=[1, 8], tensor_dtypes={},
                        payload_sha256=digest(path))
            path.with_suffix(".json").write_text(json.dumps(meta))
        row = export_case(tmp_path, case, [20, 21], rank, tmp_path / f"accepted-r{rank}")
        assert (tmp_path / f"accepted-r{rank}" / row["path"]).stat().st_mode & 0o777 == 0o644
        reports.append({"status": "rank_verified", "rank": rank, "start": 0, "stop": 1,
                        "plan_sha256": "plan", "source_sha256": "source", "rows": [row]})
    paired = compare_ranks(*reports)
    assert paired["all_rank_features_bitwise_equal"]
    assert paired["rows"][0]["rejected_rows_excluded"] == 6
    assert paired["rows"][0]["numpy_bf16_elements_compared"] == 4 * 20480
    path = tmp_path / "accepted-r0/train-case-one.safetensors"
    values = load_file(path)
    assert values["generated_token_mask"].tolist() == [False, False, False, True, True]
    values["target_features"][0, 0] += 1
    save_file(values, path)
    with pytest.raises(ValueError, match="feature bits"):
        audit_rank_export(sorted((tmp_path / "ct-r0").glob("*.safetensors")), path,
                          [10, 11, 12, 20, 21])
    reports[1]["rows"][0]["sha256"] = "changed"
    with pytest.raises(ValueError, match="payloads"):
        compare_ranks(*reports)


def test_incremental_export_refuses_sealed_split_and_unsafe_ids(tmp_path: Path) -> None:
    for split, identifier in (("test", "one"), ("train", "../outside"), ("train", "*")):
        with pytest.raises(ValueError, match="identity"):
            export_case(tmp_path, {"id": identifier, "split": split}, [], 0, tmp_path)
