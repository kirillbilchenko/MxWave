"""The full-run schedule must preserve pilot arithmetic and stop at frozen boundaries."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from integrations.glm53_mxfp4.drafter_training_schedule import TrainingSchedule
from integrations.glm53_mxfp4.prepare_drafter_fresh_capture import smoke_cases

PROTOCOL = Path("benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/"
                "drafter-context-pilot-v3-protocol.json")


def test_generalized_schedule_preserves_every_original_pilot_rate_exactly() -> None:
    schedule = TrainingSchedule.from_protocol(json.loads(PROTOCOL.read_bytes()))
    for step in range(1, 33):
        previous = (1e-5 * step / 4 if step <= 4 else
                    1e-6 + 0.5 * (1e-5 - 1e-6) * (1 + math.cos(math.pi * (step - 4) / 28)))
        assert schedule.learning_rate(step).hex() == previous.hex()
        assert schedule.evaluate_after(step, 32) == (step in (8, 16, 24, 32))
    assert schedule.evaluate_after(29, 29)


def test_full_schedule_uses_all_eligible_prompts_and_keeps_final_evaluation() -> None:
    protocol = json.loads(PROTOCOL.read_bytes())
    protocol["optimizer"].update(maximum_steps=4000, warmup_steps=200)
    protocol["validation"]["steps"] = [0, 250, 500, 1000, 2000, 3000, 4000]
    schedule = TrainingSchedule.from_protocol(protocol)
    assert schedule.learning_rate(200) == 1e-5
    assert schedule.learning_rate(4000) == 1e-6
    assert schedule.learning_rate(1000) > schedule.learning_rate(2000)
    assert schedule.evaluate_after(3912, 3912)
    assert not schedule.evaluate_after(32, 3912)
    with pytest.raises(ValueError):
        schedule.learning_rate(4001)
    with pytest.raises(ValueError):
        schedule.evaluate_after(3913, 3912)


@pytest.mark.parametrize("fault", ["seed", "budget", "warmup", "rate", "validation", "tensor"])
def test_unsupported_or_ambiguous_training_contract_fails_closed(fault: str) -> None:
    protocol = copy.deepcopy(json.loads(PROTOCOL.read_bytes()))
    if fault == "seed":
        protocol["optimizer"]["seed"] = 99
    elif fault == "budget":
        protocol["optimizer"]["maximum_steps"] = 4001
    elif fault == "warmup":
        protocol["optimizer"]["warmup_steps"] = 32
    elif fault == "rate":
        protocol["optimizer"]["peak_learning_rate"] = float("nan")
    elif fault == "validation":
        protocol["validation"]["steps"] = [0, 8, 16]
    else:
        protocol["trainable"].append("selector.weight")
    with pytest.raises(ValueError):
        TrainingSchedule.from_protocol(protocol)


def test_smoke_capture_keeps_longest_inputs_and_all_domains_without_test_access() -> None:
    rows = [{"id": f"{split}-{domain}", "split": split, "domain": domain,
             "prompt_ids": [1] * length}
            for split in ("train", "validation")
            for domain, length in (("code", 30), ("chat", 50), ("explanation", 20))]
    rows += [{"id": split + "-long", "split": split, "domain": "code",
              "prompt_ids": [1] * 3973} for split in ("train", "validation")]
    selected = smoke_cases(rows)
    assert len(selected) == 8
    assert {r["id"] for r in selected} == {r["id"] for r in rows}
