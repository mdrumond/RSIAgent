from __future__ import annotations

import json

import pytest

from benchmarks.a5kernels.phase1_registry import (
    CHECKPOINTS,
    DEFAULT_PROPOSALS,
    MAX_PROJECTS,
    PHASE1_BRIEF,
    PROJECT_REGISTRY,
    WAVES,
    CurriculumProposal,
    dry_run_plan,
    saturation_status,
)
from run_a5kernels import main


def proposal(**overrides):
    value = {
        "family": "length-knee",
        "parameters": {"length": 128},
        "hypothesis": "A bounded size may expose a performance transition.",
        "evidence_preset": "correctness-timing",
    }
    value.update(overrides)
    return CurriculumProposal.from_mapping(value)


def test_phase1_brief_and_registry_are_fixed_and_bounded():
    assert PHASE1_BRIEF.max_projects == MAX_PROJECTS == 8
    assert PHASE1_BRIEF.waves == WAVES == (1, 2, 3, 4)
    assert PHASE1_BRIEF.checkpoints == CHECKPOINTS == (0, 4, 8)
    assert len(PROJECT_REGISTRY) == 7
    assert all(item.wave in WAVES for item in PROJECT_REGISTRY)
    with pytest.raises(Exception):
        PHASE1_BRIEF.max_projects = 9


@pytest.mark.parametrize("extra", [
    {"command": "python arbitrary.py"},
    {"argv": ["bash", "-c", "anything"]},
    {"score": 1.0},
])
def test_proposal_rejects_commands_and_arbitrary_fields(extra):
    value = proposal().as_dict()
    value.update(extra)
    with pytest.raises(ValueError, match="fields must be exactly"):
        CurriculumProposal.from_mapping(value)


@pytest.mark.parametrize("parameters", [
    {"length": 31},
    {"length": 96},
    {"length": 401},
    {"length": True},
    {"length": 128, "command": "run"},
    {},
])
def test_proposal_rejects_out_of_bounds_or_arbitrary_parameters(parameters):
    with pytest.raises(ValueError, match="parameter"):
        proposal(parameters=parameters)


@pytest.mark.parametrize("length", [32, 64, 128, 256, 400])
def test_length_knee_accepts_every_host_representable_shape(length):
    assert dict(proposal(parameters={"length": length}).parameters) == {
        "length": length
    }


def test_padded_multitile_preserves_continuous_length_range():
    registered = CurriculumProposal.from_mapping({
        "family": "padded-multitile",
        "parameters": {"length": 96},
        "hypothesis": "A non-knee padded extent remains registered.",
        "evidence_preset": "correctness",
    })
    assert dict(registered.parameters) == {"length": 96}


def test_proposal_rejects_unregistered_family_and_wrong_evidence():
    with pytest.raises(ValueError, match="registered family"):
        proposal(family="arbitrary-kernel")
    with pytest.raises(ValueError, match="not registered"):
        proposal(evidence_preset="msprof-guided")


def test_direct_proposal_construction_cannot_bypass_registry_validation():
    valid = proposal()
    with pytest.raises(ValueError, match="registered ProjectFamily"):
        CurriculumProposal(
            "length-knee", valid.parameters, valid.hypothesis, valid.evidence_preset
        )
    with pytest.raises(ValueError, match="canonical registered parameter"):
        CurriculumProposal(
            valid.family, (("length", 401), ("command", "run")),
            valid.hypothesis, valid.evidence_preset,
        )


def test_duplicate_and_over_budget_plans_fail_closed():
    with pytest.raises(ValueError, match="duplicate"):
        dry_run_plan([proposal(), proposal()])
    with pytest.raises(ValueError, match="at most 8"):
        dry_run_plan([*DEFAULT_PROPOSALS, CurriculumProposal.from_mapping({
            "family": "runtime-recovery", "parameters": {"faults": 2},
            "hypothesis": "A second runtime recovery exceeds the project budget.",
            "evidence_preset": "ordinary-recovery",
        })])


def test_default_plan_meets_every_saturation_gate():
    plan = dry_run_plan()
    assert plan["mode"] == "dry-run"
    assert plan["budget"] == {"maximum": 8, "planned": 8}
    assert plan["waves"] == [1, 2, 3, 4]
    assert plan["checkpoints"] == [0, 4, 8]
    assert plan["checkpoint_status"] == {
        "0": "CONTINUE", "4": "CONTINUE", "8": "SATURATED",
    }
    assert plan["terminal_status"] == saturation_status(DEFAULT_PROPOSALS) == "SATURATED"
    projects = plan["projects"]
    assert [item["ordinal"] for item in projects] == list(range(1, 9))
    assert projects[0]["family"] == "vector-add-baseline"
    assert projects[1]["family"] == "padded-multitile"
    coverage = [tag for item in projects for tag in item["coverage"]]
    assert "functional-correctness" in coverage
    assert "padded-multitile" in coverage
    assert "ordinary-recovery" in coverage
    assert coverage.count("performance-knee") == 2
    assert "cross-layer" in coverage
    assert "msprof-guided" in coverage


def test_saturation_requires_two_distinct_knees_and_all_coverage():
    without_second_knee = tuple(
        item for item in DEFAULT_PROPOSALS
        if not (item.family.value == "length-knee" and dict(item.parameters)["length"] == 400)
    )
    assert saturation_status(without_second_knee) == "IN_PROGRESS"
    replacement = CurriculumProposal.from_mapping({
        "family": "runtime-recovery", "parameters": {"faults": 2},
        "hypothesis": "A second bounded runtime failure remains recoverable.",
        "evidence_preset": "ordinary-recovery",
    })
    eight_without_second_knee = (*without_second_knee, replacement)
    assert saturation_status(eight_without_second_knee) == "CONTINUE"


def test_plan_is_deterministic_independent_of_proposal_order():
    assert dry_run_plan(DEFAULT_PROPOSALS) == dry_run_plan(tuple(reversed(DEFAULT_PROPOSALS)))


def test_phase1_plan_cli_is_machine_readable_and_offline(capsys):
    assert main(["phase1-plan"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == dry_run_plan()
    assert "command" not in json.dumps(payload)
    assert "argv" not in json.dumps(payload)


def test_phase1_plan_cli_validates_explicit_proposals(tmp_path, capsys):
    path = tmp_path / "proposals.json"
    path.write_text(json.dumps([item.as_dict() for item in DEFAULT_PROPOSALS]))
    assert main(["phase1-plan", "--proposals", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["terminal_status"] == "SATURATED"
