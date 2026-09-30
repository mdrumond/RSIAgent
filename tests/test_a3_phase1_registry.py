from dataclasses import FrozenInstanceError
import json

import pytest

from benchmarks.a3kernels.phase1_registry import (
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


def proposal(**overrides):
    value = {
        "target": "Ascend910B4",
        "language": "ascend-c",
        "family": "length-knee",
        "parameters": {"length": 128},
        "hypothesis": "A bounded size may expose a performance transition.",
        "evidence_preset": "correctness-timing",
    }
    value.update(overrides)
    return CurriculumProposal.from_mapping(value)


def test_registry_is_fixed_bounded_and_a3_owned():
    assert PHASE1_BRIEF.max_projects == MAX_PROJECTS == 8
    assert PHASE1_BRIEF.waves == WAVES == (1, 2, 3, 4)
    assert PHASE1_BRIEF.checkpoints == CHECKPOINTS == (0, 4, 8)
    assert len(PROJECT_REGISTRY) == 7
    assert all(item.wave in WAVES for item in PROJECT_REGISTRY)
    with pytest.raises(FrozenInstanceError):
        PHASE1_BRIEF.max_projects = 9


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"target": "Ascend950"}, "A3 target"),
        ({"target": "a5"}, "A3 target"),
        ({"language": "catlass-dsl"}, "Ascend C"),
        ({"execution_profile": "bz-a5"}, "fields must be exactly"),
        ({"argv": ["bash", "-c", "anything"]}, "fields must be exactly"),
    ],
)
def test_proposal_rejects_a5_catlass_and_arbitrary_execution(changes, message):
    with pytest.raises(ValueError, match=message):
        proposal(**changes)


@pytest.mark.parametrize(
    "parameters",
    [
        {"length": 31},
        {"length": 96},
        {"length": 401},
        {"length": True},
        {"length": 128, "command": "run"},
        {},
    ],
)
def test_proposals_enforce_registered_parameter_bounds(parameters):
    with pytest.raises(ValueError, match="parameter"):
        proposal(parameters=parameters)


@pytest.mark.parametrize("length", [1, 8, 16, 32, 64, 128, 256, 400])
def test_length_knee_accepts_only_registered_shapes(length):
    assert dict(proposal(parameters={"length": length}).parameters) == {
        "length": length
    }


def test_padded_project_accepts_non_knee_extent():
    padded = proposal(
        family="padded-multitile",
        parameters={"length": 96},
        hypothesis="A padded extent exercises ordinary tail handling.",
        evidence_preset="correctness",
    )
    assert dict(padded.parameters) == {"length": 96}


def test_default_knees_keep_one_sub32_and_one_distant_large_sample():
    knees = [
        dict(item.parameters)["length"]
        for item in DEFAULT_PROPOSALS
        if item.family.value == "length-knee"
    ]
    assert knees == [16, 400]
    assert knees[0] < 32
    assert knees[1] >= 256


def test_default_plan_contains_exact_eight_project_curriculum():
    plan = dry_run_plan()

    assert plan["schema"] == "a3-ascendc-phase1-dry-plan-v1"
    assert plan["target"] == "Ascend910B4"
    assert plan["language"] == "ascend-c"
    assert plan["budget"] == {"maximum": 8, "planned": 8}
    assert plan["waves"] == [1, 2, 3, 4]
    assert plan["checkpoints"] == [0, 4, 8]
    assert plan["checkpoint_status"] == {
        "0": "CONTINUE",
        "4": "CONTINUE",
        "8": "SATURATED",
    }
    assert plan["terminal_status"] == "SATURATED"
    assert [row["family"] for row in plan["projects"]] == [
        "vector-add-baseline",
        "padded-multitile",
        "compile-recovery",
        "runtime-recovery",
        "length-knee",
        "length-knee",
        "cross-layer-launch",
        "msprof-pipe",
    ]
    assert len({row["project_id"] for row in plan["projects"]}) == 8
    assert len(plan["plan_id"]) == 64


def test_plan_order_and_identity_are_deterministic():
    forward = dry_run_plan(DEFAULT_PROPOSALS)
    reverse = dry_run_plan(tuple(reversed(DEFAULT_PROPOSALS)))
    assert forward == reverse
    assert json.dumps(forward, sort_keys=True, separators=(",", ":")) == json.dumps(
        reverse, sort_keys=True, separators=(",", ":")
    )


def test_duplicate_and_over_budget_plans_fail_closed():
    with pytest.raises(ValueError, match="duplicate"):
        dry_run_plan([proposal(), proposal()])
    ninth = proposal(
        family="runtime-recovery",
        parameters={"faults": 2},
        hypothesis="A second recovery would exceed the project budget.",
        evidence_preset="ordinary-recovery",
    )
    with pytest.raises(ValueError, match="at most 8"):
        dry_run_plan((*DEFAULT_PROPOSALS, ninth))


def test_saturation_requires_all_coverage_and_two_knees():
    short = tuple(
        item
        for item in DEFAULT_PROPOSALS
        if not (item.family.value == "length-knee" and dict(item.parameters)["length"] == 400)
    )
    assert saturation_status(short) == "IN_PROGRESS"
    replacement = proposal(
        family="runtime-recovery",
        parameters={"faults": 2},
        hypothesis="An extra recovery does not replace knee coverage.",
        evidence_preset="ordinary-recovery",
    )
    assert saturation_status((*short, replacement)) == "CONTINUE"


def test_dry_plan_has_no_execution_side_effect_surface():
    encoded = json.dumps(dry_run_plan(), sort_keys=True)
    assert "argv" not in encoded
    assert "command" not in encoded
    assert "session" not in encoded
    assert "model" not in encoded
