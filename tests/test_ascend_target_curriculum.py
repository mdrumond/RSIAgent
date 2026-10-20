import json

import pytest

from benchmarks.ascend.targets import a3_curriculum_policy
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
from benchmarks.a5kernels.phase1_curriculum_selector import (
    CurriculumAgent, CurriculumContext, _grammar,
)


def spec_mapping(operation="reduce-sum"):
    reduction = operation == "reduce-sum"
    return {
        "project_id": "native-study", "capabilities": [
            "vector-reduction" if reduction else "vector-mask"],
        "shape": [4], "dtype": "float32", "block_count": 1,
        "padding_multiple": 1, "failure_mode": "none", "timing_lengths": [],
        "profile_metric": None, "hypothesis": "Learn native vector semantics.",
        "graph": {"inputs": [{"name": "x", "shape": [4]}], "nodes": [{
            "name": "out", "op": operation, "operands": ["x"],
            "axis": 0 if reduction else None}], "output": "out"},
    }


@pytest.mark.parametrize("operation", ["reduce-sum", "mask-positive"])
def test_native_operations_do_not_require_simt(operation):
    spec = ProjectSpec.from_mapping(spec_mapping(operation))
    a3_curriculum_policy().admit(spec)
    assert not any("simt" in item.value for item in spec.capabilities)


def test_a5_grammar_remains_legacy_only():
    assert "vector-reduction" not in _grammar()["capabilities"]
    assert "simt-reduction" in _grammar()["capabilities"]


def test_a3_grammar_and_guidance_are_product_specific():
    policy = a3_curriculum_policy()
    grammar = policy.grammar(_grammar())
    assert grammar["dtypes"] == ["float32"]
    assert "vector-reduction" in grammar["capabilities"]
    text = json.dumps(grammar) + policy.guidance + policy.system
    for forbidden in ("simt", "mxfp8", "float8_", "tla.", "msdebug", "mssanitizer"):
        assert forbidden not in text.lower()


def test_dtype_admission_is_explicit_and_limited():
    value = spec_mapping()
    value["dtype"] = "bfloat16"
    spec = ProjectSpec.from_mapping(value)
    with pytest.raises(ValueError, match="dtype"):
        a3_curriculum_policy().admit(spec)
    a3_curriculum_policy(dtypes=("float32", "bfloat16")).admit(spec)
    with pytest.raises(ValueError, match="dtype"):
        a3_curriculum_policy(dtypes=("float8_e4m3fn",))


@pytest.mark.parametrize("update", [
    {"capabilities": ["simt-reduction"]},
    {"learning_focus": ["simt-barrier"]},
    {"learning_focus": ["tool-msdebug"]},
    {"dtype": "float8_e4m3fn"},
])
def test_a3_admission_rejects_a5_only_scope(update):
    value = dict(spec_mapping(), **update)
    with pytest.raises(ValueError):
        a3_curriculum_policy().admit(ProjectSpec.from_mapping(value))


def test_agent_applies_target_policy_and_repairs_invalid_dtype():
    valid = spec_mapping()
    invalid = dict(valid, dtype="bfloat16")
    calls = []
    responses = iter((invalid, valid))

    def complete(messages):
        calls.append(messages)
        return "DECISION: WAVE\n" + json.dumps({
            "wave_id": "wave-001", "pre_wave_memory_sha256": "a" * 64,
            "pre_wave_coverage": [],
            "projects": [{"spec": next(responses), "requires": []}],
        })

    context = CurriculumContext("Learn native vector kernels", 1, "a" * 64, (), ())
    decision, session = CurriculumAgent(
        complete, target_policy=a3_curriculum_policy(),
    ).select(context)
    assert decision.wave.projects[0].spec.dtype.value == "float32"
    assert session.turns == 2
    assert "A3 Ascend C" in calls[0][0]["content"]
    assert "dtype unavailable" in calls[1][-1]["content"]


def test_legacy_agent_derives_simt_labels_for_new_proposals():
    response = "DECISION: WAVE\n" + json.dumps({
        "wave_id": "wave-001", "pre_wave_memory_sha256": "a" * 64,
        "pre_wave_coverage": [],
        "projects": [{"spec": spec_mapping(), "requires": []}],
    })
    decision, _ = CurriculumAgent(lambda messages: response).select(
        CurriculumContext("Learn kernels", 1, "a" * 64, (), ()),
    )
    labels = {item.value for item in decision.wave.projects[0].spec.capabilities}
    assert "simt-reduction" in labels
    assert "vector-reduction" not in labels
