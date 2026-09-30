from dataclasses import replace
import json

import pytest

from benchmarks.a3_experiments import (
    A3EvidenceRef,
    A3ExperimentScheduler,
    A3Language,
    A3RuntimeCapabilities,
    A3Target,
    BackendModel,
    CapabilityUnavailableError,
    KnowledgeMode,
    ProfilingGuidance,
    ProgrammingLevel,
    build_a3_experiment_plan,
)


def test_default_plan_is_fixed_a3_ascendc_cartesian_product():
    plan = build_a3_experiment_plan()

    assert plan.target is A3Target.A3
    assert plan.language is A3Language.ASCEND_C
    assert len(plan.cells) == 24
    assert {
        (cell.backend_model, cell.knowledge, cell.profiling, cell.programming_level)
        for cell in plan.cells
    } == {
        (model, knowledge, profiling, level)
        for model in BackendModel
        for knowledge in KnowledgeMode
        for profiling in ProfilingGuidance
        for level in ProgrammingLevel
    }
    assert [stage.programming_level for stage in plan.curriculum] == list(
        ProgrammingLevel
    )
    assert [stage.ordinal for stage in plan.curriculum] == [1, 2, 3]


def test_plan_ids_order_and_serialization_are_stable():
    first = build_a3_experiment_plan()
    second = build_a3_experiment_plan()

    assert first == second
    assert [cell.cell_id for cell in first.cells] == [
        "a3-cell-06be7e628e1a742d",
        "a3-cell-22af0dc10707c849",
        "a3-cell-9cef0a3ff2d1fc9a",
        "a3-cell-f186780727548109",
        "a3-cell-92797aff4839cf83",
        "a3-cell-f58cec0d66ac9591",
        "a3-cell-929a8bddac922ca2",
        "a3-cell-eabbb9ad8d80fbbc",
        "a3-cell-260034c431efdac7",
        "a3-cell-e04ac278abd81c08",
        "a3-cell-586ce83ede94d45f",
        "a3-cell-6ea07bd1567ca84b",
        "a3-cell-302a6e353e8f47f4",
        "a3-cell-f2fcc4337279ad92",
        "a3-cell-a9ef7dc1581bed73",
        "a3-cell-1cea8f7a01e29ebe",
        "a3-cell-e7260fbd878bdb7c",
        "a3-cell-d36248fbb17965a7",
        "a3-cell-063a3d65762b420e",
        "a3-cell-9221cb1e0749e8e4",
        "a3-cell-6941b5fbb7001d48",
        "a3-cell-9805db9eab68abcb",
        "a3-cell-3af73f3eb5a55777",
        "a3-cell-a1a26ccb06a4c92d",
    ]
    assert first.to_json() == second.to_json()
    payload = json.loads(first.to_json())
    assert payload["schema"] == "a3-ascendc-experiment-plan-v1"
    assert payload["target"] == "a3"
    assert payload["language"] == "ascend-c"
    assert payload["cells"][0]["backend_model"] == "openai/gpt-5.6-astra"


@pytest.mark.parametrize(
    ("field", "value"),
    [("target", "a5"), ("language", "catlass-dsl")],
)
def test_active_evidence_rejects_a5_or_non_ascendc_labels(field, value):
    payload = {
        "evidence_id": "ascendc-api-reference",
        "target": "a3",
        "language": "ascend-c",
        "sha256": "a" * 64,
    }
    payload[field] = value

    with pytest.raises(ValueError, match="active A3 evidence"):
        A3EvidenceRef.from_mapping(payload)


@pytest.mark.parametrize(("field", "value"), [("evidence_id", 1), ("sha256", 1)])
def test_active_evidence_rejects_coerced_identifiers(field, value):
    payload = {
        "evidence_id": "ascendc-api-reference",
        "target": "a3",
        "language": "ascend-c",
        "sha256": "a" * 64,
    }
    payload[field] = value

    with pytest.raises(ValueError):
        A3EvidenceRef.from_mapping(payload)


def test_plan_serializes_only_active_a3_knowledge():
    evidence = A3EvidenceRef.from_mapping({
        "evidence_id": "ascendc-api-reference",
        "target": "a3",
        "language": "ascend-c",
        "sha256": "a" * 64,
    })

    plan = build_a3_experiment_plan(active_knowledge=(evidence,))

    assert plan.active_knowledge == (evidence,)
    encoded = plan.to_json()
    payload = json.loads(encoded)
    assert '"target":"a3"' in encoded
    assert '"language":"ascend-c"' in encoded
    assert {item["target"] for item in payload["active_knowledge"]} == {"a3"}
    assert {item["language"] for item in payload["active_knowledge"]} == {
        "ascend-c"
    }


def test_plan_rejects_untyped_or_duplicate_active_knowledge():
    evidence = A3EvidenceRef("docs", "a" * 64)
    with pytest.raises(ValueError, match="A3EvidenceRef"):
        build_a3_experiment_plan(active_knowledge=("historical-a5",))
    with pytest.raises(ValueError, match="unique"):
        build_a3_experiment_plan(active_knowledge=(evidence, evidence))


def test_capability_scheduler_reports_every_missing_dimension():
    cell = build_a3_experiment_plan().cells[-1]
    scheduler = A3ExperimentScheduler(
        A3RuntimeCapabilities(
            target=A3Target.A3,
            language=A3Language.ASCEND_C,
            backend_models=frozenset({BackendModel.GPT_5_6_ASTRA}),
            programming_levels=frozenset({ProgrammingLevel.FOUNDATION}),
            kdb=False,
            profiling_guidance=False,
        )
    )

    with pytest.raises(CapabilityUnavailableError) as failure:
        scheduler.schedule(cell)

    message = str(failure.value)
    assert "backend-model:openai/gpt-5.6-sol" in message
    assert "programming-level:optimized" in message
    assert "kdb" in message
    assert "profiling-guidance" in message


def test_capability_scheduler_returns_plan_without_success_claim():
    cell = build_a3_experiment_plan().cells[-1]
    scheduler = A3ExperimentScheduler(
        A3RuntimeCapabilities(
            target=A3Target.A3,
            language=A3Language.ASCEND_C,
            backend_models=frozenset(BackendModel),
            programming_levels=frozenset(ProgrammingLevel),
            kdb=True,
            profiling_guidance=True,
        )
    )

    scheduled = scheduler.schedule(cell)

    assert scheduled.cell == cell
    assert scheduled.target is A3Target.A3
    assert scheduled.language is A3Language.ASCEND_C
    assert not hasattr(scheduled, "passed")


@pytest.mark.parametrize(
    "change",
    [
        {"target": "a5"},
        {"language": "catlass-dsl"},
        {"backend_models": frozenset()},
        {"programming_levels": frozenset()},
        {"kdb": 1},
        {"profiling_guidance": "false"},
    ],
)
def test_capabilities_reject_non_a3_or_untyped_values(change):
    values = {
        "target": A3Target.A3,
        "language": A3Language.ASCEND_C,
        "backend_models": frozenset(BackendModel),
        "programming_levels": frozenset(ProgrammingLevel),
        "kdb": False,
        "profiling_guidance": False,
    }
    values.update(change)

    with pytest.raises(ValueError):
        A3RuntimeCapabilities(**values)


def test_scheduler_rejects_reconstructed_or_modified_cells():
    scheduler = A3ExperimentScheduler(
        A3RuntimeCapabilities(
            target=A3Target.A3,
            language=A3Language.ASCEND_C,
            backend_models=frozenset(BackendModel),
            programming_levels=frozenset(ProgrammingLevel),
        )
    )
    cell = build_a3_experiment_plan().cells[0]

    with pytest.raises(ValueError, match="registered A3 experiment cell"):
        scheduler.schedule(replace(cell, cell_id="a3-cell-not-registered"))
