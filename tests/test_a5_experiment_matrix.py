from __future__ import annotations

import json

import pytest

from benchmarks.a5kernels.fixtures import Language
from benchmarks.a5kernels.matrix import (
    CapabilityUnavailableError,
    ExperimentOrchestrator,
    KnowledgeMode,
    ModelProfile,
    ProfilingMode,
    RuntimeCapabilities,
    Workload,
    aggregate_report,
    initial_matrix,
)
from run_a5kernels import main


def test_initial_matrix_is_exact_cartesian_product_with_isolated_state() -> None:
    plan = initial_matrix()

    assert len(plan.cells) == 12
    assert [
        (
            cell.language.value,
            cell.knowledge.value,
            cell.profiling.value,
            cell.cell_id,
        )
        for cell in plan.cells
    ] == [
        (
            "catlass-dsl",
            "without-kdb",
            "without-profiling-guidance",
            "cell-c82737a617458aec",
        ),
        (
            "catlass-dsl",
            "without-kdb",
            "with-profiling-guidance",
            "cell-2e8ae3a01ced7240",
        ),
        (
            "catlass-dsl",
            "with-kdb",
            "without-profiling-guidance",
            "cell-97d03f4e20e35d21",
        ),
        (
            "catlass-dsl",
            "with-kdb",
            "with-profiling-guidance",
            "cell-81f240b42785f7b8",
        ),
        (
            "ascend-c",
            "without-kdb",
            "without-profiling-guidance",
            "cell-d573b7b73a53c7b0",
        ),
        (
            "ascend-c",
            "without-kdb",
            "with-profiling-guidance",
            "cell-367667930dad49c5",
        ),
        (
            "ascend-c",
            "with-kdb",
            "without-profiling-guidance",
            "cell-72fb130cb2596232",
        ),
        (
            "ascend-c",
            "with-kdb",
            "with-profiling-guidance",
            "cell-42d9d469e0c26f40",
        ),
        (
            "triton-ascend",
            "without-kdb",
            "without-profiling-guidance",
            "cell-bca23414ed1536ff",
        ),
        (
            "triton-ascend",
            "without-kdb",
            "with-profiling-guidance",
            "cell-067c16f752ae28d1",
        ),
        (
            "triton-ascend",
            "with-kdb",
            "without-profiling-guidance",
            "cell-3de8364b311ad220",
        ),
        (
            "triton-ascend",
            "with-kdb",
            "with-profiling-guidance",
            "cell-b0a9bc751432d463",
        ),
    ]
    assert len({cell.context_id for cell in plan.cells}) == 12
    assert len({cell.memory_id for cell in plan.cells}) == 12
    assert len({cell.workspace_id for cell in plan.cells}) == 12
    assert {cell.model.model_id for cell in plan.cells} == {"openai/gpt-5.6-sol"}
    assert {cell.model.name for cell in plan.cells} == {"openai-gpt-5.6-sol-v1"}
    assert {cell.model.provider for cell in plan.cells} == {"OpenAI"}
    assert all(not cell.model.allow_fallback for cell in plan.cells)
    assert plan.workloads == (Workload.SMOKE_VECTOR_ADD, Workload.SEMANTIC_GEMM)


def test_matrix_rejects_noncanonical_model_routes() -> None:
    with pytest.raises(ValueError, match="canonical A5 model route"):
        initial_matrix(ModelProfile("other", "openai/other", "OpenAI"))


def test_determinism_schedule_is_three_catlass_baseline_repeats() -> None:
    plan = initial_matrix()
    trials = plan.determinism_trials
    baseline = next(cell for cell in plan.cells if cell.cell_id == trials[0].cell_id)

    assert [trial.repeat for trial in trials] == [1, 2, 3]
    assert len({trial.trial_id for trial in trials}) == 3
    assert len({trial.context_id for trial in trials}) == 3
    assert len({trial.memory_id for trial in trials}) == 3
    assert len({trial.workspace_id for trial in trials}) == 3
    assert baseline.language is Language.CATLASS_DSL
    assert baseline.knowledge is KnowledgeMode.WITHOUT_KDB
    assert baseline.profiling is ProfilingMode.WITHOUT_GUIDANCE


def test_orchestrator_fails_closed_for_every_unavailable_capability() -> None:
    cell = next(
        cell
        for cell in initial_matrix().cells
        if cell.knowledge is KnowledgeMode.WITH_KDB
        and cell.profiling is ProfilingMode.WITH_GUIDANCE
    )
    orchestrator = ExperimentOrchestrator(
        RuntimeCapabilities(
            languages=frozenset(),
            model_ids=frozenset(),
            workloads=frozenset({Workload.SMOKE_VECTOR_ADD}),
        )
    )

    with pytest.raises(CapabilityUnavailableError) as error:
        orchestrator.schedule(cell, Workload.SEMANTIC_GEMM)

    message = str(error.value)
    assert f"runtime:{cell.language.value}" in message
    assert "model:openai/gpt-5.6-sol" in message
    assert "workload:semantic-gemm" in message
    assert "kdb" in message
    assert "profiling-guidance" in message


@pytest.mark.parametrize(
    "workloads",
    [
        frozenset(),
        {Workload.SMOKE_VECTOR_ADD},
        frozenset({"smoke-vector-add"}),
    ],
)
def test_runtime_capabilities_require_typed_nonempty_workloads(workloads) -> None:
    with pytest.raises(
        ValueError, match="workloads must be a non-empty frozenset of Workload"
    ):
        RuntimeCapabilities(
            languages=frozenset(Language),
            model_ids=frozenset({"openai/gpt-5.6-sol"}),
            workloads=workloads,
        )


def test_orchestrator_rejects_non_workload_schedule_input() -> None:
    cell = initial_matrix().cells[0]
    orchestrator = ExperimentOrchestrator(
        RuntimeCapabilities(
            languages=frozenset(Language),
            model_ids=frozenset({"openai/gpt-5.6-sol"}),
            workloads=frozenset({Workload.SMOKE_VECTOR_ADD}),
        )
    )

    with pytest.raises(ValueError, match="workload must be a Workload"):
        orchestrator.schedule(cell, "smoke-vector-add")


def test_orchestrator_schedules_without_claiming_a_result() -> None:
    cell = initial_matrix().cells[0]
    orchestrator = ExperimentOrchestrator(
        RuntimeCapabilities(
            languages=frozenset(Language),
            model_ids=frozenset({"openai/gpt-5.6-sol"}),
            workloads=frozenset({Workload.SMOKE_VECTOR_ADD}),
            kdb=True,
            profiling_guidance=True,
        )
    )

    scheduled = orchestrator.schedule(cell, Workload.SMOKE_VECTOR_ADD)

    assert scheduled.cell == cell
    assert scheduled.workload is Workload.SMOKE_VECTOR_ADD
    assert not hasattr(scheduled, "passed")


def test_matrix_cli_emits_machine_readable_plan(capsys) -> None:
    assert main(["matrix"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert len(payload["cells"]) == 12
    assert payload["workloads"] == ["smoke-vector-add", "semantic-gemm"]
    assert len(payload["determinism_trials"]) == 3


def test_report_cli_orders_correctness_before_performance(tmp_path, capsys) -> None:
    input_path = tmp_path / "metrics.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "cell_id": "cell-a",
                    "workload": "smoke-vector-add",
                    "correct": True,
                    "kernel_time_us": 10.0,
                    "exploration_succeeded": True,
                    "iterations": 2,
                    "tokens": 100,
                    "wall_time_s": 3.0,
                    "reproducible": True,
                },
                {
                    "cell_id": "cell-b",
                    "workload": "semantic-gemm",
                    "correct": False,
                    "kernel_time_us": 1.0,
                    "exploration_succeeded": False,
                    "iterations": 3,
                    "tokens": 150,
                    "wall_time_s": 4.0,
                },
            ]
        ),
        encoding="utf-8",
    )

    assert main(["report", str(input_path)]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert output.index('"correctness"') < output.index('"kernel_performance"')
    assert payload["correctness"] == {"passed": 1, "total": 2}
    assert payload["kernel_performance"]["mean_us_correct_runs"] == 10.0
    assert payload["kernel_performance"]["by_cell_workload"] == [
        {
            "cell_id": "cell-a",
            "workload": "smoke-vector-add",
            "mean_us_correct_runs": 10.0,
            "measured": 1,
        }
    ]
    assert payload["exploration_success"] == 1
    assert payload["iterations"] == 5
    assert payload["tokens"] == 250
    assert payload["wall_time_s"] == 7.0
    assert payload["reproducibility"] == {"measured": 1, "reproducible": 1}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("correct", "false"),
        ("exploration_succeeded", 1),
        ("reproducible", "true"),
    ],
)
def test_report_rejects_non_boolean_flags(tmp_path, field, value) -> None:
    from benchmarks.a5kernels.matrix import load_metrics

    payload = {
        "cell_id": "cell-a",
        "workload": "smoke-vector-add",
        "correct": True,
        "kernel_time_us": None,
        "exploration_succeeded": False,
        "iterations": 1,
        "tokens": 1,
        "wall_time_s": 1.0,
        "reproducible": True,
    }
    payload[field] = value
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps([payload]), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_metrics(path)


def test_aggregate_report_never_treats_incorrect_timing_as_performance() -> None:
    from benchmarks.a5kernels.matrix import RunMetrics

    report = aggregate_report(
        [
            RunMetrics(
                cell_id="bad",
                workload=Workload.SEMANTIC_GEMM,
                correct=False,
                kernel_time_us=0.01,
                exploration_succeeded=True,
                iterations=1,
                tokens=1,
                wall_time_s=1,
            )
        ]
    )

    assert report["kernel_performance"]["mean_us_correct_runs"] is None


def test_aggregate_report_does_not_pool_incomparable_timings() -> None:
    from benchmarks.a5kernels.matrix import RunMetrics

    def metrics(cell_id, workload, timing):
        return RunMetrics(
            cell_id=cell_id,
            workload=workload,
            correct=True,
            kernel_time_us=timing,
            exploration_succeeded=True,
            iterations=1,
            tokens=1,
            wall_time_s=1,
        )

    report = aggregate_report(
        [
            metrics("cell-b", Workload.SEMANTIC_GEMM, 100.0),
            metrics("cell-a", Workload.SMOKE_VECTOR_ADD, 10.0),
            metrics("cell-a", Workload.SMOKE_VECTOR_ADD, 20.0),
        ]
    )

    performance = report["kernel_performance"]
    assert performance["mean_us_correct_runs"] is None
    assert performance["by_cell_workload"] == [
        {
            "cell_id": "cell-a",
            "workload": "smoke-vector-add",
            "mean_us_correct_runs": 15.0,
            "measured": 2,
        },
        {
            "cell_id": "cell-b",
            "workload": "semantic-gemm",
            "mean_us_correct_runs": 100.0,
            "measured": 1,
        },
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kernel_time_us", "1.0"),
        ("kernel_time_us", float("nan")),
        ("kernel_time_us", -1.0),
        ("iterations", True),
        ("iterations", 1.5),
        ("iterations", -1),
        ("tokens", "1"),
        ("wall_time_s", float("inf")),
        ("wall_time_s", -1.0),
    ],
)
def test_report_rejects_invalid_numeric_metrics(tmp_path, field, value) -> None:
    from benchmarks.a5kernels.matrix import load_metrics

    payload = {
        "cell_id": "cell-a",
        "workload": "smoke-vector-add",
        "correct": True,
        "kernel_time_us": 1.0,
        "exploration_succeeded": True,
        "iterations": 1,
        "tokens": 1,
        "wall_time_s": 1.0,
    }
    payload[field] = value
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps([payload]), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_metrics(path)
