from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from benchmarks.a5kernels.sft_experiment import (
    AcquisitionIncomplete,
    Arm,
    BenchmarkMode,
    ContractTrack,
    LeaderboardEntry,
    PublicSourceBundle,
    RunMetric,
    StudyConfig,
    acquire_public_baselines,
    aggregate_matrix,
    build_matrix,
    synthetic_demo_metrics,
)
from run_a5_sft_experiment import main


def config() -> StudyConfig:
    return StudyConfig(
        base_model_id="accounts/fireworks/models/deepseek-v4-flash",
        adapter_model_id="accounts/example/models/rsi-lora-job-123",
        frozen_memory_id="sha256:" + "a" * 64,
        seeds=(17, 42, 91),
        max_tokens=32768,
        max_tool_calls=24,
        temperature=0.2,
        top_p=0.95,
    )


def test_four_arm_matrix_is_balanced_and_treatments_are_isolated() -> None:
    matrix = build_matrix(config())

    assert len(matrix.cells) == 4 * 2 * 3 * 3
    assert {cell.arm for cell in matrix.cells} == set(Arm)
    assert {cell.mode for cell in matrix.cells} == set(BenchmarkMode)
    assert {cell.track for cell in matrix.cells} == set(ContractTrack)
    assert {cell.seed for cell in matrix.cells} == {17, 42, 91}
    for cell in matrix.cells:
        assert cell.base_model_id == config().base_model_id
        assert cell.adapter_model_id == (
            config().adapter_model_id if cell.arm in {Arm.W, Arm.W_H} else None
        )
        assert cell.memory_id == (
            config().frozen_memory_id if cell.arm in {Arm.H, Arm.W_H} else None
        )
        assert cell.max_tokens == 32768
        assert cell.max_tool_calls == 24
        assert cell.temperature == 0.2
        assert cell.top_p == 0.95
        assert cell.study_fingerprint == config().fingerprint
        assert cell.study_seeds == (17, 42, 91)


def test_study_fingerprint_is_deterministic_and_part_of_cell_identity() -> None:
    first = config()
    same = StudyConfig(**first.__dict__)
    changed = StudyConfig(**{**first.__dict__, "max_tokens": 16384})

    assert first.fingerprint == same.fingerprint
    assert first.fingerprint != changed.fingerprint
    assert build_matrix(first).cells[0].cell_id != build_matrix(changed).cells[0].cell_id


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seeds", (17, 17, 91), "three distinct seeds"),
        ("base_model_id", "", "base_model_id"),
        ("adapter_model_id", "", "adapter_model_id"),
        ("frozen_memory_id", "memory-latest", "immutable"),
        ("frozen_memory_id", "sha256:abc", "64 lowercase hex"),
        ("frozen_memory_id", "sha256:" + "A" * 64, "64 lowercase hex"),
    ],
)
def test_study_config_rejects_missing_artifacts_or_unbalanced_seeds(field, value, message) -> None:
    values = dict(config().__dict__)
    values[field] = value
    with pytest.raises(ValueError, match=message):
        StudyConfig(**values)


def test_study_config_and_cells_are_immutable() -> None:
    study = config()
    cell = build_matrix(study).cells[0]

    with pytest.raises(FrozenInstanceError):
        study.base_model_id = "accounts/fireworks/models/changed"
    with pytest.raises(FrozenInstanceError):
        cell.adapter_model_id = "accounts/example/models/changed"


def test_leaderboard_walk_accepts_only_reproducible_public_source_bundles() -> None:
    entries = tuple(LeaderboardEntry(rank=i, submission_id=f"s{i}") for i in range(1, 6))

    def acquire(entry: LeaderboardEntry) -> PublicSourceBundle:
        if entry.rank == 1:
            raise ValueError("no public source link")
        if entry.rank == 3:
            return PublicSourceBundle(
                "https://github.com/acme/private", "abc1234", (), True, ("pytest",)
            )
        return PublicSourceBundle(
            f"https://github.com/acme/public-{entry.rank}",
            f"deadbee{entry.rank}",
            ("kernel.py", "README.md"),
            True,
            ("python", "verify.py"),
        )

    result = acquire_public_baselines(entries, acquire, minimum=3)

    assert [item.entry.rank for item in result.accepted] == [2, 4, 5]
    assert [(item.entry.rank, item.reason) for item in result.skipped] == [
        (1, "no public source link"),
        (3, "source bundle has no files"),
    ]
    assert result.examined == 5


def test_leaderboard_walk_fails_with_manifest_instead_of_inventing_sources() -> None:
    entries = (LeaderboardEntry(1, "missing"), LeaderboardEntry(2, "only-one"))

    def acquire(entry: LeaderboardEntry) -> PublicSourceBundle:
        if entry.rank == 1:
            raise ValueError("repository unavailable")
        return PublicSourceBundle(
            "https://github.com/acme/one", "deadbeef", ("kernel.py",), True,
            ("python", "verify.py"),
        )

    with pytest.raises(AcquisitionIncomplete, match="required 3.*acquired 1") as error:
        acquire_public_baselines(entries, acquire, minimum=3)

    assert len(error.value.manifest.accepted) == 1
    assert error.value.manifest.skipped[0].reason == "repository unavailable"


def test_aggregate_requires_matched_cells_and_reports_declared_effects() -> None:
    metrics = synthetic_demo_metrics(build_matrix(config()), task_id="synthetic-vector-add")

    report = aggregate_matrix(metrics)

    assert report["run_count"] == 72
    assert report["arms"]["base"]["compile_rate"] == 1.0
    assert report["arms"]["w+h"]["correctness_rate"] == 1.0
    assert report["effects"]["H-Base"]["hap"] == pytest.approx(1.0)
    assert report["effects"]["W-Base"]["hap"] == pytest.approx(2.0)
    assert report["effects"]["W+H-max(W,H)"]["hap"] == pytest.approx(1.0)
    assert report["effects"]["W+H-max(W,H)"]["latency_ms"] == pytest.approx(-1.0)
    assert report["effects"]["W+H-max(W,H)"]["cost_usd"] == pytest.approx(0.02)
    assert report["metric_directions"]["hap"] == "higher-is-better"
    assert report["metric_directions"]["latency_ms"] == "lower-is-better"
    assert report["arms"]["w+h"]["public_baseline_hap_improvement"] == pytest.approx(4.0)
    assert set(report["by_mode"]) == {mode.value for mode in BenchmarkMode}
    assert set(report["by_track"]) == {track.value for track in ContractTrack}
    assert report["by_track"]["changed-contract"]["effects"]["W-Base"]["hap"] == 2.0
    for metric in (
        "compile_rate", "correctness_rate", "hap", "latency_ms", "cost_usd",
        "regression_rate", "mean_repairs",
    ):
        assert metric in report["arms"]["base"]

    with pytest.raises(ValueError, match="complete factorial"):
        aggregate_matrix(metrics[:-1])


def test_aggregate_rejects_balanced_missing_factorial_cells() -> None:
    metrics = synthetic_demo_metrics(build_matrix(config()), task_id="synthetic-vector-add")
    incomplete = tuple(
        row for row in metrics
        if not (
            row.cell.mode is BenchmarkMode.SPEC_GENERATION
            and row.cell.track is ContractTrack.ORIGINAL
            and row.cell.seed == 17
        )
    )

    with pytest.raises(ValueError, match="complete factorial"):
        aggregate_matrix(incomplete)


@pytest.mark.parametrize(
    ("field", "value", "arm"),
    [
        ("base_model_id", "accounts/fireworks/models/other", Arm.BASE),
        ("adapter_model_id", "accounts/example/models/other-adapter", Arm.W),
        ("memory_id", "sha256:" + "b" * 64, Arm.H),
        ("max_tokens", 4096, Arm.BASE),
        ("max_tool_calls", 12, Arm.BASE),
        ("temperature", 0.7, Arm.BASE),
        ("top_p", 0.8, Arm.BASE),
    ],
)
def test_aggregate_rejects_cross_study_control_drift(field, value, arm) -> None:
    metrics = list(
        synthetic_demo_metrics(build_matrix(config()), task_id="synthetic-vector-add")
    )
    index = next(index for index, row in enumerate(metrics) if row.cell.arm is arm)
    metrics[index] = replace(
        metrics[index], cell=replace(metrics[index].cell, **{field: value})
    )

    with pytest.raises(ValueError, match="study configuration drift"):
        aggregate_matrix(metrics)


def test_study_cell_rejects_invalid_arm_treatment_assignment() -> None:
    base = build_matrix(config()).cells[0]
    with pytest.raises(ValueError, match="base arm must not use"):
        replace(base, adapter_model_id=config().adapter_model_id)


@pytest.mark.parametrize("field", ["public_baseline_id", "public_baseline_hap"])
def test_aggregate_rejects_mismatched_public_baseline(field) -> None:
    metrics = list(
        synthetic_demo_metrics(build_matrix(config()), task_id="synthetic-vector-add")
    )
    index = next(
        index for index, row in enumerate(metrics)
        if row.cell.arm is Arm.W_H
        and row.cell.mode is BenchmarkMode.OPTIMIZE_PUBLIC_BASELINE
    )
    value = "different-baseline" if field == "public_baseline_id" else 8.0
    metrics[index] = replace(metrics[index], **{field: value})

    with pytest.raises(ValueError, match="same public baseline identity and value"):
        aggregate_matrix(metrics)


def test_synthetic_demo_cli_is_deterministic_and_disclaims_execution(capsys) -> None:
    args = [
        "synthetic-demo",
        "--base-model-id", config().base_model_id,
        "--adapter-model-id", config().adapter_model_id,
        "--memory-id", config().frozen_memory_id,
    ]
    assert main(args) == 0
    first = capsys.readouterr().out
    assert main(args) == 0
    second = capsys.readouterr().out

    assert first == second
    payload = json.loads(first)
    assert payload["evidence_kind"] == "synthetic-demo-not-execution-evidence"
    assert payload["executed"] is False
    assert payload["run_count"] == 72
    assert payload["effects"]["W+H-max(W,H)"]["hap"] == 1.0


def test_run_metric_rejects_inconsistent_correctness_and_regression() -> None:
    cell = build_matrix(config()).cells[0]
    with pytest.raises(ValueError, match="correct run must compile"):
        RunMetric(
            cell=cell, task_id="task", compiled=False, correct=True, hap=1.0,
            latency_ms=1.0, cost_usd=0.1, regressed=False, repairs=0,
        )
