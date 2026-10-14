from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

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
    fixture_metrics,
)
from run_a5_sft_experiment import main


def config() -> StudyConfig:
    return StudyConfig(
        base_model_id="accounts/fireworks/models/deepseek-v4-flash",
        adapter_model_id="accounts/example/models/rsi-lora-job-123",
        frozen_memory_id="sha256:memory",
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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seeds", (17, 17, 91), "three distinct seeds"),
        ("base_model_id", "", "base_model_id"),
        ("adapter_model_id", "", "adapter_model_id"),
        ("frozen_memory_id", "memory-latest", "immutable"),
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
    metrics = fixture_metrics(build_matrix(config()), task_id="fixture-vector-add")

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

    with pytest.raises(ValueError, match="matched run keys"):
        aggregate_matrix(metrics[:-1])


def test_fixture_cli_is_deterministic_and_machine_readable(capsys) -> None:
    args = [
        "fixture",
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
    assert payload["run_count"] == 72
    assert payload["effects"]["W+H-max(W,H)"]["hap"] == 1.0


def test_run_metric_rejects_inconsistent_correctness_and_regression() -> None:
    cell = build_matrix(config()).cells[0]
    with pytest.raises(ValueError, match="correct run must compile"):
        RunMetric(
            cell=cell, task_id="task", compiled=False, correct=True, hap=1.0,
            latency_ms=1.0, cost_usd=0.1, regressed=False, repairs=0,
        )
