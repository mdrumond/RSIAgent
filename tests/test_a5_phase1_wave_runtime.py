from __future__ import annotations

import hashlib
import json
import threading
import time

import pytest

from benchmarks.a5kernels.phase1_curriculum import (
    Capability,
    DType,
    FailureMode,
    ProjectSpec,
    WaveProject,
    WaveSpec,
)
from benchmarks.a5kernels.phase1_wave_runtime import (
    LiveWaveCoordinator,
    WaveProjectResult,
)


MEMORY = b"immutable pre-wave memory"


def project(project_id: str) -> ProjectSpec:
    return ProjectSpec(
        project_id=project_id,
        capabilities=(Capability.GM_UB_MOVEMENT,),
        shape=(257,),
        dtype=DType.FLOAT32,
        block_count=1,
        padding_multiple=64,
        failure_mode=FailureMode.NONE,
        timing_lengths=(),
        profile_metric=None,
        hypothesis="Host verification determines whether the project worked.",
    )


def wave(*project_ids: str) -> WaveSpec:
    return WaveSpec(
        wave_id="wave-001",
        pre_wave_memory_sha256=hashlib.sha256(MEMORY).hexdigest(),
        pre_wave_coverage=(),
        projects=tuple(WaveProject(project(item)) for item in project_ids),
    )


def test_siblings_run_concurrently_from_identical_memory_and_publish_in_authored_order(
    tmp_path,
):
    active = 0
    widest = 0
    lock = threading.Lock()
    seen: dict[str, tuple[bytes, str]] = {}
    delays = {"first": 0.06, "second": 0.03, "third": 0.0}
    completion_order: list[str] = []

    def execute(request):
        nonlocal active, widest
        with lock:
            active += 1
            widest = max(widest, active)
            seen[request.assignment.project_id] = (
                request.pre_wave_memory,
                str(request.branch_root),
            )
        time.sleep(delays[request.assignment.project_id])
        with lock:
            active -= 1
            completion_order.append(request.assignment.project_id)
        return WaveProjectResult(
            memory_delta=f"lesson:{request.assignment.project_id}".encode(),
            verified=True,
        )

    coordinator = LiveWaveCoordinator(
        tmp_path, wave("first", "second", "third"), MEMORY, execute,
        max_parallel=2,
    )
    result = coordinator.run()

    assert widest == 2
    assert completion_order != ["first", "second", "third"]
    assert {item[0] for item in seen.values()} == {MEMORY}
    assert len({item[1] for item in seen.values()}) == 3
    assert [item.project_id for item in result.commits] == [
        "first", "second", "third",
    ]
    published = json.loads((tmp_path / "published.json").read_text())
    assert [item["project_id"] for item in published["commits"]] == [
        "first", "second", "third",
    ]


def test_failure_retains_isolated_success_but_publishes_nothing(tmp_path):
    def execute(request):
        return WaveProjectResult(
            memory_delta=request.assignment.project_id.encode(),
            verified=request.assignment.project_id != "bad",
        )

    coordinator = LiveWaveCoordinator(
        tmp_path, wave("good", "bad"), MEMORY, execute, max_parallel=2,
    )
    with pytest.raises(RuntimeError, match="not verified"):
        coordinator.run()

    assert (tmp_path / "branches/good/outcome.json").is_file()
    assert (tmp_path / "branches/bad/outcome.json").is_file()
    assert not (tmp_path / "published.json").exists()


def test_executor_exception_waits_for_and_retains_successful_sibling(tmp_path):
    sibling_finished = threading.Event()

    def execute(request):
        if request.assignment.project_id == "broken":
            raise OSError("provider disconnected")
        time.sleep(0.02)
        sibling_finished.set()
        return WaveProjectResult(b"retained", True)

    with pytest.raises(RuntimeError, match="broken"):
        LiveWaveCoordinator(
            tmp_path, wave("broken", "retained"), MEMORY, execute, 2,
        ).run()

    assert sibling_finished.is_set()
    assert (tmp_path / "branches/retained/outcome.json").is_file()
    assert not (tmp_path / "published.json").exists()


def test_resume_reuses_verified_branches_and_retries_failed_branch(tmp_path):
    calls: list[str] = []

    def first_attempt(request):
        calls.append(request.assignment.project_id)
        return WaveProjectResult(
            memory_delta=request.assignment.project_id.encode(),
            verified=request.assignment.project_id != "retry",
        )

    spec = wave("done", "retry", "also-done")
    with pytest.raises(RuntimeError):
        LiveWaveCoordinator(
            tmp_path, spec, MEMORY, first_attempt, max_parallel=3,
        ).run()
    assert set(calls) == {"done", "retry", "also-done"}

    resumed: list[str] = []

    def second_attempt(request):
        resumed.append(request.assignment.project_id)
        return WaveProjectResult(memory_delta=b"fixed", verified=True)

    result = LiveWaveCoordinator(
        tmp_path, spec, MEMORY, second_attempt, max_parallel=1,
    ).resume()
    assert resumed == ["retry"]
    assert [item.project_id for item in result.commits] == [
        "done", "retry", "also-done",
    ]
    assert LiveWaveCoordinator(
        tmp_path, spec, MEMORY,
        lambda _: pytest.fail("published wave must not execute again"), 3,
    ).resume() == result


def test_plan_and_memory_identity_must_match_on_resume(tmp_path):
    spec = wave("only")
    runner = LiveWaveCoordinator(
        tmp_path, spec, MEMORY,
        lambda _: WaveProjectResult(b"lesson", True),
        max_parallel=1,
    )
    runner.run()

    with pytest.raises(ValueError, match="memory identity"):
        LiveWaveCoordinator(tmp_path, spec, b"other", lambda _: None, 1)
    changed = WaveSpec(
        "wave-002", spec.pre_wave_memory_sha256, (), spec.projects,
    )
    with pytest.raises(ValueError, match="plan"):
        LiveWaveCoordinator(
            tmp_path, changed, MEMORY, lambda _: WaveProjectResult(b"x", True), 1,
        ).resume()
