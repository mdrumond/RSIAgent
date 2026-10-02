from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest

from benchmarks.a3kernels.phase1_protocol import VerifiedResult, attest
from benchmarks.a3kernels.profiling import (
    CandidateBinding,
    ProfileMetric,
    ProfileRequest,
    ProfilingTreatment,
    StudyDimensions,
    TimingResult,
)
from benchmarks.a3kernels.profiling_bz import BZA3ProfilingBackend, BZA3RunEvidence
from benchmarks.a3kernels.phase1_wave import CellPaths, Phase1Config, foundation_cells
from benchmarks.a3kernels.phase1_registry import DEFAULT_PROPOSALS


BINDING = CandidateBinding("a" * 64, "b" * 64)
DIMENSIONS = StudyDimensions(33, 2, 3, 5)


def verified() -> VerifiedResult:
    body = {
        "request_id": "candidate-request",
        "execution_id": BINDING.execution_id,
        "attempt_id": "attempt-1",
        "project_id": "vector-add",
        "passed": True,
        "max_abs_error": 0.0,
        "exit_code": 0,
        "output_sha256": "c" * 64,
        "source_fingerprint": BINDING.source_fingerprint,
        "evidence_sha256": "d" * 64,
        "job_handle": "bz-a3-1:verified-job",
    }
    return VerifiedResult(**body, attestation_sha256=attest(body))


def output(profile: str, *lines: str, state: str = "completed") -> str:
    return "\n".join((*lines, f"CATLASS_VALIDATION_PROFILE={profile}",
                       f"CATLASS_VALIDATION_STATE={state}",
                       f"CATLASS_VALIDATION_HANDLE={profile}:job-123",
                       "CATLASS_VALIDATION_EXIT=0")) + "\n"


def backend(tmp_path: Path, runner, *, profile: str = "bz-a3-1", result=None):
    return BZA3ProfilingBackend(
        validation_wrapper="/repo/execution-profiles/catlass-validation.sh",
        cpl_remote="/tools/cpl-remote",
        profile=profile,
        remote_candidate_directory="/home/user/work/candidate",
        evidence_directory=tmp_path / "evidence",
        physical_device=4,
        verified_results={BINDING.execution_id: result or verified()},
        process_runner=runner,
        timeout=321,
    )


def meta(mode: str, report=None) -> str:
    value = {
        "language": "ascend-c", "logical_device": 0, "mode": mode,
        "remote_report": report, "runtime": "native-ascend-c",
        "target": "Ascend910B4",
    }
    return "A3PROFILE_META=" + json.dumps(value, separators=(",", ":"))


def test_timing_uses_exact_bz_profile_native_runtime_and_logical_zero(tmp_path):
    calls = []

    def runner(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        return subprocess.CompletedProcess(
            argv, 0, output("bz-a3-1", "A3TIMING_US=10", "A3TIMING_US=12", meta("timing")), ""
        )

    result = backend(tmp_path, runner).time(BINDING, DIMENSIONS)

    assert result.samples_us == (10.0, 12.0)
    argv, kwargs = calls[0]
    assert argv[:12] == (
        "/repo/execution-profiles/catlass-validation.sh", "--profile", "bz-a3-1",
        "--operation", "a3-timing-" + result.request_id[:16], "run", "--native",
        "--runtime", "py311-torch", "--device", "4", "--timeout",
    )
    assert argv[12:15] == ("321", "--", "python")
    assert argv[-13:] == (
        "timing", "--candidate-dir", "/home/user/work/candidate",
        "--logical-device", "0", "--length", "33", "--block-count", "2",
        "--warm-up", "3", "--launch-count", "5",
    )
    assert kwargs["env"] == {
        "PATH": kwargs["env"]["PATH"], "CPL_REMOTE": "/tools/cpl-remote"
    }


@pytest.mark.parametrize(
    ("metric", "remote_metric"),
    [
        (ProfileMetric.BASIC, "ArithmeticUtilization"),
        (ProfileMetric.PIPE_UTILIZATION, "PipeUtilization"),
    ],
)
def test_profile_accepts_only_matching_bz_request_and_compact_raw_evidence(
    tmp_path, metric, remote_metric,
):
    compact = {
        "exported_kernels": ["vector_add"],
        "metric_values": [["raw_value", 0.75]],
        "timeline": [["kernel_count", 5]],
        "report_sha256": "e" * 64,
    }
    request = ProfileRequest(
        BINDING, DIMENSIONS, metric, execution_profile="bz-a3-2"
    )

    calls = []

    def runner(argv, **_kwargs):
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, output(
            "bz-a3-2",
            "A3PROFILE_COMPACT=" + json.dumps(compact, separators=(",", ":")),
            meta("profile", "/home/user/reports/raw-msprof"),
        ), "")

    concrete = backend(tmp_path, runner, profile="bz-a3-2")
    result = concrete.profile(request)
    evidence = concrete.evidence(request.default_replay_id)

    assert result.metric is metric
    assert calls[0][calls[0].index("--metric") + 1] == remote_metric
    assert result.metric_values == (("raw_value", 0.75),)
    assert result.report_sha256 == "e" * 64
    assert evidence == BZA3RunEvidence(
        replay_id=request.default_replay_id,
        request_id=request.request_id,
        mode="profile",
        profile="bz-a3-2",
        handle="bz-a3-2:job-123",
        status="completed",
        physical_device=4,
        logical_device=0,
        remote_report="/home/user/reports/raw-msprof",
        stdout_sha256=evidence.stdout_sha256,
        stderr_sha256=evidence.stderr_sha256,
    )
    assert not hasattr(result, "a5_bandwidth")

    with pytest.raises(ValueError, match="matching BZ-A3 execution profile"):
        concrete.profile(replace(request, execution_profile="bz-a3-1"))


def test_host_verified_pass_is_required_before_dispatch(tmp_path):
    calls = []
    failed = replace(verified(), passed=False)
    concrete = backend(tmp_path, lambda *a, **k: calls.append(a), result=failed)
    with pytest.raises(ValueError, match="verified passing candidate"):
        concrete.time(BINDING, DIMENSIONS)
    assert calls == []


def test_off_treatment_returns_none_without_dispatch(tmp_path):
    calls = []
    request = ProfileRequest(
        BINDING, DIMENSIONS, ProfileMetric.PIPE_UTILIZATION,
        treatment=ProfilingTreatment.OFF, execution_profile="bz-a3-1",
    )
    assert backend(tmp_path, lambda *a, **k: calls.append(a)).profile(request) is None
    assert calls == []


def test_observation_reuses_same_bz_handle_and_never_resubmits(tmp_path):
    calls = []

    def runner(argv, **_kwargs):
        calls.append(tuple(argv))
        if len(calls) == 1:
            return subprocess.CompletedProcess(
                argv, 1, output("bz-a3-1", meta("timing"), state="observation-unavailable"), "lost"
            )
        return subprocess.CompletedProcess(
            argv, 0, output("bz-a3-1", "A3TIMING_US=7", meta("timing")), ""
        )

    concrete = backend(tmp_path, runner)
    with pytest.raises(RuntimeError, match="retry retained handle bz-a3-1:job-123"):
        concrete.time(BINDING, DIMENSIONS, replay_id="recover")
    result = concrete.time(BINDING, DIMENSIONS, replay_id="recover")

    assert result.samples_us == (7.0,)
    assert "run" in calls[0] and "observe" not in calls[0]
    assert calls[1][-3:] == ("observe", "--handle", "bz-a3-1:job-123")
    assert not (tmp_path / "evidence/recover/pending.json").exists()


def test_foreign_profile_and_handle_are_rejected(tmp_path):
    def foreign_profile(argv, **_kwargs):
        return subprocess.CompletedProcess(
            argv, 0, output("bz-a3-2", "A3TIMING_US=7", meta("timing")), ""
        )
    with pytest.raises(RuntimeError, match="provenance"):
        backend(tmp_path, foreign_profile).time(BINDING, DIMENSIONS)

    def foreign_handle(argv, **_kwargs):
        text = output("bz-a3-1", "A3TIMING_US=7", meta("timing"))
        return subprocess.CompletedProcess(argv, 0, text.replace(
            "bz-a3-1:job-123", "bz-a3-2:job-123"
        ), "")
    with pytest.raises(RuntimeError, match="handle"):
        backend(tmp_path, foreign_handle).time(BINDING, DIMENSIONS)


def phase1_config(tmp_path: Path) -> Phase1Config:
    wrapper = tmp_path / "catlass-validation.sh"
    wrapper.write_text("#!/bin/sh\n")
    wrapper.chmod(0o755)
    return Phase1Config(
        state_root=tmp_path / "state",
        validation_wrapper=wrapper,
        embedding_cache=tmp_path / "embeddings",
        corpus_artifacts=tmp_path / "corpus",
        knowledge_database=tmp_path / "knowledge.sqlite3",
        knowledge_manifest=tmp_path / "manifest.json",
    )


def test_bz_live_factory_binds_user_wide_transport_profile_workspace_and_device(tmp_path):
    from benchmarks.a3kernels.live_composition import bz_live_dependencies
    from benchmarks.a3kernels.remote_candidate_bz import BZA3RemoteCandidateBackend

    cpl_remote = tmp_path / "cpl-remote"
    cpl_remote.write_text("#!/bin/sh\n")
    cpl_remote.chmod(0o755)
    config = phase1_config(tmp_path)
    dependencies = bz_live_dependencies(
        config,
        cpl_remote=str(cpl_remote),
        profile="bz-a3-2",
        remote_workspace="/home/user/rsi",
        physical_device=3,
        environ={},
    )
    paths = CellPaths(
        tmp_path / "cell", tmp_path / "cell/workspace",
        tmp_path / "cell/memory", tmp_path / "cell/evidence",
        tmp_path / "cell/terminal", tmp_path / "cell/driver.py",
    )
    bundle = dependencies.candidate_factory(
        foundation_cells()[0], DEFAULT_PROPOSALS[0], paths
    )

    assert isinstance(bundle.backend, BZA3RemoteCandidateBackend)
    assert bundle.backend.profile == "bz-a3-2"
    assert bundle.backend._cpl_remote == str(cpl_remote)
    assert bundle.backend._workspace.as_posix() == "/home/user/rsi"
    assert bundle.backend._device == 3


def test_lazy_bz_live_profile_and_timing_bind_exact_verified_candidate(
    tmp_path, monkeypatch,
):
    import benchmarks.a3kernels.live_composition as live
    from benchmarks.a3kernels.remote_candidate_bz import BZA3RemoteCandidateBackend

    config = phase1_config(tmp_path)
    managed = BZA3RemoteCandidateBackend(
        cpl_remote="/tools/cpl-remote",
        validation_wrapper=str(config.validation_wrapper),
        profile="bz-a3-1",
        remote_workspace="/home/user/rsi",
        physical_device=2,
    )
    bundle = live.RemoteCandidateBundle(managed)
    plan = bundle.planner.plan(
        '''extern "C" __global__ __aicore__ void vector_add(
        GM_ADDR input_a, GM_ADDR input_b, GM_ADDR output,
        uint32_t count, uint32_t buffer_bytes) {}''',
        request_id="cell", attempt_id="turn-1", project_id="project",
        length=33, padded_length=64, block_count=2, seed=0,
        execution_profile="bz-a3-1",
    )
    bundle.plans[plan.execution_id] = plan
    store = live.AuthoritativeResultStore(tmp_path / "authority.jsonl")
    candidate = live._RecordingCandidate(bundle, store)
    proof = verified()
    proof = replace(
        proof, execution_id=plan.execution_id,
        source_fingerprint=plan.source_fingerprint,
    )
    proof = replace(proof, attestation_sha256=attest(proof.attestation_payload()))
    candidate.verified[plan.execution_id] = proof
    paths = CellPaths(
        tmp_path / "cell", tmp_path / "cell/workspace",
        tmp_path / "cell/memory", tmp_path / "cell/evidence",
        tmp_path / "cell/terminal", tmp_path / "cell/driver.py",
    )
    seen = []

    class InjectedProfiler:
        def __init__(self, **kwargs):
            seen.append(kwargs)
        def profile(self, request):
            assert request.execution_profile == "bz-a3-1"
            return __import__(
                "benchmarks.a3kernels.profiling", fromlist=["CompactProfileResult"]
            ).CompactProfileResult.create(
                request, exported_kernels=("vector_add",),
                metric_values=(("raw", 1.0),), timeline=(("count", 1.0),),
                report_sha256="9" * 64,
            )
        def time(self, binding, dimensions):
            return TimingResult.from_samples(
                binding, dimensions, (2.0, 3.0), execution_profile="bz-a3-1"
            )

    monkeypatch.setattr(live, "BZA3ProfilingBackend", InjectedProfiler)
    lazy = live._LazyBZProfiler(
        config=config, paths=paths, candidate=candidate,
        profile="bz-a3-1", physical_device=2, cpl_remote="/tools/cpl-remote",
    )
    request = ProfileRequest(
        CandidateBinding(plan.execution_id, plan.source_fingerprint),
        DIMENSIONS, ProfileMetric.PIPE_UTILIZATION,
    )

    profiled = lazy.profile(request)
    timed = lazy.time(request.binding, DIMENSIONS)

    assert profiled.request_id != request.request_id
    assert timed.samples_us == (2.0, 3.0)
    assert seen[0]["remote_candidate_directory"].endswith(plan.execution_id)
    assert seen[0]["verified_results"] == {plan.execution_id: proof}


def test_lazy_profilers_do_not_construct_backend_for_off_treatment(tmp_path, monkeypatch):
    import benchmarks.a3kernels.live_composition as live

    constructed = []
    monkeypatch.setattr(
        live, "BZA3ProfilingBackend",
        lambda **kwargs: constructed.append(kwargs),
    )
    request = ProfileRequest(
        BINDING, DIMENSIONS, ProfileMetric.PIPE_UTILIZATION,
        treatment=ProfilingTreatment.OFF,
    )
    lazy = object.__new__(live._LazyBZProfiler)
    lazy.profile_name = "bz-a3-1"
    assert lazy.profile(request) is None
    assert constructed == []
    lazy_gz = object.__new__(live._LazyGZProfiler)
    assert lazy_gz.profile(request) is None


def test_live_factory_passes_exact_cpl_remote_to_profiler(tmp_path, monkeypatch):
    import benchmarks.a3kernels.live_composition as live
    from benchmarks.a3kernels.remote_candidate_bz import BZA3RemoteCandidateBackend

    config = phase1_config(tmp_path)
    managed = BZA3RemoteCandidateBackend(
        cpl_remote="/selected/cpl-remote",
        validation_wrapper=str(config.validation_wrapper), profile="bz-a3-1",
        remote_workspace="/home/user/rsi", physical_device=2,
    )
    bundle = live.RemoteCandidateBundle(managed)
    plan = bundle.planner.plan(
        '''extern "C" __global__ __aicore__ void vector_add(
        GM_ADDR input_a, GM_ADDR input_b, GM_ADDR output,
        uint32_t count, uint32_t buffer_bytes) {}''',
        request_id="cell", attempt_id="turn-1", project_id="project",
        length=33, padded_length=64, block_count=2, seed=0,
        execution_profile="bz-a3-1",
    )
    bundle.plans[plan.execution_id] = plan
    candidate = live._RecordingCandidate(
        bundle, live.AuthoritativeResultStore(tmp_path / "authority.jsonl")
    )
    proof = replace(
        verified(), execution_id=plan.execution_id,
        source_fingerprint=plan.source_fingerprint,
    )
    proof = replace(proof, attestation_sha256=attest(proof.attestation_payload()))
    candidate.verified[plan.execution_id] = proof
    seen = {}
    class Injected:
        def __init__(self, **kwargs): seen.update(kwargs)
        def profile(self, request): return None
    monkeypatch.setattr(live, "BZA3ProfilingBackend", Injected)
    lazy = live._LazyBZProfiler(
        config=config,
        paths=CellPaths(tmp_path / "cell", tmp_path / "workspace", tmp_path / "memory",
                        tmp_path / "evidence", tmp_path / "terminal", tmp_path / "driver"),
        candidate=candidate, profile="bz-a3-1", physical_device=2,
        cpl_remote="/selected/cpl-remote",
    )
    lazy.profile(ProfileRequest(
        CandidateBinding(plan.execution_id, plan.source_fingerprint),
        DIMENSIONS, ProfileMetric.PIPE_UTILIZATION,
    ))
    assert seen["cpl_remote"] == "/selected/cpl-remote"
