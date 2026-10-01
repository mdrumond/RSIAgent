from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest

from benchmarks.a3kernels.phase1_protocol import VerifiedResult, attest
from benchmarks.a3kernels.profiling import (
    CandidateBinding, ProfileMetric, ProfileRequest, ProfilingTreatment,
    StudyDimensions,
)
from benchmarks.a3kernels.profiling_bz import BZA3ProfilingBackend, BZA3RunEvidence


BINDING = CandidateBinding("a" * 64, "b" * 64)
DIMENSIONS = StudyDimensions(33, 2, 3, 5)


def verified() -> VerifiedResult:
    body = {
        "request_id": "candidate-request", "execution_id": BINDING.execution_id,
        "attempt_id": "attempt-1", "project_id": "vector-add", "passed": True,
        "max_abs_error": 0.0, "tolerance": 1e-5, "exit_code": 0,
        "output_sha256": "c" * 64, "source_fingerprint": BINDING.source_fingerprint,
        "evidence_sha256": "d" * 64, "job_handle": "bz-a3-1:verified-job",
    }
    return VerifiedResult(**body, attestation_sha256=attest(body))


def output(profile: str, *lines: str, state: str = "completed") -> str:
    return "\n".join((
        *lines, f"CATLASS_VALIDATION_PROFILE={profile}",
        f"CATLASS_VALIDATION_STATE={state}",
        f"CATLASS_VALIDATION_HANDLE={profile}:job-123", "CATLASS_VALIDATION_EXIT=0",
    )) + "\n"


def metadata(mode: str, report=None) -> str:
    value = {
        "language": "ascend-c", "logical_device": 0, "mode": mode,
        "remote_report": report, "runtime": "native-ascend-c", "target": "Ascend910B4",
    }
    return "A3PROFILE_META=" + json.dumps(value, separators=(",", ":"))


def backend(tmp_path: Path, runner, *, profile="bz-a3-1", result=None):
    return BZA3ProfilingBackend(
        validation_wrapper="/repo/execution-profiles/catlass-validation.sh",
        cpl_remote="/tools/cpl-remote", profile=profile,
        remote_candidate_directory="/home/user/work/candidate",
        evidence_directory=tmp_path / "evidence", physical_device=4,
        verified_results={BINDING.execution_id: result or verified()},
        process_runner=runner, timeout=321,
    )


def test_timing_uses_exact_profile_transport_and_logical_zero(tmp_path):
    calls = []

    def runner(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        return subprocess.CompletedProcess(
            argv, 0,
            output("bz-a3-1", "A3TIMING_US=10", "A3TIMING_US=12", metadata("timing")), "",
        )

    result = backend(tmp_path, runner).time(BINDING, DIMENSIONS)
    assert result.samples_us == (10.0, 12.0)
    assert result.execution_profile == "bz-a3-1"
    argv, kwargs = calls[0]
    assert argv[:12] == (
        "/repo/execution-profiles/catlass-validation.sh", "--profile", "bz-a3-1",
        "--operation", "a3-timing-" + result.request_id[:16], "run", "--native",
        "--runtime", "py311-torch", "--device", "4", "--timeout",
    )
    assert argv[12:15] == ("321", "--", "python")
    assert argv[-13:] == (
        "timing", "--candidate-dir", "/home/user/work/candidate", "--logical-device", "0",
        "--length", "33", "--block-count", "2", "--warm-up", "3", "--launch-count", "5",
    )
    assert kwargs["env"]["CPL_REMOTE"] == "/tools/cpl-remote"


@pytest.mark.parametrize(
    ("metric", "remote_metric"),
    [(ProfileMetric.BASIC, "ArithmeticUtilization"),
     (ProfileMetric.PIPE_UTILIZATION, "PipeUtilization")],
)
def test_profile_preserves_abstract_metric_and_compact_raw_evidence(
    tmp_path, metric, remote_metric,
):
    compact = {
        "exported_kernels": ["vector_add"], "metric_values": [["raw_value", 0.75]],
        "timeline": [["kernel_count", 5]], "report_sha256": "e" * 64,
    }
    request = ProfileRequest(BINDING, DIMENSIONS, metric, execution_profile="bz-a3-2")
    calls = []

    def runner(argv, **_kwargs):
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, output(
            "bz-a3-2", "A3PROFILE_COMPACT=" + json.dumps(compact),
            metadata("profile", "/home/user/reports/raw-msprof"),
        ), "")

    concrete = backend(tmp_path, runner, profile="bz-a3-2")
    result = concrete.profile(request)
    evidence = concrete.evidence(request.default_replay_id)
    assert result.metric is metric
    assert calls[0][calls[0].index("--metric") + 1] == remote_metric
    assert result.metric_values == (("raw_value", 0.75),)
    assert result.timeline == (("kernel_count", 5.0),)
    assert not hasattr(result, "a5_bandwidth")
    assert evidence == BZA3RunEvidence(
        request.default_replay_id, request.request_id, "profile", "bz-a3-2",
        "bz-a3-2:job-123", "completed", 4, 0,
        "/home/user/reports/raw-msprof", evidence.stdout_sha256, evidence.stderr_sha256,
    )


def test_requires_verified_candidate_and_matching_request_without_dispatch(tmp_path):
    calls = []
    failed = replace(verified(), passed=False)
    with pytest.raises(ValueError, match="verified passing candidate"):
        backend(tmp_path, lambda *a, **k: calls.append(a), result=failed).time(
            BINDING, DIMENSIONS
        )
    request = ProfileRequest(
        BINDING, DIMENSIONS, ProfileMetric.BASIC, execution_profile="bz-a3-2"
    )
    with pytest.raises(ValueError, match="matching BZ-A3"):
        backend(tmp_path, lambda *a, **k: calls.append(a)).profile(request)
    assert calls == []


def test_off_treatment_does_not_validate_or_dispatch(tmp_path):
    calls = []
    request = ProfileRequest(
        BINDING, DIMENSIONS, ProfileMetric.PIPE_UTILIZATION,
        treatment=ProfilingTreatment.OFF, execution_profile="bz-a3-1",
    )
    assert backend(tmp_path, lambda *a, **k: calls.append(a), result=object()).profile(request) is None
    assert calls == []


def test_observation_reuses_retained_handle_and_never_resubmits(tmp_path):
    calls = []

    def runner(argv, **_kwargs):
        calls.append(tuple(argv))
        if len(calls) == 1:
            return subprocess.CompletedProcess(
                argv, 1, output("bz-a3-1", metadata("timing"),
                                state="observation-unavailable"), "lost",
            )
        return subprocess.CompletedProcess(
            argv, 0, output("bz-a3-1", "A3TIMING_US=7", metadata("timing")), "",
        )

    concrete = backend(tmp_path, runner)
    with pytest.raises(RuntimeError, match="retry retained handle bz-a3-1:job-123"):
        concrete.time(BINDING, DIMENSIONS, replay_id="recover")
    result = concrete.time(BINDING, DIMENSIONS, replay_id="recover")
    assert result.samples_us == (7.0,)
    assert "run" in calls[0] and "observe" not in calls[0]
    assert calls[1][-3:] == ("observe", "--handle", "bz-a3-1:job-123")
    assert not (tmp_path / "evidence/recover/pending.json").exists()


def test_completed_replay_is_content_addressed_and_does_not_dispatch(tmp_path):
    calls = []

    def runner(argv, **_kwargs):
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(
            argv, 0, output("bz-a3-1", "A3TIMING_US=7", metadata("timing")), "",
        )

    first = backend(tmp_path, runner).time(BINDING, DIMENSIONS, replay_id="retained")
    second = backend(tmp_path, runner).time(BINDING, DIMENSIONS, replay_id="retained")
    assert first == second
    assert len(calls) == 1


@pytest.mark.parametrize("fault", ["profile", "handle", "metadata"])
def test_foreign_provenance_is_rejected(tmp_path, fault):
    def runner(argv, **_kwargs):
        text = output("bz-a3-1", "A3TIMING_US=7", metadata("timing"))
        if fault == "profile":
            text = text.replace("PROFILE=bz-a3-1", "PROFILE=bz-a3-2")
        elif fault == "handle":
            text = text.replace("HANDLE=bz-a3-1:", "HANDLE=bz-a3-2:")
        else:
            text = text.replace('"logical_device":0', '"logical_device":1')
        return subprocess.CompletedProcess(argv, 0, text, "")

    with pytest.raises(RuntimeError, match="foreign"):
        backend(tmp_path, runner).time(BINDING, DIMENSIONS)


def test_unsafe_replay_and_corrupt_retained_evidence_are_rejected(tmp_path):
    concrete = backend(tmp_path, lambda *_a, **_k: None)
    with pytest.raises(ValueError, match="unsafe"):
        concrete.time(BINDING, DIMENSIONS, replay_id="../escape")
    path = tmp_path / "evidence/corrupt/record.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="corrupt"):
        concrete.time(BINDING, DIMENSIONS, replay_id="corrupt")
