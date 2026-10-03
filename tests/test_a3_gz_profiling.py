from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from benchmarks.a3kernels.phase1_protocol import VerifiedResult, attest
from benchmarks.a3kernels.profiling import (
    CandidateBinding,
    ProfileMetric,
    ProfileRequest,
    StudyDimensions,
)
from benchmarks.a3kernels.profiling_gz import GZA3ProfilingBackend


BINDING = CandidateBinding("a" * 64, "b" * 64)
DIMENSIONS = StudyDimensions(33, 2, 3, 5)


def verified(binding=BINDING):
    body = {
        "request_id": "candidate-request",
        "execution_id": binding.execution_id,
        "attempt_id": "attempt-1",
        "project_id": "vector-add",
        "passed": True,
        "max_abs_error": 0.0,
        "tolerance": 1e-5,
        "exit_code": 0,
        "output_sha256": "c" * 64,
        "source_fingerprint": binding.source_fingerprint,
        "evidence_sha256": "d" * 64,
        "job_handle": "gz-a3:verified-job",
        "mismatch": None,
    }
    return VerifiedResult(**body, attestation_sha256=attest(body))


def output(*lines):
    return "\n".join((*lines, "CATLASS_VALIDATION_PROFILE=gz-a3",
                       "CATLASS_VALIDATION_STATE=completed",
                       "CATLASS_VALIDATION_HANDLE=gz-a3:job-123",
                       "CATLASS_VALIDATION_EXIT=0")) + "\n"


def backend(tmp_path, runner, *, verification=None):
    return GZA3ProfilingBackend(
        validation_wrapper="/repo/execution-profiles/catlass-validation.sh",
        remote_candidate_directory="/data2/work/a3-candidate",
        evidence_directory=tmp_path / "evidence",
        physical_device=4,
        verified_results={BINDING.execution_id: verification or verified()},
        process_runner=runner,
        timeout=321,
    )


def test_timing_uses_exact_neutral_wrapper_device_and_logical_zero(tmp_path):
    calls = []

    def runner(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        return subprocess.CompletedProcess(argv, 0, output(
            "A3TIMING_US=10", "A3TIMING_US=12",
            'A3PROFILE_META={"language":"ascend-c","logical_device":0,'
            '"mode":"timing","remote_report":null,"runtime":"native-ascend-c",'
            '"target":"Ascend910B4"}',
        ), "")

    result = backend(tmp_path, runner).time(BINDING, DIMENSIONS)

    assert result.samples_us == (10.0, 12.0)
    argv, kwargs = calls[0]
    assert argv[:12] == (
        "/repo/execution-profiles/catlass-validation.sh", "--profile", "gz-a3",
        "--operation", "a3-timing-" + result.request_id[:16], "run", "--native",
        "--runtime", "py311-torch", "--device", "4", "--timeout",
    )
    assert argv[12:15] == ("321", "--", "python")
    assert argv[-13:] == (
        "timing", "--candidate-dir", "/data2/work/a3-candidate", "--logical-device", "0",
        "--length", "33", "--block-count", "2", "--warm-up", "3",
        "--launch-count", "5",
    )
    assert kwargs["env"] == {"PATH": kwargs["env"]["PATH"]}


def test_profile_parses_compact_raw_metrics_and_retained_report(tmp_path):
    request = ProfileRequest(BINDING, DIMENSIONS, ProfileMetric.PIPE_UTILIZATION)
    compact = {
        "exported_kernels": ["vector_add"],
        "metric_values": [["raw_vector_ratio", 0.75]],
        "timeline": [["kernel_count", 5]],
        "report_sha256": "e" * 64,
    }
    meta = {
        "language": "ascend-c", "logical_device": 0, "mode": "profile",
        "remote_report": "/data2/reports/a3-profile", "runtime": "native-ascend-c",
        "target": "Ascend910B4",
    }

    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, output(
            "A3PROFILE_COMPACT=" + json.dumps(compact, separators=(",", ":")),
            "A3PROFILE_META=" + json.dumps(meta, separators=(",", ":")),
        ), "")

    concrete = backend(tmp_path, runner)
    result = concrete.profile(request)
    evidence = concrete.evidence(request.default_replay_id)

    assert result.metric_values == (("raw_vector_ratio", 0.75),)
    assert result.report_sha256 == "e" * 64
    assert evidence.handle == "gz-a3:job-123"
    assert evidence.status == "completed"
    assert evidence.remote_report == "/data2/reports/a3-profile"
    assert len(evidence.stdout_sha256) == 64


def test_host_verification_is_required_before_any_executor_call(tmp_path):
    calls = []
    failed = verified()
    failed = VerifiedResult(**{**failed.__dict__, "passed": False})
    concrete = backend(tmp_path, lambda *a, **k: calls.append(a), verification=failed)

    with pytest.raises(ValueError, match="verified passing candidate"):
        concrete.time(BINDING, DIMENSIONS)
    assert calls == []


@pytest.mark.parametrize(
    "stdout,returncode,message",
    [
        ("CATLASS_VALIDATION_STATE=failed\n", 1, "failed"),
        (output("A3TIMING_US=1"), 1, "failed"),
        (output("A3TIMING_US=1").replace("gz-a3:job-123", "bz-a5:foreign"), 0, "handle"),
        (output("A3TIMING_US=1").replace("PROFILE=gz-a3", "PROFILE=bz-a5"), 0, "provenance"),
    ],
)
def test_terminal_failure_and_foreign_provenance_are_rejected(
    tmp_path, stdout, returncode, message,
):
    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, returncode, stdout, "failure detail")
    with pytest.raises(RuntimeError, match=message):
        backend(tmp_path, runner).time(BINDING, DIMENSIONS)


def test_replay_reuses_identical_evidence_and_rejects_conflict(tmp_path):
    calls = []

    def runner(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, output(
            "A3TIMING_US=7",
            'A3PROFILE_META={"language":"ascend-c","logical_device":0,'
            '"mode":"timing","remote_report":null,"runtime":"native-ascend-c",'
            '"target":"Ascend910B4"}',
        ), "")

    first = backend(tmp_path, runner).time(BINDING, DIMENSIONS, replay_id="replay-one")
    second = backend(tmp_path, runner).time(BINDING, DIMENSIONS, replay_id="replay-one")
    assert second == first and len(calls) == 1
    with pytest.raises(ValueError, match="conflicting replay"):
        backend(tmp_path, runner).time(
            BINDING, StudyDimensions(64, 2, 3, 5), replay_id="replay-one"
        )
    assert len(calls) == 1


def test_observation_unavailable_reopens_same_handle_without_resubmission(tmp_path):
    calls = []

    def runner(argv, **_kwargs):
        calls.append(tuple(argv))
        if len(calls) == 1:
            return subprocess.CompletedProcess(argv, 1, output(
                "A3TIMING_US=7",
                'A3PROFILE_META={"language":"ascend-c","logical_device":0,'
                '"mode":"timing","remote_report":null,"runtime":"native-ascend-c",'
                '"target":"Ascend910B4"}',
            ).replace("CATLASS_VALIDATION_STATE=completed", "CATLASS_VALIDATION_STATE=observation-unavailable"), "observer lost")
        return subprocess.CompletedProcess(argv, 0, output(
            "A3TIMING_US=7",
            'A3PROFILE_META={"language":"ascend-c","logical_device":0,'
            '"mode":"timing","remote_report":null,"runtime":"native-ascend-c",'
            '"target":"Ascend910B4"}',
        ), "")

    replay = "recover-timing"
    with pytest.raises(RuntimeError, match="retry retained handle"):
        backend(tmp_path, runner).time(BINDING, DIMENSIONS, replay_id=replay)
    pending = json.loads((tmp_path / "evidence" / replay / "pending.json").read_text())
    assert pending["handle"] == "gz-a3:job-123"

    result = backend(tmp_path, runner).time(BINDING, DIMENSIONS, replay_id=replay)

    assert result.samples_us == (7.0,)
    assert len(calls) == 2
    assert "timing" in calls[0] and "observe" not in calls[0]
    assert calls[1][-3:] == ("observe", "--handle", "gz-a3:job-123")
    assert not (tmp_path / "evidence" / replay / "pending.json").exists()
    assert (tmp_path / "evidence" / replay / "record.json").is_file()


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_execution_id", "f" * 64),
        ("source_fingerprint", "f" * 64),
        ("evidence_sha256", "f" * 64),
    ],
)
def test_timing_replay_revalidates_binding_and_digest(tmp_path, field, value):
    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, output(
            "A3TIMING_US=7",
            'A3PROFILE_META={"language":"ascend-c","logical_device":0,'
            '"mode":"timing","remote_report":null,"runtime":"native-ascend-c",'
            '"target":"Ascend910B4"}',
        ), "")

    concrete = backend(tmp_path, runner)
    concrete.time(BINDING, DIMENSIONS, replay_id="corrupt-timing")
    path = tmp_path / "evidence" / "corrupt-timing" / "record.json"
    record = json.loads(path.read_text())
    record["result"][field] = value
    path.write_text(json.dumps(record))

    with pytest.raises(RuntimeError, match="does not match"):
        concrete.time(BINDING, DIMENSIONS, replay_id="corrupt-timing")


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", "f" * 64),
        ("candidate_execution_id", "f" * 64),
        ("source_fingerprint", "f" * 64),
        ("report_sha256", "f" * 64),
        ("evidence_sha256", "f" * 64),
    ],
)
def test_profile_replay_revalidates_request_binding_and_digest(tmp_path, field, value):
    request = ProfileRequest(BINDING, DIMENSIONS, ProfileMetric.PIPE_UTILIZATION)
    compact = {
        "exported_kernels": ["vector_add"],
        "metric_values": [["vector_ratio", 0.75]],
        "timeline": [["kernel_count", 5]],
        "report_sha256": "e" * 64,
    }
    meta = {
        "language": "ascend-c", "logical_device": 0, "mode": "profile",
        "remote_report": "/data2/reports/a3-profile", "runtime": "native-ascend-c",
        "target": "Ascend910B4",
    }

    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, output(
            "A3PROFILE_COMPACT=" + json.dumps(compact, separators=(",", ":")),
            "A3PROFILE_META=" + json.dumps(meta, separators=(",", ":")),
        ), "")

    concrete = backend(tmp_path, runner)
    concrete.profile(request, replay_id="corrupt-profile")
    path = tmp_path / "evidence" / "corrupt-profile" / "record.json"
    record = json.loads(path.read_text())
    record["result"][field] = value
    path.write_text(json.dumps(record))

    with pytest.raises(RuntimeError, match="does not match"):
        concrete.profile(request, replay_id="corrupt-profile")


def test_replay_revalidates_retained_evidence_identity(tmp_path):
    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, output(
            "A3TIMING_US=7",
            'A3PROFILE_META={"language":"ascend-c","logical_device":0,'
            '"mode":"timing","remote_report":null,"runtime":"native-ascend-c",'
            '"target":"Ascend910B4"}',
        ), "")

    concrete = backend(tmp_path, runner)
    concrete.time(BINDING, DIMENSIONS, replay_id="evidence-binding")
    path = tmp_path / "evidence" / "evidence-binding" / "record.json"
    record = json.loads(path.read_text())
    record["evidence"]["request_id"] = "f" * 64
    path.write_text(json.dumps(record))

    with pytest.raises(RuntimeError, match="evidence identity"):
        concrete.time(BINDING, DIMENSIONS, replay_id="evidence-binding")


def test_non_object_replay_record_is_reported_as_corrupt(tmp_path):
    path = tmp_path / "evidence" / "broken-record" / "record.json"
    path.parent.mkdir(parents=True)
    path.write_text("[]")

    with pytest.raises(RuntimeError, match="corrupt"):
        backend(tmp_path, lambda *_a, **_k: None).time(
            BINDING, DIMENSIONS, replay_id="broken-record"
        )


def test_missing_or_conflicting_compact_output_fails_without_publication(tmp_path):
    request = ProfileRequest(BINDING, DIMENSIONS, ProfileMetric.BASIC)

    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, output(
            'A3PROFILE_META={"language":"ascend-c","logical_device":0,'
            '"mode":"profile","remote_report":"/tmp/report",'
            '"runtime":"native-ascend-c","target":"Ascend910B4"}',
        ), "")

    concrete = backend(tmp_path, runner)
    with pytest.raises(RuntimeError, match="compact"):
        concrete.profile(request)
    assert not (tmp_path / "evidence" / request.default_replay_id).exists()
