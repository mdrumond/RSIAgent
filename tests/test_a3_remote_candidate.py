from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from benchmarks.a3kernels.candidate import A3CandidateBackend
from benchmarks.a3kernels.candidate import CandidateCompilation
from benchmarks.a3kernels.phase1_protocol import FailedEvidence, VerifiedResult
from benchmarks.a3kernels.remote_candidate import GZA3RemoteCandidateBackend


SOURCE = r'''#include "kernel_operator.h"
extern "C" __global__ __aicore__ void vector_add(
    GM_ADDR input_a, GM_ADDR input_b, GM_ADDR output,
    uint32_t count, uint32_t buffer_bytes) {}
'''


def _plan():
    return A3CandidateBackend(lambda *a, **k: None).plan(
        SOURCE, request_id="request", attempt_id="attempt", length=3,
        padded_length=4, block_count=2, seed=4,
    )


def _health(*, transport="asyncssh"):
    return {
        "session_metadata": {
            "execution_mode": "listener",
            "listener_ssh_transport": transport,
        }
    }


def test_bundle_is_deterministic_identity_bound_and_contains_checked_assets(tmp_path):
    backend = _backend(tmp_path, lambda *a, **k: None)
    first = backend.build_bundle(_plan(), tmp_path / "first.tar")
    second = backend.build_bundle(_plan(), tmp_path / "second.tar")

    assert first.sha256 == second.sha256
    assert first.path.read_bytes() == second.path.read_bytes()
    with tarfile.open(first.path) as archive:
        names = archive.getnames()
        manifest = json.load(archive.extractfile("manifest.json"))
    assert names == sorted(names)
    assert "a3_profile_driver.py" in names
    assert manifest["execution_id"] == _plan().execution_id
    assert manifest["source_fingerprint"] == _plan().source_fingerprint
    assert manifest["archive_schema"] == "rsi-a3-candidate-v1"


def test_remote_vertical_flow_health_upload_poll_wrapper_and_verify(tmp_path):
    calls = []

    def process(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        if "transfer" in argv and "upload" in argv:
            return _completed(argv, '{"id":"upload-7","status":"queued"}')
        if "transfer" in argv and "status" in argv:
            return _completed(argv, '{"id":"upload-7","status":"succeeded"}')
        plan = _plan(); executing = "a3-execute-" in " ".join(argv)
        output = [a + b for a, b in zip(plan.input_a, plan.input_b)]
        body = (
            ("A3REMOTE_STAGE=execute", "A3KERNEL_OUTPUT=" + json.dumps(output))
            if executing else
            ("A3REMOTE_STAGE=compile", "A3CANDIDATE_COMPILED=" + "a" * 64)
        )
        stdout = "\n".join((*body,
            "CATLASS_VALIDATION_PROFILE=gz-a3",
            "CATLASS_VALIDATION_STATE=completed",
            "CATLASS_VALIDATION_HANDLE=" + ("gz-a3:run-9" if executing else "gz-a3:compile-9"),
            "CATLASS_VALIDATION_EXIT=0", "",
        ))
        return _completed(argv, stdout)

    backend = _backend(tmp_path, process)
    compilation = backend.compile(_plan(), tmp_path / "local")
    assert isinstance(compilation, CandidateCompilation)
    result = backend.execute(compilation)

    assert isinstance(result, VerifiedResult) and result.passed
    assert result.job_handle == "gz-a3:run-9"
    upload = calls[0][0]
    assert upload[:4] == ("/checked/remote_agent_client.sh", "--server", "http://127.0.0.1:37787", "transfer")
    assert upload[4:7] == ("upload", "--remote", "a3-gz")
    assert calls[1][0][-2:] == ("--job-id", "upload-7")
    wrapper, execute = calls[2][0], calls[3][0]
    assert wrapper[:8] == (
        "/checked/catlass-validation.sh", "--profile", "gz-a3", "--operation",
        "a3-candidate-" + _plan().execution_id[:16], "run", "--native", "--runtime",
    )
    assert "py311-torch" in wrapper and "--device" in wrapper
    bash = wrapper.index("bash")
    assert wrapper[bash:bash + 2] == ("bash", "-c")
    assert str(_backend(tmp_path, process).remote_candidate_directory(_plan())) in wrapper
    assert "a3-execute-" + _plan().execution_id[:16] in execute
    assert "host_driver.py input.json" in execute[execute.index("bash") + 2]


def test_listener_gate_fails_closed_before_upload(tmp_path):
    calls = []
    backend = _backend(
        tmp_path, lambda *args, **kwargs: calls.append(args),
        health=lambda: _health(transport="paramiko"),
    )

    result = backend.compile(_plan(), tmp_path / "local")

    assert isinstance(result, FailedEvidence)
    assert result.stage == "prepare" and "asyncssh" in result.detail
    assert calls == []


def test_gz_backend_rejects_bz_plan_before_transfer(tmp_path):
    plan = A3CandidateBackend(lambda *a, **k: None).plan(
        SOURCE, request_id="request", attempt_id="attempt", length=3,
        padded_length=4, block_count=2, seed=4, execution_profile="bz-a3-1",
    )

    with pytest.raises(ValueError, match="execution profile"):
        _backend(tmp_path, lambda *a, **k: None).compile(plan, tmp_path / "local")


def test_failed_transfer_and_remote_compile_are_structured(tmp_path):
    def transfer_failure(argv, **kwargs):
        return _completed(argv, "", "upload unavailable", 2)

    upload = _backend(tmp_path / "upload", transfer_failure).compile(
        _plan(), tmp_path / "upload" / "local"
    )
    assert isinstance(upload, FailedEvidence) and upload.stage == "prepare"

    def compile_failure(argv, **kwargs):
        if "upload" in argv:
            return _completed(argv, '{"id":"u1","status":"queued"}')
        if "status" in argv:
            return _completed(argv, '{"id":"u1","status":"succeeded"}')
        return _completed(
            argv,
            "A3REMOTE_STAGE=compile\nCATLASS_VALIDATION_PROFILE=gz-a3\n"
            "CATLASS_VALIDATION_STATE=failed\nCATLASS_VALIDATION_HANDLE=gz-a3:j1\n"
            "CATLASS_VALIDATION_EXIT=2\n", "bisheng error", 2,
        )

    failed = _backend(tmp_path / "compile", compile_failure).compile(
        _plan(), tmp_path / "compile" / "local"
    )
    assert isinstance(failed, FailedEvidence)
    assert failed.stage == "compile" and "bisheng error" in failed.detail


def test_execute_failure_is_runtime_evidence_and_does_not_reupload(tmp_path):
    calls = []
    def process(argv, **kwargs):
        calls.append(tuple(argv))
        if "upload" in argv: return _completed(argv, '{"id":"u1","status":"queued"}')
        if "status" in argv: return _completed(argv, '{"id":"u1","status":"succeeded"}')
        if "a3-candidate-" in " ".join(argv):
            return _completed(argv, "A3REMOTE_STAGE=compile\nA3CANDIDATE_COMPILED=" + "a" * 64 + "\nCATLASS_VALIDATION_PROFILE=gz-a3\nCATLASS_VALIDATION_STATE=completed\nCATLASS_VALIDATION_HANDLE=gz-a3:c\nCATLASS_VALIDATION_EXIT=0")
        return _completed(argv, "A3REMOTE_STAGE=execute\nCATLASS_VALIDATION_PROFILE=gz-a3\nCATLASS_VALIDATION_STATE=failed\nCATLASS_VALIDATION_HANDLE=gz-a3:r\nCATLASS_VALIDATION_EXIT=2", "runtime failed", 2)
    backend = _backend(tmp_path, process)
    compilation = backend.compile(_plan(), tmp_path / "local")
    result = backend.execute(compilation)
    assert isinstance(result, FailedEvidence) and result.stage == "execute"
    assert sum("upload" in call for call in calls) == 1


def test_compile_observer_loss_reopens_same_handle_without_reupload(tmp_path):
    calls = []
    observations = 0
    def process(argv, **kwargs):
        nonlocal observations
        calls.append(tuple(argv))
        if "upload" in argv: return _completed(argv, '{"id":"u1","status":"queued"}')
        if "status" in argv: return _completed(argv, '{"id":"u1","status":"succeeded"}')
        if "observe" in argv:
            observations += 1
            if observations == 1:
                return _completed(argv, "", "listener temporarily unavailable", 2)
            return _completed(argv, "A3REMOTE_STAGE=compile\nA3CANDIDATE_COMPILED=" + "a" * 64 + "\nCATLASS_VALIDATION_PROFILE=gz-a3\nCATLASS_VALIDATION_STATE=completed\nCATLASS_VALIDATION_HANDLE=gz-a3:compile-pending\nCATLASS_VALIDATION_EXIT=0")
        return _completed(argv, "CATLASS_VALIDATION_PROFILE=gz-a3\nCATLASS_VALIDATION_STATE=observation-unavailable\nCATLASS_VALIDATION_HANDLE=gz-a3:compile-pending\nCATLASS_VALIDATION_EXIT=1", "observer lost", 1)

    with pytest.raises(RuntimeError, match="retry retained handle"):
        _backend(tmp_path, process).compile(_plan(), tmp_path / "local")
    with pytest.raises(RuntimeError, match="retry retained handle"):
        _backend(tmp_path, process).compile(_plan(), tmp_path / "local")
    compilation = _backend(tmp_path, process).compile(_plan(), tmp_path / "local")

    assert isinstance(compilation, CandidateCompilation)
    assert sum("upload" in call for call in calls) == 1
    assert calls[-1][-3:] == ("observe", "--handle", "gz-a3:compile-pending")


def test_execute_observer_loss_reopens_same_handle_without_resubmission(tmp_path):
    calls = []
    plan = _plan()
    output = json.dumps([a + b for a, b in zip(plan.input_a, plan.input_b)])
    def process(argv, **kwargs):
        calls.append(tuple(argv))
        if "upload" in argv: return _completed(argv, '{"id":"u1","status":"queued"}')
        if "status" in argv: return _completed(argv, '{"id":"u1","status":"succeeded"}')
        if "a3-candidate-" in " ".join(argv):
            return _completed(argv, "A3REMOTE_STAGE=compile\nA3CANDIDATE_COMPILED=" + "b" * 64 + "\nCATLASS_VALIDATION_PROFILE=gz-a3\nCATLASS_VALIDATION_STATE=completed\nCATLASS_VALIDATION_HANDLE=gz-a3:compile\nCATLASS_VALIDATION_EXIT=0")
        if "observe" in argv:
            return _completed(argv, "A3REMOTE_STAGE=execute\nA3KERNEL_OUTPUT=" + output + "\nCATLASS_VALIDATION_PROFILE=gz-a3\nCATLASS_VALIDATION_STATE=completed\nCATLASS_VALIDATION_HANDLE=gz-a3:execute-pending\nCATLASS_VALIDATION_EXIT=0")
        return _completed(argv, "CATLASS_VALIDATION_PROFILE=gz-a3\nCATLASS_VALIDATION_STATE=observation-unavailable\nCATLASS_VALIDATION_HANDLE=gz-a3:execute-pending\nCATLASS_VALIDATION_EXIT=1", "observer lost", 1)

    first = _backend(tmp_path, process)
    compilation = first.compile(plan, tmp_path / "local")
    with pytest.raises(RuntimeError, match="retry retained handle"):
        first.execute(compilation)
    result = _backend(tmp_path, process).execute(compilation)

    assert isinstance(result, VerifiedResult) and result.passed
    execute_submissions = [call for call in calls if "a3-execute-" in " ".join(call) and "run" in call]
    assert len(execute_submissions) == 1
    assert calls[-1][-3:] == ("observe", "--handle", "gz-a3:execute-pending")


def _backend(tmp_path: Path, process, *, health=_health):
    return GZA3RemoteCandidateBackend(
        client="/checked/remote_agent_client.sh",
        server="http://127.0.0.1:37787",
        validation_wrapper="/checked/catlass-validation.sh",
        remote="a3-gz", remote_workspace="/data2/research",
        physical_device=2, health_reader=health, process_runner=process,
        poll_interval=0, sleeper=lambda _: None,
        state_directory=tmp_path / "state",
    )


def _completed(argv, stdout, stderr="", code=0):
    return subprocess.CompletedProcess(argv, code, stdout + ("\n" if stdout else ""), stderr)
