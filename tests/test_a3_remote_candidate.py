from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import tarfile

from benchmarks.a3kernels.candidate import A3CandidateBackend
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
        plan = _plan()
        output = [a + b for a, b in zip(plan.input_a, plan.input_b)]
        stdout = "\n".join((
            "A3REMOTE_STAGE=compile", "A3CANDIDATE_COMPILED=" + "a" * 64,
            "A3REMOTE_STAGE=execute", "A3KERNEL_OUTPUT=" + json.dumps(output),
            "CATLASS_VALIDATION_PROFILE=gz-a3",
            "CATLASS_VALIDATION_STATE=completed",
            "CATLASS_VALIDATION_HANDLE=gz-a3:job-9",
            "CATLASS_VALIDATION_EXIT=0", "",
        ))
        return _completed(argv, stdout)

    result = _backend(tmp_path, process).run(_plan(), tmp_path / "local")

    assert isinstance(result, VerifiedResult) and result.passed
    assert result.job_handle == "gz-a3:job-9"
    upload = calls[0][0]
    assert upload[:4] == ("/checked/remote_agent_client.sh", "--server", "http://127.0.0.1:37787", "transfer")
    assert upload[4:7] == ("upload", "--remote", "a3-gz")
    assert calls[1][0][-2:] == ("--job-id", "upload-7")
    wrapper = calls[2][0]
    assert wrapper[:8] == (
        "/checked/catlass-validation.sh", "--profile", "gz-a3", "--operation",
        "a3-candidate-" + _plan().execution_id[:16], "run", "--native", "--runtime",
    )
    assert "py311-torch" in wrapper and "--device" in wrapper
    bash = wrapper.index("bash")
    assert wrapper[bash:bash + 2] == ("bash", "-c")
    assert str(_backend(tmp_path, process).remote_candidate_directory(_plan())) in wrapper


def test_listener_gate_fails_closed_before_upload(tmp_path):
    calls = []
    backend = _backend(
        tmp_path, lambda *args, **kwargs: calls.append(args),
        health=lambda: _health(transport="paramiko"),
    )

    result = backend.run(_plan(), tmp_path / "local")

    assert isinstance(result, FailedEvidence)
    assert result.stage == "prepare" and "asyncssh" in result.detail
    assert calls == []


def test_failed_transfer_and_remote_compile_are_structured(tmp_path):
    def transfer_failure(argv, **kwargs):
        return _completed(argv, "", "upload unavailable", 2)

    upload = _backend(tmp_path / "upload", transfer_failure).run(
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

    failed = _backend(tmp_path / "compile", compile_failure).run(
        _plan(), tmp_path / "compile" / "local"
    )
    assert isinstance(failed, FailedEvidence)
    assert failed.stage == "compile" and "bisheng error" in failed.detail


def _backend(tmp_path: Path, process, *, health=_health):
    return GZA3RemoteCandidateBackend(
        client="/checked/remote_agent_client.sh",
        server="http://127.0.0.1:37787",
        validation_wrapper="/checked/catlass-validation.sh",
        remote="a3-gz", remote_workspace="/data2/research",
        physical_device=2, health_reader=health, process_runner=process,
        poll_interval=0, sleeper=lambda _: None,
    )


def _completed(argv, stdout, stderr="", code=0):
    return subprocess.CompletedProcess(argv, code, stdout + ("\n" if stdout else ""), stderr)
