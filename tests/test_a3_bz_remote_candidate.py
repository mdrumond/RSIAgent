from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import subprocess

import pytest

from benchmarks.a3kernels.candidate import A3CandidateBackend, CandidateCompilation
from benchmarks.a3kernels.phase1_protocol import FailedEvidence, VerifiedResult
from benchmarks.a3kernels.remote_candidate_bz import BZA3RemoteCandidateBackend


SOURCE = r'''#include "kernel_operator.h"
extern "C" __global__ __aicore__ void vector_add(
    GM_ADDR input_a, GM_ADDR input_b, GM_ADDR output,
    uint32_t count, uint32_t buffer_bytes) {}
'''


def _plan(profile="bz-a3-1"):
    return A3CandidateBackend(lambda *a, **k: None).plan(
        SOURCE, request_id="request", attempt_id="attempt", length=3,
        padded_length=4, block_count=2, seed=4, execution_profile=profile,
    )


def _completed(argv, stdout="", stderr="", code=0):
    return subprocess.CompletedProcess(argv, code, stdout + ("\n" if stdout else ""), stderr)


def _terminal(profile, handle, *body, state="completed", code=0, stderr=""):
    stdout = "\n".join((*body,
        f"CATLASS_VALIDATION_PROFILE={profile}",
        f"CATLASS_VALIDATION_STATE={state}",
        f"CATLASS_VALIDATION_HANDLE={handle}",
        f"CATLASS_VALIDATION_EXIT={code}",
    ))
    return _completed((), stdout, stderr, code)


def _upload_terminal(profile="bz-a3-1"):
    return "\n".join((
        f"REMOTE_TARGET={profile}",
        "REMOTE_BACKEND=ssh",
        "REMOTE_OPERATION=upload",
        "REMOTE_STATE=completed",
        "REMOTE_EXIT=0",
    ))


def _backend(tmp_path: Path, process, *, profile="bz-a3-1", device=2):
    return BZA3RemoteCandidateBackend(
        cpl_remote="/skills/remote-access/scripts/cpl-remote",
        validation_wrapper="/checked/catlass-validation.sh",
        profile=profile,
        remote_workspace="/home/research",
        physical_device=device,
        process_runner=process,
        state_directory=tmp_path / "state",
    )


def test_complete_bz_flow_uploads_compiles_and_host_verifies(tmp_path, monkeypatch):
    calls = []
    plan = _plan()
    expected = [a + b for a, b in zip(plan.input_a, plan.input_b)]
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")

    def process(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        if argv[0].endswith("cpl-remote"):
            return _completed(argv, _upload_terminal())
        if "a3-candidate-" in " ".join(argv):
            result = _terminal(
                "bz-a3-1", "bz-a3-1:compile-9",
                "A3REMOTE_STAGE=compile", "A3CANDIDATE_COMPILED=" + "a" * 64,
            )
        else:
            result = _terminal(
                "bz-a3-1", "bz-a3-1:execute-9",
                "A3REMOTE_STAGE=execute", "A3KERNEL_OUTPUT=" + json.dumps(expected),
            )
        return subprocess.CompletedProcess(argv, result.returncode, result.stdout, result.stderr)

    backend = _backend(tmp_path, process)
    compilation = backend.compile(plan, tmp_path / "local")
    result = backend.execute(compilation)

    assert isinstance(compilation, CandidateCompilation)
    assert isinstance(result, VerifiedResult) and result.passed
    assert result.max_abs_error == 0
    assert result.job_handle == "bz-a3-1:execute-9"
    upload = calls[0][0]
    assert upload[:3] == (
        "/skills/remote-access/scripts/cpl-remote", "upload", "bz-a3-1",
    )
    remote_archive = PurePosixPath(upload[-1])
    assert remote_archive.parent == PurePosixPath("/home/research")
    assert remote_archive.name == f".rsi-a3-candidate-{plan.execution_id}.tar"
    compile_argv, execute_argv = calls[1][0], calls[2][0]
    wrapper_env = {
        "PATH": os.environ["PATH"],
        "CPL_REMOTE": "/skills/remote-access/scripts/cpl-remote",
    }
    for (argv, kwargs) in calls[1:]:
        assert argv[:4] == (
            "/checked/catlass-validation.sh", "--profile", "bz-a3-1", "--operation",
        )
        assert ("--device", "2") == argv[argv.index("--device"):argv.index("--device") + 2]
        assert ("--runtime", "py311-torch") == argv[argv.index("--runtime"):argv.index("--runtime") + 2]
        assert kwargs["env"] == wrapper_env
        assert not any("KEY" in name or "SECRET" in name for name in kwargs["env"])
    assert "torch.npu.set_device(0)" in plan.files[3].content


@pytest.mark.parametrize("profile", ["bz-a3-1", "bz-a3-2"])
def test_only_declared_bz_profiles_are_accepted(tmp_path, profile):
    assert _backend(tmp_path, lambda *a, **k: None, profile=profile).profile == profile


def test_bz_backend_rejects_plan_for_another_registered_profile(tmp_path):
    backend = _backend(tmp_path, lambda *a, **k: None)

    with pytest.raises(ValueError, match="execution profile"):
        backend.compile(_plan("bz-a3-2"), tmp_path / "local")
    with pytest.raises(ValueError, match="execution profile"):
        backend.compile(_plan("gz-a3"), tmp_path / "local")


@pytest.mark.parametrize("profile", ["gz-a3", "bz-a5", "bz-a3-3"])
def test_foreign_profiles_are_rejected(tmp_path, profile):
    with pytest.raises(ValueError, match="profile"):
        _backend(tmp_path, lambda *a, **k: None, profile=profile)


def test_physical_device_is_required(tmp_path):
    with pytest.raises(ValueError, match="physical_device"):
        _backend(tmp_path, lambda *a, **k: None, device=None)


def test_upload_and_compile_failures_are_structured(tmp_path):
    upload = _backend(
        tmp_path / "upload", lambda argv, **kwargs: _completed(argv, stderr="vpn down", code=2)
    ).compile(_plan(), tmp_path / "upload" / "local")
    assert isinstance(upload, FailedEvidence)
    assert upload.stage == "prepare" and "vpn down" in upload.detail

    def compile_failure(argv, **kwargs):
        if argv[0].endswith("cpl-remote"):
            return _completed(argv, _upload_terminal())
        result = _terminal(
            "bz-a3-1", "bz-a3-1:compile-failed",
            "A3REMOTE_STAGE=compile", state="failed", code=2, stderr="bisheng failed",
        )
        return subprocess.CompletedProcess(argv, result.returncode, result.stdout, result.stderr)

    failed = _backend(tmp_path / "compile", compile_failure).compile(
        _plan(), tmp_path / "compile" / "local"
    )
    assert isinstance(failed, FailedEvidence)
    assert failed.stage == "compile" and "bisheng failed" in failed.detail


def test_runtime_failure_is_structured_and_does_not_reupload(tmp_path):
    calls = []

    def process(argv, **kwargs):
        calls.append(tuple(argv))
        if argv[0].endswith("cpl-remote"):
            return _completed(argv, _upload_terminal())
        if "a3-candidate-" in " ".join(argv):
            result = _terminal(
                "bz-a3-1", "bz-a3-1:compile", "A3REMOTE_STAGE=compile",
                "A3CANDIDATE_COMPILED=" + "b" * 64,
            )
        else:
            result = _terminal(
                "bz-a3-1", "bz-a3-1:execute", "A3REMOTE_STAGE=execute",
                state="failed", code=3, stderr="runtime failed",
            )
        return subprocess.CompletedProcess(argv, result.returncode, result.stdout, result.stderr)

    backend = _backend(tmp_path, process)
    compilation = backend.compile(_plan(), tmp_path / "local")
    failed = backend.execute(compilation)

    assert isinstance(failed, FailedEvidence)
    assert failed.stage == "execute" and "runtime failed" in failed.detail
    assert sum(call[0].endswith("cpl-remote") for call in calls) == 1


def test_observation_unavailable_recovers_exact_compile_handle_without_upload(
    tmp_path, monkeypatch,
):
    calls = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")

    def process(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        if argv[0].endswith("cpl-remote"):
            return _completed(argv, _upload_terminal())
        if "observe" in argv:
            result = _terminal(
                "bz-a3-1", "bz-a3-1:compile-pending", "A3REMOTE_STAGE=compile",
                "A3CANDIDATE_COMPILED=" + "c" * 64,
            )
        else:
            result = _terminal(
                "bz-a3-1", "bz-a3-1:compile-pending",
                state="observation-unavailable", code=75, stderr="transport lost",
            )
        return subprocess.CompletedProcess(argv, result.returncode, result.stdout, result.stderr)

    with pytest.raises(RuntimeError, match="retry retained handle"):
        _backend(tmp_path, process).compile(_plan(), tmp_path / "local")
    compilation = _backend(tmp_path, process).compile(_plan(), tmp_path / "local")

    assert isinstance(compilation, CandidateCompilation)
    assert sum(call[0][0].endswith("cpl-remote") for call in calls) == 1
    observe_argv, observe_kwargs = calls[-1]
    assert observe_argv[-3:] == ("observe", "--handle", "bz-a3-1:compile-pending")
    assert observe_kwargs["env"] == {
        "PATH": os.environ["PATH"],
        "CPL_REMOTE": "/skills/remote-access/scripts/cpl-remote",
    }


def test_pending_observation_rejects_foreign_handle_and_remains_recoverable(tmp_path):
    calls = []

    def process(argv, **kwargs):
        calls.append(tuple(argv))
        if argv[0].endswith("cpl-remote"):
            return _completed(argv, _upload_terminal())
        handle = "bz-a3-2:foreign" if "observe" in argv else "bz-a3-1:pending"
        state = "completed" if "observe" in argv else "observation-unavailable"
        code = 0 if "observe" in argv else 75
        result = _terminal(
            "bz-a3-1", handle, "A3CANDIDATE_COMPILED=" + "d" * 64,
            state=state, code=code,
        )
        return subprocess.CompletedProcess(argv, result.returncode, result.stdout, result.stderr)

    backend = _backend(tmp_path, process)
    with pytest.raises(RuntimeError, match="retry retained handle"):
        backend.compile(_plan(), tmp_path / "local")
    failed = backend.compile(_plan(), tmp_path / "local")

    assert isinstance(failed, FailedEvidence)
    assert failed.stage == "compile" and "foreign handle" in failed.detail
    pending = tmp_path / "state" / _plan().execution_id / "compile.json"
    assert json.loads(pending.read_text())["handle"] == "bz-a3-1:pending"


def test_corrupt_or_cross_profile_pending_state_fails_closed(tmp_path):
    plan = _plan()
    state = tmp_path / "state" / plan.execution_id
    state.mkdir(parents=True)
    (state / "compile.json").write_text(json.dumps({
        "execution_id": plan.execution_id,
        "source_fingerprint": plan.source_fingerprint,
        "operation": "compile",
        "handle": "bz-a3-2:foreign",
    }))

    with pytest.raises(RuntimeError, match="corrupt"):
        _backend(tmp_path, lambda *a, **k: None).compile(plan, tmp_path / "local")


@pytest.mark.parametrize("stdout", [
    "",
    "REMOTE_STATE=completed",
    _upload_terminal("bz-a3-2"),
    _upload_terminal() + "\nREMOTE_TARGET=bz-a3-1",
    _upload_terminal().replace("REMOTE_STATE=completed", "REMOTE_STATE=running"),
])
def test_zero_exit_upload_rejects_missing_duplicate_foreign_or_nonterminal_markers(
    tmp_path, stdout,
):
    result = _backend(
        tmp_path, lambda argv, **kwargs: _completed(argv, stdout)
    ).compile(_plan(), tmp_path / "local")

    assert isinstance(result, FailedEvidence)
    assert result.stage == "prepare"
    assert "terminal markers" in result.detail
