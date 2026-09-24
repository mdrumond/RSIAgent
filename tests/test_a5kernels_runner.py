from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.a5kernels import (
    A5KernelRunner,
    BZSessionAdapter,
    CatlassValidationExecutor,
    Language,
    ProfileCommandExecutor,
    RunRequest,
    fixture_for,
)
from benchmarks.a5kernels.bz import (
    CommandResult,
    OUTPUT_MARKER,
    RuntimeUnavailableError,
)
from benchmarks.a5kernels.protocol import ExecutionReceipt, SourceFile


def _write_fake_catlass(root: Path, *, compatible: bool) -> str:
    package = root / "catlass"
    package.mkdir()
    (package / "__init__.py").write_text("")
    exports = (
        "AddressSpace = allocate = compile = flag = kernel = vector = object()\n"
        if compatible
        else "from_dlpack = object()\n"
    )
    (package / "tla.py").write_text(exports)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=A5 Test",
            "-c",
            "user.email=a5@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


class FakeBackend:
    def __init__(
        self,
        *,
        corrupt=False,
        exit_code=0,
        claimed_score="0",
        runtime_provenance=(),
    ):
        self.corrupt = corrupt
        self.exit_code = exit_code
        self.claimed_score = claimed_score
        self.runtime_provenance = runtime_provenance
        self.plans = []

    def execute(self, plan):
        self.plans.append(plan)
        output = tuple(a + b for a, b in zip(plan.input_a, plan.input_b))
        if self.corrupt:
            output = (output[0] + 1,) + output[1:]
        return ExecutionReceipt(
            exit_code=self.exit_code,
            output=output,
            stdout="hello from A5",
            session_handle="bz-a5:test-session",
            metadata=(("claimed_score", self.claimed_score),),
        )


@pytest.mark.parametrize("language", list(Language))
def test_each_language_fixture_runs_through_host_oracle(language):
    backend = FakeBackend(claimed_score="0")
    result = A5KernelRunner(backend).run(RunRequest(language.value, length=7, seed=42))

    assert result.passed is True
    assert result.max_abs_error == 0
    assert backend.plans[0].language == language.value
    assert backend.plans[0].source_fingerprint == result.source_fingerprint
    assert result.runtime_provenance == ()
    assert len(result.attestation_sha256) == 64


def test_agent_claim_cannot_turn_wrong_output_into_pass():
    backend = FakeBackend(corrupt=True, claimed_score="1.0")

    result = A5KernelRunner(backend).run(RunRequest(Language.CATLASS_DSL.value))

    assert result.passed is False
    assert result.max_abs_error == pytest.approx(1.0)


def test_agent_claim_cannot_turn_correct_output_into_failure():
    backend = FakeBackend(claimed_score="0.0")

    result = A5KernelRunner(backend).run(RunRequest(Language.ASCEND_C.value))

    assert result.passed is True


def test_nonzero_exit_fails_even_with_correct_output():
    result = A5KernelRunner(FakeBackend(exit_code=9, claimed_score="1")).run(
        RunRequest(Language.TRITON_ASCEND.value)
    )
    assert result.passed is False


def test_requests_and_results_are_immutable_and_repeatable():
    request = RunRequest(Language.CATLASS_DSL.value, length=4, seed=3)
    runner = A5KernelRunner(FakeBackend())

    first = runner.run(request, attempt_id="repeatable-1")
    second = runner.run(request, attempt_id="repeatable-1")

    assert first == second
    with pytest.raises(FrozenInstanceError):
        request.seed = 99


def test_fixture_registry_rejects_unknown_languages():
    with pytest.raises(ValueError, match="unsupported language"):
        fixture_for("cuda")


@pytest.mark.parametrize("length", [0, -1])
def test_invalid_lengths_are_rejected(length):
    with pytest.raises(ValueError, match="positive"):
        RunRequest(Language.CATLASS_DSL.value, length=length)


class FakeCommandExecutor:
    def __init__(self, dispatch, inspections=(), runtime_provenance=()):
        self.dispatch = dispatch
        self.inspections = list(inspections)
        self.invocations = []
        self.inspect_argv = []
        self.runtime_provenance = runtime_provenance

    def run(self, invocation):
        self.invocations.append(invocation)
        return self.dispatch

    def inspect(self, argv):
        self.inspect_argv.append(argv)
        return self.inspections.pop(0)


def test_bz_adapter_retrieves_retained_logs_and_result_before_parsing():
    command = FakeCommandExecutor(
        CommandResult(0, f"wrapper-only\n{OUTPUT_MARKER}[999]\n"),
        (
            CommandResult(0, f"banner\n{OUTPUT_MARKER}[1.25,2]\n"),
            CommandResult(0, "SESSION_STATE=finished exit=0", session_handle="bz-a5:s1"),
        ),
    )
    backend = BZSessionAdapter(
        command, session_wrapper="execution-profiles/bz-a5/session.sh"
    )
    plan = A5KernelRunner(FakeBackend()).prepare(
        RunRequest(Language.CATLASS_DSL.value, length=2), attempt_id="trial-1"
    )

    receipt = backend.execute(plan)

    assert receipt.output == (1.25, 2.0)
    assert receipt.session_handle == "bz-a5:s1"
    invocation = command.invocations[0]
    assert invocation.argv[:4] == (
        "execution-profiles/bz-a5/session.sh",
        "--name",
        f"codex-a5hello-{plan.execution_id[:8]}-trial-1",
        "run",
    )
    assert invocation.files[:-1] == plan.files
    assert invocation.files[-1].relative_path == "input.json"
    assert json.loads(invocation.files[-1].content) == {
        "input_a": list(plan.input_a),
        "input_b": list(plan.input_b),
    }
    assert invocation.remote_directory == f".a5kernels/{plan.execution_id}/trial-1"
    assert invocation.argv[-4:] == (
        "python",
        f".a5kernels/{plan.execution_id}/trial-1/host_driver.py",
        f".a5kernels/{plan.execution_id}/trial-1/kernel.py",
        f".a5kernels/{plan.execution_id}/trial-1/input.json",
    )
    assert invocation.stdin == ""
    assert [argv[-1] for argv in command.inspect_argv] == ["logs", "result"]


def test_bz_adapter_does_not_accept_stale_evidence_after_dispatch_failure():
    command = FakeCommandExecutor(
        CommandResult(9, "upload failed", "transfer error"),
        (
            CommandResult(0, f"{OUTPUT_MARKER}[3]\n"),
            CommandResult(0, "old success"),
        ),
    )
    backend = BZSessionAdapter(
        command, session_wrapper="execution-profiles/bz-a5/session.sh"
    )
    plan = A5KernelRunner(FakeBackend()).prepare(
        RunRequest(Language.CATLASS_DSL.value, length=1), attempt_id="trial-1"
    )

    receipt = backend.execute(plan)

    assert receipt.exit_code == 9
    assert receipt.output == ()
    assert receipt.stdout == "upload failed"
    assert receipt.stderr == "transfer error"
    assert command.inspect_argv == []


def test_bz_adapter_observes_only_explicitly_identified_uncertain_dispatch():
    plan = A5KernelRunner(FakeBackend()).prepare(
        RunRequest(Language.CATLASS_DSL.value, length=1), attempt_id="trial-1"
    )
    session = f"codex-a5hello-{plan.execution_id[:8]}-trial-1"
    command = FakeCommandExecutor(
        CommandResult(
            75,
            "CATLASS_VALIDATION_STATE=observation-unavailable\n"
            f"CATLASS_VALIDATION_HANDLE=bz-a5:{session}\n",
        ),
        (
            CommandResult(0, f"{OUTPUT_MARKER}[3]\n"),
            CommandResult(0, "exit=0"),
        ),
    )

    receipt = BZSessionAdapter(
        command, session_wrapper="execution-profiles/bz-a5/session.sh"
    ).execute(plan)

    assert receipt.output == (3.0,)
    assert [argv[-1] for argv in command.inspect_argv] == ["logs", "result"]


def test_catlass_fixture_uses_current_imperative_runtime_api():
    fixture = fixture_for(Language.CATLASS_DSL)
    sources = {item.relative_path: item.content for item in fixture.files}

    assert fixture.argv == ("python", "host_driver.py", "kernel.py", "input.json")
    assert set(sources) == {"host_driver.py", "kernel.py"}
    assert "tla.allocate" in sources["kernel.py"]
    assert 'tla.vec.func(mode="simd")' in sources["kernel.py"]
    assert ".mark_compact_shape_dynamic(0)" in sources["kernel.py"]
    assert 'options="--npu-arch 3510"' in sources["kernel.py"]
    assert "torch.npu.synchronize()" in sources["kernel.py"]


def test_staged_host_driver_emits_single_numeric_record_with_fake_kernel(tmp_path):
    fixture = fixture_for(Language.CATLASS_DSL)
    driver = next(item for item in fixture.files if item.relative_path == "host_driver.py")
    (tmp_path / driver.relative_path).write_text(driver.content)
    (tmp_path / "kernel.py").write_text(
        "def run(a, b):\n    return [left + right for left, right in zip(a, b)]\n"
    )
    payload = json.dumps({"input_a": [1, 2.5], "input_b": [3, -0.5]})
    (tmp_path / "input.json").write_text(payload)
    revision = _write_fake_catlass(tmp_path, compatible=True)

    result = subprocess.run(
        [sys.executable, "host_driver.py", "kernel.py", "input.json", revision],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(tmp_path), "CATLASS_SRC": str(tmp_path)},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{OUTPUT_MARKER}[4.0,2.0]\n"


def test_host_driver_preflight_rejects_legacy_catlass_before_kernel_import(tmp_path):
    fixture = fixture_for(Language.CATLASS_DSL)
    driver = next(item for item in fixture.files if item.relative_path == "host_driver.py")
    (tmp_path / driver.relative_path).write_text(driver.content)
    (tmp_path / "input.json").write_text(
        json.dumps({"input_a": [1], "input_b": [2]})
    )
    (tmp_path / "kernel.py").write_text(
        "from pathlib import Path\nPath('kernel-imported').touch()\n"
    )
    revision = _write_fake_catlass(tmp_path, compatible=False)

    result = subprocess.run(
        [sys.executable, "host_driver.py", "kernel.py", "input.json", revision],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(tmp_path), "CATLASS_SRC": str(tmp_path)},
    )

    assert result.returncode != 0
    assert "incompatible Catlass DSL runtime" in result.stderr
    assert "kernel" in result.stderr
    assert str(tmp_path / "catlass" / "tla.py") in result.stderr
    assert not (tmp_path / "kernel-imported").exists()


def test_host_driver_preflight_rejects_wrong_retained_revision(tmp_path):
    fixture = fixture_for(Language.CATLASS_DSL)
    sources = {item.relative_path: item.content for item in fixture.files}
    for name, content in sources.items():
        (tmp_path / name).write_text(content)
    (tmp_path / "input.json").write_text(
        json.dumps({"input_a": [1], "input_b": [2]})
    )
    _write_fake_catlass(tmp_path, compatible=True)

    result = subprocess.run(
        [sys.executable, "host_driver.py", "kernel.py", "input.json", "0" * 40],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(tmp_path), "CATLASS_SRC": str(tmp_path)},
    )

    assert result.returncode != 0
    assert "Catlass source revision mismatch" in result.stderr
    assert "expected " + "0" * 40 in result.stderr


@pytest.mark.parametrize("dirty_kind", ["tracked", "untracked"])
def test_host_driver_preflight_rejects_dirty_retained_source(tmp_path, dirty_kind):
    fixture = fixture_for(Language.CATLASS_DSL)
    sources = {item.relative_path: item.content for item in fixture.files}
    for name, content in sources.items():
        (tmp_path / name).write_text(content)
    (tmp_path / "input.json").write_text(
        json.dumps({"input_a": [1], "input_b": [2]})
    )
    revision = _write_fake_catlass(tmp_path, compatible=True)
    if dirty_kind == "tracked":
        dirty_path = tmp_path / "catlass" / "tla.py"
        dirty_path.write_text(dirty_path.read_text() + "changed = True\n")
    else:
        dirty_path = tmp_path / "untracked-change.py"
        dirty_path.write_text("changed = True\n")

    result = subprocess.run(
        [sys.executable, "host_driver.py", "kernel.py", "input.json", revision],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(tmp_path), "CATLASS_SRC": str(tmp_path)},
    )

    assert result.returncode != 0
    assert "Catlass retained source is not clean" in result.stderr
    assert dirty_path.name in result.stderr


def test_host_driver_preflight_rejects_ignored_imported_artifact(tmp_path):
    fixture = fixture_for(Language.CATLASS_DSL)
    sources = {item.relative_path: item.content for item in fixture.files}
    for name, content in sources.items():
        (tmp_path / name).write_text(content)
    (tmp_path / "input.json").write_text(
        json.dumps({"input_a": [1], "input_b": [2]})
    )
    revision = _write_fake_catlass(tmp_path, compatible=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "A5 Test"], check=True
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "a5@example.invalid"],
        check=True,
    )
    (tmp_path / ".gitignore").write_text("catlass/runtime_shadow.py\n")
    tracked_tla = tmp_path / "catlass" / "tla.py"
    tracked_tla.write_text("import catlass.runtime_shadow\n" + tracked_tla.read_text())
    subprocess.run(["git", "-C", str(tmp_path), "add", ".gitignore", "catlass/tla.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "load runtime artifact"], check=True)
    revision = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    (tmp_path / "catlass" / "runtime_shadow.py").write_text("SHADOW = True\n")
    assert subprocess.run(
        ["git", "-C", str(tmp_path), "status", "--porcelain", "--untracked-files=all"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout == ""

    result = subprocess.run(
        [sys.executable, "host_driver.py", "kernel.py", "input.json", revision],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(tmp_path), "CATLASS_SRC": str(tmp_path)},
    )

    assert result.returncode != 0
    assert "Catlass import provenance mismatch" in result.stderr
    assert "runtime_shadow.py" in result.stderr


def test_catlass_capacity_is_rejected_before_remote_execution():
    with pytest.raises(ValueError, match="cannot exceed 400"):
        A5KernelRunner(FakeBackend()).prepare(
            RunRequest(Language.CATLASS_DSL.value, length=401)
        )


def test_catlass_source_pads_to_full_tiles_and_preserves_public_capacity():
    source = {item.relative_path: item.content for item in fixture_for(Language.CATLASS_DSL).files}[
        "kernel.py"
    ]

    assert "PADDED_VECTOR_ELE = 448" in source
    assert "padded_length = ((original_length + VL_ELE - 1) // VL_ELE) * VL_ELE" in source
    assert "return out[:original_length].cpu().tolist()" in source
    A5KernelRunner(FakeBackend()).prepare(RunRequest(Language.CATLASS_DSL.value, length=400))


def test_catlass_executor_selects_adapter_source_revision_and_retained_evidence():
    calls = []
    revision = "9a6ac627b5f4078060287844189730cf0d184800"
    source = "/home/mariodrumond/worktrees/catlass/rsi-a5-imperative-hello"

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[0].endswith("upload.sh"):
            assert argv[1] == "--recursive"
            assert argv[3].startswith(".a5kernels/")
            assert (Path(argv[2]) / "host_driver.py").is_file()
            assert (Path(argv[2]) / "kernel.py").is_file()
            assert (Path(argv[2]) / "input.json").is_file()
            return subprocess.CompletedProcess(argv, 0, "uploaded", "")
        if argv[0].endswith("catlass-validation.sh"):
            assert argv[:10] == (
                "execution-profiles/catlass-validation.sh",
                "--profile",
                "bz-a5",
                "--operation",
                f"codex-a5hello-{plan.execution_id[:8]}-trial-1",
                "run",
                "--catlass-src",
                source,
                "--timeout",
                "600",
            )
            assert argv[-1] == revision
            assert argv[-5:-1] == (
                "python",
                f".a5kernels/{plan.execution_id}/trial-1/host_driver.py",
                f".a5kernels/{plan.execution_id}/trial-1/kernel.py",
                f".a5kernels/{plan.execution_id}/trial-1/input.json",
            )
            return subprocess.CompletedProcess(argv, 0, "completed", "")
        if argv[-1] == "logs":
            return subprocess.CompletedProcess(
                argv, 0, f"{OUTPUT_MARKER}[3.0]\n", ""
            )
        if argv[-1] == "result":
            return subprocess.CompletedProcess(argv, 0, "SESSION_STATE=completed", "")
        return subprocess.CompletedProcess(argv, 0, "SESSION_STATE=completed", "")

    backend = BZSessionAdapter(
        CatlassValidationExecutor(
            upload_wrapper="execution-profiles/bz-a5/upload.sh",
            validation_wrapper="execution-profiles/catlass-validation.sh",
            catlass_source=source,
            catlass_revision=revision,
            process_runner=run,
        ),
        session_wrapper="execution-profiles/bz-a5/session.sh",
    )
    plan = A5KernelRunner(backend).prepare(
        RunRequest(Language.CATLASS_DSL.value, length=1), attempt_id="trial-1"
    )

    receipt = backend.execute(plan)

    assert receipt.output == (3.0,)
    assert receipt.session_handle == f"bz-a5:codex-a5hello-{plan.execution_id[:8]}-trial-1"
    assert len(calls) == 4


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"catlass_source": "relative/worktree"}, "absolute retained BZ path"),
        ({"catlass_revision": "master"}, "lowercase 40-character SHA"),
        ({"validation_wrapper": "session.sh"}, "catlass-validation.sh"),
    ],
)
def test_catlass_executor_rejects_implicit_or_unpinned_configuration(
    overrides, message
):
    options = {
        "upload_wrapper": "execution-profiles/bz-a5/upload.sh",
        "validation_wrapper": "execution-profiles/catlass-validation.sh",
        "catlass_source": "/home/mariodrumond/worktrees/catlass/retained",
        "catlass_revision": "9a6ac627b5f4078060287844189730cf0d184800",
    }
    options.update(overrides)

    with pytest.raises(ValueError, match=message):
        CatlassValidationExecutor(**options)


def test_adapter_rejects_plan_bound_to_different_runtime_provenance():
    executor = FakeCommandExecutor(
        CommandResult(0, "must not run"),
        runtime_provenance=(("catlass_revision", "1" * 40),),
    )
    backend = BZSessionAdapter(
        executor, session_wrapper="execution-profiles/bz-a5/session.sh"
    )
    plan = A5KernelRunner(FakeBackend()).prepare(
        RunRequest(Language.CATLASS_DSL.value), attempt_id="trial-1"
    )

    with pytest.raises(RuntimeUnavailableError, match="provenance does not match"):
        backend.execute(plan)
    assert executor.invocations == []


def test_runtime_revision_changes_identity_session_and_attestation():
    source = "/home/mariodrumond/worktrees/catlass/retained"
    first_provenance = (
        ("catlass_revision", "1" * 40),
        ("catlass_source", source),
        ("execution_profile", "bz-a5"),
    )
    second_provenance = (
        ("catlass_revision", "2" * 40),
        ("catlass_source", source),
        ("execution_profile", "bz-a5"),
    )
    request = RunRequest(Language.CATLASS_DSL.value, length=1)
    first_result = A5KernelRunner(
        FakeBackend(runtime_provenance=first_provenance)
    ).run(request, attempt_id="same-attempt")
    second_result = A5KernelRunner(
        FakeBackend(runtime_provenance=second_provenance)
    ).run(request, attempt_id="same-attempt")

    assert first_result.request_id == second_result.request_id
    assert first_result.execution_id != second_result.execution_id
    assert first_result.runtime_provenance != second_result.runtime_provenance
    assert first_result.evidence_sha256 != second_result.evidence_sha256
    assert first_result.attestation_sha256 != second_result.attestation_sha256
    first_session = f"codex-a5hello-{first_result.execution_id[:8]}-same-attempt"
    second_session = f"codex-a5hello-{second_result.execution_id[:8]}-same-attempt"
    assert first_session != second_session


@pytest.mark.parametrize("plan_field", ["source", "argv"])
def test_plan_content_changes_identity_remote_directory_and_session(plan_field):
    plan = A5KernelRunner(FakeBackend()).prepare(
        RunRequest(Language.CATLASS_DSL.value, length=1), attempt_id="same-attempt"
    )
    if plan_field == "source":
        changed_file = SourceFile(
            plan.files[0].relative_path, plan.files[0].content + "\n# changed\n"
        )
        changed = replace(plan, files=(changed_file, *plan.files[1:]))
    else:
        assert plan.argv is not None
        changed = replace(plan, argv=(*plan.argv, "--changed"))

    assert plan.request_id == changed.request_id
    assert plan.runtime_provenance == changed.runtime_provenance
    assert plan.attempt_id == changed.attempt_id
    assert plan.execution_id != changed.execution_id

    invocations = []
    for candidate in (plan, changed):
        command = FakeCommandExecutor(CommandResult(9, "expected dispatch failure"))
        BZSessionAdapter(
            command, session_wrapper="execution-profiles/bz-a5/session.sh"
        ).execute(candidate)
        invocations.append(command.invocations[0])

    assert invocations[0].remote_directory != invocations[1].remote_directory
    assert invocations[0].argv[2] != invocations[1].argv[2]


def test_input_changes_with_same_sum_change_identity_remote_directory_and_session():
    plan = A5KernelRunner(FakeBackend()).prepare(
        RunRequest(Language.CATLASS_DSL.value, length=2), attempt_id="same-attempt"
    )
    changed = replace(plan, input_a=(1.0, 2.0), input_b=(0.0, 3.0))
    original = replace(plan, input_a=(0.0, 3.0), input_b=(1.0, 2.0))

    assert sum(original.input_a) == sum(changed.input_a)
    assert sum(original.input_b) == sum(changed.input_b)
    assert original.request_id == changed.request_id
    assert original.runtime_provenance == changed.runtime_provenance
    assert original.attempt_id == changed.attempt_id
    assert original.execution_id != changed.execution_id

    invocations = []
    for candidate in (original, changed):
        command = FakeCommandExecutor(CommandResult(9, "expected dispatch failure"))
        BZSessionAdapter(
            command, session_wrapper="execution-profiles/bz-a5/session.sh"
        ).execute(candidate)
        invocations.append(command.invocations[0])

    assert invocations[0].remote_directory != invocations[1].remote_directory
    assert invocations[0].argv[2] != invocations[1].argv[2]


@pytest.mark.parametrize("stdout", ["no record", f"{OUTPUT_MARKER}[]\n{OUTPUT_MARKER}[]"])
def test_bz_adapter_rejects_missing_or_ambiguous_output(stdout):
    backend = BZSessionAdapter(
        FakeCommandExecutor(
            CommandResult(0, "SESSION_STATE=finished exit=0"),
            (CommandResult(0, stdout), CommandResult(0, "exit=0")),
        ),
        session_wrapper="execution-profiles/bz-a5/session.sh",
    )
    plan = replace(
        A5KernelRunner(FakeBackend()).prepare(
            RunRequest(Language.ASCEND_C.value), attempt_id="trial-1"
        ),
        argv=("run-ascend-c",),
    )

    with pytest.raises(ValueError, match="exactly one"):
        backend.execute(plan)


def test_foundational_fixture_fails_closed_without_concrete_driver():
    command = FakeCommandExecutor(CommandResult(0, "unused"))
    backend = BZSessionAdapter(
        command, session_wrapper="execution-profiles/bz-a5/session.sh"
    )
    plan = A5KernelRunner(FakeBackend()).prepare(
        RunRequest(Language.TRITON_ASCEND.value), attempt_id="trial-1"
    )

    with pytest.raises(RuntimeUnavailableError, match="no executable runtime"):
        backend.execute(plan)
    assert command.invocations == []


def test_repeated_request_uses_distinct_host_owned_attempts_and_sessions():
    runner = A5KernelRunner(FakeBackend())
    request = RunRequest(Language.CATLASS_DSL.value)

    first = runner.prepare(request)
    second = runner.prepare(request)

    assert first.request_id == second.request_id
    assert first.attempt_id != second.attempt_id

    executors = []
    for plan in (first, second):
        executor = FakeCommandExecutor(
            CommandResult(0, "finished"),
            (CommandResult(0, "workload failed"), CommandResult(1, "exit=1")),
        )
        BZSessionAdapter(
            executor, session_wrapper="execution-profiles/bz-a5/session.sh"
        ).execute(replace(plan, argv=("real-driver",)))
        executors.append(executor)
    assert executors[0].invocations[0].argv[2] != executors[1].invocations[0].argv[2]


@pytest.mark.parametrize("attempt_id", ["", "spaces are unsafe", "../escape"])
def test_attempt_identifier_is_validated(attempt_id):
    with pytest.raises(ValueError, match="attempt_id"):
        A5KernelRunner(FakeBackend()).prepare(
            RunRequest(Language.CATLASS_DSL.value), attempt_id=attempt_id
        )
