from dataclasses import FrozenInstanceError, replace

import pytest

from benchmarks.a5kernels import A5KernelRunner, BZSessionAdapter, Language, RunRequest, fixture_for
from benchmarks.a5kernels.bz import (
    CommandResult,
    OUTPUT_MARKER,
    RuntimeUnavailableError,
)
from benchmarks.a5kernels.protocol import ExecutionReceipt


class FakeBackend:
    def __init__(self, *, corrupt=False, exit_code=0, claimed_score="0"):
        self.corrupt = corrupt
        self.exit_code = exit_code
        self.claimed_score = claimed_score
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
    def __init__(self, dispatch, inspections=()):
        self.dispatch = dispatch
        self.inspections = list(inspections)
        self.invocations = []
        self.inspect_argv = []

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
    plan = replace(
        A5KernelRunner(FakeBackend()).prepare(
            RunRequest(Language.CATLASS_DSL.value, length=2), attempt_id="trial-1"
        ),
        argv=("python", "real_driver.py"),
    )

    receipt = backend.execute(plan)

    assert receipt.output == (1.25, 2.0)
    assert receipt.session_handle == "bz-a5:s1"
    invocation = command.invocations[0]
    assert invocation.argv[:4] == (
        "execution-profiles/bz-a5/session.sh",
        "--name",
        f"codex-a5hello-{plan.request_id[:8]}-trial-1",
        "run",
    )
    assert invocation.files == plan.files
    assert [argv[-1] for argv in command.inspect_argv] == ["logs", "result"]


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
