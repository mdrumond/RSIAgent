from dataclasses import FrozenInstanceError

import pytest

from benchmarks.a5kernels import A5KernelRunner, BZSessionAdapter, Language, RunRequest, fixture_for
from benchmarks.a5kernels.bz import CommandResult, OUTPUT_MARKER
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

    first = runner.run(request)
    second = runner.run(request)

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
    def __init__(self, result):
        self.result = result
        self.invocations = []

    def run(self, invocation):
        self.invocations.append(invocation)
        return self.result


def test_bz_adapter_uses_named_session_and_parses_one_output_record():
    command = FakeCommandExecutor(
        CommandResult(0, f"banner\n{OUTPUT_MARKER}[1.25,2]\n", session_handle="bz-a5:s1")
    )
    backend = BZSessionAdapter(
        command, session_wrapper="execution-profiles/bz-a5/session.sh"
    )
    plan = A5KernelRunner(FakeBackend()).prepare(
        RunRequest(Language.CATLASS_DSL.value, length=2)
    )

    receipt = backend.execute(plan)

    assert receipt.output == (1.25, 2.0)
    assert receipt.session_handle == "bz-a5:s1"
    invocation = command.invocations[0]
    assert invocation.argv[:4] == (
        "execution-profiles/bz-a5/session.sh",
        "--name",
        f"codex-a5hello-{plan.request_id[:12]}",
        "run",
    )
    assert invocation.files == plan.files


@pytest.mark.parametrize("stdout", ["no record", f"{OUTPUT_MARKER}[]\n{OUTPUT_MARKER}[]"])
def test_bz_adapter_rejects_missing_or_ambiguous_output(stdout):
    backend = BZSessionAdapter(
        FakeCommandExecutor(CommandResult(0, stdout)),
        session_wrapper="execution-profiles/bz-a5/session.sh",
    )
    plan = A5KernelRunner(FakeBackend()).prepare(RunRequest(Language.ASCEND_C.value))

    with pytest.raises(ValueError, match="exactly one"):
        backend.execute(plan)
