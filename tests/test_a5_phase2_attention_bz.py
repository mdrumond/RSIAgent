from __future__ import annotations

import json

import pytest

import benchmarks.a5kernels.phase2_attention_bz as attention_bz
from benchmarks.a5kernels.bz import CommandResult
from benchmarks.a5kernels.phase2_attention import (
    ATTENTION_CASES,
    AttentionInfrastructureError,
    AttentionTargetVerifier,
    build_execution,
)
from benchmarks.a5kernels.phase2_attention_bz import (
    ATTENTION_CANDIDATE_FIXTURE,
    MAX_OBSERVED_BYTES,
    OUTPUT_MARKER,
    CatlassAttentionBackend,
    validate_attention_candidate,
)


SOURCE = """\
import catlass.tla as tla

@tla.kernel
def dense_attention(gm_q: tla.Tensor, gm_k: tla.Tensor,
                    gm_v: tla.Tensor, gm_output: tla.Tensor) -> None:
    pass
"""


class FakeCheckedExecutor:
    runtime_provenance = (("catlass_revision", "a" * 40),)

    def __init__(self, dispatch=None, inspections=()):
        self.dispatch = dispatch or CommandResult(0, "dispatched", session_handle="bz-a5:d")
        self.inspections = list(inspections)
        self.invocations = []
        self.inspect_argv = []

    def run(self, invocation):
        self.invocations.append(invocation)
        if isinstance(self.dispatch, Exception):
            raise self.dispatch
        if callable(self.dispatch):
            return self.dispatch(invocation)
        return self.dispatch

    def inspect(self, argv):
        self.inspect_argv.append(argv)
        result = self.inspections.pop(0)
        if isinstance(result, Exception):
            raise result
        if argv[-1] == "result" and result.stdout in {"", "placeholder", "exit=0"}:
            result = CommandResult(
                result.exit_code,
                f"SESSION_STATE=completed session={argv[2]} exit={result.exit_code}",
                result.stderr,
                result.session_handle,
            )
        return result


def _success(execution, *, handle="bz-a5:finished"):
    output = [0.0] * execution.case.output_elements
    return (
        CommandResult(0, "banner\n" + OUTPUT_MARKER + json.dumps(output)),
        CommandResult(0, "", session_handle=handle),
    )


def _backend(executor, *, device=3):
    return CatlassAttentionBackend(
        executor,
        session_wrapper="execution-profiles/bz-a5/session.sh",
        device=device,
        observe_timeout=321,
    )


@pytest.mark.parametrize("case", ATTENTION_CASES, ids=lambda case: case.case_id)
def test_complete_suite_stages_exact_case_and_checked_argv(tmp_path, case):
    execution = build_execution(case)
    executor = FakeCheckedExecutor(inspections=_success(execution))

    receipt = _backend(executor).execute(SOURCE, execution, tmp_path / case.case_id)

    assert receipt.exit_code == 0
    assert len(receipt.output) == case.output_elements
    assert receipt.session_handle == "bz-a5:finished"
    invocation = executor.invocations[0]
    assert invocation.physical_device == 3
    assert invocation.stdin == ""
    assert invocation.remote_directory.startswith(".a5kernels/attention/")
    assert ".." not in invocation.remote_directory
    assert invocation.argv[:4] == (
        "execution-profiles/bz-a5/session.sh", "--name",
        invocation.argv[2], "run",
    )
    assert invocation.argv[4:8] == ("--wait", "--observe-timeout", "321", "--")
    assert invocation.argv[8:15] == (
        "env", "-u", "PYTHONPYCACHEPREFIX",
        "BZ_A5_PROFILE_PHYSICAL_DEVICE=3", "A5KERNEL_BLOCK_NUM=1",
        "python", "-B",
    )
    assert all(argument.startswith(invocation.remote_directory)
               for argument in invocation.argv[-3:])
    assert [item.relative_path for item in invocation.files] == [
        "kernel.py", "host_driver.py", "input.json",
    ]
    driver = invocation.files[1].content
    assert "from_dlpack(tensor.contiguous().reshape(-1)" in driver
    assert "output.float().cpu().reshape(-1).tolist()" in driver
    payload = json.loads(invocation.files[-1].content)
    assert payload["case"]["case_id"] == case.case_id
    assert tuple(payload["q"]) == execution.q
    assert tuple(payload["k"]) == execution.k
    assert tuple(payload["v"]) == execution.v
    assert payload["scale"] == execution.scale
    staged = tmp_path / case.case_id
    assert {item.name for item in staged.iterdir()} == {
        "kernel.py", "host_driver.py", "input.json",
    }
    kernel = (staged / "kernel.py").read_text()
    assert f"BATCH = {case.batch}" in kernel
    assert f"CAUSAL = {case.causal!r}" in kernel
    assert kernel.endswith(SOURCE)
    assert [argv[-1] for argv in executor.inspect_argv] == ["logs", "result"]
    metadata = dict(receipt.metadata)
    assert metadata["dispatch_exit_code"] == "0"
    assert metadata["logs_exit_code"] == "0"
    assert metadata["result_exit_code"] == "0"


def test_same_candidate_and_case_have_deterministic_paths_and_no_source_in_argv(tmp_path):
    execution = build_execution(ATTENTION_CASES[0])
    executors = [FakeCheckedExecutor(inspections=_success(execution)) for _ in range(2)]
    for index, executor in enumerate(executors):
        _backend(executor).execute(SOURCE, execution, tmp_path / str(index))

    first, second = (item.invocations[0] for item in executors)
    assert first.remote_directory == second.remote_directory
    assert first.argv[2] == second.argv[2]
    assert first.argv[2].startswith("codex-a5attn-")
    assert len(first.argv[2]) <= 48
    assert SOURCE not in first.argv
    assert all(" " not in argument for argument in first.argv[-3:])


def test_host_driver_revision_changes_session_and_remote_identity(tmp_path, monkeypatch):
    execution = build_execution(ATTENTION_CASES[0])
    executors = [FakeCheckedExecutor(inspections=_success(execution)) for _ in range(2)]
    _backend(executors[0]).execute(SOURCE, execution, tmp_path / "original")
    monkeypatch.setattr(
        attention_bz,
        "_ATTENTION_DRIVER",
        attention_bz._ATTENTION_DRIVER + "\n# revised host harness\n",
    )
    _backend(executors[1]).execute(SOURCE, execution, tmp_path / "revised")

    first, second = (item.invocations[0] for item in executors)
    assert first.files[1].content != second.files[1].content
    assert first.remote_directory != second.remote_directory
    assert first.argv[2] != second.argv[2]


def test_runtime_provenance_changes_session_and_remote_identity(tmp_path):
    execution = build_execution(ATTENTION_CASES[0])
    executors = [
        FakeCheckedExecutor(inspections=_success(execution)) for _ in range(3)
    ]
    executors[0].runtime_provenance = (("catlass_revision", "a" * 40),)
    executors[1].runtime_provenance = (("catlass_revision", "a" * 40),)
    executors[2].runtime_provenance = (("catlass_revision", "b" * 40),)
    for index, executor in enumerate(executors):
        _backend(executor).execute(SOURCE, execution, tmp_path / str(index))

    first, repeated, changed = (item.invocations[0] for item in executors)
    assert first.remote_directory == repeated.remote_directory
    assert first.argv[2] == repeated.argv[2]
    assert first.remote_directory != changed.remote_directory
    assert first.argv[2] != changed.argv[2]


def test_same_candidate_and_case_are_isolated_by_physical_device(tmp_path):
    execution = build_execution(ATTENTION_CASES[0])
    executors = [FakeCheckedExecutor(inspections=_success(execution)) for _ in range(2)]
    _backend(executors[0], device=2).execute(SOURCE, execution, tmp_path / "device-2")
    _backend(executors[1], device=5).execute(SOURCE, execution, tmp_path / "device-5")

    first, second = (item.invocations[0] for item in executors)
    assert first.remote_directory != second.remote_directory
    assert first.argv[2] != second.argv[2]
    assert all(len(item.argv[2]) <= 48 for item in (first, second))


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("", "must not be empty"),
        ("import os\n" + SOURCE, "may import only"),
        (SOURCE.replace("dense_attention", "attention"), "must use dense_attention"),
        (SOURCE.replace("@tla.kernel", ""), "exactly @tla.kernel"),
        (SOURCE + "\nX = 1\n", "import and kernel"),
    ],
)
def test_candidate_export_contract_fails_before_dispatch(tmp_path, source, message):
    executor = FakeCheckedExecutor()
    with pytest.raises(ValueError, match=message):
        _backend(executor).execute(source, build_execution(ATTENTION_CASES[0]), tmp_path)
    assert executor.invocations == []


def test_hand_authored_fixture_uses_the_same_validated_tla_abi():
    validate_attention_candidate(ATTENTION_CANDIDATE_FIXTURE)
    assert """\
        tla.copy(local_v, gm_v)
        tla.pipe_barrier(tla.pipes.ALL)
        tla.copy(gm_output, local_v)
        tla.pipe_barrier(tla.pipes.ALL)
""" in ATTENTION_CANDIDATE_FIXTURE


@pytest.mark.parametrize("exit_code", [2, 19, 75])
def test_confirmed_nonzero_dispatch_retains_compiler_log_and_terminal(tmp_path, exit_code):
    def dispatch(invocation):
        session = invocation.argv[2]
        return CommandResult(
            exit_code,
            f"SESSION_DISPATCH=confirmed session={session} output=.codex/output.log\n",
            session_handle=f"bz-a5:{session}",
        )
    executor = FakeCheckedExecutor(dispatch=dispatch, inspections=(
        CommandResult(0, "TLA compiler: invalid tile shape"),
        CommandResult(exit_code, "placeholder", "", "bz-a5:terminal"),
    ))

    receipt = _backend(executor).execute(
        SOURCE, build_execution(ATTENTION_CASES[0]), tmp_path
    )

    assert receipt.exit_code == exit_code
    assert receipt.output == ()
    assert "TLA compiler: invalid tile shape" in receipt.stdout
    assert receipt.session_handle == "bz-a5:terminal"
    assert dict(receipt.metadata)["dispatch_exit_code"] == str(exit_code)
    assert dict(receipt.metadata)["result_exit_code"] == str(exit_code)
    assert [argv[-1] for argv in executor.inspect_argv] == ["logs", "result"]


def test_current_wrapper_completed_dispatch_retains_nonzero_result(tmp_path):
    def dispatch(invocation):
        session = invocation.argv[2]
        return CommandResult(
            2,
            f"SESSION_STATE=completed session={session} exit=2 "
            f"handle=remote:bz-a5:session:{session}\n",
            session_handle=f"bz-a5:{session}",
        )

    executor = FakeCheckedExecutor(dispatch=dispatch, inspections=(
        CommandResult(0, "TLA compiler: layout failure"),
        CommandResult(2, "placeholder", "runtime returned 2"),
    ))

    receipt = _backend(executor).execute(
        SOURCE, build_execution(ATTENTION_CASES[0]), tmp_path
    )

    assert receipt.exit_code == 2
    assert "TLA compiler: layout failure" in receipt.stdout
    assert [argv[-1] for argv in executor.inspect_argv] == ["logs", "result"]


@pytest.mark.parametrize(
    ("record", "exit_code", "expected"),
    [
        ("exit=1", 1, True),
        ("exit=1 handle=remote:bz-a5:session:example", 1, True),
        ("exit=1 handle=remote:bz-a5:session:example", 2, False),
        ("exit=01 handle=remote:bz-a5:session:example", 1, False),
        ("exit=1 handle=", 1, False),
        ("exit=1 unexpected=value", 1, False),
    ],
)
def test_completed_result_accepts_only_exact_wrapper_record(
    record, exit_code, expected,
):
    session = "codex-a5attn-example"
    stdout = f"SESSION_STATE=completed session={session} {record}\n"
    assert attention_bz._completed_result(exit_code, stdout, session) is expected


def test_confirmed_host_failure_reaches_digest_bound_target_evidence(tmp_path):
    def dispatch(invocation):
        session = invocation.argv[2]
        return CommandResult(
            19,
            f"SESSION_DISPATCH=confirmed session={session} output=.codex/output.log\n",
            session_handle=f"bz-a5:{session}",
        )

    inspections = []
    for _case in ATTENTION_CASES:
        inspections.extend((
            CommandResult(0, "TLA compiler: ordinary tile failure"),
            CommandResult(19, "placeholder", "runtime returned 19"),
        ))
    executor = FakeCheckedExecutor(dispatch=dispatch, inspections=inspections)

    evidence = AttentionTargetVerifier(_backend(executor)).verify(
        SOURCE,
        request_id="phase2-cell-1",
        attempt_id="target-1",
        execution_profile="bz-a5",
        workdir=tmp_path / "runs",
    )

    assert evidence.passed is False
    assert len(evidence.cases) == len(ATTENTION_CASES)
    assert all(item.failure == "runtime-exit" for item in evidence.cases)
    assert all(
        item.diagnostic is not None
        and "TLA compiler: ordinary tile failure" in item.diagnostic
        and "runtime returned 19" in item.diagnostic
        for item in evidence.cases
    )


def test_unconfirmed_wrapper_failure_is_infrastructure_even_with_handle(tmp_path):
    executor = FakeCheckedExecutor(
        dispatch=CommandResult(
            1, "wrapper failed before SESSION_DISPATCH", "transport error",
            "bz-a5:synthesized",
        )
    )

    with pytest.raises(AttentionInfrastructureError, match="unavailable"):
        _backend(executor).execute(
            SOURCE, build_execution(ATTENTION_CASES[0]), tmp_path
        )


def test_completed_dispatch_with_mismatched_exit_is_infrastructure(tmp_path):
    def dispatch(invocation):
        session = invocation.argv[2]
        return CommandResult(
            2,
            f"SESSION_STATE=completed session={session} exit=1\n",
            session_handle=f"bz-a5:{session}",
        )

    executor = FakeCheckedExecutor(dispatch=dispatch)
    with pytest.raises(AttentionInfrastructureError, match="unavailable"):
        _backend(executor).execute(
            SOURCE, build_execution(ATTENTION_CASES[0]), tmp_path
        )
    assert executor.inspect_argv == []


def test_runtime_exit_is_semantic_and_retains_all_exit_metadata(tmp_path):
    execution = build_execution(ATTENTION_CASES[0])
    executor = FakeCheckedExecutor(inspections=(
        CommandResult(0, "ordinary runtime log"),
        CommandResult(17, "placeholder", "kernel failed", "bz-a5:r"),
    ))

    receipt = _backend(executor).execute(SOURCE, execution, tmp_path)

    assert receipt.exit_code == 17
    assert receipt.output == ()
    assert receipt.session_handle == "bz-a5:r"
    assert dict(receipt.metadata) == {
        "case_id": execution.case.case_id,
        "dispatch_exit_code": "0",
        "logs_exit_code": "0",
        "result_exit_code": "17",
        "remote_directory": executor.invocations[0].remote_directory,
    }


@pytest.mark.parametrize(
    "records",
    [
        "no marker",
        OUTPUT_MARKER + "[]\n" + OUTPUT_MARKER + "[]",
        OUTPUT_MARKER + "not-json",
        OUTPUT_MARKER + "[true]",
        OUTPUT_MARKER + "[NaN]",
    ],
)
def test_malformed_output_is_a_semantic_parse_failure(tmp_path, records):
    execution = build_execution(ATTENTION_CASES[0])
    executor = FakeCheckedExecutor(inspections=(
        CommandResult(0, records), CommandResult(0, "exit=0"),
    ))

    with pytest.raises(ValueError):
        _backend(executor).execute(SOURCE, execution, tmp_path)


def test_output_element_and_raw_byte_bounds_are_enforced(tmp_path):
    execution = build_execution(ATTENTION_CASES[0])
    too_many = [0.0] * (execution.case.output_elements + 1)
    executor = FakeCheckedExecutor(inspections=(
        CommandResult(0, OUTPUT_MARKER + json.dumps(too_many)),
        CommandResult(0, "exit=0"),
    ))
    with pytest.raises(ValueError, match="bounded"):
        _backend(executor).execute(SOURCE, execution, tmp_path / "elements")

    executor = FakeCheckedExecutor(inspections=(
        CommandResult(0, "x" * (MAX_OBSERVED_BYTES + 1)),
        CommandResult(0, "exit=0"),
    ))
    with pytest.raises(AttentionInfrastructureError, match="exceeded"):
        _backend(executor).execute(SOURCE, execution, tmp_path / "bytes")


@pytest.mark.parametrize("stage", ["dispatch", "logs", "result"])
def test_transport_and_observation_uncertainty_is_infrastructure(tmp_path, stage):
    execution = build_execution(ATTENTION_CASES[0])
    if stage == "dispatch":
        executor = FakeCheckedExecutor(dispatch=OSError("transport down"))
    elif stage == "logs":
        executor = FakeCheckedExecutor(inspections=(OSError("observe down"),))
    else:
        executor = FakeCheckedExecutor(inspections=(
            CommandResult(0, "logs"), CommandResult(255, "connection lost"),
        ))
    with pytest.raises(AttentionInfrastructureError):
        _backend(executor).execute(SOURCE, execution, tmp_path)


def test_explicit_uncertain_dispatch_observes_the_same_retained_session(tmp_path):
    execution = build_execution(ATTENTION_CASES[0])
    probe = FakeCheckedExecutor(inspections=_success(execution))
    backend = _backend(probe)
    # Session identity is deterministic, so obtain it from an ordinary dispatch.
    backend.execute(SOURCE, execution, tmp_path / "probe")
    session_name = probe.invocations[0].argv[2]
    uncertain = CommandResult(
        75,
        "CATLASS_VALIDATION_STATE=observation-unavailable\n"
        f"CATLASS_VALIDATION_HANDLE=bz-a5:{session_name}\n",
    )
    executor = FakeCheckedExecutor(dispatch=uncertain, inspections=_success(execution))

    receipt = _backend(executor).execute(SOURCE, execution, tmp_path / "retry")

    assert receipt.exit_code == 0
    assert executor.invocations[0].argv[2] == session_name
    assert all(argv[2] == session_name for argv in executor.inspect_argv)


def test_invalid_configuration_is_rejected():
    executor = FakeCheckedExecutor()
    with pytest.raises(ValueError, match="session.sh"):
        CatlassAttentionBackend(executor, session_wrapper="ssh")
    with pytest.raises(ValueError, match="device"):
        _backend(executor, device=-1)
