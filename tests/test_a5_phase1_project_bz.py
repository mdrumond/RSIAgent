from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.a5kernels.bz import CommandResult
from benchmarks.a5kernels.evidence import EvidenceLedger
from benchmarks.a5kernels.phase1_project_bz import (
    PROJECT_RESULT_MARKER,
    CatlassProjectBackend,
    ProjectInfrastructureError,
)
from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
from tests.test_a5_phase1_project_executor import _spec


class FakeExecutor:
    runtime_provenance = (("catlass_revision", "a" * 40), ("execution_profile", "bz-a5"))

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.invocations = []
        self.inspections = []
        self._outcome = None
        self.dispatch_result = None
        self.observations = []

    def run(self, invocation):
        self.invocations.append(invocation)
        self._outcome = self.outcomes.pop(0)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self.dispatch_result or CommandResult(
            0, "dispatch accepted", session_handle=None
        )

    def inspect(self, argv):
        self.inspections.append(argv)
        session = argv[argv.index("--name") + 1]
        if argv[-1] == "logs":
            return CommandResult(
                0, PROJECT_RESULT_MARKER + json.dumps(self._outcome, separators=(",", ":"))
            )
        return CommandResult(
            0, f"SESSION_STATE=completed session={session} exit=0"
        )

    def observe(self, handle):
        self.observations.append(handle)
        session = handle.removeprefix("bz-a5:")
        return CommandResult(
            0,
            "\n".join((
                handle.replace("bz-a5:", "CATLASS_VALIDATION_HANDLE=bz-a5:"),
                "CATLASS_VALIDATION_STATE=completed",
            )),
            session_handle=f"bz-a5:{session}",
        )


def _backend(tmp_path: Path, executor: FakeExecutor) -> CatlassProjectBackend:
    return CatlassProjectBackend(
        executor,
        session_wrapper="execution-profiles/bz-a5/session.sh",
        evidence_root=tmp_path / "retained",
        device=3,
    )


def _request(tmp_path: Path, family: str):
    executor = FakeExecutor([])
    project = ProjectExecutor(executor, EvidenceLedger(tmp_path / "unused.jsonl"))
    prepared = project.prepare(_spec(family))
    return project._request(prepared, "# exact candidate bytes\n")


@pytest.mark.parametrize("family", ["movement", "simt", "cube"])
def test_bundle_is_deterministic_but_independent_backends_use_fresh_sessions(
    tmp_path, family,
):
    request = _request(tmp_path, family)
    outcomes = [{"status": "compiled", "diagnostic": "compiled"}] * 2
    first_executor, second_executor = FakeExecutor(outcomes[:1]), FakeExecutor(outcomes[1:])
    first = _backend(tmp_path / "one", first_executor).compile(request)
    second = _backend(tmp_path / "two", second_executor).compile(request)

    one, two = first_executor.invocations[0], second_executor.invocations[0]
    assert first.artifact_id == second.artifact_id
    assert first.artifact_id.startswith("bundle:")
    assert one.argv[:2] == two.argv[:2]
    assert one.argv[3:] == two.argv[3:]
    assert one.argv[2] != two.argv[2]
    assert one.argv[2].endswith("-000001")
    assert two.argv[2].endswith("-000001")
    assert one.remote_directory == two.remote_directory
    assert one.physical_device == two.physical_device == 3
    assert tuple((item.relative_path, item.content) for item in one.files) == tuple(
        (item.relative_path, item.content) for item in two.files
    )
    files = {item.relative_path: item.content for item in one.files}
    manifest = json.loads(files["manifest.json"])
    assert files["kernel.py"] == request.candidate_source
    assert manifest["family"] == request.family.value
    assert len(manifest["project_sha256"]) == 64
    assert manifest["abi"] == request.abi.as_dict()
    assert manifest["output"]["physical_shape"] == list(request.expected.physical_shape)
    assert manifest["block_count"] == request.spec.block_count
    assert "torch.npu.synchronize()" in files["host_driver.py"]
    assert "tla.compile" in files["host_driver.py"]
    assert "artifact(*tla_inputs, tla_output" in files["host_driver.py"]
    assert "catlass-validation.sh" not in " ".join(one.argv)
    assert one.argv[0] == "execution-profiles/bz-a5/session.sh"


def test_launch_returns_the_complete_physical_storage_and_retains_bundle(tmp_path):
    request = _request(tmp_path, "movement")
    physical = list(request.expected.storage)
    executor = FakeExecutor([
        {"status": "compiled", "diagnostic": "compiled"},
        {"status": "completed", "diagnostic": "launch synchronized", "output": physical},
    ])
    backend = _backend(tmp_path, executor)

    compilation = backend.compile(request)
    runtime = backend.run(request, compilation)

    assert runtime.passed is True
    assert runtime.output == tuple(physical)
    assert len(runtime.output) == len(request.expected.storage)
    retained = (
        tmp_path / "retained" /
        compilation.artifact_id.removeprefix("bundle:")
    )
    assert (retained / "kernel.py").read_bytes() == request.candidate_source.encode()
    assert json.loads((retained / "manifest.json").read_text())["output"][
        "storage_elements"
    ] == len(physical)
    assert len(executor.invocations) == 2
    assert executor.invocations[0].files == executor.invocations[1].files
    assert executor.invocations[1].argv != executor.invocations[0].argv
    compile_session = executor.invocations[0].argv[2]
    run_session = executor.invocations[1].argv[2]
    assert compile_session.endswith("-000001")
    assert run_session.endswith("-000001")
    for digest in (compilation.evidence_sha256, runtime.evidence_sha256):
        evidence_path = tmp_path / "retained" / "cas" / digest[:2] / f"{digest}.json"
        retained_evidence = json.loads(evidence_path.read_text())
        assert retained_evidence["identity"] == compilation.artifact_id.removeprefix(
            "bundle:"
        )
        assert set(retained_evidence["command_results"]) == {
            "dispatch", "logs", "result",
        }


def test_candidate_failures_are_observations_not_infrastructure(tmp_path):
    request = _request(tmp_path, "simt")
    compiler = _backend(tmp_path / "compile", FakeExecutor([
        {"status": "compile-failure", "diagnostic": "frontend rejected candidate"}
    ])).compile(request)
    assert compiler.passed is False
    assert compiler.artifact_id is None

    executor = FakeExecutor([
        {"status": "compiled", "diagnostic": "compiled"},
        {"status": "runtime-failure", "diagnostic": "device launch failed"},
    ])
    backend = _backend(tmp_path / "runtime", executor)
    runtime = backend.run(request, backend.compile(request))
    assert runtime.passed is False
    assert runtime.output is None
    assert runtime.diagnostic == "device launch failed"


@pytest.mark.parametrize(
    "outcome",
    [RuntimeError("transport unavailable"), {"status": "infrastructure-failure", "diagnostic": "Catlass unavailable"}],
)
def test_transport_and_host_infrastructure_fail_separately(tmp_path, outcome):
    request = _request(tmp_path, "cube")
    with pytest.raises(ProjectInfrastructureError):
        _backend(tmp_path, FakeExecutor([outcome])).compile(request)


def test_failed_current_dispatch_cannot_reuse_stale_deterministic_logs(tmp_path):
    request = _request(tmp_path, "movement")
    executor = FakeExecutor([
        {"status": "compiled", "diagnostic": "stale prior result"}
    ])
    executor.dispatch_result = CommandResult(1, "", "upload failed")

    with pytest.raises(ProjectInfrastructureError, match="current session"):
        _backend(tmp_path, executor).compile(request)

    assert executor.inspections == []


def test_confirmed_retries_use_distinct_attempts_but_uncertain_dispatch_reattaches(tmp_path):
    request = _request(tmp_path, "movement")
    outcome = {"status": "compiled", "diagnostic": "compiled"}
    executor = FakeExecutor([outcome, outcome])
    backend = _backend(tmp_path, executor)

    backend.compile(request)
    backend.compile(request)

    sessions = [invocation.argv[2] for invocation in executor.invocations]
    assert sessions[0].endswith("-000001")
    assert sessions[1].endswith("-000002")
    assert sessions[0] != sessions[1]

    class UncertainExecutor(FakeExecutor):
        def run(self, invocation):
            self.invocations.append(invocation)
            self._outcome = self.outcomes.pop(0)
            session = invocation.argv[2]
            return CommandResult(
                75,
                "\n".join((
                    "CATLASS_VALIDATION_STATE=observation-unavailable",
                    f"CATLASS_VALIDATION_HANDLE=bz-a5:{session}",
                )),
                session_handle=f"bz-a5:{session}",
            )

    uncertain = UncertainExecutor([outcome])
    _backend(tmp_path / "reattach", uncertain).compile(request)
    expected_session = uncertain.invocations[0].argv[2]
    assert uncertain.observations == [f"bz-a5:{expected_session}"]


def test_evidence_bound_scales_for_admitted_large_padded_outputs(tmp_path):
    request = _request(tmp_path, "movement")
    large_record = "x" * (4 * 1024 * 1024 + 1)
    executor = FakeExecutor([
        {"status": "completed", "diagnostic": large_record,
         "output": list(request.expected.storage)}
    ])
    backend = _backend(tmp_path, executor)
    identity, files = backend._bundle(request)
    record, evidence = backend._execute(
        identity, files, "run", 200_000
    )
    assert record["status"] == "completed"
    assert len(evidence) == 64


def test_backend_rejects_non_checked_session_wrapper(tmp_path):
    with pytest.raises(ValueError, match="session.sh"):
        CatlassProjectBackend(
            FakeExecutor([]), session_wrapper="ssh", evidence_root=tmp_path
        )
