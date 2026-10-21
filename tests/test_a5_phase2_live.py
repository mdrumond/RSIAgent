from __future__ import annotations
from dataclasses import replace

from dataclasses import asdict
import hashlib
import json

import pytest

import benchmarks.a5kernels.phase2_live as phase2_live_module
from benchmarks.a5kernels.phase2_attention import (
    CANDIDATE_ABI_SHA256,
    HOST_TARGET_GRADING_SHA256,
    AttentionInfrastructureError,
    AttentionReceipt,
    AttentionTargetVerifier,
    attention_reference,
)
from benchmarks.a5kernels.phase2_live import (
    DEFAULT_MAX_SECONDS,
    Phase2Dependencies,
    Phase2LiveRunner,
)
from tests.test_a5_phase2_memory import _snapshot


SOURCE = """\
import catlass.tla as tla

@tla.kernel
def dense_attention(q, k, v, output):
    pass
"""


class _Backend:
    def __init__(self, outcomes=(True,)):
        self.outcomes = iter(outcomes)
        self.passed = True
        self.calls = []

    def execute(self, source, execution, workdir):
        self.calls.append((source, execution.case.case_id, workdir))
        output = attention_reference(execution)
        if not self.passed:
            output = (output[0] + 1.0, *output[1:])
        return AttentionReceipt(0, output, session_handle="bz-a5:test")


def _deps(snapshot, backend, source_calls):
    verifier = AttentionTargetVerifier(backend)

    def source(request):
        source_calls.append(request)
        try:
            backend.passed = next(backend.outcomes)
        except StopIteration:
            pass
        return SOURCE + f"\n# attempt {request.attempt}\n"

    return Phase2Dependencies(
        snapshot.destination,
        source,
        lambda candidate, request, attempt, workdir: verifier.verify(
            candidate,
            request_id=request,
            attempt_id=attempt,
            execution_profile="bz-a5",
            workdir=workdir,
        ),
    )


def test_pass_is_host_owned_and_protocol_binds_full_treatment(tmp_path):
    _composition, _source, destination, snapshot = _snapshot(tmp_path)
    calls = []
    backend = _Backend((True,))
    root = tmp_path / "phase2"

    terminal = Phase2LiveRunner(
        phase2_root=root,
        snapshot=snapshot,
        dependencies=_deps(snapshot, backend, calls),
    ).run()

    cell_root = root / "cells" / destination.cell_id
    protocol = json.loads((cell_root / "protocol.json").read_text())
    attempt = json.loads(
        (cell_root / "attempts" / "attempt-01" / "attempt.json").read_text()
    )
    assert terminal["status"] == "passed"
    assert terminal["completed_attempts"] == 1
    assert terminal["evidence_head_sha256"]
    assert protocol["schema"] == "a5-catlass-phase2-live-protocol-v3"
    assert protocol["semantic_retry"]["preceding_attempt_only"] is True
    assert protocol["semantic_retry"]["candidate_source_max_bytes"] == 256 * 1024
    assert protocol["identity"] == {
        "target_suite_sha256": destination.suite_sha256,
        "candidate_abi_sha256": CANDIDATE_ABI_SHA256,
        "host_target_grading_sha256": HOST_TARGET_GRADING_SHA256,
        "source_snapshot_id": snapshot.snapshot_id,
        "cell": destination.as_dict(),
    }
    assert attempt["candidate_abi_sha256"] == CANDIDATE_ABI_SHA256
    assert attempt["host_target_grading_sha256"] == HOST_TARGET_GRADING_SHA256
    assert attempt["protocol_sha256"] == protocol["protocol_sha256"]
    assert calls[0].phase1_memory == snapshot.memory_entries()
    assert calls[0].phase2_memory == ()
    assert calls[0].repair_context is None
    assert len(backend.calls) == 4


def test_failed_host_receipt_advances_memory_and_next_attempt(tmp_path):
    _composition, _source, _destination, snapshot = _snapshot(tmp_path)
    source_calls = []
    backend = _Backend((False, True))
    runner = Phase2LiveRunner(
        phase2_root=tmp_path / "phase2",
        snapshot=snapshot,
        dependencies=_deps(snapshot, backend, source_calls),
    )

    terminal = runner.run()

    assert terminal["status"] == "passed"
    assert terminal["completed_attempts"] == 2
    assert len(source_calls) == 2
    assert source_calls[1].phase2_memory[0]["passed"] is False
    repair = source_calls[1].repair_context
    assert repair is not None
    assert repair["prior_attempt"] == 1
    assert repair["candidate_source"] == SOURCE + "\n# attempt 1\n"
    assert repair["candidate_source_truncated"] is False
    assert repair["candidate_sha256"] == source_calls[1].phase2_memory[0][
        "candidate_sha256"
    ]
    assert repair["evidence_sha256"] == source_calls[1].phase2_memory[0][
        "evidence_sha256"
    ]
    assert repair["host_evidence"] == source_calls[1].phase2_memory[0]
    assert "repair the preceding candidate" in repair["instruction"]
    failed_case = source_calls[1].phase2_memory[0]["failed_cases"][0]
    assert failed_case == {
        "case_id": failed_case["case_id"],
        "failure": "numerical-mismatch",
        "exit_code": 0,
        "max_abs_error": failed_case["max_abs_error"],
        "max_rel_error": failed_case["max_rel_error"],
        "mismatch_index": 0,
        "mismatch_expected": failed_case["mismatch_expected"],
        "mismatch_actual": failed_case["mismatch_actual"],
    }
    assert failed_case["max_abs_error"] >= 1.0
    assert "agent" not in json.dumps(source_calls[1].phase2_memory).lower()
    entries = [json.loads(line) for line in runner.memory_path.read_text().splitlines()]
    assert [entry["attempt"] for entry in entries] == [1, 2]
    assert [entry["passed"] for entry in entries] == [False, True]


def test_long_host_diagnostic_head_and_tail_reach_next_attempt(tmp_path):
    class DiagnosticBackend(_Backend):
        def execute(self, source, execution, workdir):
            if not self.passed:
                return AttentionReceipt(
                    1,
                    (),
                    stdout=(
                        "Traceback (most recent call last):\n"
                        + "x" * 2000
                        + "\nValueError: dynamic loop value escaped its scope"
                    ),
                    session_handle="bz-a5:test",
                )
            return super().execute(source, execution, workdir)

    _composition, _source, _destination, snapshot = _snapshot(tmp_path)
    source_calls = []
    runner = Phase2LiveRunner(
        phase2_root=tmp_path / "phase2",
        snapshot=snapshot,
        dependencies=_deps(
            snapshot, DiagnosticBackend((False, True)), source_calls
        ),
    )

    terminal = runner.run()

    assert terminal["status"] == "passed"
    diagnostic = source_calls[1].phase2_memory[0]["failed_cases"][0]["diagnostic"]
    assert diagnostic.startswith("stdout:\nTraceback (most recent call last):")
    assert "\n...[truncated]...\n" in diagnostic
    assert diagnostic.endswith("dynamic loop value escaped its scope")
    assert len(diagnostic.encode("utf-8")) <= 512


def test_retry_candidate_source_is_bounded_and_digest_bound(tmp_path):
    _composition, _source, _destination, snapshot = _snapshot(tmp_path)
    source_calls = []
    backend = _Backend((False, True))
    verifier = AttentionTargetVerifier(backend)
    oversized = SOURCE + "\n# " + "x" * (300 * 1024)

    def source(request):
        source_calls.append(request)
        try:
            backend.passed = next(backend.outcomes)
        except StopIteration:
            pass
        return oversized if request.attempt == 1 else SOURCE

    dependencies = Phase2Dependencies(
        snapshot.destination,
        source,
        lambda candidate, request, attempt, workdir: verifier.verify(
            candidate, request_id=request, attempt_id=attempt,
            execution_profile="bz-a5", workdir=workdir,
        ),
    )
    terminal = Phase2LiveRunner(
        phase2_root=tmp_path / "phase2", snapshot=snapshot,
        dependencies=dependencies,
    ).run()

    assert terminal["status"] == "passed"
    repair = source_calls[1].repair_context
    assert repair is not None
    assert repair["candidate_source_truncated"] is True
    assert len(repair["candidate_source"].encode("utf-8")) <= 256 * 1024
    assert repair["candidate_source_bytes"] == len(oversized.encode("utf-8"))
    assert repair["candidate_context_sha256"] == hashlib.sha256(
        repair["candidate_source"].encode("utf-8")
    ).hexdigest()


def test_exhausted_semantic_attempts_emit_failed_terminal(tmp_path):
    _composition, _source, _destination, snapshot = _snapshot(tmp_path)
    calls = []
    now = [0.]
    dependencies = _deps(snapshot, _Backend((False,) * 3), calls)
    def source(request):
        now[0] += 1
        return dependencies.source(request)
    runner = Phase2LiveRunner(
        phase2_root=tmp_path / "phase2", snapshot=snapshot,
        dependencies=replace(dependencies, source=source), max_seconds=3, clock=lambda: now[0],
    )

    terminal = runner.run()

    assert terminal["status"] == "failed"
    assert terminal["completed_attempts"] == 3
    assert len(calls) == 3


def test_infrastructure_is_incomplete_and_resumes_same_persisted_source(tmp_path):
    _composition, _source, _destination, snapshot = _snapshot(tmp_path)
    calls = []
    backend = _Backend((True,))
    dependencies = _deps(snapshot, backend, calls)
    real_verify = dependencies.verify
    verify_calls = []

    def interrupted(source, request, attempt, workdir):
        verify_calls.append((source, request, attempt))
        if len(verify_calls) == 1:
            raise AttentionInfrastructureError("device unavailable")
        return real_verify(source, request, attempt, workdir)

    dependencies = Phase2Dependencies(
        snapshot.destination, dependencies.source, interrupted,
    )
    runner = Phase2LiveRunner(
        phase2_root=tmp_path / "phase2", snapshot=snapshot,
        dependencies=dependencies,
    )

    with pytest.raises(AttentionInfrastructureError):
        runner.run()
    assert not runner.memory_path.exists()
    assert not runner.terminal_path.exists()

    terminal = runner.run()
    assert terminal["status"] == "passed"
    assert len(calls) == 1
    assert verify_calls[0] == verify_calls[1]


def test_resume_does_not_redo_committed_attempt_or_terminal(tmp_path):
    _composition, _source, _destination, snapshot = _snapshot(tmp_path)
    calls = []
    runner = Phase2LiveRunner(
        phase2_root=tmp_path / "phase2", snapshot=snapshot,
        dependencies=_deps(snapshot, _Backend((True,)), calls),
    )
    expected = runner.run()
    memory = runner.memory_path.read_bytes()

    repeated = runner.run()

    assert repeated == expected
    assert runner.memory_path.read_bytes() == memory
    assert len(calls) == 1


def test_resume_rejects_changed_candidate_abi(tmp_path, monkeypatch):
    _composition, _source, _destination, snapshot = _snapshot(tmp_path)
    runner = Phase2LiveRunner(
        phase2_root=tmp_path / "phase2", snapshot=snapshot,
        dependencies=_deps(snapshot, _Backend((True,)), []),
    )
    runner.run()
    retained = json.loads(runner.protocol_path.read_text())

    monkeypatch.setattr(phase2_live_module, "CANDIDATE_ABI_SHA256", "f" * 64)
    assert runner._protocol_body()["identity"] != retained["identity"]
    with pytest.raises(ValueError, match="immutable Phase 2 artifact conflicts"):
        runner.run()


def test_resume_rejects_changed_host_grading_contract(tmp_path, monkeypatch):
    _composition, _source, _destination, snapshot = _snapshot(tmp_path)
    runner = Phase2LiveRunner(
        phase2_root=tmp_path / "phase2", snapshot=snapshot,
        dependencies=_deps(snapshot, _Backend((True,)), []),
    )
    runner.run()
    retained = json.loads(runner.protocol_path.read_text())

    monkeypatch.setattr(
        phase2_live_module, "HOST_TARGET_GRADING_SHA256", "e" * 64,
    )
    assert runner._protocol_body()["identity"] != retained["identity"]
    with pytest.raises(ValueError, match="immutable Phase 2 artifact conflicts"):
        runner.run()


def test_resume_reverifies_snapshot_and_rejects_tamper(tmp_path):
    _composition, _source, _destination, snapshot = _snapshot(tmp_path)
    runner = Phase2LiveRunner(
        phase2_root=tmp_path / "phase2", snapshot=snapshot,
        dependencies=_deps(snapshot, _Backend((True,)), []),
    )
    runner.run()
    snapshot.memory_path.write_bytes(snapshot.memory_path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="bytes changed"):
        runner.run()


def test_resume_rejects_missing_committed_host_receipt(tmp_path):
    _composition, _source, _destination, snapshot = _snapshot(tmp_path)
    runner = Phase2LiveRunner(
        phase2_root=tmp_path / "phase2", snapshot=snapshot,
        dependencies=_deps(snapshot, _Backend((True,)), []),
    )
    runner.run()
    (runner.root / "attempts" / "attempt-01" / "evidence.json").unlink()

    with pytest.raises(ValueError, match="incomplete"):
        runner.run()


def test_rejects_cell_identity_and_output_path_mismatch(tmp_path):
    _composition, _source, _destination, snapshot = _snapshot(tmp_path)
    dependencies = _deps(snapshot, _Backend((True,)), [])
    wrong = tmp_path / "phase2" / "not-cells" / snapshot.destination.cell_id

    with pytest.raises(ValueError, match="root|cell"):
        Phase2LiveRunner(
            phase2_root=wrong, snapshot=snapshot, dependencies=dependencies,
        )

    changed = asdict(snapshot.destination)
    changed["cell_id"] = "a5-phase2-cell-" + "0" * 16
    with pytest.raises((TypeError, ValueError)):
        Phase2Dependencies(changed, dependencies.source, dependencies.verify)


def test_time_window_allows_more_than_three_semantic_attempts(tmp_path):
    _composition, _source, _destination, snapshot = _snapshot(tmp_path)
    calls = []
    terminal = Phase2LiveRunner(phase2_root=tmp_path / 'phase2', snapshot=snapshot,
        dependencies=_deps(snapshot, _Backend((False, False, False, True)), calls)).run()
    assert terminal['status'] == 'passed'
    assert terminal['completed_attempts'] == len(calls) == 4
    assert calls[-1].remaining_seconds > 0
