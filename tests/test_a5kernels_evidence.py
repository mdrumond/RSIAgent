import json

import pytest

from benchmarks.a5kernels import A5KernelRunner, BZSessionAdapter, EvidenceLedger, RunRequest
from benchmarks.a5kernels.bz import RuntimeUnavailableError
from benchmarks.a5kernels.evidence import (
    GENESIS_HASH,
    EvidenceKind,
    canonical_bytes,
    canonical_digest,
)
from benchmarks.a5kernels.protocol import ExecutionReceipt


class FakeBackend:
    def __init__(self, corrupt=False):
        self.corrupt = corrupt

    def execute(self, plan):
        output = tuple(a + b for a, b in zip(plan.input_a, plan.input_b))
        if self.corrupt:
            output = output[:-1]
        return ExecutionReceipt(
            exit_code=0,
            output=output,
            stdout="remote log",
            session_handle="bz-a5:evidence-test",
            metadata=(("agent_claim", "passed"),),
        )


class NoDispatchExecutor:
    def run(self, invocation):
        raise RuntimeUnavailableError("runtime unavailable")

    def inspect(self, argv):
        raise AssertionError("fail-closed fixture must not inspect a session")


def test_canonical_serialization_is_order_independent_and_strict():
    left = {"unicode": "café", "nested": {"b": (2, 3), "a": 1}}
    right = {"nested": {"a": 1, "b": [2, 3]}, "unicode": "café"}

    assert canonical_bytes(left) == canonical_bytes(right)
    assert canonical_digest(left) == canonical_digest(right)
    with pytest.raises(ValueError, match="non-finite"):
        canonical_bytes({"bad": float("nan")})
    with pytest.raises(TypeError, match="string keys"):
        canonical_bytes({1: "bad"})


def test_ledger_appends_and_reopens_a_verified_chain(tmp_path):
    path = tmp_path / "run.evidence.jsonl"
    ledger = EvidenceLedger(path)

    first = ledger.append(EvidenceKind.REQUEST, {"seed": 7})
    second = ledger.append(EvidenceKind.ACTION, {"argv": ("python", "kernel.py")})
    reopened = EvidenceLedger(path)

    assert first.previous_sha256 == GENESIS_HASH
    assert second.previous_sha256 == first.entry_sha256
    assert reopened.entries == (first, second)
    assert reopened.head_sha256 == second.entry_sha256
    assert path.read_bytes().endswith(b"\n")


def test_two_ledger_instances_refresh_the_chain_before_appending(tmp_path):
    path = tmp_path / "shared.evidence.jsonl"
    first = EvidenceLedger(path)
    second = EvidenceLedger(path)

    request = first.append(EvidenceKind.REQUEST, {"seed": 7})
    action = second.append(EvidenceKind.ACTION, {"argv": ["python", "kernel.py"]})

    reopened = EvidenceLedger(path)
    assert [entry.sequence for entry in reopened.entries] == [0, 1]
    assert action.previous_sha256 == request.entry_sha256


def test_ledger_rejects_payload_tampering(tmp_path):
    path = tmp_path / "run.evidence.jsonl"
    EvidenceLedger(path).append(EvidenceKind.RESULT, {"passed": True})
    record = json.loads(path.read_text())
    record["payload"]["passed"] = False
    path.write_text(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="invalid evidence digest"):
        EvidenceLedger(path)


def test_runner_records_host_owned_lifecycle_without_agent_metadata(tmp_path):
    ledger = EvidenceLedger(tmp_path / "run.evidence.jsonl")
    result = A5KernelRunner(FakeBackend(), evidence_ledger=ledger).run(
        RunRequest("catlass-dsl", length=3, seed=9)
    )

    assert result.passed
    assert [entry.kind for entry in ledger.entries] == [
        "request",
        "action",
        "artifact",
        "result",
    ]
    assert ledger.entries[0].payload["request_id"] == result.request_id
    assert ledger.entries[0].payload["execution_id"] == result.execution_id
    assert ledger.entries[0].payload["attempt_id"] == result.attempt_id
    assert ledger.entries[0].payload["request"]["seed"] == 9
    assert ledger.entries[1].payload["attempt_id"] == result.attempt_id
    assert ledger.entries[2].payload["source_fingerprint"] == result.source_fingerprint
    assert ledger.entries[3].payload["attestation_sha256"] == result.attestation_sha256
    assert "agent_claim" not in path_text(ledger.path)


def test_runner_records_non_finite_failure_metric_portably(tmp_path):
    ledger = EvidenceLedger(tmp_path / "failed.evidence.jsonl")
    result = A5KernelRunner(FakeBackend(corrupt=True), evidence_ledger=ledger).run(
        RunRequest("ascend-c", length=3)
    )

    assert not result.passed
    payload = ledger.entries[-1].payload
    assert payload["max_abs_error"] is None
    assert payload["max_abs_error_status"] == "non-finite"
    EvidenceLedger(ledger.path)


def test_fail_closed_fixture_records_terminal_attempt_lifecycle(tmp_path):
    ledger = EvidenceLedger(tmp_path / "unavailable.evidence.jsonl")
    backend = BZSessionAdapter(
        NoDispatchExecutor(), session_wrapper="execution-profiles/bz-a5/session.sh"
    )

    with pytest.raises(RuntimeUnavailableError):
        A5KernelRunner(backend, evidence_ledger=ledger).run(
            RunRequest("catlass-dsl"), attempt_id="unavailable-1"
        )

    assert [entry.kind for entry in ledger.entries] == [
        "request",
        "action",
        "artifact",
        "result",
    ]
    result = ledger.entries[-1].payload
    assert result == {
        "request_id": ledger.entries[0].payload["request_id"],
        "execution_id": ledger.entries[0].payload["execution_id"],
        "attempt_id": "unavailable-1",
        "status": "execution_error",
        "error_type": "RuntimeUnavailableError",
    }


def path_text(path):
    return path.read_text(encoding="utf-8")
