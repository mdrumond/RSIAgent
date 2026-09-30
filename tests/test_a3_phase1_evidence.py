import json

import pytest

from benchmarks.a3kernels.phase1_evidence import (
    GENESIS_HASH,
    EvidenceKind,
    EvidenceLedger,
    canonical_bytes,
    canonical_digest,
)
from benchmarks.a3kernels.phase1_protocol import ExecutionPlan, SourceFile


def test_canonical_serialization_is_exact_and_portable():
    left = {"unicode": "café", "nested": {"b": (2, 3), "a": 1}}
    right = {"nested": {"a": 1, "b": [2, 3]}, "unicode": "café"}

    assert canonical_bytes(left) == canonical_bytes(right)
    assert canonical_digest(left) == canonical_digest(right)
    assert canonical_bytes(left) == (
        b'{"nested":{"a":1,"b":[2,3]},"unicode":"caf\\u00e9"}'
    )
    with pytest.raises(ValueError, match="non-finite"):
        canonical_bytes({"bad": float("nan")})
    with pytest.raises(TypeError, match="string keys"):
        canonical_bytes({1: "bad"})


def test_ledger_round_trip_preserves_append_continuity(tmp_path):
    path = tmp_path / "a3.evidence.jsonl"
    first_writer = EvidenceLedger(path)
    second_writer = EvidenceLedger(path)

    first = first_writer.append(EvidenceKind.PLAN, {"request_id": "r1"})
    second = second_writer.append(EvidenceKind.RESULT, {"passed": True})
    reopened = EvidenceLedger(path)

    assert first.previous_sha256 == GENESIS_HASH
    assert second.previous_sha256 == first.entry_sha256
    assert reopened.entries == (first, second)
    assert reopened.head_sha256 == second.entry_sha256
    assert path.read_bytes().endswith(b"\n")


def test_ledger_detects_tampering(tmp_path):
    path = tmp_path / "a3.evidence.jsonl"
    EvidenceLedger(path).append(EvidenceKind.RESULT, {"passed": True})
    entry = json.loads(path.read_text())
    entry["payload"]["passed"] = False
    path.write_text(json.dumps(entry) + "\n")

    with pytest.raises(ValueError, match="digest"):
        EvidenceLedger(path)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"target": "Ascend950"}, "target"),
        ({"language": "catlass-dsl"}, "language"),
        ({"execution_profile": "bz-a5"}, "profile"),
        ({"runtime_provenance": [["target", "a5"]]}, "provenance"),
    ],
)
def test_ledger_rejects_a5_catlass_or_foreign_provenance(tmp_path, payload, message):
    with pytest.raises(ValueError, match=message):
        EvidenceLedger(tmp_path / "bad.jsonl").append(EvidenceKind.ACTION, payload)


def test_deterministic_local_round_trip_records_source_identity(tmp_path):
    plan = ExecutionPlan(
        request_id="request-proof",
        attempt_id="attempt-proof",
        project_id="vector-add",
        files=(SourceFile("kernel.cpp", "// deterministic proof\n"),),
        argv=("python", "host_driver.py"),
        input_a=(1.0, 2.0),
        input_b=(3.0, 5.0),
    )
    expected = tuple(a + b for a, b in zip(plan.input_a, plan.input_b))
    ledger = EvidenceLedger(tmp_path / "proof.jsonl")
    ledger.append(EvidenceKind.PLAN, plan.as_dict())
    ledger.append(
        EvidenceKind.RESULT,
        {"execution_id": plan.execution_id, "output": expected, "passed": expected == (4.0, 7.0)},
    )

    reopened = EvidenceLedger(ledger.path)
    assert reopened.entries[0].payload["source_fingerprint"] == plan.source_fingerprint
    assert reopened.entries[-1].payload["passed"] is True
    assert reopened.head_sha256 == ledger.head_sha256
