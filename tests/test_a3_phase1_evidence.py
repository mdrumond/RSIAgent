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


def test_ledger_rejects_unterminated_entry_before_append(tmp_path):
    path = tmp_path / "a3.evidence.jsonl"
    ledger = EvidenceLedger(path)
    ledger.append(EvidenceKind.PLAN, {"request_id": "r1"})
    path.write_bytes(path.read_bytes().removesuffix(b"\n"))

    with pytest.raises(ValueError, match="unterminated"):
        ledger.append(EvidenceKind.RESULT, {"passed": True})
    with pytest.raises(ValueError, match="unterminated"):
        EvidenceLedger(path)


def test_ledger_recovers_only_torn_tail_and_continues_chain_deterministically(tmp_path):
    path = tmp_path / "interrupted.jsonl"
    reference_path = tmp_path / "reference.jsonl"
    for target in (path, reference_path):
        ledger = EvidenceLedger(target)
        ledger.append(EvidenceKind.PLAN, {"request_id": "r1"})
        ledger.append(EvidenceKind.ACTION, {"action": "compile"})
    committed = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b'{"sequence":2,"kind":"result"')

    assert EvidenceLedger.recover_incomplete_tail(path) == 1
    assert path.read_bytes() == committed
    resumed = EvidenceLedger(path)
    reference = EvidenceLedger(reference_path)
    resumed_entry = resumed.append(EvidenceKind.RESULT, {"passed": True})
    reference_entry = reference.append(EvidenceKind.RESULT, {"passed": True})

    assert resumed_entry == reference_entry
    assert path.read_bytes() == reference_path.read_bytes()


def test_ledger_recovery_rejects_newline_terminated_malformed_record(tmp_path):
    path = tmp_path / "committed-malformed.jsonl"
    EvidenceLedger(path).append(EvidenceKind.PLAN, {"request_id": "r1"})
    with path.open("ab") as stream:
        stream.write(b"not-json\n")

    with pytest.raises(ValueError, match="line 2"):
        EvidenceLedger.recover_incomplete_tail(path)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"target": "Ascend950"}, "target"),
        ({"language": "catlass-dsl"}, "language"),
        ({"execution_profile": "bz-a5"}, "profile"),
        ({"runtime_provenance": [["target", "a5"]]}, "target"),
    ],
)
def test_ledger_rejects_a5_catlass_or_foreign_provenance(tmp_path, payload, message):
    with pytest.raises(ValueError, match=message):
        EvidenceLedger(tmp_path / "bad.jsonl").append(EvidenceKind.ACTION, payload)


def test_provenance_checks_only_explicit_route_fields(tmp_path):
    digest_with_a5 = "0" * 20 + "a5" + "1" * 42
    payload = {
        "runtime_provenance": [
            ["artifact_sha256", digest_with_a5],
            ["note", "compare catlass history without routing to it"],
            ["execution_profile", "gz-a3"],
            ["language", "ascend-c"],
        ]
    }

    entry = EvidenceLedger(tmp_path / "explicit.jsonl").append(
        EvidenceKind.ARTIFACT, payload
    )

    assert entry.payload["runtime_provenance"][0][1] == digest_with_a5


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("target", "a5", "target"),
        ("language", "catlass-dsl", "language"),
        ("execution_profile", "bz-a5", "profile"),
        ("runtime", "catlass", "runtime"),
    ],
)
def test_nested_provenance_rejects_explicit_foreign_route(field, value, message, tmp_path):
    with pytest.raises(ValueError, match=message):
        EvidenceLedger(tmp_path / "foreign.jsonl").append(
            EvidenceKind.ACTION,
            {"runtime_provenance": [["artifact_sha256", "a5" * 32], [field, value]]},
        )


def test_deterministic_local_round_trip_records_source_identity(tmp_path):
    plan = ExecutionPlan(
        request_id="request-proof",
        attempt_id="attempt-proof",
        project_id="vector-add",
        files=(SourceFile("kernel.cpp", "// deterministic proof\n"),),
        argv=("python", "host_driver.py"),
        input_a=(1.0, 2.0),
        input_b=(3.0, 5.0),
        execution_profile="bz-a3-1",
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


@pytest.mark.parametrize("profile", ["bz-a3-1", "bz-a3-2", "gz-a3"])
def test_ledger_accepts_authoritative_and_compatibility_a3_profiles(tmp_path, profile):
    entry = EvidenceLedger(tmp_path / f"{profile}.jsonl").append(
        EvidenceKind.ACTION,
        {"execution_profile": profile},
    )

    assert entry.payload["execution_profile"] == profile
