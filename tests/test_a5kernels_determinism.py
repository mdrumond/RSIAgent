from dataclasses import replace
import json

import pytest

from benchmarks.a5kernels.determinism import (
    AUDITED_FIELDS,
    MetricsPacket,
    ReplaySnapshot,
    audit_replays,
    main,
    run_three_replays,
    snapshot_from_ledger,
)
from benchmarks.a5kernels.evidence import EvidenceKind, EvidenceLedger
from benchmarks.a5kernels.protocol import RunRequest


def snapshot(attempt_id: str, **changes) -> ReplaySnapshot:
    values = {
        "request_id": RunRequest("catlass-dsl", length=2, seed=7).request_id,
        "attempt_id": attempt_id,
        "actions": ({"language": "catlass-dsl", "argv": ["python", "run.py"]},),
        "source_artifacts": ({
            "sequence": 2,
            "source_sha256": {"kernel.py": "source-1"},
            "artifact_sha256": {"source_bundle": "artifact-1"},
        },),
        "evidence": ("evidence-1",),
        "output": {"output_sha256": "output-1", "passed": True},
        "metrics": {"latency_us": 10.0, "max_abs_error": 0.0},
    }
    values.update(changes)
    return ReplaySnapshot(**values)


def write_ledger(
    path,
    attempt_id: str,
    *,
    output="output-1",
    include_action=True,
    include_artifact=True,
) -> EvidenceLedger:
    request = RunRequest("catlass-dsl", length=2, seed=7)
    identity = {"request_id": request.request_id, "attempt_id": attempt_id}
    ledger = EvidenceLedger(path)
    ledger.append(EvidenceKind.REQUEST, {**identity, "request": request.__dict__})
    if include_action:
        ledger.append(EvidenceKind.ACTION, {**identity, "language": request.language})
    if include_artifact:
        ledger.append(
            EvidenceKind.ARTIFACT,
            {
                **identity,
                "source_sha256": {"kernel.py": "source-1"},
                "artifact_sha256": {"source_bundle": "artifact-1"},
            },
        )
    ledger.append(
        EvidenceKind.RESULT,
        {
            **identity,
            "output_sha256": output,
            "passed": True,
            "exit_code": 0,
            "max_abs_error": 0.0,
        },
    )
    return ledger


def test_runner_captures_exactly_three_unique_attempts_for_one_request():
    request = RunRequest("catlass-dsl", length=2, seed=7)
    calls = []

    def capture(seen_request, attempt_id):
        calls.append((seen_request, attempt_id))
        return snapshot(attempt_id)

    report = run_three_replays(request, capture, audit_id="audit-7")

    assert [attempt for _, attempt in calls] == [
        "audit-7-replay-1",
        "audit-7-replay-2",
        "audit-7-replay-3",
    ]
    assert all(seen is request for seen, _ in calls)
    assert report.run_count == 3
    assert report.fields_with_observed_divergence == ()
    assert "do not prove determinism" in report.interpretation


@pytest.mark.parametrize("returned", ["request", "attempt"])
def test_runner_rejects_misidentified_capture(returned):
    request = RunRequest("catlass-dsl", length=2, seed=7)

    def capture(_request, attempt_id):
        if returned == "request":
            return replace(snapshot(attempt_id), request_id="other")
        return snapshot("other-attempt")

    with pytest.raises(ValueError, match=f"different {returned}_id"):
        run_three_replays(request, capture, audit_id="audit-7")


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("actions", ({"language": "catlass-dsl", "argv": ["python", "other.py"]},)),
        (
            "source_artifacts",
            ({
                "sequence": 2,
                "source_sha256": {"kernel.py": "source-2"},
                "artifact_sha256": {},
            },),
        ),
        ("evidence", ("evidence-2",)),
        ("output", {"output_sha256": "output-2", "passed": True}),
        ("metrics", {"latency_us": 11.0, "max_abs_error": 0.0}),
    ],
)
def test_audit_reports_each_required_divergence_dimension(field, changed):
    report = audit_replays(
        [snapshot("one"), snapshot("two"), snapshot("three", **{field: changed})]
    )

    assert report.fields_with_observed_divergence == (field,)
    assert report.comparisons[field].matches is False
    assert report.comparisons[field].divergent_pairs == 2


@pytest.mark.parametrize(
    "replays, message",
    [
        ([snapshot("one"), snapshot("two")], "exactly three"),
        ([snapshot("same"), snapshot("same"), snapshot("three")], "unique"),
        (
            [snapshot("one"), snapshot("two"), replace(snapshot("three"), request_id="other")],
            "share one request_id",
        ),
    ],
)
def test_audit_rejects_incomplete_or_ambiguous_replay_sets(replays, message):
    with pytest.raises(ValueError, match=message):
        audit_replays(replays)


def test_verified_ledger_converts_to_snapshot(tmp_path):
    path = tmp_path / "one.jsonl"
    write_ledger(path, "one")

    ledger = EvidenceLedger(path)
    replay = snapshot_from_ledger(
        ledger.entries,
        metrics=MetricsPacket(
            request_id=ledger.entries[0].payload["request_id"],
            attempt_id="one",
            ledger_head_sha256=ledger.head_sha256,
            metrics={"latency_us": 4.5},
        ),
    )

    assert replay.attempt_id == "one"
    assert replay.actions == ({"language": "catlass-dsl"},)
    assert replay.source_artifacts[0]["source_sha256"]["kernel.py"] == "source-1"
    assert replay.output["passed"] is True
    assert replay.metrics["supplemental"]["latency_us"] == 4.5
    assert len(replay.evidence) == 4


@pytest.mark.parametrize(
    ("missing", "options", "message"),
    [
        ("action", {"include_action": False}, "exactly one action"),
        ("artifact", {"include_artifact": False}, "at least one artifact"),
    ],
)
def test_snapshot_rejects_missing_execution_evidence(
    tmp_path, missing, options, message
):
    ledger = write_ledger(tmp_path / f"missing-{missing}.jsonl", "one", **options)

    with pytest.raises(ValueError, match=message):
        snapshot_from_ledger(ledger.entries)


@pytest.mark.parametrize("field", ["source_sha256", "artifact_sha256"])
def test_snapshot_rejects_artifacts_without_hash_evidence(tmp_path, field):
    request = RunRequest("catlass-dsl", length=2, seed=7)
    identity = {"request_id": request.request_id, "attempt_id": "one"}
    ledger = EvidenceLedger(tmp_path / f"missing-{field}.jsonl")
    ledger.append(EvidenceKind.REQUEST, {**identity, "request": request.__dict__})
    ledger.append(EvidenceKind.ACTION, {**identity, "language": request.language})
    ledger.append(
        EvidenceKind.ARTIFACT,
        {
            **identity,
            "source_sha256": {"kernel.py": "source-1"},
            "artifact_sha256": {"source_bundle": "artifact-1"},
            field: {},
        },
    )
    ledger.append(EvidenceKind.RESULT, {**identity, "passed": True, "exit_code": 0})

    with pytest.raises(ValueError, match=f"non-empty {field} mapping"):
        snapshot_from_ledger(ledger.entries)


def test_cli_writes_compact_machine_readable_three_replay_report(tmp_path):
    ledgers = []
    metrics = []
    for index in range(1, 4):
        ledger = tmp_path / f"replay-{index}.jsonl"
        evidence = write_ledger(ledger, f"attempt-{index}")
        ledgers.append(str(ledger))
        metric = tmp_path / f"metrics-{index}.json"
        metric.write_text(json.dumps({
            "request_id": evidence.entries[0].payload["request_id"],
            "attempt_id": f"attempt-{index}",
            "ledger_head_sha256": evidence.head_sha256,
            "metrics": {"latency_us": 4.5},
        }), encoding="utf-8")
        metrics.append(str(metric))
    output = tmp_path / "report.json"

    assert main([*ledgers, "--metrics", *metrics, "--output", str(output)]) == 0

    raw = output.read_text(encoding="utf-8")
    report = json.loads(raw)
    assert "\n " not in raw
    assert report["run_count"] == 3
    assert set(report["comparisons"]) == set(AUDITED_FIELDS)
    assert report["fields_with_observed_divergence"] == []
    assert "do not prove determinism" in report["interpretation"]


def test_metrics_packet_rejects_reserved_keys_and_wrong_ledger(tmp_path):
    path = tmp_path / "one.jsonl"
    ledger = write_ledger(path, "one")
    identity = {
        "request_id": ledger.entries[0].payload["request_id"],
        "attempt_id": "one",
        "ledger_head_sha256": ledger.head_sha256,
    }
    with pytest.raises(ValueError, match="reserved ledger keys"):
        MetricsPacket.from_mapping({**identity, "metrics": {"exit_code": 0}})
    packet = MetricsPacket.from_mapping({**identity, "metrics": {"latency_us": 4.5}})

    with pytest.raises(ValueError, match="does not match"):
        snapshot_from_ledger(
            ledger.entries, metrics=replace(packet, attempt_id="another-attempt")
        )


def test_snapshot_preserves_ordered_artifact_generations(tmp_path):
    path = tmp_path / "one.jsonl"
    ledger = write_ledger(path, "one")
    identity = ledger.entries[0].payload
    ledger.append(
        EvidenceKind.ARTIFACT,
        {
            "request_id": identity["request_id"],
            "attempt_id": "one",
            "source_sha256": {"kernel.py": "source-2"},
            "artifact_sha256": {"source_bundle": "artifact-2"},
        },
    )

    replay = snapshot_from_ledger(ledger.entries)

    assert [item["source_sha256"]["kernel.py"] for item in replay.source_artifacts] == [
        "source-1",
        "source-2",
    ]
