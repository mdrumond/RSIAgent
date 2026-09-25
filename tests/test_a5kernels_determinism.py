from dataclasses import replace
import json
import random

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
from benchmarks.a5kernels.evidence import (
    EvidenceEntry,
    EvidenceKind,
    EvidenceLedger,
    canonical_digest,
)
from benchmarks.a5kernels.protocol import ExecutionPlan, RunRequest, canonical_hash
from benchmarks.a5kernels.fixtures import fixture_for


SOURCE_ONE = "1" * 64
SOURCE_TWO = "2" * 64
ARTIFACT_ONE = "a" * 64
ARTIFACT_TWO = "b" * 64
OUTPUT_ONE = "e" * 64


def input_digest(request):
    rng = random.Random(request.seed)
    return canonical_digest({
        "input_a": tuple(rng.uniform(-1.0, 1.0) for _ in range(request.length)),
        "input_b": tuple(rng.uniform(-1.0, 1.0) for _ in range(request.length)),
    })


def fixture_argv(request):
    argv = fixture_for(request.language).argv
    return list(argv) if argv is not None else None


def snapshot(attempt_id: str, **changes) -> ReplaySnapshot:
    values = {
        "request_id": RunRequest("catlass-dsl", length=2, seed=7).request_id,
        "attempt_id": attempt_id,
        "actions": ({"language": "catlass-dsl", "argv": ["python", "run.py"]},),
        "source_artifacts": ({
            "sequence": 2,
            "source_sha256": {"kernel.py": SOURCE_ONE},
            "artifact_sha256": {"source_bundle": ARTIFACT_ONE},
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
    output=OUTPUT_ONE,
    include_action=True,
    include_artifact=True,
    language="catlass-dsl",
) -> EvidenceLedger:
    request = RunRequest(language, length=2, seed=7)
    fixture = fixture_for(language)
    rng = random.Random(request.seed)
    input_a = tuple(rng.uniform(-1.0, 1.0) for _ in range(request.length))
    input_b = tuple(rng.uniform(-1.0, 1.0) for _ in range(request.length))
    plan = ExecutionPlan(
        request_id=request.request_id,
        attempt_id=attempt_id,
        language=language,
        files=fixture.files,
        argv=fixture.argv,
        input_a=input_a,
        input_b=input_b,
    )
    identity = {
        "request_id": request.request_id,
        "execution_id": plan.execution_id,
        "attempt_id": attempt_id,
    }
    ledger = EvidenceLedger(path)
    ledger.append(EvidenceKind.REQUEST, {**identity, "request": request.__dict__})
    if include_action:
        ledger.append(
            EvidenceKind.ACTION,
            {
                **identity,
                "language": request.language,
                "argv": fixture_argv(request),
                "inputs_sha256": input_digest(request),
            },
        )
    if include_artifact:
        ledger.append(
            EvidenceKind.ARTIFACT,
            {
                **identity,
                "source_fingerprint": plan.source_fingerprint,
                "source_sha256": {
                    source.relative_path: source.sha256 for source in fixture.files
                },
                "artifact_sha256": {"source_bundle": plan.source_fingerprint},
            },
        )
    execution_evidence = {
        "retained_logs": "",
        "diagnostics": "",
        "result_exit_code": 0,
        "output_sha256": output,
    }
    evidence_sha256 = canonical_hash({
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "session_handle": None,
        "output_sha256": output,
        "execution_id": plan.execution_id,
        "runtime_provenance": plan.runtime_provenance,
    })
    ledger.append(
        EvidenceKind.RESULT,
        {
            **identity,
            "output_sha256": output,
            "status": "verified",
            "passed": True,
            "exit_code": 0,
            "max_abs_error": 0.0,
            "runtime_provenance": plan.runtime_provenance,
            "session_handle": None,
            "evidence_sha256": evidence_sha256,
            "execution_evidence": execution_evidence,
        },
    )
    return ledger


def replace_entry_payload(entries, index, payload):
    rebuilt = list(entries[:index])
    previous = rebuilt[-1].entry_sha256 if rebuilt else "0" * 64
    original = entries[index]
    rebuilt.append(
        EvidenceEntry.create(original.sequence, original.kind, payload, previous)
    )
    for entry in entries[index + 1:]:
        rebuilt.append(
            EvidenceEntry.create(
                entry.sequence, entry.kind, entry.payload, rebuilt[-1].entry_sha256
            )
        )
    return rebuilt


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
    request = RunRequest("catlass-dsl", length=2, seed=7)
    assert replay.actions == ({
        "language": "catlass-dsl",
        "argv": fixture_argv(request),
        "inputs_sha256": input_digest(request),
    },)
    fixture = fixture_for("catlass-dsl")
    assert replay.source_artifacts[0]["source_sha256"] == {
        source.relative_path: source.sha256 for source in fixture.files
    }
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


def test_snapshot_rejects_identity_only_action(tmp_path):
    ledger = write_ledger(tmp_path / "empty-action.jsonl", "one")
    action = ledger.entries[1]
    payload = {
        key: action.payload[key]
        for key in ("request_id", "execution_id", "attempt_id")
    }

    with pytest.raises(ValueError, match="substantive action evidence"):
        snapshot_from_ledger(replace_entry_payload(ledger.entries, 1, payload))


def test_snapshot_accepts_runner_null_argv_boundary(tmp_path):
    ledger = write_ledger(
        tmp_path / "null-argv.jsonl", "one", language="ascend-c"
    )
    replay = snapshot_from_ledger(ledger.entries)

    assert replay.actions[0]["argv"] is None


def test_snapshot_rejects_numeric_input_digest(tmp_path):
    ledger = write_ledger(tmp_path / "numeric-input.jsonl", "one")
    action = ledger.entries[1]

    with pytest.raises(ValueError, match="substantive action evidence"):
        snapshot_from_ledger(
            replace_entry_payload(
                ledger.entries,
                1,
                {**action.payload, "inputs_sha256": int("1" * 64)},
            )
        )


def test_snapshot_binds_request_id_to_recorded_request(tmp_path):
    ledger = write_ledger(tmp_path / "missing-request.jsonl", "one")
    request = ledger.entries[0]
    payload = {key: value for key, value in request.payload.items() if key != "request"}

    with pytest.raises(ValueError, match="valid recorded request"):
        snapshot_from_ledger(replace_entry_payload(ledger.entries, 0, payload))


def test_snapshot_binds_action_language_to_recorded_request(tmp_path):
    ledger = write_ledger(tmp_path / "wrong-language.jsonl", "one")
    action = ledger.entries[1]

    with pytest.raises(ValueError, match="action language"):
        snapshot_from_ledger(
            replace_entry_payload(
                ledger.entries, 1, {**action.payload, "language": "ascend-c"}
            )
        )


def test_snapshot_rejects_unregistered_request_language(tmp_path):
    ledger = write_ledger(tmp_path / "unsupported-language.jsonl", "one")
    request_entry = ledger.entries[0]
    unsupported = RunRequest("made-up", length=2, seed=7)
    request_payload = {
        **request_entry.payload,
        "request_id": unsupported.request_id,
        "request": unsupported.__dict__,
    }
    rebuilt = replace_entry_payload(ledger.entries, 0, request_payload)
    for index in range(1, len(rebuilt)):
        payload = {
            **rebuilt[index].payload,
            "request_id": unsupported.request_id,
        }
        if index == 1:
            payload["language"] = unsupported.language
        rebuilt = replace_entry_payload(rebuilt, index, payload)

    with pytest.raises(ValueError, match="unsupported language"):
        snapshot_from_ledger(rebuilt)


def test_snapshot_binds_input_digest_to_recorded_request(tmp_path):
    ledger = write_ledger(tmp_path / "wrong-inputs.jsonl", "one")
    action = ledger.entries[1]

    with pytest.raises(ValueError, match="input digest"):
        snapshot_from_ledger(
            replace_entry_payload(
                ledger.entries, 1, {**action.payload, "inputs_sha256": "f" * 64}
            )
        )


def test_snapshot_binds_argv_to_registered_fixture(tmp_path):
    ledger = write_ledger(tmp_path / "wrong-argv.jsonl", "one")
    action = ledger.entries[1]

    with pytest.raises(ValueError, match="registered fixture"):
        snapshot_from_ledger(
            replace_entry_payload(
                ledger.entries, 1, {**action.payload, "argv": ["python", "other.py"]}
            )
        )


def test_snapshot_rejects_request_over_fixture_length_limit(tmp_path):
    ledger = write_ledger(tmp_path / "oversized.jsonl", "one")
    oversized = RunRequest("catlass-dsl", length=401, seed=7)
    request = ledger.entries[0]
    rebuilt = replace_entry_payload(
        ledger.entries,
        0,
        {
            **request.payload,
            "request_id": oversized.request_id,
            "request": oversized.__dict__,
        },
    )
    for index in range(1, len(rebuilt)):
        rebuilt = replace_entry_payload(
            rebuilt,
            index,
            {**rebuilt[index].payload, "request_id": oversized.request_id},
        )

    with pytest.raises(ValueError, match="fixture length limit"):
        snapshot_from_ledger(rebuilt)


def test_snapshot_rejects_non_sha_output_digest(tmp_path):
    ledger = write_ledger(tmp_path / "invalid-output.jsonl", "one")
    result = ledger.entries[-1]

    with pytest.raises(ValueError, match="verified replay output and metrics"):
        snapshot_from_ledger(
            replace_entry_payload(
                ledger.entries,
                len(ledger.entries) - 1,
                {**result.payload, "output_sha256": "not-a-sha"},
            )
        )


def test_snapshot_rejects_numeric_output_digest(tmp_path):
    ledger = write_ledger(tmp_path / "numeric-output.jsonl", "one")
    result = ledger.entries[-1]

    with pytest.raises(ValueError, match="verified replay output and metrics"):
        snapshot_from_ledger(
            replace_entry_payload(
                ledger.entries,
                len(ledger.entries) - 1,
                {**result.payload, "output_sha256": int("1" * 64)},
            )
        )


@pytest.mark.parametrize("field", ["source_sha256", "artifact_sha256"])
def test_snapshot_rejects_artifacts_without_hash_evidence(tmp_path, field):
    request = RunRequest("catlass-dsl", length=2, seed=7)
    identity = {
        "request_id": request.request_id,
        "execution_id": "execution-1",
        "attempt_id": "one",
    }
    ledger = EvidenceLedger(tmp_path / f"missing-{field}.jsonl")
    ledger.append(EvidenceKind.REQUEST, {**identity, "request": request.__dict__})
    ledger.append(
        EvidenceKind.ACTION,
        {
            **identity,
            "language": request.language,
            "argv": fixture_argv(request),
            "inputs_sha256": input_digest(request),
        },
    )
    ledger.append(
        EvidenceKind.ARTIFACT,
        {
            **identity,
            "source_sha256": {"kernel.py": SOURCE_ONE},
            "artifact_sha256": {"source_bundle": ARTIFACT_ONE},
            field: {},
        },
    )
    ledger.append(
        EvidenceKind.RESULT,
        {
            **identity,
            "status": "verified",
            "output_sha256": OUTPUT_ONE,
            "passed": True,
            "exit_code": 0,
            "max_abs_error": 0.0,
        },
    )

    with pytest.raises(ValueError, match=f"non-empty {field} mapping"):
        snapshot_from_ledger(ledger.entries)


def test_snapshot_rejects_execution_error_without_output(tmp_path):
    ledger = write_ledger(tmp_path / "execution-error.jsonl", "one")
    result = ledger.entries[-1]
    identity = {
        key: result.payload[key]
        for key in ("request_id", "execution_id", "attempt_id")
    }

    with pytest.raises(ValueError, match="verified replay output"):
        snapshot_from_ledger(
            replace_entry_payload(
                ledger.entries, len(ledger.entries) - 1,
                {**identity, "status": "execution_error"},
            )
        )


@pytest.mark.parametrize("digest", ["not-a-sha", "A" * 64])
def test_snapshot_rejects_non_sha_artifact_values(tmp_path, digest):
    ledger = write_ledger(tmp_path / "invalid-artifact.jsonl", "one")
    entries = list(ledger.entries)
    artifact = entries[2]
    replacement = EvidenceEntry.create(
        artifact.sequence,
        artifact.kind,
        {**artifact.payload, "source_sha256": {"kernel.py": digest}},
        artifact.previous_sha256,
    )
    rebuilt = entries[:2] + [replacement]
    for entry in entries[3:]:
        rebuilt.append(
            EvidenceEntry.create(
                entry.sequence, entry.kind, entry.payload, rebuilt[-1].entry_sha256
            )
        )

    with pytest.raises(ValueError, match="non-empty source_sha256 mapping"):
        snapshot_from_ledger(rebuilt)


@pytest.mark.parametrize("value", ["zero", True, -1.0, None])
def test_snapshot_rejects_invalid_max_error_evidence(tmp_path, value):
    ledger = write_ledger(tmp_path / "invalid-error.jsonl", "one")
    entries = list(ledger.entries)
    result = entries[-1]
    replacement = EvidenceEntry.create(
        result.sequence,
        result.kind,
        {**result.payload, "max_abs_error": value},
        result.previous_sha256,
    )

    with pytest.raises(ValueError, match="verified replay output and metrics"):
        snapshot_from_ledger([*entries[:-1], replacement])


@pytest.mark.parametrize(
    "changes",
    [
        {"passed": True, "exit_code": 1},
        {
            "passed": True,
            "max_abs_error": None,
            "max_abs_error_status": "non-finite",
        },
    ],
)
def test_snapshot_rejects_contradictory_pass_outcomes(tmp_path, changes):
    ledger = write_ledger(tmp_path / "contradictory-result.jsonl", "one")
    result = ledger.entries[-1]

    with pytest.raises(ValueError, match="contradictory outcome"):
        snapshot_from_ledger(
            replace_entry_payload(
                ledger.entries,
                len(ledger.entries) - 1,
                {**result.payload, **changes},
            )
        )


def test_snapshot_rejects_mixed_execution_ids(tmp_path):
    ledger = write_ledger(tmp_path / "mixed-execution.jsonl", "one")
    entries = list(ledger.entries)
    action = entries[1]
    replacement = EvidenceEntry.create(
        action.sequence,
        action.kind,
        {**action.payload, "execution_id": "execution-2"},
        action.previous_sha256,
    )
    rebuilt = [entries[0], replacement]
    for entry in entries[2:]:
        rebuilt.append(
            EvidenceEntry.create(
                entry.sequence, entry.kind, entry.payload, rebuilt[-1].entry_sha256
            )
        )

    with pytest.raises(ValueError, match="one execution_id"):
        snapshot_from_ledger(rebuilt)


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


def test_snapshot_rejects_artifact_generation_outside_registered_fixture(tmp_path):
    path = tmp_path / "one.jsonl"
    ledger = write_ledger(path, "one")
    identity = ledger.entries[0].payload
    ledger.append(
        EvidenceKind.ARTIFACT,
            {
                "request_id": identity["request_id"],
                "execution_id": identity["execution_id"],
                "attempt_id": "one",
            "source_sha256": {"kernel.py": SOURCE_TWO},
            "artifact_sha256": {"source_bundle": ARTIFACT_TWO},
        },
    )

    with pytest.raises(ValueError, match="registered fixture"):
        snapshot_from_ledger(ledger.entries)


def test_snapshot_rejects_consistent_but_fabricated_execution_id(tmp_path):
    ledger = write_ledger(tmp_path / "fabricated-execution.jsonl", "one")
    entries = ledger.entries
    for index, entry in enumerate(tuple(entries)):
        entries = replace_entry_payload(
            entries, index, {**entries[index].payload, "execution_id": "execution-1"}
        )

    with pytest.raises(ValueError, match="reconstructed plan"):
        snapshot_from_ledger(entries)


@pytest.mark.parametrize("mutation", ["missing", "exit-code", "digest"])
def test_snapshot_rejects_invalid_execution_evidence(tmp_path, mutation):
    ledger = write_ledger(tmp_path / f"evidence-{mutation}.jsonl", "one")
    result = ledger.entries[-1]
    payload = dict(result.payload)
    if mutation == "missing":
        payload.pop("execution_evidence")
    elif mutation == "exit-code":
        payload["execution_evidence"] = {
            **payload["execution_evidence"], "result_exit_code": 1
        }
    else:
        payload["evidence_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="execution evidence|evidence digest"):
        snapshot_from_ledger(
            replace_entry_payload(
                ledger.entries, len(ledger.entries) - 1, payload
            )
        )
