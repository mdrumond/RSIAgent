from dataclasses import replace
import hashlib
import json

import pytest

import run_a5_phase1

from benchmarks.a5kernels.catlass_harness import HarnessResult
from benchmarks.a5kernels.phase1_experiments import (
    PINNED_CATLASS_REVISION,
    ProgrammingGuideIdentity,
    build_cells,
)
from benchmarks.a5kernels.phase1_live import (
    BACKOFF_SECONDS,
    InfrastructureFailure,
    Phase1GateRunner,
    QualificationBlocked,
    dry_run_manifest,
    load_gate_report,
    phase1_manifest,
    qualification_agents,
    qualification_tasks,
    require_gate,
    write_gate_report,
)
from benchmarks.a5kernels.protocol import canonical_hash


GUIDE = ProgrammingGuideIdentity(
    "catlass-dsl-programming-guide-v1",
    "a" * 64,
    "b" * 40,
    PINNED_CATLASS_REVISION,
)


def result(contract, source, attempt_id, *, verdict="PASS", code=0):
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    fields = dict(
        schema="a5-catlass-harness-result-v1",
        contract=contract.value,
        attempt_id=attempt_id,
        source_sha256=source_sha,
        source_fingerprint="c" * 64,
        catlass_revision=PINNED_CATLASS_REVISION,
        execution_profile="bz-a5",
        device=0,
        session_handle=f"bz-a5:{attempt_id}",
        host_verdict=verdict,
        exit_code=code,
        max_abs_error=0.0 if verdict == "PASS" else 1.0,
        mismatch_index=None if verdict == "PASS" else 0,
        mismatch_expected=None if verdict == "PASS" else 1.0,
        mismatch_actual=None if verdict == "PASS" else 0.0,
        output_sha256="d" * 64,
        retained_evidence_sha256="e" * 64,
    )
    return HarnessResult(**fields, record_sha256=canonical_hash(fields))


class PassingHarness:
    def __init__(self, outcome=None):
        self.calls = []
        self.outcome = outcome

    def run(self, source, contract, attempt_id):
        self.calls.append((source, contract, attempt_id))
        if self.outcome is not None:
            value = self.outcome(source, contract, attempt_id, len(self.calls))
            if value is not None:
                return value
        return result(contract, source, attempt_id)


def provider(agent, contract, revision, feedback):
    return f"# {agent.agent_id} {contract.value} revision {revision}\n"


def test_registry_has_six_independent_agents_and_eighteen_contract_tasks():
    agents = qualification_agents()
    tasks = qualification_tasks()

    assert len(agents) == 6
    assert len({agent.agent_id for agent in agents}) == 6
    assert {agent.model.provider for agent in agents} == {"OpenAI", "DeepSeek"}
    assert {agent.ordinal for agent in agents} == {1, 2, 3}
    assert len(tasks) == 18
    assert len({task.task_id for task in tasks}) == 18
    assert {task.agent for task in tasks} == set(agents)
    assert {task.contract.value for task in tasks} == {
        "padded-simd", "multiblock-simt", "cube-matmul",
    }


def test_dry_run_binds_one_guide_to_gate_and_all_eight_cells():
    manifest = dry_run_manifest(GUIDE)
    encoded = json.dumps(manifest, sort_keys=True)

    assert manifest["qualification"]["task_count"] == 18
    assert manifest["qualification"]["required_host_passes"] == 2
    assert manifest["qualification"]["max_semantic_revisions"] == 3
    assert manifest["qualification"]["infrastructure_retry"] == {
        "max_retries": 3, "backoff_seconds": 120,
    }
    assert len(manifest["phase1"]["cells"]) == 8
    assert {
        cell["guide"]["guide_sha256"]
        for cell in manifest["phase1"]["cells"]
    } == {GUIDE.guide_sha256}
    assert "guide_path" not in encoded
    assert "/home/" not in encoded
    assert "secret" not in encoded.lower()


def test_complete_gate_runs_every_contract_twice_and_admits_phase1():
    harness = PassingHarness()
    report = Phase1GateRunner(GUIDE, harness, provider, sleep=lambda _: None).run()

    assert report.ready is True
    assert len(report.records) == 18
    assert len(harness.calls) == 36
    assert all(len(record.revisions) == 1 for record in report.records)
    assert all(
        [item.host_verdict for item in record.revisions[0].host_results]
        == ["PASS", "PASS"]
        for record in report.records
    )
    require_gate(report, GUIDE)
    manifest = phase1_manifest(GUIDE, report)
    assert manifest["status"] == "ready"
    assert len(manifest["pending_cells"]) == 8


def test_semantic_failure_gets_at_most_three_source_revisions():
    def always_fail(source, contract, attempt_id, _count):
        return result(contract, source, attempt_id, verdict="FAIL", code=1)

    harness = PassingHarness(always_fail)
    with pytest.raises(QualificationBlocked, match="semantic revisions") as caught:
        Phase1GateRunner(GUIDE, harness, provider, sleep=lambda _: None).run()

    assert caught.value.task.task_id == qualification_tasks()[0].task_id
    assert len(caught.value.revisions) == 3
    assert [entry.revision for entry in caught.value.revisions] == [1, 2, 3]
    assert len(harness.calls) == 6


def test_infrastructure_retries_do_not_consume_semantic_revision():
    sleeps = []
    source_calls = 0

    def flaky_provider(agent, contract, revision, feedback):
        nonlocal source_calls
        source_calls += 1
        if source_calls <= 3:
            raise InfrastructureFailure(f"transport {source_calls}")
        assert revision == 1
        return provider(agent, contract, revision, feedback)

    report = Phase1GateRunner(
        GUIDE, PassingHarness(), flaky_provider, sleep=sleeps.append,
    ).run()

    first = report.records[0].revisions[0]
    assert first.revision == 1
    assert first.source_infrastructure_retries == 3
    assert sleeps == [BACKOFF_SECONDS] * 3


def test_fourth_infrastructure_failure_blocks_without_semantic_result():
    sleeps = []

    def unavailable(*_args):
        raise InfrastructureFailure("provider unavailable")

    with pytest.raises(InfrastructureFailure, match="provider unavailable"):
        Phase1GateRunner(
            GUIDE, PassingHarness(), unavailable, sleep=sleeps.append,
        ).run()
    assert sleeps == [BACKOFF_SECONDS] * 3


def test_gate_validation_rejects_missing_failed_or_drifted_evidence():
    report = Phase1GateRunner(
        GUIDE, PassingHarness(), provider, sleep=lambda _: None,
    ).run()

    with pytest.raises(ValueError, match="exactly 18"):
        require_gate(replace(report, records=report.records[:-1]), GUIDE)
    bad_revision = replace(
        report.records[0].revisions[0].host_results[0],
        catlass_revision="0" * 40,
    )
    bad_attempt = replace(
        report.records[0].revisions[0],
        host_results=(bad_revision, report.records[0].revisions[0].host_results[1]),
    )
    bad_record = replace(
        report.records[0], revisions=(bad_attempt,), admitted=True,
    )
    with pytest.raises(ValueError, match="Catlass revision"):
        require_gate(replace(report, records=(bad_record, *report.records[1:])), GUIDE)

    other_guide = replace(GUIDE, guide_sha256="9" * 64)
    with pytest.raises(ValueError, match="guide"):
        require_gate(report, other_guide)


def test_gate_report_round_trip_is_digest_checked_and_path_free(tmp_path):
    report = Phase1GateRunner(
        GUIDE, PassingHarness(), provider, sleep=lambda _: None,
    ).run()
    path = tmp_path / "private" / "qualification.json"
    write_gate_report(report, path)
    loaded = load_gate_report(path)

    assert loaded == report
    assert str(tmp_path) not in path.read_text()
    require_gate(loaded, GUIDE)

    value = json.loads(path.read_text())
    value["ready"] = False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="digest"):
        load_gate_report(path)

    write_gate_report(report, path)
    value = json.loads(path.read_text())
    value["guide_path"] = "/private/guide.md"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="load"):
        load_gate_report(path)


def test_phase1_manifest_fails_closed_without_gate():
    with pytest.raises(ValueError, match="qualification gate"):
        phase1_manifest(GUIDE, None)


def test_phase1_manifest_resume_excludes_only_registered_completed_cells():
    report = Phase1GateRunner(
        GUIDE, PassingHarness(), provider, sleep=lambda _: None,
    ).run()
    cells = build_cells(GUIDE)
    manifest = phase1_manifest(GUIDE, report, completed_cell_ids=[cells[0].cell_id])

    assert manifest["status"] == "ready"
    assert manifest["completed_cells"] == [cells[0].cell_id]
    assert len(manifest["pending_cells"]) == 7
    with pytest.raises(ValueError, match="registered"):
        phase1_manifest(GUIDE, report, completed_cell_ids=["a5-cell-unknown"])


def test_cli_plan_and_report_need_neither_credentials_nor_npu(tmp_path, capsys):
    assert run_a5_phase1.main([
        "plan", "--guide-sha256", GUIDE.guide_sha256,
        "--cpl-skills-revision", GUIDE.cpl_skills_revision,
    ]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["qualification"]["task_count"] == 18

    report = Phase1GateRunner(
        GUIDE, PassingHarness(), provider, sleep=lambda _: None,
    ).run()
    state = tmp_path / "gate.json"
    write_gate_report(report, state)
    assert run_a5_phase1.main(["report", "--gate", str(state)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "admitted": 18,
        "guide_sha256": GUIDE.guide_sha256,
        "qualification_sha256": report.report_sha256,
        "ready": True,
        "schema": report.schema,
    }


def test_cli_run_and_resume_are_gate_guarded(tmp_path, capsys):
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="load"):
        run_a5_phase1.main(["run", "--gate", str(missing)])

    report = Phase1GateRunner(
        GUIDE, PassingHarness(), provider, sleep=lambda _: None,
    ).run()
    state = tmp_path / "gate.json"
    write_gate_report(report, state)
    first = build_cells(GUIDE)[0].cell_id
    assert run_a5_phase1.main([
        "resume", "--gate", str(state), "--completed-cell", first,
    ]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["completed_cells"] == [first]
    assert len(manifest["pending_cells"]) == 7
