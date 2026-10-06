import hashlib
import json

import pytest

from benchmarks.a3_experiments import ProgrammingLevel
from benchmarks.a3kernels.live_composition import AuthoritativeResultStore
from benchmarks.a3kernels.phase1_evidence import EvidenceKind, EvidenceLedger, canonical_digest
from benchmarks.a3kernels.phase1_memory import HostFact, Phase1LearningJournal, ProjectMemory
from benchmarks.a3kernels.phase1_registry import DEFAULT_PROPOSALS
from benchmarks.a3kernels.phase1_wave import Phase1Wave, foundation_cells
from benchmarks.a3kernels.phase2_memory import (
    Phase2Learning,
    Phase2LearningJournal,
    admit_phase1_snapshot,
    phase2_cell_pairs,
)
from benchmarks.a3kernels.trial import trial_protocol_sha256


def _phase1_cell(tmp_path, *, execution_profile="bz-a3-1"):
    cell = foundation_cells()[0]
    root = tmp_path / "phase1-cell"
    root.mkdir()
    evidence = EvidenceLedger(root / "evidence.jsonl")
    final = evidence.append(EvidenceKind.RESULT, {
        "target": "Ascend910B4", "language": "ascend-c",
        "result": "synthetic-complete-phase1",
    })
    authority = AuthoritativeResultStore(root / "authority.jsonl")
    journal = Phase1LearningJournal(
        root / "memory.jsonl", DEFAULT_PROPOSALS,
        cell_id=cell.cell_id, lineage_id=f"phase1-{cell.cell_id}",
        evidence_resolver=authority,
        trial_protocol_sha256=trial_protocol_sha256(),
    )
    for ordinal, proposal in enumerate(DEFAULT_PROPOSALS, 1):
        fact = f"{ordinal:064x}"
        source = f"{ordinal + 20:064x}"
        authority.register("host-verification", fact, proposal.project_id, source, True)
        journal.append(ProjectMemory(
            proposal, proposal.project_id, cell.cell_id, f"phase1-{cell.cell_id}",
            source, (f"complete project {ordinal}",),
            (HostFact("host-verification", f"host pass {ordinal}", fact),),
        ))
    terminal = Phase1Wave._terminal(cell, {
        "status": "passed", "terminal_reason": "completed",
        "completed_projects": 8, "failed_project_id": None,
        "evidence_sha256": final.entry_sha256,
        "infrastructure_retries_used": 0,
        "infrastructure_retry_evidence_sha256": canonical_digest([]),
    }, execution_profile)
    (root / "terminal.json").write_text(
        json.dumps(terminal, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return cell, root


def test_phase2_mapping_preserves_treatment_and_changes_only_level():
    pairs = phase2_cell_pairs()
    assert len(pairs) == 8
    for source, destination in pairs:
        assert source.programming_level is ProgrammingLevel.FOUNDATION
        assert destination.programming_level is ProgrammingLevel.TILED
        assert (
            source.target, source.language, source.backend_model,
            source.knowledge, source.profiling,
        ) == (
            destination.target, destination.language, destination.backend_model,
            destination.knowledge, destination.profiling,
        )


def test_admission_copies_exact_complete_phase1_and_appends_separate_learning(tmp_path):
    source_cell, source = _phase1_cell(tmp_path)
    original = (source / "memory.jsonl").read_bytes()
    snapshot = admit_phase1_snapshot(
        source, tmp_path / "phase2",
    )
    manifest = snapshot.manifest

    assert snapshot.memory_path.read_bytes() == original
    assert manifest["phase1_memory_sha256"] == hashlib.sha256(original).hexdigest()
    assert manifest["phase1_memory_entries"] == 8
    assert manifest["phase1_memory_head_sha256"] == json.loads(
        original.splitlines()[-1]
    )["entry_sha256"]
    assert manifest["phase1_root_sha256"] == canonical_digest({
        "schema": "a3-phase2-phase1-root-v1",
        "authority_sha256": hashlib.sha256((source / "authority.jsonl").read_bytes()).hexdigest(),
        "evidence_sha256": hashlib.sha256((source / "evidence.jsonl").read_bytes()).hexdigest(),
        "memory_sha256": hashlib.sha256((source / "memory.jsonl").read_bytes()).hexdigest(),
        "terminal_sha256": hashlib.sha256((source / "terminal.json").read_bytes()).hexdigest(),
    })
    assert manifest["source_cell_id"] == source_cell.cell_id
    assert manifest["destination_cell_id"] != source_cell.cell_id
    assert manifest["trial_protocol_sha256"] == trial_protocol_sha256()

    phase2_authority = AuthoritativeResultStore(tmp_path / "phase2-authority.jsonl")
    phase2_authority.register(
        "host-verification", "a" * 64, DEFAULT_PROPOSALS[0].project_id,
        "d" * 64, False,
    )
    learning = Phase2LearningJournal(
        tmp_path / "phase2" / "learning.jsonl", snapshot, phase2_authority,
    )
    first = learning.append(Phase2Learning(
        "target-attempt", "held-out target failed at tile tail",
        DEFAULT_PROPOSALS[0].project_id, "d" * 64,
        (HostFact("host-verification", "host mismatch", "a" * 64, success=False),),
    ))
    reopened = Phase2LearningJournal(learning.path, snapshot, phase2_authority)
    assert reopened.read() == (first,)
    assert snapshot.memory_path.read_bytes() == original
    assert reopened.context()["phase1"]["entries"] == 8
    assert reopened.context()["phase2"]["entries"] == 1


@pytest.mark.parametrize("terminal_change", [
    {"status": "failed"}, {"completed_projects": 7},
    {"trial_protocol_sha256": "f" * 64},
])
def test_admission_rejects_incomplete_or_conflicting_terminal(tmp_path, terminal_change):
    _cell, source = _phase1_cell(tmp_path)
    terminal = json.loads((source / "terminal.json").read_text())
    terminal.update(terminal_change)
    (source / "terminal.json").write_text(json.dumps(terminal) + "\n")
    with pytest.raises(ValueError, match="terminal|protocol"):
        admit_phase1_snapshot(source, tmp_path / "phase2")


def test_admission_rejects_memory_tampering(tmp_path):
    _cell, source = _phase1_cell(tmp_path)
    lines = (source / "memory.jsonl").read_text().splitlines()
    changed = json.loads(lines[3]); changed["memory"]["actions"] = ["curated"]
    lines[3] = json.dumps(changed)
    (source / "memory.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="journal|resume"):
        admit_phase1_snapshot(source, tmp_path / "phase2")


def test_admission_rejects_non_authoritative_gz_a3_phase1(tmp_path):
    _cell, source = _phase1_cell(tmp_path, execution_profile="gz-a3")
    with pytest.raises(ValueError, match="authoritative BZ-A3"):
        admit_phase1_snapshot(source, tmp_path / "phase2")


def test_snapshot_and_phase2_journal_fail_if_phase1_bytes_are_rewritten(tmp_path):
    _cell, source = _phase1_cell(tmp_path)
    snapshot = admit_phase1_snapshot(source, tmp_path / "phase2")
    authority = AuthoritativeResultStore(tmp_path / "phase2-authority.jsonl")
    authority.register(
        "runtime", "b" * 64, DEFAULT_PROPOSALS[0].project_id, "d" * 64, True,
    )
    learning = Phase2LearningJournal(
        tmp_path / "phase2" / "learning.jsonl", snapshot, authority,
    )
    value = Phase2Learning(
        "practice", "learned a tail rule", DEFAULT_PROPOSALS[0].project_id,
        "d" * 64, (HostFact("runtime", "host run passed", "b" * 64),),
    )
    learning.append(value)
    snapshot.memory_path.write_bytes(snapshot.memory_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="immutable"):
        learning.read()
    with pytest.raises(ValueError, match="immutable"):
        learning.append(value)


def test_existing_snapshot_is_idempotent_but_conflicts_fail_closed(tmp_path):
    _cell, source = _phase1_cell(tmp_path)
    destination = tmp_path / "phase2"
    first = admit_phase1_snapshot(source, destination)
    second = admit_phase1_snapshot(source, destination)
    assert second.manifest == first.manifest

    (destination / "phase1-terminal.json").write_text("{}\n")
    with pytest.raises(ValueError, match="immutable|conflict"):
        admit_phase1_snapshot(source, destination)


def test_phase2_learning_requires_nonempty_resolved_host_evidence(tmp_path):
    _cell, source = _phase1_cell(tmp_path)
    snapshot = admit_phase1_snapshot(source, tmp_path / "phase2")
    authority = AuthoritativeResultStore(tmp_path / "phase2-authority.jsonl")
    project = DEFAULT_PROPOSALS[0].project_id
    with pytest.raises(ValueError, match="host fact"):
        Phase2Learning("practice", "unsupported", project, "d" * 64, ())

    learning = Phase2LearningJournal(
        tmp_path / "phase2" / "learning.jsonl", snapshot, authority,
    )
    unresolved = Phase2Learning(
        "practice", "unsupported", project, "d" * 64,
        (HostFact("runtime", "claimed host result", "c" * 64),),
    )
    with pytest.raises(ValueError, match="authoritative"):
        learning.append(unresolved)
    assert not learning.path.exists()


def test_phase2_learning_binds_project_candidate_kind_digest_and_success(tmp_path):
    _cell, source = _phase1_cell(tmp_path)
    snapshot = admit_phase1_snapshot(source, tmp_path / "phase2")
    project = DEFAULT_PROPOSALS[0].project_id
    fact = HostFact("runtime", "host result", "c" * 64)
    value = Phase2Learning("practice", "lesson", project, "d" * 64, (fact,))

    for field, changed in (
        ("kind", "compile"), ("project_id", "e" * 64),
        ("candidate_sha256", "e" * 64), ("success", False),
    ):
        kwargs = dict(
            kind="runtime", evidence_sha256="c" * 64, project_id=project,
            candidate_sha256="d" * 64, success=True,
        )
        kwargs[field] = changed
        authority = AuthoritativeResultStore(tmp_path / field / "authority.jsonl")
        authority.register(**kwargs)
        journal = Phase2LearningJournal(tmp_path / field / "learning.jsonl", snapshot, authority)
        with pytest.raises(ValueError, match=field.replace("_id", "").replace("_sha256", "")):
            journal.append(value)
