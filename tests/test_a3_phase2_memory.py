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


ROOT_ID = "1" * 64


def _phase1_cell(tmp_path):
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
    }, "bz-a3-1")
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
        source, tmp_path / "phase2", phase1_root_sha256=ROOT_ID,
    )
    manifest = snapshot.manifest

    assert snapshot.memory_path.read_bytes() == original
    assert manifest["phase1_memory_sha256"] == hashlib.sha256(original).hexdigest()
    assert manifest["phase1_memory_entries"] == 8
    assert manifest["phase1_memory_head_sha256"] == json.loads(
        original.splitlines()[-1]
    )["entry_sha256"]
    assert manifest["phase1_root_sha256"] == ROOT_ID
    assert manifest["source_cell_id"] == source_cell.cell_id
    assert manifest["destination_cell_id"] != source_cell.cell_id
    assert manifest["trial_protocol_sha256"] == trial_protocol_sha256()

    learning = Phase2LearningJournal(tmp_path / "phase2" / "learning.jsonl", snapshot)
    first = learning.append(Phase2Learning(
        "target-attempt", "held-out target failed at tile tail", ("a" * 64,)
    ))
    reopened = Phase2LearningJournal(learning.path, snapshot)
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
        admit_phase1_snapshot(source, tmp_path / "phase2", phase1_root_sha256=ROOT_ID)


def test_admission_rejects_memory_tampering_and_wrong_root_identity(tmp_path):
    _cell, source = _phase1_cell(tmp_path)
    with pytest.raises(ValueError, match="root"):
        admit_phase1_snapshot(source, tmp_path / "bad", phase1_root_sha256="bad")

    lines = (source / "memory.jsonl").read_text().splitlines()
    changed = json.loads(lines[3]); changed["memory"]["actions"] = ["curated"]
    lines[3] = json.dumps(changed)
    (source / "memory.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="journal|resume"):
        admit_phase1_snapshot(source, tmp_path / "phase2", phase1_root_sha256=ROOT_ID)


def test_snapshot_and_phase2_journal_fail_if_phase1_bytes_are_rewritten(tmp_path):
    _cell, source = _phase1_cell(tmp_path)
    snapshot = admit_phase1_snapshot(source, tmp_path / "phase2", phase1_root_sha256=ROOT_ID)
    learning = Phase2LearningJournal(tmp_path / "phase2" / "learning.jsonl", snapshot)
    learning.append(Phase2Learning("practice", "learned a tail rule", ("b" * 64,)))
    snapshot.memory_path.write_bytes(snapshot.memory_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="immutable"):
        learning.read()
    with pytest.raises(ValueError, match="immutable"):
        learning.append(Phase2Learning("practice", "must not append", ("c" * 64,)))


def test_existing_snapshot_is_idempotent_but_conflicts_fail_closed(tmp_path):
    _cell, source = _phase1_cell(tmp_path)
    destination = tmp_path / "phase2"
    first = admit_phase1_snapshot(source, destination, phase1_root_sha256=ROOT_ID)
    second = admit_phase1_snapshot(source, destination, phase1_root_sha256=ROOT_ID)
    assert second.manifest == first.manifest

    (destination / "phase1-terminal.json").write_text("{}\n")
    with pytest.raises(ValueError, match="immutable|conflict"):
        admit_phase1_snapshot(source, destination, phase1_root_sha256=ROOT_ID)
