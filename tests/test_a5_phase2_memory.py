from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from benchmarks.a5kernels.evidence import canonical_bytes, canonical_digest
from benchmarks.a5kernels.phase1_composition import CellDisposition, Phase1CellComposition
from benchmarks.a5kernels.phase1_experiments import ProfilingGuidance
from benchmarks.a5kernels.phase1_registry import DEFAULT_PROPOSALS
from benchmarks.a5kernels.phase2_attention import SUITE_SHA256
from benchmarks.a5kernels.phase2_memory import (
    CELL_SCHEMA_V2,
    SNAPSHOT_SCHEMA,
    A5Phase1Snapshot,
    admit_phase1_snapshot,
    phase2_cell_pairs,
    phase2_cells,
)
from tests.test_a5_phase1_composition import _dependencies, _guide_and_report


def _completed_source(tmp_path):
    guide_path, guide, report = _guide_and_report(tmp_path)
    selected = None

    def only_one(cell):
        nonlocal selected
        if selected is None:
            selected = cell
        return (
            CellDisposition.runnable()
            if cell == selected
            else CellDisposition.pending("test isolation")
        )

    composition = Phase1CellComposition(
        tmp_path / "phase1",
        guide_path,
        guide,
        report,
        _dependencies(tmp_path / "deps", []),
        cell_disposition=only_one,
        sleeper=lambda _seconds: None,
    )
    records = composition.run()
    assert len(records) == 1
    assert selected is not None
    destination = dict(phase2_cell_pairs(guide, report.report_sha256))[selected]
    return composition, selected, destination


def _snapshot(tmp_path):
    composition, source, destination = _completed_source(tmp_path)
    snapshot = admit_phase1_snapshot(composition, destination, tmp_path / "phase2")
    return composition, source, destination, snapshot


def test_phase2_registry_has_exactly_eight_runnable_a5_treatments(tmp_path):
    _path, guide, report = _guide_and_report(tmp_path)

    cells = phase2_cells(guide, report.report_sha256)
    pairs = phase2_cell_pairs(guide, report.report_sha256)

    assert len(cells) == len(pairs) == 8
    assert len({cell.cell_id for cell in cells}) == 8
    assert tuple(destination for _source, destination in pairs) == cells
    assert all(source.runnable for source, _destination in pairs)
    assert all(
        source.profiling is not ProfilingGuidance.NEW
        for source, _destination in pairs
    )
    assert {cell.source_cell_id for cell in cells} == {
        source.cell_id for source, _destination in pairs
    }
    assert {cell.suite_sha256 for cell in cells} == {SUITE_SHA256}
    assert {cell.qualification_sha256 for cell in cells} == {
        report.report_sha256
    }
    assert {cell.guide for cell in cells} == {guide}
    assert {cell.catlass_revision for cell in cells} == {
        guide.catlass_revision
    }
    assert {cell.target.value for cell in cells} == {"a5"}
    assert {cell.language.value for cell in cells} == {"catlass-dsl"}
    assert all(cell.as_dict()["cell_id"] == cell.cell_id for cell in cells)


def test_same_guide_preserves_legacy_v1_identity_exactly(tmp_path):
    _path, guide, report = _guide_and_report(tmp_path)

    implicit = phase2_cells(guide, report.report_sha256)
    explicit = phase2_cells(guide, report.report_sha256, destination_guide=guide)

    assert explicit == implicit
    assert implicit[0].cell_id == "a5-phase2-cell-4bf906a8bc06c72c"
    assert implicit[0].schema == "a5-catlass-phase2-cell-v1"
    assert "source_guide" not in implicit[0].as_dict()


def test_different_guides_have_deterministic_dual_guide_identity(tmp_path):
    _path, source_guide, report = _guide_and_report(tmp_path)
    destination_guide = replace(source_guide, guide_sha256="a" * 64)

    first = phase2_cells(
        source_guide,
        report.report_sha256,
        destination_guide=destination_guide,
    )
    second = phase2_cells(
        source_guide,
        report.report_sha256,
        destination_guide=destination_guide,
    )

    assert first == second
    assert [cell.cell_id for cell in first] != [
        cell.cell_id for cell in phase2_cells(source_guide, report.report_sha256)
    ]
    assert {cell.schema for cell in first} == {CELL_SCHEMA_V2}
    assert {cell.source_guide for cell in first} == {source_guide}
    assert {cell.guide for cell in first} == {destination_guide}
    assert all(
        cell.as_dict()["source_guide"] == source_guide.as_dict() for cell in first
    )
    assert all(cell.as_dict()["guide"] == destination_guide.as_dict() for cell in first)


def test_dual_guide_identity_rejects_tampering(tmp_path):
    _path, source_guide, report = _guide_and_report(tmp_path)
    destination_guide = replace(source_guide, guide_sha256="a" * 64)
    cell = phase2_cells(
        source_guide,
        report.report_sha256,
        destination_guide=destination_guide,
    )[0]

    with pytest.raises(ValueError, match="ID|identity"):
        replace(cell, guide=replace(destination_guide, guide_sha256="c" * 64))
    with pytest.raises(ValueError, match="source|guide|schema"):
        replace(cell, source_guide=destination_guide)


def test_phase2_identity_changes_with_qualification_and_rejects_bad_digest(tmp_path):
    _path, guide, report = _guide_and_report(tmp_path)
    first = phase2_cells(guide, report.report_sha256)
    second = phase2_cells(guide, "f" * 64)

    assert [cell.source_cell_id for cell in first] == [
        cell.source_cell_id for cell in second
    ]
    assert [cell.cell_id for cell in first] != [cell.cell_id for cell in second]
    with pytest.raises(ValueError, match="qualification"):
        phase2_cells(guide, "not-a-digest")


def test_admission_byte_copies_complete_memory_and_terminal(tmp_path):
    composition, source, destination, snapshot = _snapshot(tmp_path)
    source_root = composition.root / "cells" / source.cell_id

    assert snapshot.root == (
        tmp_path / "phase2" / "cells" / destination.cell_id / "snapshot"
    ).resolve()
    assert snapshot.memory_path.read_bytes() == (source_root / "memory.jsonl").read_bytes()
    assert snapshot.terminal_path.read_bytes() == (source_root / "terminal.json").read_bytes()
    assert snapshot.manifest_path.is_file()
    assert snapshot.snapshot_id == snapshot.manifest["snapshot_id"]
    assert snapshot.manifest["schema"] == SNAPSHOT_SCHEMA
    assert snapshot.manifest["phase1_memory_entries"] == len(DEFAULT_PROPOSALS)
    assert snapshot.manifest["phase1_memory_head_sha256"] == json.loads(
        snapshot.memory_path.read_bytes().splitlines()[-1]
    )["entry_sha256"]
    assert snapshot.manifest["phase1_evidence_head_sha256"] == json.loads(
        snapshot.terminal_path.read_text()
    )["evidence_head_sha256"]
    assert snapshot.manifest["destination_identity"] == destination.as_dict()
    assert A5Phase1Snapshot.open(snapshot.root, destination) == snapshot

    # Idempotent admission is allowed only when every immutable byte agrees.
    assert admit_phase1_snapshot(
        composition, destination, tmp_path / "phase2"
    ) == snapshot


@pytest.mark.parametrize("artifact", ["phase1-memory.jsonl", "phase1-terminal.json"])
def test_open_rejects_tampered_snapshot_artifacts(tmp_path, artifact):
    _composition, _source, destination, snapshot = _snapshot(tmp_path)
    path = snapshot.root / artifact
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="bytes changed|memory journal"):
        A5Phase1Snapshot.open(snapshot.root, destination)


def test_open_rejects_tampered_manifest_and_wrong_destination(tmp_path):
    composition, source, destination, snapshot = _snapshot(tmp_path)
    manifest = json.loads(snapshot.manifest_path.read_text())
    manifest["phase1_memory_entries"] = 7
    snapshot.manifest_path.write_bytes(canonical_bytes(manifest) + b"\n")

    with pytest.raises(ValueError, match="manifest"):
        A5Phase1Snapshot.open(snapshot.root, destination)

    snapshot.manifest_path.write_bytes(canonical_bytes(snapshot.manifest) + b"\n")
    alternatives = [
        cell
        for cell in phase2_cells(source.guide, composition.qualification.report_sha256)
        if cell != destination
    ]
    with pytest.raises(ValueError, match="identity"):
        A5Phase1Snapshot.open(snapshot.root, alternatives[0])


def test_open_rejects_rehashed_terminal_with_changed_source_identity(tmp_path):
    _composition, _source, destination, snapshot = _snapshot(tmp_path)
    terminal = json.loads(snapshot.terminal_path.read_text())
    terminal["cell_id"] = "a5-cell-" + "0" * 16
    body = {key: value for key, value in terminal.items() if key != "record_sha256"}
    terminal["record_sha256"] = canonical_digest(body)
    terminal_bytes = canonical_bytes(terminal) + b"\n"
    snapshot.terminal_path.write_bytes(terminal_bytes)
    manifest = dict(snapshot.manifest)
    manifest["phase1_terminal_sha256"] = hashlib.sha256(terminal_bytes).hexdigest()
    manifest_body = {
        key: value for key, value in manifest.items() if key != "snapshot_id"
    }
    manifest["snapshot_id"] = canonical_digest(manifest_body)
    snapshot.manifest_path.write_bytes(canonical_bytes(manifest) + b"\n")

    with pytest.raises(ValueError, match="terminal identity"):
        A5Phase1Snapshot.open(snapshot.root, destination)


def test_snapshot_uses_source_guide_while_cell_records_destination_guide(tmp_path):
    composition, source, legacy_destination = _completed_source(tmp_path)
    source_guide = source.guide
    destination_guide = replace(source_guide, guide_sha256="a" * 64)
    destination = next(
        cell
        for cell in phase2_cells(
            source_guide,
            composition.qualification.report_sha256,
            destination_guide=destination_guide,
        )
        if cell.source_cell_id == legacy_destination.source_cell_id
    )

    snapshot = admit_phase1_snapshot(composition, destination, tmp_path / "phase2")

    assert destination.source_guide == source_guide
    assert destination.guide == destination_guide
    assert json.loads(snapshot.terminal_path.read_text())["guide_sha256"] == (
        source_guide.guide_sha256
    )
    assert A5Phase1Snapshot.open(snapshot.root, destination) == snapshot

    terminal = json.loads(snapshot.terminal_path.read_text())
    terminal["guide_sha256"] = destination_guide.guide_sha256
    terminal_body = {
        key: value for key, value in terminal.items() if key != "record_sha256"
    }
    terminal["record_sha256"] = canonical_digest(terminal_body)
    terminal_bytes = canonical_bytes(terminal) + b"\n"
    snapshot.terminal_path.write_bytes(terminal_bytes)
    manifest = dict(snapshot.manifest)
    manifest["phase1_terminal_sha256"] = hashlib.sha256(terminal_bytes).hexdigest()
    manifest_body = {
        key: value for key, value in manifest.items() if key != "snapshot_id"
    }
    manifest["snapshot_id"] = canonical_digest(manifest_body)
    snapshot.manifest_path.write_bytes(canonical_bytes(manifest) + b"\n")

    with pytest.raises(ValueError, match="terminal identity"):
        A5Phase1Snapshot.open(snapshot.root, destination)


def test_admission_rejects_incomplete_phase1_before_publication(tmp_path):
    guide_path, guide, report = _guide_and_report(tmp_path)
    composition = Phase1CellComposition.one_project_smoke(
        tmp_path / "phase1",
        guide_path,
        guide,
        report,
        _dependencies(tmp_path / "deps", []),
        sleeper=lambda _seconds: None,
    )
    composition.run()
    source, destination = phase2_cell_pairs(guide, report.report_sha256)[0]

    with pytest.raises(ValueError, match="eight|complete"):
        admit_phase1_snapshot(composition, destination, tmp_path / "phase2")
    assert not (tmp_path / "phase2").exists()


def test_admission_rejects_source_evidence_or_treatment_mismatch(tmp_path):
    composition, source, destination = _completed_source(tmp_path)
    wrong_qualification = next(
        cell
        for cell in phase2_cells(source.guide, "f" * 64)
        if cell.source_cell_id == source.cell_id
    )
    with pytest.raises(ValueError, match="destination identity"):
        admit_phase1_snapshot(
            composition, wrong_qualification, tmp_path / "wrong-qualification"
        )

    evidence = composition.root / "cells" / source.cell_id / "evidence.jsonl"
    evidence.write_bytes(evidence.read_bytes() + b"{}\n")

    with pytest.raises(ValueError, match="evidence|chain"):
        admit_phase1_snapshot(composition, destination, tmp_path / "phase2")

    alternatives = [
        cell
        for cell in phase2_cells(source.guide, composition.qualification.report_sha256)
        if cell.source_cell_id != source.cell_id
    ]
    with pytest.raises(ValueError, match="source|terminal"):
        admit_phase1_snapshot(composition, alternatives[0], tmp_path / "other")


def test_admission_rejects_conflicting_or_partial_snapshot(tmp_path):
    composition, _source, destination = _completed_source(tmp_path)
    root = tmp_path / "phase2" / "cells" / destination.cell_id / "snapshot"
    root.mkdir(parents=True)
    (root / "phase1-memory.jsonl").write_text("conflict\n", encoding="utf-8")

    with pytest.raises(ValueError, match="partial|conflict|manifest"):
        admit_phase1_snapshot(composition, destination, tmp_path / "phase2")


def test_manifest_snapshot_id_binds_all_identity_and_source_digests(tmp_path):
    _composition, _source, _destination, snapshot = _snapshot(tmp_path)
    body = {
        key: value
        for key, value in snapshot.manifest.items()
        if key != "snapshot_id"
    }
    assert snapshot.snapshot_id == canonical_digest(body)
