from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.a5kernels.evidence import canonical_bytes, canonical_digest
from benchmarks.a5kernels.phase1_experiments import A5BackendModel
from benchmarks.a5kernels.phase2_adaptive_memory import admit_adaptive_snapshot
from benchmarks.a5kernels.phase2_live import MAX_ATTEMPTS, Phase2LiveRunner
from benchmarks.a5kernels.phase2_memory import QUALIFICATION_SCHEMA, phase2_cells
from benchmarks.a5kernels.phase3_core import admit_phase2_lineage
from tests.test_a5_phase2_live import _Backend, _deps
from tests.test_a5_phase2_memory import _snapshot
from tests.test_a5_phase2_adaptive_memory import _source


def _write_qualification(root, cell, runtime_revision="b" * 40):
    provenance = [
        ["catlass_revision", runtime_revision],
        ["catlass_source", "/remote/catlass"],
        ["execution_profile", "bz-a5"],
        ["manifest_sha256", "a" * 64],
    ]
    case = {"case_id": "copy-v", "seed": 7}
    output = [1.0, 2.0]
    body = {
        "schema": QUALIFICATION_SCHEMA,
        "cell_id": cell.cell_id,
        "catlass_source": "/remote/catlass",
        "catlass_revision": cell.catlass_revision,
        "runtime_catlass_revision": runtime_revision,
        "runtime_provenance": provenance,
        "physical_device": 3,
        "case": case,
        "case_sha256": canonical_digest(case),
        "candidate_sha256": "c" * 64,
        "input_sha256": "d" * 64,
        "status": "transport-qualified",
        "exit_code": 0,
        "session_handle": "bz-a5:qualification",
        "output": output,
        "output_sha256": canonical_digest(output),
        "metadata": [["result_exit_code", "0"]],
    }
    value = {**body, "record_sha256": canonical_digest(body)}
    path = root / "cells" / cell.cell_id / "qualification.json"
    path.write_bytes(canonical_bytes(value) + b"\n")
    return path


def _phase2(tmp_path, outcomes=(True,), runtime_revision="b" * 40):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _composition, _source, _destination, snapshot = _snapshot(tmp_path)
    root = tmp_path / "phase2"
    runner = Phase2LiveRunner(
        phase2_root=root,
        snapshot=snapshot,
        dependencies=_deps(snapshot, _Backend(outcomes), []),
    )
    terminal = runner.run()
    _write_qualification(root, snapshot.destination, runtime_revision)
    return root, snapshot.destination, runner, terminal


def test_admission_freezes_only_authenticated_actor_visible_memory(tmp_path):
    root, cell, _runner, terminal = _phase2(tmp_path)
    assert terminal["status"] == "passed"

    frozen = admit_phase2_lineage(root, cell, tmp_path / "phase3")

    assert cell.model.backend_model is A5BackendModel.GPT_5_6_SOL
    assert frozen.manifest["source_cell"] == cell.as_dict()
    assert frozen.manifest["phase2_status"] == "passed"
    assert frozen.manifest["phase1_entries"] == 8
    assert frozen.manifest["phase2_entries"] == 1
    assert frozen.manifest["runtime_catlass_revision"] == "b" * 40
    assert frozen.manifest["runtime_catlass_revision"] != cell.catlass_revision
    assert dict(frozen.manifest["runtime_provenance"])["catlass_revision"] == (
        "b" * 40
    )
    qualification = json.loads(
        (root / "cells" / cell.cell_id / "qualification.json").read_text()
    )
    assert frozen.manifest["transport_qualification_sha256"] == (
        qualification["record_sha256"]
    )
    memory = frozen.actor_memory()
    assert memory["phase1"]
    assert memory["phase2"][0]["passed"] is True
    serialized = canonical_bytes(memory).decode()
    assert "candidate_source" not in serialized
    assert "host_target_grading" not in serialized
    assert str(root.resolve()) not in serialized
    with pytest.raises(TypeError):
        memory["phase1"][0]["new"] = "mutation"
    with pytest.raises(TypeError):
        frozen.manifest["new"] = "mutation"
    with pytest.raises(TypeError):
        frozen.manifest["runtime_provenance"][0][1] = "mutation"

    reopened = type(frozen).open(frozen.root, cell)
    assert reopened.snapshot_id == frozen.snapshot_id
    assert admit_phase2_lineage(root, cell, tmp_path / "phase3") == frozen


def test_legacy_frozen_bytes_remain_unchanged(tmp_path):
    root, cell, _runner, _terminal = _phase2(tmp_path)
    frozen = admit_phase2_lineage(root, cell, tmp_path / "phase3")
    assert hashlib.sha256(frozen.memory_path.read_bytes()).hexdigest() == (
        "07917374b5df7f9d4af1cc772593f60cf650fe2db9fce06e05f34240e5a3366e"
    )
    assert hashlib.sha256(frozen.manifest_path.read_bytes()).hexdigest() == (
        "6f66e2221c5d504364ed20dfaa389286a53a9daaecd4b5b3f96dc86936c6d358"
    )


@pytest.mark.parametrize("new_guidance", [False, True])
def test_reviewed_adaptive_phase2_freezes_into_phase3(tmp_path, new_guidance):
    source_root, source, review, memory = _source(tmp_path, new=new_guidance)
    root = tmp_path / "phase2"
    snapshot = admit_adaptive_snapshot(source_root, b"", review, source, root)
    original = {path.relative_to(snapshot.root): path.read_bytes()
                for path in snapshot.root.rglob("*") if path.is_file()}
    runner = Phase2LiveRunner(
        phase2_root=root,
        snapshot=snapshot,
        dependencies=_deps(snapshot, _Backend(), []),
    )
    assert runner.run()["status"] == "passed"
    _write_qualification(root, snapshot.destination)

    frozen = admit_phase2_lineage(root, snapshot.destination, tmp_path / "phase3")

    assert frozen.manifest["source_snapshot_id"] == snapshot.snapshot_id
    assert frozen.manifest["phase1_entries"] == 1
    assert canonical_bytes(frozen.actor_memory()["phase1"]) == canonical_bytes([memory])
    assert frozen.actor_memory()["phase2"][0]["passed"] is True
    assert type(frozen).open(frozen.root, snapshot.destination) == frozen
    assert original == {path.relative_to(snapshot.root): path.read_bytes()
                        for path in snapshot.root.rglob("*") if path.is_file()}

    (snapshot.root / "review.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="adaptive snapshot bytes changed"):
        admit_phase2_lineage(root, snapshot.destination, tmp_path / "tampered-phase3")


def test_frozen_copy_rejects_memory_and_manifest_tampering(tmp_path):
    root, cell, _runner, _terminal = _phase2(tmp_path)
    frozen = admit_phase2_lineage(root, cell, tmp_path / "phase3")
    frozen.memory_path.write_bytes(frozen.memory_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="bytes changed"):
        frozen.verify()

    # Restore through a distinct handoff, then alter a reserialized manifest.
    second = admit_phase2_lineage(root, cell, tmp_path / "other-phase3")
    manifest = json.loads(second.manifest_path.read_text())
    manifest["phase2_entries"] = 2
    second.manifest_path.write_bytes(canonical_bytes(manifest) + b"\n")
    with pytest.raises(ValueError, match="manifest"):
        type(second).open(second.root, cell)


def test_actor_memory_parses_the_same_bytes_it_authenticates(tmp_path, monkeypatch):
    root, cell, _runner, _terminal = _phase2(tmp_path)
    frozen = admit_phase2_lineage(root, cell, tmp_path / "phase3")
    trusted = frozen.memory_path.read_bytes()
    changed = json.loads(trusted)
    changed["phase2"][0]["passed"] = False
    changed_bytes = canonical_bytes(changed) + b"\n"
    real_read_bytes = Path.read_bytes
    reads = 0

    def changing_read(path):
        nonlocal reads
        if path == frozen.memory_path:
            reads += 1
            return trusted if reads == 1 else changed_bytes
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", changing_read)

    assert frozen.actor_memory()["phase2"][0]["passed"] is True
    assert reads == 1


def test_admission_rejects_failed_and_non_openai_phase2(tmp_path):
    root, cell, _runner, terminal = _phase2(
        tmp_path / "failed", (False,) * MAX_ATTEMPTS
    )
    assert terminal["status"] == "failed"
    with pytest.raises(ValueError, match="passed"):
        admit_phase2_lineage(root, cell, tmp_path / "failed-p3")

    deepseek = next(
        item for item in phase2_cells(cell.phase1_guide, cell.qualification_sha256)
        if item.model.backend_model is A5BackendModel.DEEPSEEK_FLASH
    )
    with pytest.raises(ValueError, match="OpenAI"):
        admit_phase2_lineage(root, deepseek, tmp_path / "deepseek-p3")


@pytest.mark.parametrize(
    ("field", "replacement", "rehash"),
    [
        ("record_sha256", "0" * 64, False),
        ("cell_id", "a5-phase2-cell-0000000000000000", True),
        ("runtime_catlass_revision", "c" * 40, True),
        ("catlass_source", "/different/catlass", True),
    ],
)
def test_admission_rejects_tampered_qualification(
    tmp_path, field, replacement, rehash,
):
    root, cell, _runner, _terminal = _phase2(tmp_path)
    path = root / "cells" / cell.cell_id / "qualification.json"
    value = json.loads(path.read_text())
    value[field] = replacement
    if rehash:
        body = {key: item for key, item in value.items() if key != "record_sha256"}
        value["record_sha256"] = canonical_digest(body)
    path.write_bytes(canonical_bytes(value) + b"\n")

    with pytest.raises(ValueError, match="qualification"):
        admit_phase2_lineage(root, cell, tmp_path / "phase3")


def test_admission_rejects_missing_qualification(tmp_path):
    root, cell, _runner, _terminal = _phase2(tmp_path)
    (root / "cells" / cell.cell_id / "qualification.json").unlink()

    with pytest.raises(ValueError, match="qualification"):
        admit_phase2_lineage(root, cell, tmp_path / "phase3")


@pytest.mark.parametrize("artifact", ["protocol", "memory", "terminal"])
def test_admission_rejects_source_changed_after_authentication(
    tmp_path, monkeypatch, artifact,
):
    root, cell, _runner, _terminal = _phase2(tmp_path)
    real_load_terminal = Phase2LiveRunner.load_terminal

    def load_then_change(runner):
        terminal = real_load_terminal(runner)
        path = {
            "protocol": runner.protocol_path,
            "memory": runner.memory_path,
            "terminal": runner.terminal_path,
        }[artifact]
        payload = path.read_bytes()
        path.write_bytes(payload.rstrip(b"\n") + b" \n")
        return terminal

    monkeypatch.setattr(Phase2LiveRunner, "load_terminal", load_then_change)

    with pytest.raises(ValueError, match="changed during authentication"):
        admit_phase2_lineage(root, cell, tmp_path / "phase3")


@pytest.mark.parametrize("artifact", ["protocol", "memory", "candidate", "evidence"])
def test_admission_rejects_tampered_phase2_trust_anchors(tmp_path, artifact):
    root, cell, runner, _terminal = _phase2(tmp_path)
    paths = {
        "protocol": runner.protocol_path,
        "memory": runner.memory_path,
        "candidate": runner.root / "attempts" / "attempt-01" / "candidate.py",
        "evidence": runner.root / "attempts" / "attempt-01" / "evidence.json",
    }
    paths[artifact].write_bytes(paths[artifact].read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="Phase 2|candidate|evidence|identity|invalid"):
        admit_phase2_lineage(root, cell, tmp_path / "phase3")
