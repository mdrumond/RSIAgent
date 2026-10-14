from __future__ import annotations

import json

import pytest

from benchmarks.a5kernels.evidence import canonical_bytes
from benchmarks.a5kernels.phase1_experiments import A5BackendModel
from benchmarks.a5kernels.phase2_live import MAX_ATTEMPTS, Phase2LiveRunner
from benchmarks.a5kernels.phase2_memory import phase2_cells
from benchmarks.a5kernels.phase3_core import admit_phase2_lineage
from tests.test_a5_phase2_live import _Backend, _deps
from tests.test_a5_phase2_memory import _snapshot


def _phase2(tmp_path, outcomes=(True,)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _composition, _source, _destination, snapshot = _snapshot(tmp_path)
    root = tmp_path / "phase2"
    runner = Phase2LiveRunner(
        phase2_root=root,
        snapshot=snapshot,
        dependencies=_deps(snapshot, _Backend(outcomes), []),
    )
    terminal = runner.run()
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

    reopened = type(frozen).open(frozen.root, cell)
    assert reopened.snapshot_id == frozen.snapshot_id
    assert admit_phase2_lineage(root, cell, tmp_path / "phase3") == frozen


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
