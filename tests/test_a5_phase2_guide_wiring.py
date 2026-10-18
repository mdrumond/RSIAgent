from __future__ import annotations

from argparse import Namespace

import pytest

import run_a5_campaign
import run_a5_phase2
import benchmarks.a5kernels.phase2_production as phase2_production_module
from benchmarks.a5kernels.phase1_experiments import admit_programming_guide
from benchmarks.a5kernels.phase1_live import write_gate_report
from benchmarks.a5kernels.phase2_memory import CELL_SCHEMA_V2, phase2_cell_pairs
from benchmarks.a5kernels.phase2_production import (
    Phase2ProductionConfig,
    build_phase2_production,
)
from benchmarks.a5kernels.priority_controller import Job, JobKind
from tests.test_a5_phase2_production import (
    CopyVBackend,
    ENVIRONMENT,
    FakeExecutor,
    _transport,
)
from tests.test_a5_phase2_memory import _completed_source
from tests.test_a5_phase1_composition import _guide_and_report


CPL_REVISION = "d" * 40


def test_production_uses_phase1_guide_for_snapshot_and_phase2_guide_for_agent(tmp_path):
    composition, source, _legacy_destination = _completed_source(tmp_path)
    path = tmp_path / "phase2-guide.md"
    text = "Current Phase 2 Catlass programming guidance.\n"
    path.write_text(text, encoding="utf-8")
    guide = admit_programming_guide(
        path,
        cpl_skills_revision=CPL_REVISION,
        catlass_revision=composition.guide.catlass_revision,
    )
    destination = dict(phase2_cell_pairs(
        composition.guide,
        composition.qualification.report_sha256,
        destination_guide=guide,
    ))[source]
    root = tmp_path / "phase2"
    config = Phase2ProductionConfig(
        phase1_root=composition.root,
        gate=composition.qualification,
        guide_path=tmp_path / "agent-guide.md",
        phase2_root=root,
        snapshot_path=root / "cells" / destination.cell_id / "snapshot",
        cell_id=destination.cell_id,
        tla_root=tmp_path / "tla",
        catlass_source="/remote/catlass",
        device=3,
        phase2_guide_path=path,
        phase2_cpl_skills_revision=CPL_REVISION,
    )

    snapshot, dependencies = build_phase2_production(
        config,
        environment=ENVIRONMENT,
        transport=_transport,
        executor=FakeExecutor(destination.catlass_revision),
        backend=CopyVBackend(),
    )

    assert destination.schema == CELL_SCHEMA_V2
    assert destination.source_guide == composition.guide
    assert snapshot.destination == destination
    assert dependencies.cell == destination
    assert dependencies.source.guide_text == text


@pytest.mark.parametrize("missing", ["path", "revision"])
def test_production_rejects_partial_phase2_guide_admission(tmp_path, missing):
    composition, _source, destination = _completed_source(tmp_path)
    root = tmp_path / "phase2"
    values = {
        "phase2_guide_path": tmp_path / "phase2-guide.md",
        "phase2_cpl_skills_revision": CPL_REVISION,
    }
    values["phase2_guide_path" if missing == "path" else "phase2_cpl_skills_revision"] = None
    with pytest.raises(ValueError, match="Phase 2 guide"):
        Phase2ProductionConfig(
            phase1_root=composition.root,
            gate=composition.qualification,
            guide_path=tmp_path / "agent-guide.md",
            phase2_root=root,
            snapshot_path=root / "cells" / destination.cell_id / "snapshot",
            cell_id=destination.cell_id,
            tla_root=tmp_path / "tla",
            catlass_source="/remote/catlass",
            device=3,
            **values,
        )


def test_phase2_guide_rejects_content_changed_during_admission(
    monkeypatch, tmp_path
):
    composition, _source, destination = _completed_source(tmp_path)
    path = tmp_path / "phase2-guide.md"
    path.write_text("admitted guide\n", encoding="utf-8")
    config = Phase2ProductionConfig(
        phase1_root=composition.root,
        gate=composition.qualification,
        guide_path=tmp_path / "agent-guide.md",
        phase2_root=tmp_path / "phase2",
        snapshot_path=tmp_path / "phase2" / "cells" / destination.cell_id / "snapshot",
        cell_id=destination.cell_id,
        tla_root=tmp_path / "tla",
        catlass_source="/remote/catlass",
        device=3,
        phase2_guide_path=path,
        phase2_cpl_skills_revision=CPL_REVISION,
    )
    admit = phase2_production_module.admit_programming_guide

    def replace_after_admission(*args, **kwargs):
        guide = admit(*args, **kwargs)
        path.write_text("changed guide\n", encoding="utf-8")
        return guide

    monkeypatch.setattr(
        phase2_production_module,
        "admit_programming_guide",
        replace_after_admission,
    )

    with pytest.raises(ValueError, match="changed during admission"):
        phase2_production_module._phase2_guide(config, composition)


def test_phase2_cli_forwards_separate_guide_admission(monkeypatch, tmp_path):
    captured = []
    _path, _guide, gate = _guide_and_report(tmp_path)
    monkeypatch.setattr(run_a5_phase2, "load_gate_report", lambda _path: gate)
    monkeypatch.setattr(
        run_a5_phase2,
        "build_phase2_production",
        lambda config: captured.append(config) or (object(), object()),
    )
    args = run_a5_phase2._parser().parse_args([
        "run", "--root", str(tmp_path / "phase2"),
        "--snapshot", str(tmp_path / "snapshot"), "--cell-id", "cell",
        "--phase1-root", str(tmp_path / "phase1"), "--gate", str(tmp_path / "gate"),
        "--guide", str(tmp_path / "phase1-guide"),
        "--phase2-guide", str(tmp_path / "phase2-guide"),
        "--phase2-cpl-skills-revision", CPL_REVISION,
        "--tla-root", str(tmp_path / "tla"),
        "--catlass-source", "/remote/catlass", "--device", "3",
        "--compile-probes", "3",
    ])

    run_a5_phase2._production_factory(args)

    assert captured[0].gate is gate
    assert captured[0].phase2_guide_path == tmp_path / "phase2-guide"
    assert captured[0].phase2_cpl_skills_revision == CPL_REVISION
    assert captured[0].compile_probes == 3


def test_campaign_only_forwards_phase2_guide_to_phase2_workers(tmp_path):
    args = Namespace(
        gate=tmp_path / "gate", guide=tmp_path / "phase1-guide",
        phase2_guide=tmp_path / "phase2-guide",
        phase2_cpl_skills_revision=CPL_REVISION,
        phase1_root=tmp_path / "phase1", phase2_root=tmp_path / "phase2",
        tla_root=tmp_path / "tla", profiling_skill_root=tmp_path / "profiling",
        catlass_source="/remote/catlass", bge_cache=tmp_path / "cache",
        kdb=tmp_path / "kdb", collection="catlass", env_file=None,
    )
    destination = type("Destination", (), {
        "cell_id": "phase2-cell", "source_cell_id": "phase1-cell",
    })()
    command = run_a5_campaign._command_factory(args, (destination,))

    phase1 = command(Job(JobKind.PHASE1_RUN, "phase1-cell"), 2)
    phase2 = command(Job(JobKind.PHASE2_RUN, "phase2-cell"), 2)

    assert "--phase2-guide" not in phase1
    assert phase2[phase2.index("--phase2-guide") + 1] == str(args.phase2_guide)
    assert phase2[phase2.index("--phase2-cpl-skills-revision") + 1] == CPL_REVISION


def test_campaign_registers_v2_destinations_from_separate_guide(tmp_path):
    phase1_guide_path, phase1_guide, gate = _guide_and_report(tmp_path)
    gate_path = tmp_path / "gate.json"
    write_gate_report(gate, gate_path)
    phase2_guide_path = tmp_path / "phase2-guide.md"
    phase2_guide_path.write_text("Current Phase 2 guide.\n", encoding="utf-8")
    phase2_guide = admit_programming_guide(
        phase2_guide_path,
        cpl_skills_revision=CPL_REVISION,
        catlass_revision=phase1_guide.catlass_revision,
    )
    args = Namespace(
        gate=gate_path, guide=phase1_guide_path,
        phase2_guide=phase2_guide_path,
        phase2_cpl_skills_revision=CPL_REVISION,
        controller_root=tmp_path / "controller",
        phase1_root=tmp_path / "phase1", phase2_root=tmp_path / "phase2",
        tla_root=tmp_path / "tla", profiling_skill_root=tmp_path / "profiling",
        catlass_source="/remote/catlass", bge_cache=tmp_path / "cache",
        kdb=tmp_path / "kdb", collection="catlass", env_file=None,
    )

    controller = run_a5_campaign._controller(args)

    assert len(controller.destinations) == 8
    assert {item.schema for item in controller.destinations} == {CELL_SCHEMA_V2}
    assert {item.phase1_guide for item in controller.destinations} == {phase1_guide}
    assert {item.guide for item in controller.destinations} == {phase2_guide}
