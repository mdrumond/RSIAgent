from __future__ import annotations

import json

import pytest

import run_a5_phase1
from benchmarks.a5kernels.phase1_composition import (
    LiveDependencies,
    Phase1CellComposition,
    select_worker_cell,
)
from benchmarks.a5kernels.phase1_experiments import (
    PendingTreatmentError,
    build_cells,
)
from benchmarks.a5kernels.phase1_live import write_gate_report
from tests.test_a5_phase1_composition import (
    _dependencies,
    _guide_and_report,
)


def _runtime(tmp_path, state, guide_path, root, *selection):
    return [
        "--gate", str(state),
        "--guide", str(guide_path),
        "--root", str(root),
        "--one-project-smoke",
        "--tla-root", str(tmp_path),
        "--profiling-skill-root", str(tmp_path),
        "--catlass-source", "/remote/catlass",
        "--bge-cache", str(tmp_path),
        "--kdb", str(tmp_path / "kdb.sqlite"),
        "--collection", "catlass",
        "--device", "0",
        *selection,
    ]


def test_worker_selection_is_canonical_and_fails_closed(tmp_path):
    _path, guide, _report = _guide_and_report(tmp_path)
    cells = build_cells(guide)
    ready = next(cell for cell in cells if cell.runnable)
    pending = next(cell for cell in cells if not cell.runnable)

    assert select_worker_cell(guide, None) is None
    assert select_worker_cell(guide, (ready.cell_id,)) == ready
    with pytest.raises(ValueError, match="duplicate"):
        select_worker_cell(guide, (ready.cell_id, ready.cell_id))
    with pytest.raises(ValueError, match="exactly one"):
        select_worker_cell(guide, (cells[0].cell_id, cells[1].cell_id))
    with pytest.raises(ValueError, match="unknown"):
        select_worker_cell(guide, ("a5-cell-unknown",))
    with pytest.raises(PendingTreatmentError, match="pending"):
        select_worker_cell(guide, (pending.cell_id,))


def test_smoke_help_requires_a_distinct_full_campaign_root(capsys):
    with pytest.raises(SystemExit) as stopped:
        run_a5_phase1._parser().parse_args(["run", "--help"])
    assert stopped.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "use a root distinct from every full campaign" in help_text


def test_cli_single_cell_isolated_resume_and_terminal_reuse(tmp_path, capsys):
    gate_root = tmp_path / "gate"
    gate_root.mkdir()
    guide_path, guide, report = _guide_and_report(gate_root)
    state = tmp_path / "gate.json"
    write_gate_report(report, state)
    selected = next(cell for cell in build_cells(guide) if cell.runnable)
    root = tmp_path / "run"
    calls = []
    dependencies = _dependencies(tmp_path / "dependencies", calls)
    dependency_roots = []

    def builder(paths, **_kwargs):
        dependency_roots.append(paths.results_root)
        return dependencies

    argv = _runtime(
        tmp_path, state, guide_path, root,
        "--cell-id", selected.cell_id,
    )
    assert run_a5_phase1.main(
        ["run", *argv], dependency_builder=builder,
    ) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["completed_cell_ids"] == [selected.cell_id]
    assert len(calls) == 1
    assert dependency_roots == [root / "cells" / selected.cell_id]
    assert {path.name for path in (root / "cells").iterdir()} == {
        selected.cell_id
    }
    cell_root = root / "cells" / selected.cell_id
    assert (cell_root / "memory.jsonl").exists()
    assert (cell_root / "evidence.jsonl").exists()
    assert (cell_root / "terminal.json").exists()

    assert run_a5_phase1.main(
        ["resume", *argv], dependency_builder=builder,
    ) == 0
    assert json.loads(capsys.readouterr().out) == first
    assert len(calls) == 1


def test_single_cell_resume_preserves_ordered_progressive_memory(tmp_path):
    gate_root = tmp_path / "gate"
    gate_root.mkdir()
    guide_path, guide, report = _guide_and_report(gate_root)
    selected = next(cell for cell in build_cells(guide) if cell.runnable)
    root = tmp_path / "run"
    first_calls = []
    base = _dependencies(tmp_path / "first-dependencies", first_calls)

    def stop_at_third(request, *args):
        if request.ordinal == 3:
            raise RuntimeError("intentional worker interruption")
        return base.execute(request, *args)

    interrupted = Phase1CellComposition(
        root,
        guide_path,
        guide,
        report,
        LiveDependencies(stop_at_third, base.knowledge_factory),
    )
    with pytest.raises(RuntimeError, match="worker interruption"):
        interrupted.run_cell(selected.cell_id)
    assert [call["ordinal"] for call in first_calls] == [1, 2]
    assert {path.name for path in (root / "cells").iterdir()} == {
        selected.cell_id
    }

    resumed_calls = []
    resumed = Phase1CellComposition(
        root,
        guide_path,
        guide,
        report,
        _dependencies(tmp_path / "resumed-dependencies", resumed_calls),
    )
    terminal = resumed.run_cell(selected.cell_id)
    assert terminal["completed_projects"] == 8
    assert [call["ordinal"] for call in resumed_calls] == list(range(3, 9))

    assert resumed.run_cell(selected.cell_id) == terminal
    assert [call["ordinal"] for call in resumed_calls] == list(range(3, 9))


@pytest.mark.parametrize("selection", [
    ("--cell-id", "a5-cell-unknown"),
    ("--cell-id", "pending"),
])
def test_invalid_cli_selection_precedes_dependency_creation(
    tmp_path, selection,
):
    gate_root = tmp_path / "gate"
    gate_root.mkdir()
    guide_path, guide, report = _guide_and_report(gate_root)
    state = tmp_path / "gate.json"
    write_gate_report(report, state)
    if selection[-1] == "pending":
        pending = next(cell for cell in build_cells(guide) if not cell.runnable)
        selection = ("--cell-id", pending.cell_id)
    created = []

    def builder(*_args, **_kwargs):
        created.append(True)
        return _dependencies(tmp_path / "dependencies", [])

    with pytest.raises((ValueError, PendingTreatmentError)):
        run_a5_phase1.main(
            ["run", *_runtime(
                tmp_path, state, guide_path, tmp_path / "run", *selection,
            )],
            dependency_builder=builder,
        )
    assert created == []
    assert not (tmp_path / "run").exists()
