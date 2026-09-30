import json
from pathlib import Path

import pytest

from benchmarks.a3kernels.phase1_wave import (
    Phase1Config, Phase1Wave, foundation_cells, full_dry_run,
)


def config(tmp_path):
    wrapper = tmp_path / "catlass-validation.sh"
    wrapper.write_text("#!/bin/sh\n")
    wrapper.chmod(0o755)
    embedding = tmp_path / "embedding"; embedding.mkdir()
    corpus = tmp_path / "corpus"; corpus.mkdir()
    database = tmp_path / "kdb.sqlite"; database.write_bytes(b"db")
    manifest = tmp_path / "manifest.json"; manifest.write_text("{}")
    profile_driver = tmp_path / "a3_profile_driver.py"; profile_driver.write_text("# fixed\n")
    return Phase1Config(tmp_path / "state", wrapper, embedding, corpus, database, manifest, profile_driver)


def test_all_eight_project_dry_run_has_no_side_effects(tmp_path):
    cfg = config(tmp_path)
    before = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")}
    plan = full_dry_run()
    after = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")}
    assert len(plan["projects"]) == 8 and plan["mode"] == "dry-run"
    assert before == after and not cfg.state_root.exists()


def test_foundation_selection_is_exact_two_by_two_by_two():
    cells = foundation_cells()
    assert len(cells) == len({cell.cell_id for cell in cells}) == 8
    assert len({cell.backend_model for cell in cells}) == 2
    assert len({cell.knowledge for cell in cells}) == 2
    assert len({cell.profiling for cell in cells}) == 2
    assert {cell.programming_level.value for cell in cells} == {"foundation"}


def test_preflight_checks_credentials_paths_and_fixed_adapter(tmp_path):
    cfg = config(tmp_path)
    report = cfg.preflight({"OPENROUTER_API_KEY": "gpt", "DEEPSEEK_API_KEY": "ds"})
    assert report["ready"] is True and all(report["checks"].values())
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        cfg.preflight({"OPENROUTER_API_KEY": "gpt"})
    cfg.profile_driver.write_text("foreign catlass evidence")
    with pytest.raises(ValueError, match="foreign"):
        cfg.preflight({"OPENROUTER_API_KEY": "gpt", "DEEPSEEK_API_KEY": "ds"})


def test_fake_eight_cell_wave_is_isolated_terminal_and_reportable(tmp_path):
    cfg = config(tmp_path)
    seen = []
    def execute(cell, paths):
        seen.append((cell.cell_id, paths))
        assert paths.workspace.parent == cfg.state_root / "cells" / cell.cell_id
        assert paths.profile_driver.read_text() == "# fixed\n"
        return {"status": "passed", "evidence_sha256": cell.cell_id.removeprefix("a3-cell-") * 4}
    wave = Phase1Wave(cfg, execute)
    records = wave.run()
    assert len(records) == len(seen) == 8
    assert {record["status"] for record in records} == {"passed"}
    report = wave.report()
    assert report["counts"] == {"passed": 8}
    assert len({str(paths.memory) for _, paths in seen}) == 8


def test_interruption_resume_does_not_duplicate_terminal_cells(tmp_path):
    cfg = config(tmp_path)
    calls = []
    def interrupted(cell, paths):
        calls.append(cell.cell_id)
        if len(calls) == 4: raise KeyboardInterrupt()
        return {"status": "passed", "evidence_sha256": "a" * 64}
    wave = Phase1Wave(cfg, interrupted)
    with pytest.raises(KeyboardInterrupt): wave.run()
    assert len(wave.report()["records"]) == 3
    resumed_calls = []
    records = Phase1Wave(
        cfg, lambda cell, paths: (
            resumed_calls.append(cell.cell_id) or {"status": "passed", "evidence_sha256": "b" * 64}
        ),
    ).resume()
    assert len(records) == 8 and len(resumed_calls) == 5
    assert not set(calls[:3]) & set(resumed_calls)


def test_foreign_or_conflicting_terminal_record_is_rejected(tmp_path):
    cfg = config(tmp_path)
    wave = Phase1Wave(cfg, lambda *_: {"status": "passed", "evidence_sha256": "a" * 64})
    first = foundation_cells()[0]
    paths = wave.paths(first); paths.root.mkdir(parents=True)
    paths.terminal.write_text(json.dumps({"cell_id": first.cell_id, "target": "a5", "language": "catlass-dsl", "status": "passed", "evidence_sha256": "a" * 64}))
    with pytest.raises(ValueError, match="foreign"):
        wave.resume()


def test_cli_dry_run_and_report(capsys, tmp_path):
    import run_a3_phase1
    assert run_a3_phase1.main(["dry-run"]) == 0
    assert len(json.loads(capsys.readouterr().out)["projects"]) == 8
    cfg = config(tmp_path)
    assert run_a3_phase1.main(["report", "--state-root", str(cfg.state_root)]) == 0
    assert json.loads(capsys.readouterr().out)["counts"] == {}
