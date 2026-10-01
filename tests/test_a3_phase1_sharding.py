import json

import pytest

from benchmarks.a3kernels.phase1_wave import (
    Phase1Config,
    Phase1Wave,
    foundation_cells,
    select_foundation_cells,
)


def config(tmp_path):
    wrapper = tmp_path / "catlass-validation.sh"
    wrapper.write_text("#!/bin/sh\n")
    wrapper.chmod(0o755)
    embedding = tmp_path / "embedding"
    embedding.mkdir()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    database = tmp_path / "kdb.sqlite"
    database.write_bytes(b"db")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    return Phase1Config(
        tmp_path / "state", wrapper, embedding, corpus, database, manifest
    )


def outcome(cell, _paths):
    return {
        "status": "passed",
        "evidence_sha256": cell.cell_id.removeprefix("a3-cell-") * 4,
    }


def test_explicit_selection_is_canonical_and_rejects_invalid_ids():
    cells = foundation_cells()
    selected = select_foundation_cells(
        cell_ids=(cells[5].cell_id, cells[1].cell_id)
    )
    assert selected == (cells[1], cells[5])

    with pytest.raises(ValueError, match="duplicate cell ID"):
        select_foundation_cells(cell_ids=(cells[0].cell_id, cells[0].cell_id))
    with pytest.raises(ValueError, match="unknown foundation cell ID"):
        select_foundation_cells(cell_ids=("a3-cell-unknown",))


@pytest.mark.parametrize(
    ("shard_count", "shard_index"),
    [(0, 0), (-1, 0), (2, -1), (2, 2), (None, 0), (2, None), (True, 0)],
)
def test_shard_validation_rejects_invalid_coordinates(shard_count, shard_index):
    with pytest.raises(ValueError, match="shard"):
        select_foundation_cells(
            shard_count=shard_count,
            shard_index=shard_index,
        )


def test_two_shards_are_deterministic_disjoint_and_cover_foundation_cells():
    first = select_foundation_cells(shard_count=2, shard_index=0)
    second = select_foundation_cells(shard_count=2, shard_index=1)
    assert first == select_foundation_cells(shard_count=2, shard_index=0)
    assert not set(first) & set(second)
    assert set(first) | set(second) == set(foundation_cells())


def test_selected_wave_reuses_terminals_without_staging_other_cells(tmp_path):
    cfg = config(tmp_path)
    cells = foundation_cells()
    selected = select_foundation_cells(cell_ids=(cells[2].cell_id, cells[6].cell_id))
    calls = []

    def execute(cell, paths):
        calls.append(cell.cell_id)
        return outcome(cell, paths)

    first = Phase1Wave(cfg, execute, cells=selected).run()
    second = Phase1Wave(cfg, execute, cells=selected).resume()
    assert first == second
    assert calls == [cell.cell_id for cell in selected]
    assert {path.name for path in (cfg.state_root / "cells").iterdir()} == set(calls)


def test_two_shard_runtime_proof_aggregates_one_complete_report(tmp_path):
    cfg = config(tmp_path)
    shard_zero = select_foundation_cells(shard_count=2, shard_index=0)
    shard_one = select_foundation_cells(shard_count=2, shard_index=1)
    assert len(Phase1Wave(cfg, outcome, cells=shard_zero).run()) == 4
    assert len(Phase1Wave(cfg, outcome, cells=shard_one).run()) == 4

    report = Phase1Wave(cfg, outcome).report()
    assert report["counts"] == {"passed": 8}
    assert [row["cell_id"] for row in report["records"]] == [
        cell.cell_id for cell in foundation_cells()
    ]


def test_cli_repeatable_cell_selection_and_shard_routing(tmp_path, monkeypatch, capsys):
    import run_a3_phase1

    cfg = config(tmp_path)
    cells = foundation_cells()
    created = []

    class FakeComposition:
        def __init__(self, _config, _dependencies):
            pass

        def execute(self, cell, paths):
            return outcome(cell, paths)

    monkeypatch.setattr(run_a3_phase1, "LiveComposition", FakeComposition)
    monkeypatch.setattr(
        run_a3_phase1,
        "managed_live_dependencies",
        lambda *args, **kwargs: created.append("dependencies") or object(),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")

    common = [
        "--state-root", str(cfg.state_root),
        "--validation-wrapper", str(cfg.validation_wrapper),
        "--embedding-cache", str(cfg.embedding_cache),
        "--corpus-artifacts", str(cfg.corpus_artifacts),
        "--knowledge-database", str(cfg.knowledge_database),
        "--knowledge-manifest", str(cfg.knowledge_manifest),
        "--remote-client", "/checked/remote_agent_client.sh",
        "--server", "http://127.0.0.1:37787",
        "--remote", "bz-a3-1",
        "--remote-workspace", "/data2/research",
        "--physical-device", "0",
    ]
    assert run_a3_phase1.main([
        "run", *common,
        "--cell-id", cells[7].cell_id,
        "--cell-id", cells[3].cell_id,
        "--shard-count", "2", "--shard-index", "1",
    ]) == 0
    records = json.loads(capsys.readouterr().out)
    expected = select_foundation_cells(
        cell_ids=(cells[7].cell_id, cells[3].cell_id),
        shard_count=2,
        shard_index=1,
    )
    assert [record["cell_id"] for record in records] == [cell.cell_id for cell in expected]
    assert created == ["dependencies"]


def test_cli_rejects_unknown_cell_before_remote_dependency_creation(
    tmp_path, monkeypatch,
):
    import run_a3_phase1

    cfg = config(tmp_path)
    created = []
    monkeypatch.setattr(
        run_a3_phase1,
        "managed_live_dependencies",
        lambda *args, **kwargs: created.append("dependencies"),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    argv = [
        "run",
        "--state-root", str(cfg.state_root),
        "--validation-wrapper", str(cfg.validation_wrapper),
        "--embedding-cache", str(cfg.embedding_cache),
        "--corpus-artifacts", str(cfg.corpus_artifacts),
        "--knowledge-database", str(cfg.knowledge_database),
        "--knowledge-manifest", str(cfg.knowledge_manifest),
        "--remote-client", "/checked/remote_agent_client.sh",
        "--server", "http://127.0.0.1:37787",
        "--remote", "bz-a3-1",
        "--remote-workspace", "/data2/research",
        "--physical-device", "0",
        "--cell-id", "a3-cell-unknown",
    ]
    with pytest.raises(ValueError, match="unknown foundation cell ID"):
        run_a3_phase1.main(argv)
    assert created == []
