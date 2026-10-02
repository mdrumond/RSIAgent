import json
from types import SimpleNamespace

import pytest

from benchmarks.a3_experiments import KnowledgeMode
from benchmarks.a3kernels.phase1_evidence import canonical_digest
from benchmarks.a3kernels.phase1_wave import (
    SMOKE_PROPOSALS,
    Phase1Config,
    SmokeWave,
    foundation_cells,
    select_foundation_cells,
    smoke_dry_run,
)
from benchmarks.a3kernels.knowledge import CollectionManifest


def config(tmp_path):
    wrapper = tmp_path / "catlass-validation.sh"
    wrapper.write_text("#!/bin/sh\n")
    wrapper.chmod(0o755)
    embedding = tmp_path / "embedding"; embedding.mkdir()
    corpus = tmp_path / "corpus"; corpus.mkdir()
    database = tmp_path / "knowledge.sqlite3"; database.write_bytes(b"db")
    manifest = tmp_path / "manifest.json"; manifest.write_text("{}")
    return Phase1Config(
        tmp_path / "state", wrapper, embedding, corpus, database, manifest
    )


def outcome(*_args):
    return {"status": "passed", "evidence_sha256": "a" * 64}


def knowledge_identity(seed="a"):
    return {
        "embedding_snapshot_sha256": seed * 64,
        "knowledge_database_sha256": "b" * 64,
        "knowledge_manifest_sha256": "c" * 64,
        "knowledge_collection_sha256": "d" * 64,
        "knowledge_probe_sha256": "e" * 64,
    }


def test_smoke_plan_is_one_registered_baseline_and_not_phase1_plan():
    plan = smoke_dry_run()
    assert plan["schema"] == "a3-ascendc-foundation-smoke-plan-v1"
    assert plan["mode"] == "smoke"
    assert len(plan["projects"]) == len(SMOKE_PROPOSALS) == 1
    assert plan["projects"][0]["family"] == "vector-add-baseline"


def test_smoke_paths_and_terminals_are_isolated_from_full_phase1(tmp_path):
    cfg = config(tmp_path)
    cell = next(c for c in foundation_cells() if c.knowledge is KnowledgeMode.WITHOUT_KDB)
    wave = SmokeWave(
        cfg, outcome, cells=(cell,), execution_profile="bz-a3-1"
    )
    record = wave.run()[0]
    assert wave.paths(cell).root.parent == cfg.state_root / "smoke-cells"
    assert record["schema"] == "a3-phase1-smoke-cell-terminal-v1"
    assert record["knowledge_identity"] is None
    assert record["plan_fingerprint"] == canonical_digest(smoke_dry_run())
    assert record["proposal_set_sha256"] == canonical_digest(
        [proposal.as_dict() for proposal in SMOKE_PROPOSALS]
    )
    assert not (cfg.state_root / "cells" / cell.cell_id).exists()


def test_no_kdb_smoke_never_authenticates_or_opens_artifacts(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    for path in (
        cfg.embedding_cache, cfg.corpus_artifacts, cfg.knowledge_database,
        cfg.knowledge_manifest,
    ):
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()
    cell = next(c for c in foundation_cells() if c.knowledge is KnowledgeMode.WITHOUT_KDB)
    monkeypatch.setattr(
        "benchmarks.a3kernels.phase1_wave.authenticate_smoke_knowledge",
        lambda *_: pytest.fail("no-KDB selection touched knowledge artifacts"),
    )
    credential = (
        "OPENAI_API_KEY" if "openai" in cell.backend_model.value
        else "DEEPSEEK_API_KEY"
    )
    report = cfg.smoke_preflight({credential: "secret"}, cells=(cell,))
    assert report["knowledge_identity"] is None
    SmokeWave(cfg, outcome, cells=(cell,), execution_profile="bz-a3-1").run()


def test_kdb_authentication_happens_before_executor(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cell = next(c for c in foundation_cells() if c.knowledge is KnowledgeMode.WITH_KDB)
    events = []
    monkeypatch.setattr(
        "benchmarks.a3kernels.phase1_wave.authenticate_smoke_knowledge",
        lambda *_: events.append("authenticated-query") or knowledge_identity(),
    )
    wave = SmokeWave(
        cfg,
        lambda *_: events.append("provider-executor") or outcome(),
        cells=(cell,), execution_profile="bz-a3-1",
    )
    wave.run()
    assert events == ["authenticated-query", "provider-executor"]


def test_smoke_kdb_authentication_opens_read_only_and_runs_fixed_query(
    tmp_path, monkeypatch,
):
    import benchmarks.a3kernels.phase1_wave as wave_module

    cfg = config(tmp_path)
    manifest = CollectionManifest(
        "a3", "a3", "model", "revision", 3, "schema", 80, 20, (), "f" * 64
    )
    events = []

    class Database:
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            events.append("closed")
        def manifest(self, collection):
            events.append(("manifest", collection))
            return manifest
        def query(self, collection, query, *, limit):
            events.append(("query", collection, query, limit))
            return [SimpleNamespace(chunk_id="1" * 64, content_sha256="2" * 64)]

    monkeypatch.setattr(
        wave_module, "authenticated_snapshot_identity", lambda _path: "a" * 64
    )
    monkeypatch.setattr(
        wave_module.CollectionManifest, "from_json", lambda _value: manifest
    )
    monkeypatch.setattr(wave_module, "PinnedBGEEmbeddings", lambda **_kw: object())
    monkeypatch.setattr(
        wave_module.KnowledgeDB, "open_read_only",
        lambda path, embeddings: events.append(("open", path, embeddings)) or Database(),
    )

    identity = wave_module.authenticate_smoke_knowledge(cfg)

    assert identity["embedding_snapshot_sha256"] == "a" * 64
    assert events[0][0] == "open"
    assert events[1] == ("manifest", "a3")
    assert events[2] == (
        "query", "a3", "A3 Ascend C vector addition tensor movement", 1,
    )
    assert events[3] == "closed"


def test_smoke_kdb_probe_fails_closed_when_query_has_no_result(tmp_path, monkeypatch):
    import benchmarks.a3kernels.phase1_wave as wave_module

    cfg = config(tmp_path)
    manifest = SimpleNamespace(collection="a3", fingerprint="f" * 64)

    class Database:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def manifest(self, _collection): return manifest
        def query(self, *_args, **_kwargs): return []

    monkeypatch.setattr(
        wave_module, "authenticated_snapshot_identity", lambda _path: "a" * 64
    )
    monkeypatch.setattr(
        wave_module.CollectionManifest, "from_json", lambda _value: manifest
    )
    monkeypatch.setattr(wave_module, "PinnedBGEEmbeddings", lambda **_kw: object())
    monkeypatch.setattr(
        wave_module.KnowledgeDB, "open_read_only", lambda *_args: Database()
    )
    with pytest.raises(ValueError, match="no authenticated result"):
        wave_module.authenticate_smoke_knowledge(cfg)


@pytest.mark.parametrize(
    "field",
    [
        "execution_profile", "model_profile_sha256", "proposal_set_sha256",
        "profile_driver_sha256", "host_fixture_sha256", "knowledge_identity",
    ],
)
def test_smoke_resume_rejects_changed_bound_identity(tmp_path, monkeypatch, field):
    cfg = config(tmp_path)
    cell = next(c for c in foundation_cells() if c.knowledge is KnowledgeMode.WITH_KDB)
    identity = knowledge_identity()
    monkeypatch.setattr(
        "benchmarks.a3kernels.phase1_wave.authenticate_smoke_knowledge",
        lambda *_: identity,
    )
    wave = SmokeWave(
        cfg, outcome, cells=(cell,), execution_profile="bz-a3-1"
    )
    wave.run()
    terminal = wave.paths(cell).terminal
    record = json.loads(terminal.read_text())
    record[field] = (
        knowledge_identity("f") if field == "knowledge_identity" else "f" * 64
    )
    terminal.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="foreign|conflicting"):
        wave.resume()


def test_smoke_shards_are_deterministic_disjoint_and_use_unique_paths(tmp_path):
    cfg = config(tmp_path)
    first = select_foundation_cells(shard_count=2, shard_index=0)
    second = select_foundation_cells(shard_count=2, shard_index=1)
    assert not set(first) & set(second)
    assert set(first) | set(second) == set(foundation_cells())
    first_paths = {
        SmokeWave(cfg, outcome, cells=(cell,), execution_profile="bz-a3-1").paths(cell).root
        for cell in first
    }
    second_paths = {
        SmokeWave(cfg, outcome, cells=(cell,), execution_profile="bz-a3-2").paths(cell).root
        for cell in second
    }
    assert first_paths.isdisjoint(second_paths)


def test_cli_smoke_selects_one_project_composition_and_smoke_terminal(
    tmp_path, monkeypatch, capsys,
):
    import run_a3_phase1

    cfg = config(tmp_path)
    cell = next(c for c in foundation_cells() if c.knowledge is KnowledgeMode.WITHOUT_KDB)
    selected = {}

    class Composition:
        def __init__(self, _cfg, _deps, *, proposals, lineage_prefix):
            selected["proposals"] = tuple(proposals)
            selected["lineage_prefix"] = lineage_prefix
        def execute(self, *_args):
            return outcome()

    monkeypatch.setattr(run_a3_phase1, "LiveComposition", Composition)
    monkeypatch.setattr(
        run_a3_phase1, "bz_live_dependencies", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        run_a3_phase1, "_bz_preflight", lambda *_args: {"state": "completed"}
    )
    credential = (
        "OPENAI_API_KEY" if "openai" in cell.backend_model.value
        else "DEEPSEEK_API_KEY"
    )
    monkeypatch.setenv(credential, "secret")
    argv = [
        "smoke", "--state-root", str(cfg.state_root),
        "--validation-wrapper", str(cfg.validation_wrapper),
        "--profile", "bz-a3-1", "--cpl-remote", "/checked/cpl-remote",
        "--remote-workspace", "/remote/rsi", "--physical-device", "0",
        "--cell-id", cell.cell_id,
    ]
    assert run_a3_phase1.main(argv) == 0
    record = json.loads(capsys.readouterr().out)[0]
    assert record["schema"] == "a3-phase1-smoke-cell-terminal-v1"
    assert selected == {
        "proposals": SMOKE_PROPOSALS,
        "lineage_prefix": "smoke",
    }
