from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.knowledge import KnowledgeDB
from benchmarks.a5kernels.production import (
    PILOT_CELL_ID,
    PILOT_WORKLOAD,
    ProductionPaths,
    _probe_catlass_runtime,
    build_production_trial,
    preflight,
    run_pilot,
)


class TinyEmbeddings:
    model = "BAAI/bge-small-en-v1.5"
    revision = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

    def __init__(self, **_kwargs):
        pass

    def embed(self, texts):
        return [[1.0, float(index + 1)] for index, _text in enumerate(texts)]

    embed_query = embed


def _query_schema(database, collection):
    row = database.connection.execute(
        "SELECT fts_table FROM collections WHERE name = ?", (collection,)
    ).fetchone()
    assert row is not None
    expected = database._fts_table_name(collection)
    assert row["fts_table"] == expected
    assert database.connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (expected,)
    ).fetchone() is not None


def _inputs(tmp_path: Path) -> tuple[ProductionPaths, list[Path]]:
    tla = tmp_path / "tla"
    profile = tmp_path / "profiling"
    wrappers = [
        tla / "execution-profiles/catlass-validation.sh",
        tla / "execution-profiles/bz-a5/session.sh",
        tla / "execution-profiles/bz-a5/upload.sh",
        tla / "execution-profiles/bz-a5/catlass-provenance.sh",
        profile / "scripts/collect_profile.sh",
    ]
    for wrapper in wrappers:
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
        wrapper.chmod(0o755)
    cache = tmp_path / "cache"
    cache.mkdir()
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = source_root / "vector.py"
    source.write_text("@tla.kernel\ndef vector_add(a, b, c): pass\n", encoding="utf-8")
    database_path = tmp_path / "knowledge.sqlite3"
    with KnowledgeDB(database_path, TinyEmbeddings()) as database:
        database.index(
            source_root, [source], collection="catlass-pilot", language="en"
        )
    paths = ProductionPaths(
        tla_root=tla,
        profiling_skill_root=profile,
        catlass_source="/retained/catlass",
        catlass_revision="a" * 40,
        bge_cache=cache,
        kdb=database_path,
        collection="catlass-pilot",
        results_root=tmp_path / "results" / "pilot-001",
        device=3,
    )
    paths.results_root.parent.mkdir()
    return paths, wrappers


def test_preflight_is_fail_closed_and_does_not_expose_secret(tmp_path):
    paths, _wrappers = _inputs(tmp_path)
    probed = []
    report = preflight(
        paths,
        environment={"OPENROUTER_API_KEY": ""},
        embedding_probe=lambda backend: probed.append((backend.model, backend.revision)),
        runtime_probe=lambda _paths: None,
        query_probe=_query_schema,
    )

    assert report["ready"] is False
    assert report["checks"]["openrouter_credential"] is False
    assert report["checks"]["kdb_manifest"] is True
    assert probed == [(TinyEmbeddings.model, TinyEmbeddings.revision)]
    assert "OPENROUTER_API_KEY" not in json.dumps(report)


def test_dotenv_credential_syntax_matches_runtime_parser(tmp_path):
    paths, _wrappers = _inputs(tmp_path)
    env_file = tmp_path / ".env"
    base = {"RSIAGENT_ENV_FILE": str(env_file)}

    env_file.write_text(" OPENROUTER_API_KEY=not-runtime-visible\n", encoding="utf-8")
    rejected = preflight(
        paths, environment=base, embedding_probe=lambda _backend: None,
        runtime_probe=lambda _paths: None, query_probe=_query_schema,
    )
    env_file.write_text("OPENROUTER_API_KEY=runtime-visible\n", encoding="utf-8")
    accepted = preflight(
        paths, environment=base, embedding_probe=lambda _backend: None,
        runtime_probe=lambda _paths: None, query_probe=_query_schema,
    )

    assert rejected["checks"]["openrouter_credential"] is False
    assert accepted["checks"]["openrouter_credential"] is True


def test_preflight_checks_complete_local_inputs_without_model_or_bz_call(tmp_path):
    paths, _wrappers = _inputs(tmp_path)
    report = preflight(
        paths,
        environment={"OPENROUTER_API_KEY": "present-but-never-reported"},
        embedding_probe=lambda _backend: None,
        runtime_probe=lambda _paths: None,
        query_probe=_query_schema,
    )

    assert report == {
        "ready": True,
        "cell_id": PILOT_CELL_ID,
        "workload": PILOT_WORKLOAD,
        "checks": {name: True for name in report["checks"]},
    }
    assert "present-but-never-reported" not in json.dumps(report)


def test_preflight_rejects_legacy_shared_fts_without_migrating(
    monkeypatch, tmp_path
):
    paths, _wrappers = _inputs(tmp_path)
    with sqlite3.connect(paths.kdb) as current:
        manifest_json = current.execute(
            "SELECT manifest_json FROM collections WHERE name = ?",
            (paths.collection,),
        ).fetchone()[0]
    legacy = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(legacy) as connection:
        connection.executescript(
            """
            CREATE TABLE collections (
                name TEXT PRIMARY KEY, language TEXT NOT NULL,
                fingerprint TEXT NOT NULL, manifest_json TEXT NOT NULL
            );
            CREATE TABLE chunks (
                rowid INTEGER PRIMARY KEY, collection TEXT NOT NULL,
                chunk_id TEXT NOT NULL, path TEXT NOT NULL,
                start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
                text TEXT NOT NULL, vector_json TEXT NOT NULL,
                UNIQUE(collection, chunk_id)
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                text, content='chunks', content_rowid='rowid'
            );
            """
        )
        connection.execute(
            "INSERT INTO collections VALUES (?, ?, ?, ?)",
            (paths.collection, "en", "legacy", manifest_json),
        )
    monkeypatch.setattr(
        "benchmarks.a5kernels.embeddings.PinnedBGEEmbeddings", TinyEmbeddings
    )

    report = preflight(
        replace(paths, kdb=legacy),
        environment={"OPENROUTER_API_KEY": "configured"},
        runtime_probe=lambda _paths: None,
    )

    assert report["checks"]["kdb_manifest"] is True
    assert report["checks"]["kdb_query"] is False
    assert report["ready"] is False
    with sqlite3.connect(legacy) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(collections)")
        }
    assert "fts_table" not in columns


def test_preflight_rejects_query_hit_with_corrupted_chunk_bytes(
    monkeypatch, tmp_path
):
    paths, _wrappers = _inputs(tmp_path)
    with sqlite3.connect(paths.kdb) as connection:
        connection.execute(
            "UPDATE chunks SET text = text || ? WHERE collection = ?",
            ("corrupted", paths.collection),
        )
    monkeypatch.setattr(
        "benchmarks.a5kernels.embeddings.PinnedBGEEmbeddings", TinyEmbeddings
    )

    report = preflight(
        paths,
        environment={"OPENROUTER_API_KEY": "configured"},
        runtime_probe=lambda _paths: None,
    )

    assert report["checks"]["kdb_manifest"] is True
    assert report["checks"]["kdb_query"] is False
    assert report["ready"] is False


def test_preflight_validates_corrupted_chunk_beyond_first_query_hit(
    monkeypatch, tmp_path
):
    paths, _wrappers = _inputs(tmp_path)
    text = "unrelated second chunk\n"
    chunk_id = hashlib.sha256(
        f"vector.py\0{1}\0{1}\0{text}".encode()
    ).hexdigest()
    with sqlite3.connect(paths.kdb) as connection:
        table = connection.execute(
            "SELECT fts_table FROM collections WHERE name = ?", (paths.collection,)
        ).fetchone()[0]
        cursor = connection.execute(
            """INSERT INTO chunks
               (collection, chunk_id, path, start_line, end_line, text, vector_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (paths.collection, chunk_id, "vector.py", 1, 1, text, "[1.0,-1.0]"),
        )
        connection.execute(
            f"INSERT INTO {table}(rowid, text) VALUES (?, ?)",
            (cursor.lastrowid, text),
        )
        connection.execute(
            "UPDATE chunks SET text = ? WHERE rowid = ?",
            (text + "corrupted", cursor.lastrowid),
        )
    monkeypatch.setattr(
        "benchmarks.a5kernels.embeddings.PinnedBGEEmbeddings", TinyEmbeddings
    )
    with KnowledgeDB.open_read_only(paths.kdb, TinyEmbeddings()) as database:
        first = database.query(paths.collection, "A5 preflight", limit=1)
        assert first and first[0].chunk_id != chunk_id

    report = preflight(
        paths,
        environment={"OPENROUTER_API_KEY": "configured"},
        runtime_probe=lambda _paths: None,
    )

    assert report["checks"]["kdb_manifest"] is True
    assert report["checks"]["kdb_query"] is False
    assert report["ready"] is False


def test_hugging_face_home_routes_its_hub_cache_to_embeddings(tmp_path):
    paths, _wrappers = _inputs(tmp_path)
    (paths.bge_cache / "hub").mkdir()
    observed = []
    preflight(
        paths,
        environment={"OPENROUTER_API_KEY": "configured"},
        embedding_probe=lambda backend: observed.append(backend.cache_dir),
        runtime_probe=lambda _paths: None,
        query_probe=_query_schema,
    )
    assert observed == [paths.bge_cache / "hub"]


def test_existing_results_directory_prevents_accidental_overwrite(tmp_path):
    paths, _wrappers = _inputs(tmp_path)
    paths.results_root.mkdir()
    report = preflight(
        paths,
        environment={"OPENROUTER_API_KEY": "configured"},
        embedding_probe=lambda _backend: None,
        runtime_probe=lambda _paths: None,
        query_probe=_query_schema,
    )
    assert report["ready"] is False
    assert report["checks"]["results_absent"] is False


def test_remote_runtime_failure_is_bounded_and_fails_closed(tmp_path):
    paths, _wrappers = _inputs(tmp_path)

    def unavailable(_paths):
        raise RuntimeError("remote secret path and transport details")

    report = preflight(
        paths,
        environment={"OPENROUTER_API_KEY": "configured"},
        embedding_probe=lambda _backend: None,
        runtime_probe=unavailable,
        query_probe=_query_schema,
    )

    assert report["ready"] is False
    assert report["checks"]["catlass_runtime"] is False
    assert "remote secret" not in json.dumps(report)


def test_runtime_probe_binds_exact_source_and_revision(monkeypatch, tmp_path):
    paths, _wrappers = _inputs(tmp_path)
    captured = {}

    class FakeExecutor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        runtime_provenance = (("catlass_revision", "a" * 40),)

    monkeypatch.setattr(
        "benchmarks.a5kernels.bz.CatlassValidationExecutor", FakeExecutor
    )
    _probe_catlass_runtime(paths)

    assert captured["catlass_source"] == "/retained/catlass"
    assert captured["catlass_revision"] == "a" * 40
    assert captured["validation_wrapper"].endswith("catlass-validation.sh")
    assert captured["upload_wrapper"].endswith("upload.sh")


def test_missing_provenance_wrapper_prevents_remote_probe(tmp_path):
    paths, _wrappers = _inputs(tmp_path)
    paths.provenance_wrapper.unlink()
    calls = []
    report = preflight(
        paths,
        environment={"OPENROUTER_API_KEY": "configured"},
        embedding_probe=lambda _backend: None,
        runtime_probe=lambda _paths: calls.append(True),
        query_probe=_query_schema,
    )

    assert report["checks"]["catlass_provenance_wrapper"] is False
    assert report["checks"]["catlass_runtime"] is False
    assert calls == []


def test_run_pilot_does_not_construct_actor_after_remote_preflight_failure(
    monkeypatch, tmp_path
):
    paths, _wrappers = _inputs(tmp_path)
    monkeypatch.setattr(
        "benchmarks.a5kernels.production.preflight",
        lambda _paths: {"ready": False, "checks": {"catlass_runtime": False}},
    )
    monkeypatch.setattr(
        "benchmarks.a5kernels.production.build_production_trial",
        lambda _paths: pytest.fail("Actor runtime must not be constructed"),
    )

    with pytest.raises(RuntimeError, match="catlass_runtime"):
        run_pilot(paths)


def test_production_composition_uses_read_only_kdb_and_no_guidance(tmp_path):
    paths, _wrappers = _inputs(tmp_path)
    production = build_production_trial(paths)
    orchestrator = production._orchestrator
    cell = next(
        cell
        for cell in __import__(
            "benchmarks.a5kernels.matrix", fromlist=["initial_matrix"]
        ).initial_matrix().cells
        if cell.cell_id == PILOT_CELL_ID
    )
    knowledge = orchestrator.knowledge_factory(tmp_path / "memory", cell)
    profiling = orchestrator.profiling_factory(cell)

    assert knowledge.enabled is True
    assert knowledge.collection == "catlass-pilot"
    with pytest.raises(Exception):
        knowledge.database.connection.execute("DELETE FROM chunks")
    assert profiling._controller._enabled is False
    assert orchestrator.backend._device == paths.device
    # Final evaluation remains wired even though intermediate guidance is off.
    assert profiling._controller._backend._catlass_source == "/retained/catlass"
    knowledge.close()
    production._database.close()


def test_production_database_closes_when_orchestrator_raises(tmp_path):
    paths, _wrappers = _inputs(tmp_path)
    production = build_production_trial(paths)
    production._orchestrator.run = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("actor failed")
    )

    with pytest.raises(RuntimeError, match="actor failed"):
        production.run(object(), object())
    with pytest.raises(Exception):
        production._database.connection.execute("SELECT 1")
    with pytest.raises(RuntimeError, match="one-shot"):
        production.run(object(), object())


def test_run_pilot_routes_only_fixed_cell_and_workload(monkeypatch, tmp_path):
    paths, _wrappers = _inputs(tmp_path)
    calls = []

    class FakeOrchestrator:
        def run(self, cell, workload):
            calls.append((cell.cell_id, workload.value))
            return SimpleNamespace(
                verified=SimpleNamespace(passed=True, attestation_sha256="a" * 64),
                outcome=SimpleNamespace(
                    status="done", iterations=4, tokens=None, wall_time_s=1.25),
                evidence_sha256="e" * 64,
                final_profile={"duration_us": 2.5},
                workspace=tmp_path / "workspace",
            )

    monkeypatch.setattr(
        "benchmarks.a5kernels.production.preflight", lambda _paths: {"ready": True}
    )
    monkeypatch.setattr(
        "benchmarks.a5kernels.production.build_production_trial",
        lambda _paths: FakeOrchestrator(),
    )
    result = run_pilot(paths)

    assert calls == [(PILOT_CELL_ID, PILOT_WORKLOAD)]
    assert result["passed"] is True
    assert result["final_profile"] == {"duration_us": 2.5}
    from benchmarks.a5kernels.matrix import RunMetrics

    metrics = RunMetrics.from_mapping(result)
    assert metrics.correct is True
    assert metrics.kernel_time_us == 2.5
    assert metrics.exploration_succeeded is True
    assert metrics.tokens is None
