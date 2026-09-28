from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.knowledge import KnowledgeDB
from benchmarks.a5kernels.production import (
    PILOT_CELL_ID,
    PILOT_WORKLOAD,
    ProductionPaths,
    build_production_trial,
    preflight,
    run_pilot,
)


class TinyEmbeddings:
    model = "BAAI/bge-small-en-v1.5"
    revision = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

    def embed(self, texts):
        return [[1.0, float(index + 1)] for index, _text in enumerate(texts)]

    embed_query = embed


def _inputs(tmp_path: Path) -> tuple[ProductionPaths, list[Path]]:
    tla = tmp_path / "tla"
    profile = tmp_path / "profiling"
    wrappers = [
        tla / "execution-profiles/catlass-validation.sh",
        tla / "execution-profiles/bz-a5/session.sh",
        tla / "execution-profiles/bz-a5/upload.sh",
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
    )

    assert report["ready"] is False
    assert report["checks"]["openrouter_credential"] is False
    assert report["checks"]["kdb_manifest"] is True
    assert probed == [(TinyEmbeddings.model, TinyEmbeddings.revision)]
    assert "OPENROUTER_API_KEY" not in json.dumps(report)


def test_preflight_checks_complete_local_inputs_without_model_or_bz_call(tmp_path):
    paths, _wrappers = _inputs(tmp_path)
    report = preflight(
        paths,
        environment={"OPENROUTER_API_KEY": "present-but-never-reported"},
        embedding_probe=lambda _backend: None,
    )

    assert report == {
        "ready": True,
        "cell_id": PILOT_CELL_ID,
        "workload": PILOT_WORKLOAD,
        "checks": {name: True for name in report["checks"]},
    }
    assert "present-but-never-reported" not in json.dumps(report)


def test_hugging_face_home_routes_its_hub_cache_to_embeddings(tmp_path):
    paths, _wrappers = _inputs(tmp_path)
    (paths.bge_cache / "hub").mkdir()
    observed = []
    preflight(
        paths,
        environment={"OPENROUTER_API_KEY": "configured"},
        embedding_probe=lambda backend: observed.append(backend.cache_dir),
    )
    assert observed == [paths.bge_cache / "hub"]


def test_existing_results_directory_prevents_accidental_overwrite(tmp_path):
    paths, _wrappers = _inputs(tmp_path)
    paths.results_root.mkdir()
    report = preflight(
        paths,
        environment={"OPENROUTER_API_KEY": "configured"},
        embedding_probe=lambda _backend: None,
    )
    assert report["ready"] is False
    assert report["checks"]["results_absent"] is False


def test_production_composition_uses_read_only_kdb_and_no_guidance(tmp_path):
    paths, _wrappers = _inputs(tmp_path)
    orchestrator = build_production_trial(paths)
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
    # Final evaluation remains wired even though intermediate guidance is off.
    assert profiling._controller._backend._catlass_source == "/retained/catlass"
    knowledge.close()


def test_run_pilot_routes_only_fixed_cell_and_workload(monkeypatch, tmp_path):
    paths, _wrappers = _inputs(tmp_path)
    calls = []

    class FakeOrchestrator:
        def run(self, cell, workload):
            calls.append((cell.cell_id, workload.value))
            return SimpleNamespace(
                verified=SimpleNamespace(passed=True, attestation_sha256="a" * 64),
                outcome=SimpleNamespace(iterations=4, tokens=None, wall_time_s=1.25),
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
