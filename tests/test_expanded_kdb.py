"""Production ingestion and gated retrieval; no model download required."""

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.a5kernels.expanded_kdb import ingest, query
from benchmarks.a5kernels.knowledge import CollectionManifest, KnowledgeDB


class Embeddings:
    model = "test-model"
    revision = "test-revision"

    def __init__(self):
        self.batches = []

    def embed(self, texts):
        self.batches.append(len(texts))
        return [[1., float("vector" in text), float("matrix" in text)] for text in texts]


def inventory(tmp_path, *, names=("ops-nn", "triton-kernels")):
    sources = []
    for name in names:
        root = tmp_path / name
        root.mkdir()
        data = ("vector add kernel matrix\n" * 201).encode()
        (root / "kernel.cpp").write_bytes(data)
        sources.append({"name": name, "root": str(root), "status": "available",
                        "repository": f"https://gitcode.com/Ascend/{name}.git",
                        "revision": "a" * 40, "licenses": [],
                        "files": [{"path": "kernel.cpp", "sha256": hashlib.sha256(data).hexdigest(),
                                   "bytes": len(data)}]})
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps({"schema_version": 1, "sources": sources}))
    return path


def test_batch_ingestion_and_multiple_gated_collections_with_provenance(tmp_path):
    embeddings = Embeddings()
    path = inventory(tmp_path)
    db_path = tmp_path / "kdb.sqlite3"
    report = ingest(path, db_path, embeddings, batch_size=2)
    assert len(report["collections"]) == 2
    assert max(embeddings.batches) <= 2
    assert sum(embeddings.batches) == 8
    assert not db_path.with_suffix(".sqlite3.staging").exists()
    result = query(db_path, embeddings, "vector", tmp_path / "journal", limit=10)
    assert {hit["citation"]["collection"] for hit in result["results"]} == {
        entry["collection"] for entry in report["collections"]}
    for hit in result["results"]:
        citation = hit["citation"]
        assert citation["revision"] == "a" * 40
        assert citation["repository"].startswith("https://gitcode.com/")
        assert citation["source_sha256"] == hashlib.sha256(
            (tmp_path / "ops-nn" / "kernel.cpp").read_bytes()).hexdigest()
        assert hit["vector_rank"] is not None
        assert citation["start_line"] <= citation["end_line"]
    scoped = query(db_path, embeddings, "vector", tmp_path / "scope-journal",
                   kernel_language="triton")
    assert len(scoped["collections"]) == 1
    assert scoped["collections"][0].startswith("triton-")
    with pytest.raises(ValueError, match="no collections"):
        query(db_path, embeddings, "vector", tmp_path / "empty-journal",
              kernel_language="catlass-dsl")


def test_legacy_manifest_roundtrip_does_not_invent_kernel_language(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    path = root / "guide.py"
    path.write_text("vector add\n")
    with KnowledgeDB(tmp_path / "legacy.sqlite3", Embeddings()) as database:
        manifest = database.index(root, [path], collection="old-en", language="en")
        legacy_json = manifest.to_json()
        assert "kernel_languages" not in legacy_json
        assert "repository" not in legacy_json
        restored = CollectionManifest.from_json(legacy_json)
        assert restored.kernel_languages == ()
        assert restored.to_json() == legacy_json
        assert database.query("old-en", "vector")[0].text == "vector add\n"


def test_unverified_source_never_publishes_over_existing_index(tmp_path):
    path = inventory(tmp_path)
    db = tmp_path / "published.sqlite3"
    db.write_bytes(b"retained prior index")
    (tmp_path / "ops-nn" / "kernel.cpp").write_text("modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        ingest(path, db, Embeddings())
    assert db.read_bytes() == b"retained prior index"


@pytest.mark.parametrize("remove", [False, True])
def test_ingestion_report_pins_inventory_bytes_read_before_indexing(tmp_path, remove):
    path = inventory(tmp_path, names=("ops-nn",))
    original = path.read_bytes()
    class ChangedInventory(Embeddings):
        def embed(self, texts):
            if path.exists():
                if remove:
                    path.unlink()
                else:
                    path.write_text('{"schema_version":1,"sources":[]}')
            return super().embed(texts)
    db = tmp_path / "published.sqlite3"
    report = ingest(path, db, ChangedInventory())
    assert report["inventory_sha256"] == hashlib.sha256(original).hexdigest()
    assert json.loads(db.with_suffix(".sqlite3.inventory.json").read_bytes()) == report
    with KnowledgeDB.open_read_only(db, Embeddings()) as database:
        assert len(database.manifest(report["collections"][0]["collection"]).sources) == 1


def test_source_change_between_verification_and_index_snapshot_never_publishes(tmp_path, monkeypatch):
    path = inventory(tmp_path, names=("ops-nn",))
    db = tmp_path / "published.sqlite3"
    original = b"retained prior index"
    db.write_bytes(original)
    original_index = KnowledgeDB.index

    def changed_index(database, root, paths, **kwargs):
        (root / "kernel.cpp").write_text("changed vector kernel\n")
        return original_index(database, root, paths, **kwargs)

    monkeypatch.setattr(KnowledgeDB, "index", changed_index)
    with pytest.raises(ValueError, match="changed after inventory verification.*new DB path"):
        ingest(path, db, Embeddings())
    assert db.read_bytes() == original
    assert not db.with_suffix(".sqlite3.inventory.json").exists()
    # Retain staging evidence; do not silently delete the mismatching collection.
    assert db.with_suffix(".sqlite3.staging").exists()


def test_failed_embedding_rolls_back_collection_and_resume_uses_completed_collection(tmp_path):
    path = inventory(tmp_path)
    db = tmp_path / "kdb.sqlite3"

    class Interrupt(Embeddings):
        def embed(self, texts):
            if len(self.batches) == 2:
                raise RuntimeError("interrupted second collection")
            return super().embed(texts)

    with pytest.raises(RuntimeError, match="interrupted"):
        ingest(path, db, Interrupt(), batch_size=2)
    assert not db.exists()
    embeddings = Embeddings()
    report = ingest(path, db, embeddings, batch_size=2)
    assert len(report["collections"]) == 2
    assert sum(embeddings.batches) == 4


def test_reference_exposure_and_document_language_are_independent(tmp_path):
    path = inventory(tmp_path, names=("cannbench",))
    raw = json.loads(path.read_text())
    raw["sources"][0].update(document_language="zh", kernel_languages=["ascend-c"],
                             reference_exposed=True)
    path.write_text(json.dumps(raw))
    db_path = tmp_path / "kdb.sqlite3"
    report = ingest(path, db_path, Embeddings())
    with KnowledgeDB.open_read_only(db_path, Embeddings()) as database:
        manifest = database.manifest(report["collections"][0]["collection"])
        assert manifest.language == "zh"
        assert manifest.kernel_languages == ("ascend-c",)
    hits = query(db_path, Embeddings(), "vector", tmp_path / "journal")
    assert hits["results"][0]["citation"]["reference_exposed"] is True


def test_repeated_citation_checks_reuse_immutable_manifest(tmp_path):
    path = inventory(tmp_path, names=("ops-nn",))
    db_path = tmp_path / "kdb.sqlite3"
    report = ingest(path, db_path, Embeddings())
    collection = report["collections"][0]["collection"]
    with KnowledgeDB.open_read_only(db_path, Embeddings()) as database:
        statements = []
        database.connection.set_trace_callback(statements.append)
        hits = database.query(collection, "vector")
        for _ in range(3):
            for hit in hits:
                database.validate_citation(collection, chunk_id=hit.chunk_id, path=hit.path,
                                           start_line=hit.start_line, end_line=hit.end_line)
        manifest_reads = [statement for statement in statements
                          if statement.startswith("SELECT manifest_json FROM collections")]
        assert len(manifest_reads) == 1
