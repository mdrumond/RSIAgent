from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from benchmarks.a3kernels.artifact_prepare import (
    prepare_phase1_artifacts,
    validate_manifest_contract,
)
from benchmarks.a3kernels.corpus import CorpusSpec
from benchmarks.a3kernels.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
)
from benchmarks.a3kernels.knowledge import (
    CollectionManifest, KnowledgeDB, SourceFingerprint,
)


class FakeEmbeddings:
    model = DEFAULT_EMBEDDING_MODEL
    revision = DEFAULT_EMBEDDING_REVISION
    dimension = 3

    def embed_documents(self, texts):
        terms = ("datacopy", "vector", "pipeline")
        return [
            [float(text.casefold().count(term)) for term in terms]
            for text in texts
        ]

    def embed_queries(self, texts):
        return self.embed_documents(texts)


def fixture_spec(tmp_path):
    values = {
        "docs/data-copy.md": b"DataCopy moves tensor values.\n",
        "kernels/add.cpp": b"vector add pipeline\n",
    }
    raw = {
        "name": "a3-fixture-en",
        "collection": "a3-fixture-en-v1",
        "target": "a3",
        "language": "en",
        "repository": "https://example.invalid/a3.git",
        "revision": "a" * 40,
        "files": [
            {"path": path, "sha256": hashlib.sha256(data).hexdigest()}
            for path, data in values.items()
        ],
    }
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path, values


def paths(tmp_path):
    return {
        "embedding_cache": tmp_path / "embeddings",
        "corpus_artifacts": tmp_path / "corpus",
        "knowledge_database": tmp_path / "database" / "knowledge.sqlite3",
        "knowledge_manifest": tmp_path / "manifest" / "manifest.json",
    }


def test_offline_prepare_builds_reopens_queries_and_records_provenance(tmp_path):
    spec_path, values = fixture_spec(tmp_path)
    config = paths(tmp_path)
    config["embedding_cache"].mkdir()
    fetched = []
    factories = []

    def fetch(repository, revision, requested):
        fetched.append((repository, revision, requested))
        return values

    def embeddings(cache):
        factories.append(cache)
        return FakeEmbeddings()

    report = prepare_phase1_artifacts(
        **config, corpus_spec_path=spec_path,
        fetcher=fetch, embedding_factory=embeddings,
    )

    assert report["schema"] == "a3-phase1-artifacts-v1"
    assert report["reused"] is False
    assert report["source_count"] == 2
    assert fetched == [(
        "https://example.invalid/a3.git", "a" * 40,
        ("docs/data-copy.md", "kernels/add.cpp"),
    )]
    assert factories == [config["embedding_cache"]]
    manifest = CollectionManifest.from_json(
        config["knowledge_manifest"].read_text(encoding="utf-8")
    )
    assert report["fingerprint"] == manifest.fingerprint
    with KnowledgeDB.open_read_only(config["knowledge_database"], FakeEmbeddings()) as db:
        hits = db.query(manifest.collection, "DataCopy vector", limit=2)
    assert {hit.path for hit in hits} == set(values)
    assert all(hit.source_revision == "a" * 40 for hit in hits)
    assert {hit.content_sha256 for hit in hits} == {
        hashlib.sha256(value).hexdigest() for value in values.values()
    }


def test_verified_reuse_does_not_fetch_or_rebuild(tmp_path):
    spec_path, values = fixture_spec(tmp_path)
    config = paths(tmp_path)
    config["embedding_cache"].mkdir()
    first = prepare_phase1_artifacts(
        **config, corpus_spec_path=spec_path,
        fetcher=lambda *_: values, embedding_factory=lambda _cache: FakeEmbeddings(),
    )
    before = {
        name: config[name].read_bytes()
        for name in ("knowledge_database", "knowledge_manifest")
    }
    second = prepare_phase1_artifacts(
        **config, corpus_spec_path=spec_path,
        fetcher=lambda *_: (_ for _ in ()).throw(AssertionError("must stay offline")),
        embedding_factory=lambda _cache: FakeEmbeddings(),
    )
    assert second == {**first, "reused": True}
    assert before == {
        name: config[name].read_bytes()
        for name in ("knowledge_database", "knowledge_manifest")
    }


def test_manifest_contract_rejects_self_consistent_but_unregistered_sources(tmp_path):
    spec_path, values = fixture_spec(tmp_path)
    config = paths(tmp_path)
    config["embedding_cache"].mkdir()
    prepare_phase1_artifacts(
        **config, corpus_spec_path=spec_path,
        fetcher=lambda *_: values, embedding_factory=lambda _cache: FakeEmbeddings(),
    )
    manifest = CollectionManifest.from_json(
        config["knowledge_manifest"].read_text(encoding="utf-8")
    )
    foreign = replace(
        manifest,
        sources=(
            SourceFingerprint("foreign.cpp", "f" * 64, "https://other.invalid/x", "b" * 40),
        ),
    )
    with pytest.raises(ValueError, match="pinned inputs"):
        validate_manifest_contract(CorpusSpec.load(spec_path), foreign, FakeEmbeddings())


@pytest.mark.parametrize("remaining", ["database", "manifest"])
def test_partial_publication_fails_closed(tmp_path, remaining):
    spec_path, _values = fixture_spec(tmp_path)
    config = paths(tmp_path)
    config["embedding_cache"].mkdir()
    selected = (
        config["knowledge_database"] if remaining == "database"
        else config["knowledge_manifest"]
    )
    selected.parent.mkdir()
    selected.write_bytes(b"partial")

    with pytest.raises(ValueError, match="partial artifact publication"):
        prepare_phase1_artifacts(
            **config, corpus_spec_path=spec_path,
            fetcher=lambda *_: (_ for _ in ()).throw(AssertionError("must not fetch")),
            embedding_factory=lambda _cache: FakeEmbeddings(),
        )
    assert selected.read_bytes() == b"partial"


@pytest.mark.parametrize("corruption", ["manifest", "database", "corpus"])
def test_reuse_rejects_corrupt_or_mismatched_artifacts(tmp_path, corruption):
    spec_path, values = fixture_spec(tmp_path)
    config = paths(tmp_path)
    config["embedding_cache"].mkdir()
    prepare_phase1_artifacts(
        **config, corpus_spec_path=spec_path,
        fetcher=lambda *_: values, embedding_factory=lambda _cache: FakeEmbeddings(),
    )
    spec = CorpusSpec.load(spec_path)
    if corruption == "manifest":
        config["knowledge_manifest"].write_text("{}", encoding="utf-8")
    elif corruption == "database":
        config["knowledge_database"].write_bytes(b"not sqlite")
    else:
        (spec.source_root(config["corpus_artifacts"]) / "kernels/add.cpp").write_text(
            "changed", encoding="utf-8"
        )

    with pytest.raises(Exception):
        prepare_phase1_artifacts(
            **config, corpus_spec_path=spec_path,
            fetcher=lambda *_: (_ for _ in ()).throw(AssertionError("must not fetch")),
            embedding_factory=lambda _cache: FakeEmbeddings(),
        )


def test_cli_prepare_routes_paths_and_emits_json(monkeypatch, capsys, tmp_path):
    import run_a3_phase1

    captured = {}
    monkeypatch.setattr(
        run_a3_phase1, "prepare_phase1_artifacts",
        lambda **kwargs: captured.update(kwargs) or {"ready": True},
    )
    argv = ["prepare"]
    for option, value in paths(tmp_path).items():
        argv.extend(("--" + option.replace("_", "-"), str(value)))
    assert run_a3_phase1.main(argv) == 0
    assert json.loads(capsys.readouterr().out) == {"ready": True}
    assert captured == paths(tmp_path)
