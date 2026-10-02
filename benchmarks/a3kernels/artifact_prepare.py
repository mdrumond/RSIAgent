"""Transactional preparation of the pinned A3 Phase 1 knowledge artifacts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

from .corpus import (
    CorpusFetcher,
    CorpusSpec,
    fetch_pinned_git_files,
    load_documents,
    prepare_corpus,
)
from .embeddings import PinnedBGEEmbeddings
from .knowledge import (
    CollectionManifest,
    EmbeddingBackend,
    KnowledgeDB,
    SourceFingerprint,
)


EmbeddingFactory = Callable[[Path], EmbeddingBackend]
DEFAULT_CORPUS_SPEC = Path(__file__).with_name("corpora") / "a3-ascendc-en.json"
_VERIFY_QUERY = "Ascend C DataCopy vector add pipeline"


def _default_embeddings(cache: Path) -> EmbeddingBackend:
    return PinnedBGEEmbeddings(cache_dir=cache, local_files_only=True)


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _expected_sources(spec: CorpusSpec) -> tuple[SourceFingerprint, ...]:
    return tuple(
        SourceFingerprint(item.path, item.sha256, spec.repository, spec.revision)
        for item in spec.files
    )


def validate_manifest_contract(
    spec: CorpusSpec,
    manifest: CollectionManifest,
    embeddings: EmbeddingBackend,
) -> None:
    """Bind a manifest to the exact registered corpus and embedding backend."""
    if (
        manifest.collection != spec.collection
        or manifest.target != "a3"
        or manifest.sources != _expected_sources(spec)
        or (manifest.embedding_model, manifest.embedding_revision, manifest.dimension)
        != (embeddings.model, embeddings.revision, embeddings.dimension)
    ):
        raise ValueError("published A3 manifest does not match pinned inputs")


def _verify(
    spec: CorpusSpec,
    corpus_artifacts: Path,
    database_path: Path,
    manifest_path: Path,
    embeddings: EmbeddingBackend,
) -> CollectionManifest:
    documents = load_documents(spec, corpus_artifacts)
    manifest = CollectionManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    validate_manifest_contract(spec, manifest, embeddings)
    with KnowledgeDB.open_read_only(database_path, embeddings) as database:
        check = database.connection.execute("PRAGMA integrity_check").fetchone()
        if check is None or check[0] != "ok":
            raise ValueError("published A3 knowledge database failed integrity check")
        if database.manifest(spec.collection) != manifest:
            raise ValueError("published A3 database and manifest do not match")
        rows = database.connection.execute(
            "SELECT DISTINCT path, source_revision, content_sha256 FROM chunks "
            "WHERE collection = ?", (spec.collection,),
        ).fetchall()
        provenance = {
            (row["path"], row["source_revision"], row["content_sha256"])
            for row in rows
        }
        expected = {
            (item.path, item.source_revision, item.content_sha256)
            for item in documents
        }
        if not rows or provenance != expected:
            raise ValueError("published A3 database has incomplete source provenance")
        database.query(spec.collection, _VERIFY_QUERY, limit=1)
    return manifest


def _publish_file(staged: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _exists(destination):
        raise FileExistsError(f"artifact appeared during publication: {destination}")
    with staged.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(staged, destination)
    directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def prepare_phase1_artifacts(
    *,
    embedding_cache: Path,
    corpus_artifacts: Path,
    knowledge_database: Path,
    knowledge_manifest: Path,
    corpus_spec_path: Path = DEFAULT_CORPUS_SPEC,
    fetcher: CorpusFetcher = fetch_pinned_git_files,
    embedding_factory: EmbeddingFactory = _default_embeddings,
) -> dict[str, object]:
    """Prepare and verify the immutable, local-only A3 corpus and KDB.

    The database is published first and the atomically replaced manifest is its
    commit marker.  An interrupted pair is never repaired or silently replaced;
    a later invocation fails closed so an operator can inspect the artifact.
    """
    spec = CorpusSpec.load(corpus_spec_path)
    database_path = knowledge_database.resolve()
    manifest_path = knowledge_manifest.resolve()
    database_exists, manifest_exists = _exists(database_path), _exists(manifest_path)
    if database_exists != manifest_exists:
        raise ValueError("partial artifact publication: database and manifest must coexist")

    embeddings = embedding_factory(embedding_cache)
    if database_exists:
        manifest = _verify(
            spec, corpus_artifacts, database_path, manifest_path, embeddings,
        )
        reused = True
    else:
        prepare_corpus(spec, corpus_artifacts, fetcher=fetcher)
        documents = load_documents(spec, corpus_artifacts)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            tempfile.TemporaryDirectory(
                prefix=".a3-kdb-", dir=database_path.parent,
            ) as database_temporary,
            tempfile.TemporaryDirectory(
                prefix=".a3-manifest-", dir=manifest_path.parent,
            ) as manifest_temporary,
        ):
            staged_database = Path(database_temporary) / "knowledge.sqlite3"
            with KnowledgeDB.create(staged_database, embeddings) as database:
                manifest = database.index(documents)
                database.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            staged_manifest = Path(manifest_temporary) / "manifest.json"
            staged_manifest.write_text(manifest.to_json(), encoding="utf-8")
            _publish_file(staged_database, database_path)
            _publish_file(staged_manifest, manifest_path)
        manifest = _verify(
            spec, corpus_artifacts, database_path, manifest_path, embeddings,
        )
        reused = False

    return {
        "schema": "a3-phase1-artifacts-v1",
        "ready": True,
        "reused": reused,
        "collection": manifest.collection,
        "fingerprint": manifest.fingerprint,
        "embedding_model": manifest.embedding_model,
        "embedding_revision": manifest.embedding_revision,
        "source_count": len(manifest.sources),
    }


__all__ = [
    "DEFAULT_CORPUS_SPEC", "prepare_phase1_artifacts", "validate_manifest_contract",
]
