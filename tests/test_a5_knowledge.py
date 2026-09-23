from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from benchmarks.a5kernels.knowledge import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    KnowledgeDB,
    chunk_source,
)


class FakeEmbeddings:
    model = DEFAULT_EMBEDDING_MODEL
    revision = DEFAULT_EMBEDDING_REVISION

    def embed(self, texts):
        # Stable and dependency-free; dimensions roughly represent kernel terms.
        terms = ("vector", "matrix", "pipeline")
        return [[float(text.lower().count(term)) for term in terms] for text in texts]


def test_chunk_source_uses_80_lines_with_20_line_overlap():
    data = "".join(f"line {number}\n" for number in range(1, 142)).encode()

    chunks = chunk_source("src/kernel.py", data)

    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [
        (1, 80),
        (61, 140),
        (121, 141),
    ]
    assert chunks[0].text.splitlines()[-1] == "line 80"
    assert chunks[1].text.splitlines()[0] == "line 61"
    assert chunk_source("src/kernel.py", data) == chunks


def _write_sources(root: Path) -> tuple[Path, Path]:
    first = root / "vector.py"
    second = root / "matrix.py"
    first.write_text("vector add kernel\n" * 85, encoding="utf-8")
    second.write_text("matrix multiply pipeline\n" * 10, encoding="utf-8")
    return first, second


def test_index_manifest_is_stable_and_collection_is_immutable(tmp_path):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    sources = _write_sources(source_root)
    manifest_path = tmp_path / "manifest.json"

    with KnowledgeDB(tmp_path / "kdb.sqlite", FakeEmbeddings()) as database:
        manifest = database.index(
            source_root,
            reversed(sources),
            collection="catlass-v1",
            language="catlass",
            manifest_path=manifest_path,
        )
        repeated = database.index(
            source_root,
            sources,
            collection="catlass-v1",
            language="catlass",
        )
        assert repeated == manifest
        assert manifest.sources[0].path == "matrix.py"
        assert manifest.embedding_revision == DEFAULT_EMBEDDING_REVISION
        assert manifest_path.read_text() == manifest.to_json()

        sources[0].write_text("changed\n", encoding="utf-8")
        with pytest.raises(ValueError, match="immutable"):
            database.index(
                source_root,
                sources,
                collection="catlass-v1",
                language="catlass",
            )


def test_hybrid_query_is_language_isolated_and_repeatable(tmp_path):
    catlass = tmp_path / "catlass"
    ascendc = tmp_path / "ascendc"
    catlass.mkdir()
    ascendc.mkdir()
    catlass_file = catlass / "guide.txt"
    ascendc_file = ascendc / "guide.txt"
    catlass_file.write_text("vector vector kernel\n", encoding="utf-8")
    ascendc_file.write_text("vector secret ascend c\n", encoding="utf-8")

    with KnowledgeDB(tmp_path / "kdb.sqlite", FakeEmbeddings()) as database:
        database.index(catlass, [catlass_file], collection="catlass", language="catlass")
        database.index(ascendc, [ascendc_file], collection="ascendc", language="ascendc")

        first = database.query("catlass", "how does vector-add work?", limit=5)
        second = database.query("catlass", "vector", limit=5)

    assert first[0].chunk_id == second[0].chunk_id
    assert [hit.path for hit in first] == ["guide.txt"]
    assert "secret" not in first[0].text
    assert first[0].lexical_rank == 1
    assert first[0].vector_rank == 1


def test_vector_ties_use_chunk_hash_as_stable_tiebreaker(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    sources = []
    for name in ("z.txt", "a.txt"):
        path = root / name
        path.write_text("unrelated\n", encoding="utf-8")
        sources.append(path)

    with KnowledgeDB(tmp_path / "kdb.sqlite", FakeEmbeddings()) as database:
        database.index(root, sources, collection="docs", language="catlass")
        hits = database.query("docs", "vector", limit=2)

    assert [hit.chunk_id for hit in hits] == sorted(hit.chunk_id for hit in hits)
    expected = hashlib.sha256(("a.txt\0" + "1\0" + "1\0unrelated\n").encode()).hexdigest()
    assert expected in {hit.chunk_id for hit in hits}
