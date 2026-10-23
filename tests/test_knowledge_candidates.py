"""Observable candidate selection across independent reference and memory stores."""
from pathlib import Path

import pytest

from benchmarks.a5kernels.knowledge import (
    DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_REVISION, KnowledgeDB,
)
from benchmarks.a5kernels.knowledge_agent import (
    KnowledgeAgent, KnowledgeQuery, ProgressiveMemoryJournal,
)


class Embeddings:
    model = DEFAULT_EMBEDDING_MODEL
    revision = DEFAULT_EMBEDDING_REVISION

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[text.count("cube"), text.count("add")] for text in texts]


def indexed(tmp_path, name, documents, backend):
    root = tmp_path / name
    root.mkdir()
    sources = []
    for filename, text in documents.items():
        source = root / filename
        source.write_text(text)
        sources.append(source)
    db = KnowledgeDB(tmp_path / (name + ".sqlite"), backend)
    db.index(root, sources, collection=name, language="ascendc")
    return db


def agent(tmp_path, stores):
    return KnowledgeAgent(enabled=True, databases=stores,
                          journal=ProgressiveMemoryJournal(tmp_path / "queries.jsonl"))


def test_small_memory_does_not_outrank_stronger_global_vectors(tmp_path):
    backend = Embeddings()
    refs = indexed(tmp_path, "references", {
        f"cube_{n:03}.cpp": f"cube matmul example {n}\n" for n in range(120)
    }, backend)
    mem = indexed(tmp_path, "memory", {"lesson.txt": "cube add add add lesson\n"}, backend)
    gate = agent(tmp_path, {"memory": mem, "references": refs})
    pool = gate.candidate_pool(KnowledgeQuery("cube"))
    assert len(pool.results) == 100
    vector_slots = [0, 1, 3, 5, 6, 8]
    assert all(pool.results[i].citation.collection == "references" for i in vector_slots)
    assert sum(row.citation.collection == "memory" for row in pool.page()) <= 1
    assert all(pool.results[i].score == pytest.approx(1) for i in vector_slots)
    assert [pool.results[i].vector_rank for i in vector_slots] == [1, 2, 3, 5, 6, 8]
    assert pool.results[2].citation.collection == "memory"
    assert sum(row.citation.collection == "memory" for row in pool.results) == 1
    assert len(gate.query(KnowledgeQuery("cube", limit=20))) == 20
    gate.close()
    # The gate owns views, not its caller's open handles.
    assert refs.manifest("references").collection == "references"
    refs.close()
    mem.close()


def test_keyword_lane_keeps_exact_identifier_with_weak_vector_match(tmp_path):
    backend = Embeddings()
    refs = indexed(tmp_path, "references", {
        **{f"cube_{n:03}.cpp": f"cube example {n}\n" for n in range(20)},
        "identifier.cpp": "rare_api_identifier implementation\n",
    }, backend)
    gate = agent(tmp_path, {"references": refs})
    pool = gate.candidate_pool(KnowledgeQuery("cube rare_api_identifier"), size=10)
    exact = next(row for row in pool.results if row.citation.path == "identifier.cpp")
    assert exact.score == 0
    assert exact.lexical_rank == 1
    assert exact.vector_rank is None
    assert pool.results.index(exact) == 2
    default_results = gate.query(KnowledgeQuery("cube rare_api_identifier"))
    assert len(default_results) == 5
    assert any(row.citation.path == "identifier.cpp" for row in default_results)
    gate.close()
    refs.close()


def test_identical_and_overlapping_chunks_are_not_repeated(tmp_path):
    backend = Embeddings()
    body = "".join(f"cube source line {n}\n" for n in range(150))
    refs = indexed(tmp_path, "references", {"long.cpp": body,
                                          "duplicate_a.cpp": "cube identical\n",
                                          "duplicate_b.cpp": "cube identical\n"}, backend)
    mirror = indexed(tmp_path, "mirror", {"mirror.cpp": "cube identical\n"}, backend)
    gate = agent(tmp_path, {"references": refs, "mirror": mirror})
    pool = gate.candidate_pool(KnowledgeQuery("cube"))
    assert sum(row.text == "cube identical\n" for row in pool.results) == 1
    spans = [(row.citation.start_line, row.citation.end_line) for row in pool.results
             if row.citation.path == "long.cpp"]
    assert spans
    assert all(end < other_start or other_end < start for i, (start, end) in enumerate(spans)
               for other_start, other_end in spans[i + 1:])
    for row in pool.results:
        db = gate.databases[row.citation.collection]
        c = row.citation
        db.validate_citation(c.collection, chunk_id=c.chunk_hash, path=c.path,
                            start_line=c.start_line, end_line=c.end_line)
    gate.close()
    refs.close()
    mirror.close()


def test_paging_is_stable_without_retrieval_and_fills_available_slots(tmp_path, monkeypatch):
    backend = Embeddings()
    refs = indexed(tmp_path, "references", {
        f"cube_{n:03}.cpp": f"cube example {n}\n" for n in range(15)
    }, backend)
    gate = agent(tmp_path, {"references": refs})
    pool = gate.candidate_pool(KnowledgeQuery("unmatched"), size=10)
    calls = backend.calls
    monkeypatch.setattr(gate.databases["references"], "ranked_candidates",
                        lambda *a, **kw: pytest.fail("paging rescanned the database"))
    assert pool.page(0, 5) + pool.page(5, 5) == pool.results
    assert pool.page(10) == ()
    assert backend.calls == calls
    assert len(pool.results) == 10
    gate.close()
    refs.close()


def test_scan_obeys_elapsed_budget_callback(tmp_path):
    backend = Embeddings()
    refs = indexed(tmp_path, "references", {"cube.cpp": "cube\n"}, backend)
    gate = agent(tmp_path, {"references": refs})
    checks = []

    def check():
        checks.append(True)
        if len(checks) == 3:
            raise TimeoutError("allocation exhausted during scan")

    with pytest.raises(TimeoutError):
        gate.candidate_pool(KnowledgeQuery("cube"), check_remaining=check)
    assert len(checks) == 3
    gate.close()
    refs.close()


def test_one_readonly_view_per_store_and_disabled_gate_never_retrieves(tmp_path, monkeypatch):
    backend = Embeddings()
    refs = indexed(tmp_path, "first", {"one.cpp": "cube one\n"}, backend)
    source = tmp_path / "first" / "one.cpp"
    refs.index(source.parent, [source], collection="second", language="ascendc")
    gate = agent(tmp_path, {"first": refs, "second": refs})
    assert gate.databases["first"] is gate.databases["second"]
    assert len(gate.candidate_pool(KnowledgeQuery("cube")).results) == 1
    view = gate.database
    closed = []
    original = view.close
    monkeypatch.setattr(view, "close", lambda: (closed.append(True), original()))
    gate.close()
    assert closed == [True]
    refs.close()
    disabled = KnowledgeAgent(enabled=False,
        journal=ProgressiveMemoryJournal(tmp_path / "disabled.jsonl"))
    assert disabled.candidate_pool(KnowledgeQuery("cube")).results == ()
    assert disabled.query(KnowledgeQuery("cube")) == ()
