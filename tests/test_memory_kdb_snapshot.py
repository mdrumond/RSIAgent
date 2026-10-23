from dataclasses import replace
import shutil

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.knowledge import KnowledgeDB
from benchmarks.a5kernels.knowledge_agent import KnowledgeAgent, KnowledgeQuery, ProgressiveMemoryJournal
from benchmarks.a5kernels.memory_snapshot import MemorySnapshot, MemorySource, export_snapshot
from benchmarks.a5kernels.phase1_memory import AgentLesson, HostFact, Phase1ProjectMemory
from benchmarks.a5kernels.phase1_registry import DEFAULT_PROPOSALS


class Embeddings:
    model = "test"
    revision = "test"

    def embed(self, texts):
        return [[1., float(text.count("pipeline"))] for text in texts]


def source(root):
    store = ContentAddressedArtifacts(root / "artifacts")
    digest = store.put_bytes(b"pipeline code example\n" * 20)
    memory = Phase1ProjectMemory(
        proposal=DEFAULT_PROPOSALS[0], source_revision="revision-1", actions=("run",),
        host_facts=(HostFact("runtime", "completed", digest),),
        lessons=(AgentLesson("data-movement", "pipeline lesson", "copy", "aligned",
                            "one shape", (digest,), None, None, "medium"),),
        no_new_knowledge=False, admitted_artifact_sha256=(digest,), artifact_store="artifacts",
    )
    return MemorySource(memory, root, {"target": "A3", "toolchain": "retained-version"}), digest


def test_fresh_consumer_retrieves_and_reads_without_producer(tmp_path):
    original, digest = source(tmp_path / "producer")
    snapshot = export_snapshot([original], tmp_path / "snapshot")
    with KnowledgeDB(tmp_path / "kdb.sqlite", Embeddings()) as database:
        snapshot.index(database, collection="memory-v1")
        shutil.rmtree(original.root)
        # Re-open the exported bundle to model an independent consumer.
        snapshot = MemorySnapshot(snapshot.root)
        gate = KnowledgeAgent(enabled=True, journal=ProgressiveMemoryJournal(tmp_path / "queries.jsonl"),
                              database=database, collections=("memory-v1",), snapshots={"memory-v1": snapshot})
        try:
            results = gate.query(KnowledgeQuery("pipeline", limit=10))
            assert results and all(item.citation.collection == "memory-v1" for item in results)
            page = gate.read_artifact("memory-v1", digest, limit=8)
            assert page.data == b"pipeline" and page.next_offset == 8
            rest = gate.read_artifact("memory-v1", digest, offset=8)
            assert page.data + rest.data == b"pipeline code example\n" * 20
            with pytest.raises(ValueError):
                gate.read_artifact("other", digest)
            with pytest.raises(ValueError):
                gate.read_artifact("memory-v1", "f" * 64)
            with pytest.raises(ValueError):
                gate.read_artifact("memory-v1", digest, limit=65537)
        finally:
            gate.close()


def test_reproducible_dedup_preserves_provenance_and_evidence(tmp_path):
    first, digest = source(tmp_path / "producer")
    second = replace(first, provenance={"target": "A3", "run": "second"})
    one = export_snapshot([first, second, first], tmp_path / "one")
    two = export_snapshot([second, first], tmp_path / "two")
    assert one.fingerprint == two.fingerprint
    assert one.manifest["memory_records"] == 1
    assert len(one.manifest["artifacts"]) == 1
    assert len(one.manifest["artifacts"][digest]["provenance"]) == 2
    import json
    memory_doc = next((one.root / "documents").glob("memory-*.json"))
    published = json.loads(memory_doc.read_text())["memory"]
    assert published["agent_lessons"] == first.memory.as_dict()["agent_lessons"]
    assert published["host_facts"] == first.memory.as_dict()["host_facts"]
    assert "artifact_store" not in published
    assert one.direct_documents() == two.direct_documents()
    with pytest.raises(FileExistsError):
        export_snapshot([first], one.root)


def test_corrupt_or_missing_evidence_excludes_record(tmp_path):
    good, digest = source(tmp_path / "producer")
    store = ContentAddressedArtifacts(good.root / "artifacts")
    store._path(digest).write_bytes(b"corrupt")
    snapshot = export_snapshot([good], tmp_path / "bad")
    assert snapshot.manifest["memory_records"] == 0
    assert snapshot.manifest["excluded"][0]["artifact_sha256"] == digest
    assert not snapshot.manifest["artifacts"]
    store._path(digest).unlink()
    missing = export_snapshot([good], tmp_path / "missing")
    assert missing.manifest["excluded"][0]["reason"] == "FileNotFoundError"


def test_gate_rejects_different_snapshot_and_tampered_artifacts(tmp_path):
    original, digest = source(tmp_path / "producer")
    one = export_snapshot([original], tmp_path / "one")
    two = export_snapshot([replace(original, provenance={"target": "different"})], tmp_path / "two")
    with KnowledgeDB(tmp_path / "kdb.sqlite", Embeddings()) as database:
        one.index(database, collection="memory-v1")
        with pytest.raises(ValueError, match="does not match"):
            KnowledgeAgent(enabled=True, journal=ProgressiveMemoryJournal(tmp_path / "q"),
                           database=database, collection="memory-v1", snapshots={"memory-v1": two})
        with pytest.raises(ValueError, match="immutable"):
            two.index(database, collection="memory-v1")
    ContentAddressedArtifacts(one.root / "artifacts")._path(digest).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        one.read_artifact(digest)
