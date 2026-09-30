from __future__ import annotations

import hashlib
import json
import math
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from benchmarks.a3kernels import corpus
from benchmarks.a3kernels.corpus import CorpusSpec, load_documents
from benchmarks.a3kernels.embeddings import (
    DEFAULT_EMBEDDING_REVISION,
    EmbeddingLoadError,
    PinnedBGEEmbeddings,
    validate_local_snapshot,
)


CORPORA = Path(corpus.__file__).with_name("corpora")


class RecordingTokenizer:
    def __init__(self):
        self.batches: list[list[str]] = []

    def __call__(self, texts, **_kwargs):
        self.batches.append(list(texts))
        token = MagicMock()
        token.to.return_value = token
        return {"input_ids": token, "attention_mask": token}


class FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def use_deterministic_algorithms(enabled):
        assert enabled is True


class FakeEncoder:
    def __init__(self, tokenizer, *, revision=DEFAULT_EMBEDDING_REVISION):
        self.tokenizer = tokenizer
        self.config = SimpleNamespace(_commit_hash=revision, hidden_size=384)

    def to(self, device):
        assert device == "cpu"
        return self

    def eval(self):
        return self

    def __call__(self, **_tokens):
        rows = [[1.0, 2.0] * 192 for _ in self.tokenizer.batches[-1]]
        pooled = MagicMock()
        pooled.detach.return_value.cpu.return_value.tolist.return_value = rows
        states = MagicMock()
        states.__getitem__.return_value = pooled
        return SimpleNamespace(last_hidden_state=states)


def _backend(tokenizer, *, revision=DEFAULT_EMBEDDING_REVISION):
    encoder = FakeEncoder(tokenizer, revision=revision)
    return PinnedBGEEmbeddings(
        tokenizer_loader=lambda *_args, **_kwargs: tokenizer,
        model_loader=lambda *_args, **_kwargs: encoder,
        torch_module=FakeTorch(),
    )


def test_checked_in_a3_corpus_is_ascendc_only_and_strictly_pinned():
    paths = sorted(CORPORA.glob("*.json"))
    specs = [CorpusSpec.load(path) for path in paths]

    assert [spec.name for spec in specs] == ["a3-ascendc-en"]
    assert all(spec.target == "a3" and spec.language == "en" for spec in specs)
    assert all(len(spec.revision) == 40 and spec.files for spec in specs)
    identities = "\n".join(
        value
        for spec in specs
        for value in [spec.name, spec.collection, spec.repository, *(item.path for item in spec.files)]
    ).lower()
    assert "catlass" not in identities
    assert "a5" not in identities


def test_spec_rejects_wrong_target_unknown_fields_and_bad_order(tmp_path):
    raw = {
        "name": "a3-docs-en",
        "collection": "a3-docs-en-v1",
        "target": "a3",
        "language": "en",
        "repository": "https://example.invalid/docs.git",
        "revision": "a" * 40,
        "files": [
            {"path": "z.md", "sha256": "b" * 64},
            {"path": "a.md", "sha256": "c" * 64},
        ],
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({**raw, "target": "a5"}))
    with pytest.raises(ValueError, match="target must be 'a3'"):
        CorpusSpec.load(path)
    path.write_text(json.dumps({**raw, "catlass": True}))
    with pytest.raises(ValueError, match="fields must be exactly"):
        CorpusSpec.load(path)
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unique and sorted"):
        CorpusSpec.load(path)


def test_load_documents_is_ordered_and_fails_closed_on_missing_or_drift(tmp_path):
    data = {"a.md": b"first\n", "z.cpp": b"second\n"}
    raw = {
        "name": "a3-docs-en", "collection": "a3-docs-en-v1", "target": "a3",
        "language": "en", "repository": "https://example.invalid/docs.git",
        "revision": "a" * 40,
        "files": [
            {"path": path, "sha256": hashlib.sha256(value).hexdigest()}
            for path, value in data.items()
        ],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(raw))
    spec = CorpusSpec.load(spec_path)
    root = spec.source_root(tmp_path / "artifacts")
    root.mkdir(parents=True)
    for path, value in data.items():
        (root / path).write_bytes(value)

    documents = load_documents(spec, tmp_path / "artifacts")
    assert [item.path for item in documents] == ["a.md", "z.cpp"]
    assert [item.text for item in documents] == ["first\n", "second\n"]
    assert documents[0].source_revision == spec.revision
    assert documents[0].content_sha256 == hashlib.sha256(data["a.md"]).hexdigest()

    (root / "a.md").unlink()
    with pytest.raises(ValueError, match="exactly match"):
        load_documents(spec, tmp_path / "artifacts")
    (root / "a.md").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_documents(spec, tmp_path / "artifacts")


def test_query_and_documents_use_separate_encoding_paths():
    tokenizer = RecordingTokenizer()
    backend = _backend(tokenizer)

    documents = backend.embed_documents(["DataCopy"])
    queries = backend.embed_queries(["vector add"])

    assert tokenizer.batches == [
        ["DataCopy"],
        ["Represent this sentence for searching relevant passages: vector add"],
    ]
    assert documents == backend.embed_documents(["DataCopy"])
    assert all(math.isclose(sum(value * value for value in row), 1.0) for row in queries)


def test_loader_rejects_unverified_revision_before_returning_vectors():
    with pytest.raises(EmbeddingLoadError, match="provenance is unverified"):
        _backend(RecordingTokenizer(), revision="main").embed_documents(["kernel"])


def test_snapshot_validation_is_offline_and_fails_closed(tmp_path):
    cache = tmp_path / "hub"
    snapshot = cache / "models--BAAI--bge-small-en-v1.5" / "snapshots" / DEFAULT_EMBEDDING_REVISION
    snapshot.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (snapshot / name).write_text(name)

    assert validate_local_snapshot(cache) == snapshot
    (snapshot / "config.json").unlink()
    with pytest.raises(EmbeddingLoadError, match="incomplete"):
        validate_local_snapshot(cache)
    with pytest.raises(EmbeddingLoadError, match="not present"):
        validate_local_snapshot(tmp_path / "missing")


def test_fake_model_indexing_is_deterministic(tmp_path):
    spec = CorpusSpec.load(CORPORA / "a3-ascendc-en.json")
    root = spec.source_root(tmp_path)
    root.mkdir(parents=True)
    # Use a derived fixture spec so the proof is independent of network access.
    texts = {"a.md": b"DataCopy movement\n", "b.cpp": b"Add vector\n"}
    fixture = spec.with_files(texts)
    fixture_root = fixture.source_root(tmp_path)
    fixture_root.mkdir(parents=True, exist_ok=True)
    for path, value in texts.items():
        (fixture_root / path).write_bytes(value)
    backend = _backend(RecordingTokenizer())

    first = [(doc.digest, vector) for doc, vector in zip(
        load_documents(fixture, tmp_path),
        backend.embed_documents([doc.text for doc in load_documents(fixture, tmp_path)]),
    )]
    second = [(doc.digest, vector) for doc, vector in zip(
        load_documents(fixture, tmp_path),
        backend.embed_documents([doc.text for doc in load_documents(fixture, tmp_path)]),
    )]
    assert first == second
