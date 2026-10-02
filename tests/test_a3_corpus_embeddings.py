from __future__ import annotations

import hashlib
import json
import math
import threading
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from benchmarks.a3kernels import corpus
from benchmarks.a3kernels import embeddings as embedding_module
from benchmarks.a3kernels.corpus import CorpusSpec, load_documents, prepare_corpus
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


@pytest.mark.parametrize("invalid_path", [".", "nested/"])
def test_spec_rejects_paths_without_a_real_filename(tmp_path, invalid_path):
    raw = {
        "name": "a3-docs-en", "collection": "a3-docs-en-v1", "target": "a3",
        "language": "en", "repository": "https://example.invalid/docs.git",
        "revision": "a" * 40,
        "files": [{"path": invalid_path, "sha256": "b" * 64}],
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="unsafe corpus path"):
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


def test_prepare_fetches_exact_allowlist_atomically_and_reuses_verified_tree(tmp_path):
    values = {"a.md": b"first\n", "nested/z.cpp": b"second\n"}
    base = CorpusSpec.load(CORPORA / "a3-ascendc-en.json")
    spec = base.with_files(values)
    calls = []

    def fetch(repository, revision, paths):
        calls.append((repository, revision, paths))
        return dict(values)

    root = prepare_corpus(spec, tmp_path, fetcher=fetch)
    assert [(item.path, item.text) for item in load_documents(spec, tmp_path)] == [
        ("a.md", "first\n"), ("nested/z.cpp", "second\n")
    ]
    assert prepare_corpus(
        spec, tmp_path,
        fetcher=lambda *_args: (_ for _ in ()).throw(AssertionError("must reuse")),
    ) == root
    assert calls == [(spec.repository, spec.revision, tuple(values))]


@pytest.mark.parametrize("failure", ["missing", "extra", "hash"])
def test_prepare_mismatch_never_publishes_partial_sources(tmp_path, failure):
    values = {"a.md": b"first\n", "z.cpp": b"second\n"}
    spec = CorpusSpec.load(CORPORA / "a3-ascendc-en.json").with_files(values)
    fetched = dict(values)
    if failure == "missing":
        fetched.pop("z.cpp")
    elif failure == "extra":
        fetched["other.md"] = b"other\n"
    else:
        fetched["a.md"] = b"changed\n"

    with pytest.raises(ValueError, match="allowlisted|hash mismatch"):
        prepare_corpus(spec, tmp_path, fetcher=lambda *_args: fetched)
    assert not spec.source_root(tmp_path).exists()


def test_prepare_rejects_invalid_utf8_before_publication(tmp_path):
    values = {"guide.md": b"\xff\xfe"}
    spec = CorpusSpec.load(CORPORA / "a3-ascendc-en.json").with_files(values)

    with pytest.raises(ValueError, match="not UTF-8"):
        prepare_corpus(spec, tmp_path, fetcher=lambda *_args: values)

    assert not spec.source_root(tmp_path).exists()


def test_concurrent_prepare_has_one_publisher_and_verified_reuse(tmp_path):
    values = {"a.md": b"first\n", "nested/z.cpp": b"second\n"}
    spec = CorpusSpec.load(CORPORA / "a3-ascendc-en.json").with_files(values)
    start = threading.Barrier(2)
    calls = []

    def fetch(*_args):
        calls.append(True)
        return dict(values)

    def prepare():
        start.wait()
        return prepare_corpus(spec, tmp_path, fetcher=fetch)

    with ThreadPoolExecutor(max_workers=2) as pool:
        roots = tuple(pool.map(lambda _index: prepare(), range(2)))

    assert roots == (spec.source_root(tmp_path),) * 2
    assert len(calls) == 1
    assert [doc.text for doc in load_documents(spec, tmp_path)] == [
        "first\n", "second\n",
    ]


def test_query_and_documents_use_separate_encoding_paths():
    tokenizer = RecordingTokenizer()
    backend = _backend(tokenizer)

    documents = backend.embed_documents(["DataCopy"])
    queries = backend.embed_queries(["vector add"])
    query = backend.embed_query("vector add")

    assert tokenizer.batches == [
        ["DataCopy"],
        ["Represent this sentence for searching relevant passages: vector add"],
        ["Represent this sentence for searching relevant passages: vector add"],
    ]
    assert documents == backend.embed_documents(["DataCopy"])
    assert query == queries[0]
    assert len(query) == 384
    assert all(math.isclose(sum(value * value for value in row), 1.0) for row in queries)
    with pytest.raises(TypeError, match="query must be a string"):
        backend.embed_query(["vector add"])


def test_loader_rejects_unverified_revision_before_returning_vectors():
    with pytest.raises(EmbeddingLoadError, match="provenance is unverified"):
        _backend(RecordingTokenizer(), revision="main").embed_documents(["kernel"])


def test_default_loader_requires_an_authenticated_cache_path():
    with pytest.raises(EmbeddingLoadError, match="cache_dir is required"):
        PinnedBGEEmbeddings().embed_documents(["kernel"])


def test_snapshot_validation_is_offline_and_authenticates_contents(
    tmp_path, monkeypatch
):
    cache = tmp_path / "hub"
    snapshot = cache / "models--BAAI--bge-small-en-v1.5" / "snapshots" / DEFAULT_EMBEDDING_REVISION
    snapshot.mkdir(parents=True)
    contents = {
        "config.json": b"config",
        "tokenizer.json": b"tokenizer",
        "model.safetensors": b"weights",
    }
    monkeypatch.setattr(
        embedding_module,
        "_SNAPSHOT_SHA256",
        tuple(
            (name, hashlib.sha256(content).hexdigest())
            for name, content in contents.items()
        ),
    )
    for name, content in contents.items():
        (snapshot / name).write_bytes(content)

    assert validate_local_snapshot(cache) == snapshot
    (snapshot / "tokenizer.json").write_bytes(b"drifted")
    with pytest.raises(EmbeddingLoadError, match="content mismatch: tokenizer.json"):
        validate_local_snapshot(cache)
    (snapshot / "tokenizer.json").write_bytes(contents["tokenizer.json"])
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
