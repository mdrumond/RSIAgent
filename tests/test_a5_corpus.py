import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.a5kernels import corpus
from benchmarks.a5kernels.corpus import (
    CorpusFile,
    CorpusSpec,
    index_corpus,
    prepare_corpus,
    verify_prepared,
)
from benchmarks.a5kernels.knowledge import KnowledgeDB


CORPORA = Path(corpus.__file__).with_name("corpora")


class FakeEmbeddings:
    model = "fake/model"
    revision = "f" * 40

    def embed(self, texts):
        return [[float("DataCopy" in text), 1.0] for text in texts]

    def embed_query(self, texts):
        return [[1.0, 1.0] for _text in texts]


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path, files: dict[str, bytes]) -> tuple[Path, str]:
    repository = tmp_path / "upstream"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Corpus Test")
    _git(repository, "config", "user.email", "corpus@example.invalid")
    for relative, data in files.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "fixture")
    return repository, _git(repository, "rev-parse", "HEAD")


def _spec(repository: Path, revision: str, files: dict[str, bytes]) -> CorpusSpec:
    return CorpusSpec(
        name="test-en",
        collection="test-en-v1",
        language="en",
        repository=str(repository),
        revision=revision,
        files=tuple(
            CorpusFile(path, hashlib.sha256(data).hexdigest())
            for path, data in sorted(files.items())
        ),
    )


def test_checked_in_corpora_are_strict_pinned_english_allowlists():
    specs = [CorpusSpec.load(path) for path in sorted(CORPORA.glob("*.json"))]

    assert [spec.name for spec in specs] == ["ascendc-en", "catlass-en"]
    assert all(spec.language == "en" for spec in specs)
    assert all(len(spec.revision) == 40 for spec in specs)
    assert all(spec.repository.startswith("https://") for spec in specs)
    assert all(spec.files for spec in specs)
    assert all(tuple(sorted(spec.files, key=lambda item: item.path)) == spec.files for spec in specs)
    assert not any("../cat_dev" in path.read_text() for path in CORPORA.glob("*.json"))
    catlass = next(spec for spec in specs if spec.name == "catlass-en")
    assert catlass.repository == "https://gitcode.com/cann/catlass.git"
    assert all(item.path.startswith("python/tla_dsl/") for item in catlass.files)
    assert all("/docs/en/" in item.path or item.path.endswith(".py") for item in catlass.files)
    assert any(item.path.endswith("basic_vadd.py") for item in catlass.files)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"language": "zh"}, "language 'en'"),
        ({"revision": "main"}, "40-character"),
        ({"repository": "git@example.invalid:repo"}, "canonical HTTPS"),
        ({"name": "../docs"}, "artifact-directory component"),
        ({"files": []}, "at least one"),
        (
            {"files": [{"path": "../secret", "sha256": "a" * 64}]},
            "unsafe corpus path",
        ),
    ],
)
def test_spec_rejects_unpinned_or_unsafe_inputs(tmp_path, update, message):
    value = {
        "name": "docs-en",
        "collection": "docs-en-v1",
        "language": "en",
        "repository": "https://example.invalid/docs.git",
        "revision": "a" * 40,
        "files": [{"path": "README.md", "sha256": "b" * 64}],
    }
    value.update(update)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match=message):
        CorpusSpec.load(path)


def test_spec_rejects_unknown_fields_and_unsorted_duplicates(tmp_path):
    path = tmp_path / "spec.json"
    base = {
        "name": "docs-en",
        "collection": "docs-en-v1",
        "language": "en",
        "repository": "https://example.invalid/docs.git",
        "revision": "a" * 40,
        "files": [
            {"path": "z.md", "sha256": "b" * 64},
            {"path": "a.md", "sha256": "c" * 64},
        ],
    }
    path.write_text(json.dumps({**base, "branch": "main"}))
    with pytest.raises(ValueError, match="fields must be exactly"):
        CorpusSpec.load(path)

    path.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="unique and sorted"):
        CorpusSpec.load(path)


def test_prepare_fetches_exact_commit_and_only_allowlisted_verified_files(tmp_path):
    files = {"docs/guide.md": b"English guide\n", "examples/add.cpp": b"DataCopy();\n"}
    repository, revision = _repository(tmp_path, {**files, "ignored.txt": b"not curated\n"})
    spec = _spec(repository, revision, files)
    artifacts = tmp_path / "artifacts"

    root = prepare_corpus(spec, artifacts)

    assert root == spec.source_root(artifacts)
    assert [(path.relative_to(root).as_posix(), path.read_bytes()) for path in verify_prepared(spec, artifacts)] == list(files.items())
    assert not (root / "ignored.txt").exists()
    assert prepare_corpus(spec, artifacts) == root


def test_prepare_hash_failure_never_publishes_partial_corpus(tmp_path):
    files = {"README.md": b"upstream bytes\n"}
    repository, revision = _repository(tmp_path, files)
    spec = CorpusSpec(
        name="test-en",
        collection="test-en-v1",
        language="en",
        repository=str(repository),
        revision=revision,
        files=(CorpusFile("README.md", "0" * 64),),
    )
    artifacts = tmp_path / "artifacts"

    with pytest.raises(ValueError, match="hash mismatch for README.md"):
        prepare_corpus(spec, artifacts)

    assert not spec.source_root(artifacts).exists()


def test_verify_rejects_tampering_and_unlisted_files(tmp_path):
    files = {"README.md": b"original\n"}
    repository, revision = _repository(tmp_path, files)
    spec = _spec(repository, revision, files)
    root = prepare_corpus(spec, tmp_path / "artifacts")
    (root / "README.md").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="prepared hash mismatch"):
        verify_prepared(spec, tmp_path / "artifacts")

    (root / "README.md").write_bytes(files["README.md"])
    (root / "extra.md").write_text("extra")
    with pytest.raises(ValueError, match="exactly match its allowlist"):
        verify_prepared(spec, tmp_path / "artifacts")


def test_index_uses_verified_sources_pinned_backend_and_exports_manifest(tmp_path, monkeypatch):
    files = {"docs/guide.md": b"Use DataCopy for local tensors.\n"}
    repository, revision = _repository(tmp_path, files)
    spec = _spec(repository, revision, files)
    artifacts = tmp_path / "artifacts"
    prepare_corpus(spec, artifacts)
    cache = tmp_path / "model-cache"
    created = []

    def embeddings(*, cache_dir, local_files_only):
        created.append((cache_dir, local_files_only))
        return FakeEmbeddings()

    monkeypatch.setattr(corpus, "PinnedBGEEmbeddings", embeddings)
    database_path = tmp_path / "kdb" / "knowledge.sqlite3"
    manifest_path = tmp_path / "kdb" / "test.manifest.json"

    index_corpus(spec, artifacts, database_path, manifest_path, model_cache=cache)

    assert created == [(cache, True)]
    with KnowledgeDB(database_path, FakeEmbeddings()) as database:
        manifest = database.manifest(spec.collection)
        assert manifest.language == "en"
        assert manifest.sources[0].path == "docs/guide.md"
        assert database.query(spec.collection, "DataCopy", limit=1)[0].path == "docs/guide.md"
    assert json.loads(manifest_path.read_text())["fingerprint"] == manifest.fingerprint
