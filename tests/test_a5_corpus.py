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

    assert [spec.name for spec in specs] == [
        "a5-ascendc-architecture-en", "ascendc-en", "catlass-en",
    ]
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


def test_a5_cross_layer_reference_has_exact_pin_hash_and_selection():
    path = CORPORA / "a5-ascendc-architecture-en.json"
    spec = CorpusSpec.load(path)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "2f109bab00ac79449a0e862f3291351e4e1403480b0cc5fb0106544a53c34926"
    )
    assert spec.repository == "https://github.com/huawei-cpl-zurich/data-movement-benchmarks"
    assert spec.revision == "23f39974d9175ea39ddad6e9fe84a7510ff8a56e"
    assert spec.collection == "a5-ascendc-architecture-en-23f3997"
    assert [item.path for item in spec.files] == [
        "README.md", "analyse_msprof.py", "compute_results_a5.metadata.json",
        "compute_throughput.asc", "load_results_a5_simt_float4.metadata.json",
        "mte2_fixpipe_contention.asc", "mte2_prefetch.asc", "results_a5.metadata.json",
        "run_compute_benchmarks.py", "run_load_benchmarks.py", "run_stride_sweep.py",
        "simt_load.asc", "stride_results_a5.metadata.json",
        "sub32_stride_results_a5.metadata.json",
    ]
    assert CorpusSpec.load(path) == spec


@pytest.mark.parametrize("failure", ["revision", "missing", "drift"])
def test_a5_cross_layer_prepare_fails_closed_on_upstream_mismatch(
    tmp_path, monkeypatch, failure,
):
    spec = CorpusSpec.load(CORPORA / "a5-ascendc-architecture-en.json")
    commands = []

    def run(argv, *, cwd):
        commands.append(argv)
        if argv[1] == "rev-parse":
            return "0" * 40 if failure == "revision" else spec.revision
        return ""

    def blob(argv, *, cwd):
        commands.append(argv)
        assert argv == ["git", "show", f"{spec.revision}:README.md"]
        if failure == "missing":
            raise RuntimeError("git show failed: missing pinned README.md")
        return b"changed upstream bytes\n"

    monkeypatch.setattr(corpus, "_run", run)
    monkeypatch.setattr(corpus, "_run_bytes", blob)
    message = {"revision": "revision mismatch", "missing": "missing pinned", "drift": "hash mismatch"}[failure]
    with pytest.raises((ValueError, RuntimeError), match=message):
        prepare_corpus(spec, tmp_path / "artifacts")
    assert ["git", "fetch", "--quiet", "--depth=1", "origin", spec.revision] in commands
    assert not spec.source_root(tmp_path / "artifacts").exists()


def test_missing_cross_layer_snapshot_cannot_be_indexed_or_downloaded(tmp_path, monkeypatch):
    spec = CorpusSpec.load(CORPORA / "a5-ascendc-architecture-en.json")

    def forbidden(*args, **kwargs):
        raise AssertionError("missing sources must fail before model or network access")

    monkeypatch.setattr(corpus, "PinnedBGEEmbeddings", forbidden)
    monkeypatch.setattr(corpus, "_run_bytes", forbidden)
    with pytest.raises(ValueError, match="exactly match its allowlist"):
        index_corpus(spec, tmp_path / "missing", tmp_path / "db", tmp_path / "manifest")
    assert not (tmp_path / "db").exists()


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
        (
            {"files": [{"path": "docs//guide.md", "sha256": "a" * 64}]},
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


def test_prepare_surfaces_git_fetch_diagnostics(tmp_path):
    missing = tmp_path / "missing-upstream"
    spec = CorpusSpec(
        name="test-en",
        collection="test-en-v1",
        language="en",
        repository=str(missing),
        revision="a" * 40,
        files=(CorpusFile("README.md", "0" * 64),),
    )

    with pytest.raises(RuntimeError, match=r"git fetch failed: .*missing-upstream"):
        prepare_corpus(spec, tmp_path / "artifacts")


def test_prepare_surfaces_git_show_diagnostics(tmp_path):
    repository, revision = _repository(tmp_path, {"README.md": b"available\n"})
    spec = _spec(repository, revision, {"missing.md": b"hypothetical\n"})

    with pytest.raises(RuntimeError, match=r"git show failed: .*missing\.md"):
        prepare_corpus(spec, tmp_path / "artifacts")


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


def test_index_uses_verified_snapshot_when_prepared_source_changes(tmp_path, monkeypatch):
    original = b"Use DataCopy for local tensors.\n"
    files = {"docs/guide.md": original}
    repository, revision = _repository(tmp_path, files)
    spec = _spec(repository, revision, files)
    artifacts = tmp_path / "artifacts"
    prepared = prepare_corpus(spec, artifacts) / "docs/guide.md"

    class MutatingEmbeddings(FakeEmbeddings):
        def embed(self, texts):
            prepared.write_bytes(b"changed after verification\n")
            return super().embed(texts)

    monkeypatch.setattr(corpus, "PinnedBGEEmbeddings", lambda **_kwargs: MutatingEmbeddings())
    database_path = tmp_path / "knowledge.sqlite3"
    manifest_path = tmp_path / "manifest.json"

    index_corpus(spec, artifacts, database_path, manifest_path)

    with KnowledgeDB(database_path, FakeEmbeddings()) as database:
        hit = database.query(spec.collection, "DataCopy", limit=1)[0]
        assert hit.text == original.decode()
        assert database.manifest(spec.collection).sources[0].sha256 == hashlib.sha256(original).hexdigest()


def test_index_rejects_database_manifest_path_collision(tmp_path, monkeypatch):
    files = {"README.md": b"verified\n"}
    repository, revision = _repository(tmp_path, files)
    spec = _spec(repository, revision, files)
    artifacts = tmp_path / "artifacts"
    prepare_corpus(spec, artifacts)
    monkeypatch.setattr(corpus, "PinnedBGEEmbeddings", lambda **_kwargs: FakeEmbeddings())
    shared = tmp_path / "kdb" / "shared-output"

    with pytest.raises(ValueError, match="database and manifest paths must be different"):
        index_corpus(spec, artifacts, shared, shared.parent / "." / shared.name)

    assert not shared.exists()


@pytest.mark.parametrize("output", ["database", "manifest"])
def test_index_rejects_outputs_inside_prepared_sources(tmp_path, monkeypatch, output):
    files = {"README.md": b"verified\n"}
    repository, revision = _repository(tmp_path, files)
    spec = _spec(repository, revision, files)
    artifacts = tmp_path / "artifacts"
    root = prepare_corpus(spec, artifacts)
    monkeypatch.setattr(corpus, "PinnedBGEEmbeddings", lambda **_kwargs: FakeEmbeddings())
    database = root / "output.sqlite3" if output == "database" else tmp_path / "kdb.sqlite3"
    manifest = root / "manifest.json" if output == "manifest" else tmp_path / "manifest.json"

    with pytest.raises(ValueError, match="must be outside prepared sources"):
        index_corpus(spec, artifacts, database, manifest)

    assert set(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()) == set(files)
