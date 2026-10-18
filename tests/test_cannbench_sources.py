import io
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
import zipfile

import pytest

from benchmarks.a5kernels import cannbench_sources as source


def bundle(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return stream.getvalue()


OP = {"function_name": "exp", "rel_path": "level1/exp"}


def test_unpack_preserves_wrappers_and_license_and_rejects_unrelated_code():
    files, identity = source.unpack(bundle({
        "level1/exp/kernel.cpp": "void Exp() {}", "utils.py": "helper = 1",
        "LICENSE": "MIT", "../escape.py": "bad", "weights.bin": b"xx",
    }), OP, {})
    assert set(files) == {"level1/exp/kernel.cpp", "utils.py", "LICENSE"}
    _, same_identity = source.unpack(bundle({
        "other/exp/kernel.cpp": "void Exp() {}", "other/exp/launch.py": "different wrapper",
        "README.md": "changed",
    }), OP, {})
    assert identity == same_identity
    with pytest.raises(ValueError, match="ranked operator"):
        source.unpack(bundle({"matmul/kernel.cpp": "matmul"}), OP, {})


def test_single_operator_submission_identifies_generic_kernel():
    files, _ = source.unpack(bundle({"kernel.py": "code"}), OP,
                             {"selected_operators": ["exp"]})
    assert "kernel.py" in files


@pytest.mark.parametrize("placeholder", ["", " \n", "# TODO: implement exp\n", "pass\n", "...\n",
                                         '"""Not implemented yet."""', "def exp():\n    pass\n",
                                         "def exp(): raise NotImplementedError\n"])
def test_placeholder_operator_falls_back_to_next_ranked_implementation(tmp_path, placeholder):
    class PlaceholderArchive(FakeArchive):
        def get(self, url):
            if url.endswith("job0"):
                data = bundle({"exp/__init__.py": placeholder, "README.md": "exp implementation"})
                return data, {"status": 200, "sha256": source.digest(data), "url": url}
            return super().get(url)
    result = source.acquire_operator(OP, PlaceholderArchive(), tmp_path, "1.1.2", 1)
    assert [item["rank"] for item in result["sources"]] == [2]
    assert "placeholder" in result["skipped"][0]["attempts"][0]["source_error"]


def test_kernel_named_archive_root_does_not_include_wrappers_in_identity(tmp_path):
    class KernelRootArchive(FakeArchive):
        def get(self, url):
            job_id = url.split("/")[-1]
            data = bundle({"cannbench-kernels-main/exp/kernel.cpp": "AscendC::Exp(x,y,n);",
                           "cannbench-kernels-main/exp/launch.py": f"wrapper = '{job_id}'",
                           "cannbench-kernels-main/README.md": "repository"})
            return data, {"status": 200, "sha256": source.digest(data), "url": url}
    result = source.acquire_operator(OP, KernelRootArchive(), tmp_path, "1.1.2", 2)
    assert len(result["sources"]) == 1
    assert result["shortfall"] == 1
    assert all("Duplicate" in item["reason"] for item in result["skipped"])


class FakeArchive:
    def __init__(self):
        self.offsets = []

    def json(self, url):
        if "/leaderboard?" in url:
            offset = int(url.split("offset=")[1])
            self.offsets.append(offset)
            entries = [{"best_job_id": f"job{i}", "score": 100 - i} for i in range(5)]
            return {"entries": entries[offset:offset + 2], "total": 5}
        job_id = url.split("/")[-1]
        return {"job": {"id": job_id, "selected_operators": ["exp"],
                         "submission_download_url": f"/source/{job_id}"}}

    def get(self, url):
        job_id = url.split("/")[-1]
        if job_id == "job0":
            return None, {"status": 401, "url": url}
        # Rank 3 duplicates rank 2; rank 4 and 5 are the next available.
        value = "first" if job_id in ("job1", "job2") else job_id
        data = bundle({"kernel.py": value, "LICENSE": "MIT"})
        return data, {"status": 200, "sha256": source.digest(data), "url": url}


def test_ranked_fallback_pagination_and_code_deduplication(tmp_path):
    archive = FakeArchive()
    result = source.acquire_operator(OP, archive, tmp_path, "1.1.2", 3)
    assert archive.offsets == [0, 2, 4]
    assert [item["rank"] for item in result["sources"]] == [2, 4, 5]
    assert result["shortfall"] == 0
    assert len(result["skipped"]) == 2
    assert "Duplicate" in result["skipped"][1]["reason"]
    assert result["sources"][0]["license_files"] == ["LICENSE"]
    for item in result["sources"]:
        assert item["status"] == "complete"
        assert item["licenses"] == ["LICENSE"]
        assert item["kernel_languages"] == ["common"]
        assert len(item["revision"]) == 64
        for record in item["files"]:
            data = (Path(item["root"]) / record["path"]).read_bytes()
            assert record["bytes"] == len(data)
            assert record["sha256"] == source.digest(data)
    assert (tmp_path / "sources/exp/rank-2-job1/kernel.py").read_text() == "first"


def test_exhausted_candidates_report_real_shortfall(tmp_path):
    result = source.acquire_operator(OP, FakeArchive(), tmp_path, "1.1.2", 4)
    assert len(result["sources"]) == 3
    assert result["shortfall"] == 1
    assert result["exhausted"]
    assert result["examined_candidates"] == 5


def test_metadata_failure_does_not_claim_all_available_sources_exhausted(tmp_path):
    class InterruptedArchive(FakeArchive):
        def json(self, url):
            if url.endswith("/job1"):
                raise RuntimeError("transport unavailable")
            return super().json(url)
    result = source.acquire_operator(OP, InterruptedArchive(), tmp_path, "1.1.2", 4)
    assert result["shortfall"] == 1
    assert result["retryable_failures"] == 1
    assert not result["exhausted"]


def test_public_archive_resumes_success_and_denied_without_network(tmp_path, monkeypatch):
    url = "https://example.invalid/source"
    key = source.digest(url.encode())
    (tmp_path / f"{key}.body").write_bytes(b"source")
    (tmp_path / f"{key}.json").write_text(json.dumps({
        "status": 200, "sha256": source.digest(b"source"), "url": url,
    }))
    monkeypatch.setattr(source, "urlopen", lambda *args, **kw: pytest.fail("network used"))
    assert source.PublicArchive(tmp_path).get(url)[0] == b"source"
    (tmp_path / f"{key}.json").write_text(json.dumps({"status": 401, "url": url}))
    assert source.PublicArchive(tmp_path).get(url)[0] is None


def test_only_published_source_pointers_are_followed():
    assert source.candidate_urls({"submission_download_url": "/download",
                                  "submission": {"download_url": "/download"}}) == [
                                      source.BASE + "/download"]


@pytest.mark.parametrize(("code", "expected"), [
    ('#include "kernel_operator.h"\nAscendC::DataCopy(x,y,n);', ["ascend-c"]),
    ("import triton\n@triton.jit\ndef kernel(): pass", ["triton"]),
    ("from catlass import tla\n@tla.kernel()\ndef kernel(): pass", ["catlass-dsl"]),
    ("from pypto import jit\n@jit\ndef kernel(): pass", ["common"]),
])
def test_kernel_language_is_derived_from_code_not_a3_label(code, expected):
    assert source.kernel_languages({"kernel.py": code.encode()}) == expected


@pytest.mark.parametrize("status", [408, 429])
def test_transient_downloads_are_not_exhausted_and_retry_on_resume(tmp_path, monkeypatch, status):
    class TransientArchive(FakeArchive):
        def get(self, url):
            if url.endswith("job0"):
                return None, {"status": status, "url": url}
            return super().get(url)
    result = source.acquire_operator(OP, TransientArchive(), tmp_path, "1.1.2", 4)
    assert result["retryable_failures"] == 1
    assert not result["exhausted"]
    calls = []
    def failing_open(request, **kwargs):
        calls.append(request.full_url)
        raise HTTPError(request.full_url, status, "transient", {}, None)
    monkeypatch.setattr(source, "urlopen", failing_open)
    archive = source.PublicArchive(tmp_path / "responses")
    archive.get("https://example.invalid/code.py")
    archive.get("https://example.invalid/code.py")
    assert len(calls) == 2


def test_raw_public_source_is_downloadable_and_schema_ready(tmp_path):
    class RawArchive(FakeArchive):
        def json(self, url):
            if "/api/jobs/" in url:
                return {"job": {"source_url": "https://example.invalid/kernel.py"}}
            return super().json(url)
        def get(self, url):
            data = b"import triton\n@triton.jit\ndef kernel(): pass"
            return data, {"status": 200, "sha256": source.digest(data), "url": url}
    result = source.acquire_operator(OP, RawArchive(), tmp_path, "1.1.2", 1)
    item = result["sources"][0]
    assert item["status"] == "complete"
    assert item["kernel_languages"] == ["triton"]
    assert item["files"][0]["path"] == "exp/kernel.py"
    assert len(item["revision"]) == 64


def test_unsupported_public_pointer_does_not_claim_exhaustion(tmp_path):
    class HtmlArchive(FakeArchive):
        def get(self, url):
            if url.endswith("job0"):
                data = b"<html>repository page</html>"
                return data, {"status": 200, "sha256": source.digest(data), "url": url}
            return super().get(url)
    result = source.acquire_operator(OP, HtmlArchive(), tmp_path, "1.1.2", 4)
    assert result["unsupported_sources"] == 1
    assert not result["exhausted"]
    assert "Unsupported" in result["skipped"][0]["attempts"][0]["source_error"]


def test_other_operator_languages_and_upstream_manifest_are_preserved(tmp_path):
    class MultiArchive(FakeArchive):
        def get(self, url):
            data = bundle({"exp/kernel.cpp": "AscendC::Exp(x,y,n);",
                           "matmul/kernel.py": "import triton", "source-manifest.json": "upstream"})
            return data, {"status": 200, "sha256": source.digest(data), "url": url}
    result = source.acquire_operator(OP, MultiArchive(), tmp_path, "1.1.2", 1)
    item = result["sources"][0]
    assert item["kernel_languages"] == ["ascend-c"]
    root = Path(item["root"])
    assert (root / "source-manifest.json").read_text() == "upstream"
    assert (root / "matmul/kernel.py").exists()
    assert (tmp_path / "source-manifests/exp/rank-1-job0.json").exists()
    for record in item["files"]:
        data = (root / record["path"]).read_bytes()
        assert record["sha256"] == source.digest(data)
        assert record["bytes"] == len(data)


@pytest.mark.parametrize("given_revision", [None, "a" * 40, "published-branch"])
def test_published_https_git_source_freezes_commit_without_credentials(tmp_path, monkeypatch, given_revision):
    calls = []
    payload = bundle({"exp/kernel.cpp": "AscendC::Exp(x,y,n);", "LICENSE": "MIT"})
    def git_run(command, **kwargs):
        calls.append((command, kwargs))
        if "clone" in command:
            Path(command[-1]).mkdir()
        data = payload if "archive" in command else ("a" * 40 + "\n").encode()
        return SimpleNamespace(stdout=data)
    monkeypatch.setattr(source.subprocess, "run", git_run)
    job = {"repository_url": "https://example.invalid/repo.git"}
    if given_revision:
        job["source_revision"] = given_revision
    files, _, record = source.published_source(job["repository_url"], job, OP,
                                              FakeArchive(), tmp_path)
    assert record["revision"] == "a" * 40
    assert "exp/kernel.cpp" in files
    assert any("fetch" in command for command, _ in calls) == (given_revision == "published-branch")
    assert all(kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0" for _, kwargs in calls)
    assert all("credential.helper=" in command for command, _ in calls)


@pytest.mark.parametrize("given_revision", [None, "a" * 40, "a" * 7, "published-branch"])
def test_git_acquisition_resume_keeps_frozen_source_offline(tmp_path, monkeypatch, given_revision):
    payload = bundle({"exp/kernel.cpp": "AscendC::Exp(x,y,n);"})
    offline = False
    calls = []
    def git_run(command, **kwargs):
        calls.append(command)
        if "clone" in command:
            assert not offline
            Path(command[-1]).mkdir()
        if "fetch" in command:
            assert not offline, "cached source attempted to contact its unavailable origin"
        # Simulate a moved HEAD/branch after the initial acquisition.
        revision = "a" * 40 if not offline or "a" * 40 + "^{commit}" in command else "b" * 40
        return SimpleNamespace(stdout=payload if "archive" in command else (revision + "\n").encode())
    monkeypatch.setattr(source.subprocess, "run", git_run)
    class RepositoryArchive(FakeArchive):
        def json(self, url):
            if "/api/jobs/" in url:
                job = {"id": url.split("/")[-1], "repository_url": "https://example.invalid/repo.git"}
                if given_revision:
                    job["source_revision"] = given_revision
                return {"job": job}
            return super().json(url)
        def get(self, url):
            pytest.fail("repository was dispatched as a download")
    first = source.acquire_operator(OP, RepositoryArchive(), tmp_path, "1.1.2", 1)
    offline = True
    calls.clear()
    resumed = source.acquire_operator(OP, RepositoryArchive(), tmp_path, "1.1.2", 1)
    assert resumed["sources"][0]["revision"] == first["sources"][0]["revision"] == "a" * 40
    assert resumed["shortfall"] == resumed["retryable_failures"] == 0
    assert not any("fetch" in command or "clone" in command for command in calls)


def test_published_commit_missing_from_clone_is_fetched(tmp_path, monkeypatch):
    calls = []
    revision = "a" * 40
    def git_run(command, **kwargs):
        calls.append(command)
        if "clone" in command:
            Path(command[-1]).mkdir()
        if "--verify" in command:
            raise source.subprocess.CalledProcessError(128, command)
        return SimpleNamespace(stdout=b"archive" if "archive" in command else (revision + "\n").encode())
    monkeypatch.setattr(source.subprocess, "run", git_run)
    _, record = source.repository_source("https://example.invalid/repo.git",
                                         {"source_revision": revision}, tmp_path)
    assert record["revision"] == revision
    assert any(command[-3:] == ["fetch", "origin", revision] for command in calls)


@pytest.mark.parametrize("nested", [False, True])
def test_root_relative_repository_pointer_uses_git_acquisition(tmp_path, monkeypatch, nested):
    pointer = {"repository_url": "/published/repo.git"}
    job = {"submission": pointer} if nested else pointer
    calls = []
    payload = bundle({"exp/kernel.cpp": "AscendC::Exp(x,y,n);"})
    def repository(url, received_job, root):
        calls.append((url, received_job, root))
        return payload, {"status": 200, "revision": "a" * 40}
    monkeypatch.setattr(source, "repository_source", repository)
    class NoDownloads:
        def get(self, url):
            pytest.fail("repository was dispatched as a download")
    url, = source.candidate_urls(job)
    files, _, record = source.published_source(url, job, OP, NoDownloads(), tmp_path)
    assert calls == [(source.BASE + "/published/repo.git", job, tmp_path)]
    assert "exp/kernel.cpp" in files
    assert record["revision"] == "a" * 40
