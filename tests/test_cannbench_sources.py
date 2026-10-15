import io
import json
from pathlib import Path
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
