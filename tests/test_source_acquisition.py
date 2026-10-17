import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from benchmarks.a5kernels.source_acquisition import acquire_source, acquire_sources, git, main


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    root = tmp_path / "upstream"
    root.mkdir()
    git("init", "--quiet", cwd=root)
    git("config", "user.name", "test", cwd=root)
    git("config", "user.email", "test@example.test", cwd=root)
    git("remote", "add", "origin", "https://example.test/kernel.git", cwd=root)
    (root / "kernel.py").write_text("def add(a, b):\n    return a + b\n")
    (root / "LICENSE").write_text("Example license\n")
    (root / "Third_Party_Open_Source_Software_Notice").write_text("Third-party notices\n")
    (root / "中文.md").write_text("文档\n")
    (root / "opaque.cpp").write_bytes(b"\xff\xfe")
    (root / "binary.h").write_bytes(b"hello\0world")
    (root / "build").mkdir()
    (root / "build/generated.py").write_text("generated = True\n")
    (root / "link.py").symlink_to("kernel.py")
    git("add", ".", cwd=root)
    git("commit", "--quiet", "-m", "sources", cwd=root)
    published = tmp_path / "published.git"
    git("clone", "--quiet", "--bare", str(root), str(published))

    def local_transport(*args, **kwargs):
        if args[0] == "fetch":
            args = tuple(str(published) if arg == "origin" else arg for arg in args)
        return git(*args, **kwargs)

    monkeypatch.setattr("benchmarks.a5kernels.source_acquisition.git", local_transport)
    return root


def spec():
    return {"name": "kernel", "repository": "https://example.test/kernel.git", "kernel_languages": ["triton"]}


def test_acquires_pinned_real_source_with_hashes_and_exclusions(upstream, tmp_path):
    result = acquire_source(spec(), tmp_path / "artifacts", upstream)
    assert result["revision"] == git("rev-parse", "HEAD", cwd=upstream)
    assert result["status"] == "complete"
    assert result["kernel_languages"] == ["triton"]
    assert result["licenses"] == ["LICENSE", "Third_Party_Open_Source_Software_Notice"]
    files = {file["path"]: file for file in result["files"]}
    assert set(files) == {"kernel.py", "LICENSE", "Third_Party_Open_Source_Software_Notice", "中文.md"}
    assert files["kernel.py"]["sha256"] == hashlib.sha256((upstream / "kernel.py").read_bytes()).hexdigest()
    omitted = {item["path"]: item["reason"] for item in result["omitted"]}
    assert omitted["binary.h"] == "binary"
    assert omitted["opaque.cpp"] == "not-utf8"
    assert omitted["link.py"] == "symlink-or-submodule"
    assert omitted["build/generated.py"] == "excluded-generated-or-non-source"


def test_pinned_resume_is_offline_and_checks_modifications(upstream, tmp_path, monkeypatch):
    first = acquire_source(spec(), tmp_path / "artifacts", upstream)
    pinned = {**spec(), "revision": first["revision"]}
    real_run = subprocess.run

    def no_network(args, **kwargs):
        assert args[1] not in {"clone", "fetch"}
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", no_network)
    second = acquire_source(pinned, tmp_path / "artifacts")
    assert first["files"] == second["files"]
    (Path(first["root"]) / "kernel.py").write_text("modified\n")
    with pytest.raises(ValueError, match="modifications"):
        acquire_source(pinned, tmp_path / "artifacts")


def test_local_clone_must_match_real_upstream(upstream, tmp_path):
    wrong = {**spec(), "repository": "https://different.test/source.git"}
    with pytest.raises(ValueError, match="origin"):
        acquire_source(wrong, tmp_path / "artifacts", upstream)


@pytest.mark.parametrize("pinned", [False, True])
def test_reuse_rejects_clean_unpublished_commit(upstream, tmp_path, pinned):
    (upstream / "kernel.py").write_text("unpublished = True\n")
    git("add", "kernel.py", cwd=upstream)
    git("commit", "--quiet", "-m", "local only", cwd=upstream)
    revision = git("rev-parse", "HEAD", cwd=upstream)
    requested = {**spec(), "revision": revision} if pinned else spec()
    assert git("status", "--porcelain", cwd=upstream) == ""
    with pytest.raises(ValueError, match="not reachable from upstream"):
        acquire_source(requested, tmp_path / "artifacts", upstream)
    assert not (tmp_path / "artifacts" / "kernel" / revision).exists()


def test_reuse_can_select_published_ancestor_of_local_head(upstream, tmp_path):
    revision = git("rev-parse", "HEAD", cwd=upstream)
    (upstream / "kernel.py").write_text("local = True\n")
    git("add", "kernel.py", cwd=upstream)
    git("commit", "--quiet", "-m", "local only", cwd=upstream)
    result = acquire_source({**spec(), "revision": revision}, tmp_path / "artifacts", upstream)
    assert result["revision"] == revision
    assert (Path(result["root"]) / "kernel.py").read_text().startswith("def add")


def test_partial_success_preserved_and_errors_sanitized(upstream, tmp_path):
    bad = {"name": "missing", "repository": "https://wrong.test/missing.git"}
    result = acquire_sources([spec(), bad], tmp_path / "artifacts", reuse={"kernel": upstream, "missing": upstream})
    assert [source["status"] for source in result["sources"]] == ["complete", "failed"]
    assert result["sources"][1]["error"] == "ValueError"


def test_duplicate_names_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        acquire_sources([spec(), spec()], tmp_path)


@pytest.mark.parametrize("change", [{"name": "../bad"}, {"repository": "ssh://example.test/repo"}, {"revision": "main"}])
def test_invalid_source_rejected(change, tmp_path):
    with pytest.raises(ValueError):
        acquire_source({**spec(), **change}, tmp_path)


def test_cli_publishes_inventory(upstream, tmp_path):
    config, output = tmp_path / "config.json", tmp_path / "inventory.json"
    config.write_text(json.dumps({"sources": [spec()]}))
    assert main(["--config", str(config), "--destination", str(tmp_path / "downloads"), "--inventory", str(output), "--reuse", f"kernel={upstream}"]) == 0
    assert json.loads(output.read_text())["sources"][0]["files"]


def test_autocrlf_checkout_exports_canonical_blobs_without_changing_input(upstream, tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.autocrlf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")
    original = (upstream / "kernel.py").read_bytes()
    result = acquire_source(spec(), tmp_path / "artifacts", upstream)
    root = Path(result["root"])
    assert root.name == result["revision"] + "-raw"
    checkout = root.with_name(result["revision"])
    assert b"\r\n" in (checkout / "kernel.py").read_bytes()
    assert (root / "kernel.py").read_bytes() == original
    assert (upstream / "kernel.py").read_bytes() == original
    assert git("status", "--porcelain", cwd=checkout) == ""
    pinned = {**spec(), "revision": result["revision"]}
    assert acquire_source(pinned, tmp_path / "artifacts")["root"] == str(root)
