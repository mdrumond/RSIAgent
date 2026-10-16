import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from benchmarks.ascend.training_annex import inventory_campaign, package_campaign, report_digest


def write(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value))
    return target


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "campaign"
    write(root, "state.json", {"completed_waves": ["wave-001"],
                              "memory_bytes_base64": ["bGVzc29u"]})
    write(root, "campaign-identity.json", {"execution_profile": "test-profile"})
    write(root, "waves/wave-001/published.json", {"commits": []})
    branch = root / "waves/wave-001/branches/project"
    source = b"exact source version"
    digest = hashlib.sha256(source).hexdigest()
    obj = branch / "artifacts/objects" / digest[:2] / digest[2:]
    obj.parent.mkdir(parents=True)
    obj.write_bytes(source)
    (branch / "evidence.jsonl").write_text(json.dumps({
        "kind": "artifact", "payload": {"sha256": digest, "role": "source"}}) + "\n")
    write(root, "native-evidence/failure.json", {"stage": "compile", "exit_code": 1})
    return root


def test_portable_package_preserves_memory_failed_attempts_and_sources(campaign, tmp_path):
    write(campaign, ".env", {"secret": "excluded"})
    write(campaign, "waves/wave-001/.env.local", {"secret": "excluded"})
    write(campaign, "weights/model.json", {"weight": "excluded"})
    result = package_campaign(campaign, tmp_path / "package.tar.gz")
    assert result["complete"]
    with tarfile.open(tmp_path / "package.tar.gz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        assert not any(".env" in name or "weights" in name for name in archive.getnames())
        for entry in manifest["files"]:
            data = archive.extractfile("campaign/" + entry["path"]).read()
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert json.load(archive.extractfile("campaign/state.json"))["memory_bytes_base64"]
        assert json.load(archive.extractfile("campaign/native-evidence/failure.json"))["exit_code"] == 1


def test_missing_cas_and_publication_are_explicit(campaign):
    next(campaign.glob("waves/**/objects/*/*")).unlink()
    (campaign / "waves/wave-001/published.json").unlink()
    result = inventory_campaign(campaign)
    assert {i["kind"] for i in result["issues"]} == {
        "missing-or-corrupt-object", "missing-wave-publication"}


def test_pending_requires_explicit_checkpoint_and_never_overwrites(campaign, tmp_path):
    write(campaign, "pending.json", {"wave": "wave-002"})
    target = tmp_path / "checkpoint.tar.gz"
    with pytest.raises(ValueError, match="incomplete"):
        package_campaign(campaign, target)
    assert not target.exists()
    result = package_campaign(campaign, target, allow_pending=True)
    assert result["package_kind"] == "pending-checkpoint"
    with pytest.raises(FileExistsError):
        package_campaign(campaign, target, allow_pending=True)
    with pytest.raises(ValueError, match="outside"):
        package_campaign(campaign, campaign / "archive.tar.gz", allow_pending=True)


def test_raw_report_descriptor_and_verified_package(campaign, tmp_path):
    store = tmp_path / "reports"
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "summary.csv").write_text("metric,value\npipe,3\n")
    digest = report_digest(raw)
    record = {"evidence": {"profile": "registry-target", "remote_report": "/runtime/report"},
              "result": {"summary": {"report_sha256": digest}}}
    write(campaign, "measurements/run/record.json", record)
    manifest = inventory_campaign(campaign, artifact_store=store)
    report = manifest["remote_reports"][0]
    assert report["target"] == "registry-target"
    assert report["record_origin"] == "measurements/run/record.json"
    assert report["status"] == "missing"
    store.mkdir()
    raw.rename(store / report["id"])
    manifest = package_campaign(campaign, tmp_path / "raw.tar.gz", artifact_store=store)
    assert manifest["remote_reports"][0]["status"] == "available"
    assert manifest["raw_files"][0]["path"].endswith("summary.csv")
    (store / report["id"] / "summary.csv").write_text("changed")
    assert inventory_campaign(campaign, artifact_store=store)["remote_reports"][0]["status"] == "digest-mismatch"


def test_phase2_uncommitted_attempt_is_pending(tmp_path):
    root = tmp_path / "phase2"
    write(root, "cells/cell/protocol.json", {"schema": "phase2-protocol"})
    write(root, "cells/cell/attempts/attempt-01/attempt.json", {
        "attempt_id": "one", "attempt_sha256": "identity"})
    assert inventory_campaign(root)["boundary"]["pending"]
    write(root, "cells/cell/attempts/attempt-01/evidence.json", {
        "passed": False, "evidence_sha256": "evidence"})
    assert inventory_campaign(root)["boundary"]["pending"]
    write(root, "cells/cell/memory.jsonl", {
        "attempt_identity_sha256": "identity", "evidence_sha256": "evidence"})
    assert inventory_campaign(root)["complete"]


def test_inventory_is_read_only_and_uses_relative_paths(campaign):
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in campaign.rglob("*") if p.is_file()}
    first = inventory_campaign(campaign)
    assert first == inventory_campaign(campaign)
    assert all(not Path(entry["path"]).is_absolute() for entry in first["files"])
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in campaign.rglob("*") if p.is_file()}


def test_legacy_traces_and_launcher_provenance_are_retained(campaign, tmp_path):
    write(campaign, "sft-traces/one.json", {"conversation": []})
    write(campaign, "provider/one/provider.json", {"request_id": "one"})
    write(campaign, "campaign-run.json", {"phase": "phase1"})
    write(campaign, "collector-state.json", {"local_runtime": "excluded"})
    write(campaign, ".env", {"private": "excluded"})
    names = {entry["path"] for entry in inventory_campaign(campaign)["files"]}
    assert {"sft-traces/one.json", "provider/one/provider.json", "campaign-run.json"} <= names
    assert "collector-state.json" not in names
    target = tmp_path / "legacy.tar.gz"
    package_campaign(campaign, target)
    with tarfile.open(target) as archive:
        for path in ("sft-traces/one.json", "provider/one/provider.json", "campaign-run.json"):
            assert archive.extractfile("campaign/" + path).read() == (campaign / path).read_bytes()
        assert "campaign/.env" not in archive.getnames()


def test_concurrent_mutation_aborts_archive(campaign, tmp_path, monkeypatch):
    original = tarfile.TarFile.addfile
    changed = False

    def mutate(archive, info, *args, **kwargs):
        nonlocal changed
        result = original(archive, info, *args, **kwargs)
        if not changed:
            changed = True
            write(campaign, "pending.json", {"new": "work"})
        return result

    monkeypatch.setattr(tarfile.TarFile, "addfile", mutate)
    target = tmp_path / "raced.tar.gz"
    with pytest.raises(ValueError, match="changed"):
        package_campaign(campaign, target)
    assert not target.exists()
