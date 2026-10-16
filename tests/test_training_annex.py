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
    from benchmarks.a5kernels.phase1_curriculum import WaveSpec
    from benchmarks.a5kernels.phase1_wave_runtime import LiveWaveCoordinator, WaveProjectResult
    from benchmarks.a5kernels.phase1_project import ExecutionEvidence, EvidenceClassification

    root = tmp_path / "campaign"
    write(root, "state.json", {"completed_waves": ["wave-001"],
                              "memory_bytes_base64": ["bGVzc29u"]})
    write(root, "campaign-identity.json", {"execution_profile": "test-profile"})
    memory = b"initial"
    wave = WaveSpec.from_mapping({"wave_id": "wave-001", "pre_wave_coverage": [],
        "pre_wave_memory_sha256": hashlib.sha256(memory).hexdigest(), "projects": [{
            "requires": [], "spec": {"project_id": "project", "capabilities": ["gm-ub-movement"],
            "shape": [32], "dtype": "float32", "block_count": 1, "padding_multiple": 1,
            "failure_mode": "none", "timing_lengths": [], "profile_metric": None,
            "hypothesis": "Verify a copy using the host reference.", "graph": {
                "inputs": [{"name": "x", "shape": [32]}], "nodes": [{"name": "y",
                "op": "copy", "operands": ["x"], "axis": None}], "output": "y"}}}]})
    LiveWaveCoordinator(root / "waves/wave-001", wave, memory, lambda _: WaveProjectResult(
        b"memory delta", ExecutionEvidence(EvidenceClassification.HOST_VERIFIED, "verified"),
        b'{"host":"verified"}'), 1).run()
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


@pytest.mark.parametrize("artifact,mutation", [
    ("candidate.cpp", "remove"), ("candidate.cpp", "corrupt"),
    ("attempt.json", "remove"), ("attempt.json", "corrupt"),
    ("evidence.json", "remove"), ("evidence.json", "corrupt"),
    ("directory", "remove"),
])
def test_phase2_uncommitted_attempt_is_pending(tmp_path, artifact, mutation):
    root = tmp_path / "phase2"
    write(root, "cells/cell/protocol.json", {"schema": "phase2-protocol"})
    attempt = {"attempt_id": "one"}
    canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    attempt["attempt_sha256"] = canonical(attempt)
    write(root, "cells/cell/attempts/attempt-01/attempt.json", attempt)
    assert inventory_campaign(root)["boundary"]["pending"]
    source = root / "cells/cell/attempts/attempt-01/candidate.cpp"
    source.write_text("source")
    evidence = {"passed": False, "candidate_sha256": hashlib.sha256(b"source").hexdigest()}
    evidence["evidence_sha256"] = canonical(evidence)
    write(root, "cells/cell/attempts/attempt-01/evidence.json", evidence)
    assert inventory_campaign(root)["boundary"]["pending"]
    write(root, "cells/cell/memory.jsonl", {
        "attempt": 1, "attempt_identity_sha256": attempt["attempt_sha256"],
        "evidence_sha256": evidence["evidence_sha256"],
        "candidate_sha256": evidence["candidate_sha256"], "passed": False})
    assert inventory_campaign(root)["complete"]
    if artifact == "directory":
        source.parent.rename(tmp_path / "removed-attempt")
    else:
        target = source.parent / artifact
        if mutation == "remove":
            target.unlink()
        else:
            target.write_text("corrupt" if artifact == "candidate.cpp" else '{"changed":true}')
    assert "missing-or-corrupt-committed-attempt" in {i["kind"] for i in inventory_campaign(root)["issues"]}
    with pytest.raises(ValueError, match="incomplete"):
        package_campaign(root, tmp_path / "missing-commit.tar.gz")


@pytest.mark.parametrize("artifact,mutation", [
    ("branches/project/memory-delta.bin", "remove"),
    ("branches/project/memory-delta.bin", "corrupt"),
    ("branches/project/execution-evidence.bin", "remove"),
    ("branches/project/execution-evidence.bin", "corrupt"),
    ("consolidated-memory.json", "corrupt"),
])
def test_phase1_publication_authenticates_retained_commits(campaign, artifact, mutation):
    target = campaign / "waves/wave-001" / artifact
    if mutation == "remove":
        target.unlink()
    else:
        target.write_text("corrupted retained evidence")
    assert "invalid-wave-publication" in {i["kind"] for i in inventory_campaign(campaign)["issues"]}
    with pytest.raises(ValueError, match="incomplete"):
        package_campaign(campaign, campaign.parent / "incomplete.tar.gz")


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


def test_a5_cas_capture_uses_collection_receipt_not_compact_archive_hash(campaign, tmp_path):
    capture = {"capture": {"execution_id": "execution", "replay_id": "profile",
                          "evidence": {"remote_tree": "/runtime/full-msprof",
                                       "remote_tree_retained": True,
                                       "archive_sha256": "a" * 64}},
               "archive_encoding": "base64", "archive": "Y29tcGFjdA=="}
    data = json.dumps(capture).encode()
    digest = hashlib.sha256(data).hexdigest()
    obj = campaign / "artifacts/objects" / digest[:2] / digest[2:]
    obj.parent.mkdir(parents=True)
    obj.write_bytes(data)
    write(campaign, "measurements/inline/record.json", {"payload": capture})
    store = tmp_path / "raw-store"
    result = inventory_campaign(campaign, artifact_store=store)
    report = result["remote_reports"][0]
    assert len(result["remote_reports"]) == 2
    assert result["remote_reports"][1]["id"] == report["id"]
    assert report["target"] == "test-profile"
    assert report["expected_sha256"] is None
    assert report["status"] == "digest-unavailable"
    raw = store / report["id"]
    raw.mkdir(parents=True)
    (raw / "full.csv").write_text("full raw report")
    raw_digest = report_digest(raw)
    receipt = {"descriptor_id": report["id"], "source_sha256": raw_digest,
               "downloaded_sha256": raw_digest, "digest_origin": "collection-time"}
    write(store, report["id"] + ".receipt.json", receipt)
    collected = inventory_campaign(campaign, artifact_store=store)["remote_reports"][0]
    assert collected["id"] == report["id"]
    assert collected["expected_sha256"] is None
    assert collected["resolved_sha256"] == raw_digest
    assert collected["digest_origin"] == "collection-time"
    assert collected["status"] == "available"
    target = tmp_path / "a5.tar.gz"
    package_campaign(campaign, target, artifact_store=store)
    with tarfile.open(target) as archive:
        assert json.load(archive.extractfile("reports/" + report["id"] + ".receipt.json")) == receipt
        assert archive.getnames().count("reports/" + report["id"] + "/full.csv") == 1
    receipt["source_sha256"] = "b" * 64
    write(store, report["id"] + ".receipt.json", receipt)
    assert inventory_campaign(campaign, artifact_store=store)["remote_reports"][0]["status"] == "digest-unavailable"
