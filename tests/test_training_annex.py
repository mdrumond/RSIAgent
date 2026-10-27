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


def test_package_preserves_feedback_and_investigation_lineage(campaign, tmp_path):
    feedback = {'feedback_id': 'observation', 'outcome': {'completed': False},
                'investigation': {'status': 'deferred', 'reason': 'isolate padded shape',
                                  'retries': ['original-target-retry']}}
    phase1_feedback = {'selected_ids': ['observation'], 'source_root': '/original/feedback'}
    investigations = {'projects': [{'project_id': 'probe-1', 'feedback_id': 'observation'}]}
    files = {'feedback/observation.json': feedback, 'phase2-feedback.json': phase1_feedback,
             'failure-investigations.json': investigations}
    for path, value in files.items():
        write(campaign, path, value)
    destination = tmp_path / 'feedback.tar.gz'
    result = package_campaign(campaign, destination)
    assert result['complete']
    with tarfile.open(destination) as archive:
        for path, value in files.items():
            assert json.load(archive.extractfile('campaign/' + path)) == value


@pytest.mark.parametrize("mutation", [None, "missing-bundle", "corrupt-object"])
def test_a5_production_bundle_and_measurement_trees_are_portable(campaign, tmp_path, mutation):
    from benchmarks.a5kernels.evidence import EvidenceLedger
    from benchmarks.a5kernels.phase1_project_bz import CatlassProjectBackend
    from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
    from benchmarks.a5kernels.profiling_bz import BZProfileBackend
    from tests.test_a5_phase1_project_bz import FakeExecutor
    from tests.test_a5_phase1_project_executor import _spec
    backend = CatlassProjectBackend(FakeExecutor([]), session_wrapper="/checked/session.sh",
                                   evidence_root=campaign / "project-bundles")
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "host-ledger"))
    prepared = executor.prepare(_spec("movement"))
    request = executor.request(prepared, "# exact retained Catlass source\n")
    identity, files = backend._bundle(request)
    digest = backend._retain_evidence({"schema": "a5-phase1-project-observation-v1",
        "identity": identity, "mode": "compile", "record": {"status": "compile-failure"}})
    profiler = object.__new__(BZProfileBackend)
    profiler._evidence_directory = campaign / "measurement-evidence"
    profiler._retain_timing_evidence("replay-one", "exact checked-wrapper stdout\n")
    cas = campaign / "project-bundles/cas" / digest[:2] / (digest + ".json")
    if mutation == "missing-bundle":
        (campaign / "project-bundles" / identity / "kernel.py").unlink()
    elif mutation == "corrupt-object":
        cas.write_text('{}')
    if mutation:
        assert not inventory_campaign(campaign)["complete"]
        with pytest.raises(ValueError, match="incomplete"):
            package_campaign(campaign, tmp_path / "a5.tar.gz")
        return
    assert package_campaign(campaign, tmp_path / "a5.tar.gz")["complete"]
    with tarfile.open(tmp_path / "a5.tar.gz") as archive:
        for source in files:
            assert archive.extractfile(f"campaign/project-bundles/{identity}/{source.relative_path}").read() == source.content.encode()
        assert archive.extractfile("campaign/" + cas.relative_to(campaign).as_posix()).read() == cas.read_bytes()
        assert archive.extractfile("campaign/measurement-evidence/replay-one/timing-stdout.txt").read() == b"exact checked-wrapper stdout\n"


def test_missing_cas_and_publication_are_explicit(campaign):
    next(campaign.glob("waves/**/objects/*/*")).unlink()
    (campaign / "waves/wave-001/published.json").unlink()
    result = inventory_campaign(campaign)
    assert {i["kind"] for i in result["issues"]} == {
        "missing-or-corrupt-object", "missing-wave-publication"}


@pytest.fixture
def adaptive_campaign(tmp_path, request):
    from dataclasses import replace
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
    from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
    from benchmarks.a5kernels.phase1_memory import HostFact
    from tests.test_a5_phase1_adaptive import _project, _result, _wave, SEED

    root = tmp_path / "adaptive"
    root.mkdir()
    (root / "initial-memory.bin").write_bytes(SEED)
    count = 0
    partial = getattr(request, "param", False)
    def complete(messages):
        nonlocal count
        count += 1
        spec = _project(f"project-{count}", "gm-ub-movement")
        spec["learning_focus"] = ["pipe-mte2" if count == 1 else "pipe-mte3"]
        specs = [spec, _project("offline", "simd-vector")] if partial else [spec]
        return _wave(messages[-1]["content"], specs)
    def execute(request):
        if request.spec.project_id == "offline":
            raise OSError("retained remote observation unavailable")
        objects = ContentAddressedArtifacts(request.branch_root / "artifacts")
        source = objects.put_bytes(b"source")
        host = objects.put_bytes(b"host verification")
        annex = objects.put_bytes(b"extra admitted evidence")
        result = _result(request.spec, True, "verified")
        return replace(result, memory=replace(result.memory, source_revision=source,
            host_facts=(HostFact("host-verification", "host passed", host),),
            lessons=(), no_new_knowledge=True, admitted_artifact_sha256=(annex,),
            artifact_store="artifacts"))
    AdaptivePhase1Coordinator(root, "Learn pipes", SEED, CurriculumAgent(complete),
                             execute, max_parallel=1, project_target=2, allow_partial=partial).run()
    assert inventory_campaign(root)["complete"]
    return root


@pytest.mark.parametrize("payload", [b"source", b"host verification", b"extra admitted evidence"])
@pytest.mark.parametrize("mutation", ["remove", "corrupt"])
def test_memory_only_cas_references_are_required(adaptive_campaign, tmp_path, payload, mutation):
    digest = hashlib.sha256(payload).hexdigest()
    obj = adaptive_campaign / "waves/wave-001/branches/project-1/artifacts/objects" / digest[:2] / digest[2:]
    if mutation == "remove":
        obj.unlink()
    else:
        obj.write_bytes(b"corrupt")
    issues = inventory_campaign(adaptive_campaign)["issues"]
    assert any(issue["kind"] == "missing-or-corrupt-object" and issue["sha256"] == digest for issue in issues)
    with pytest.raises(ValueError, match="incomplete"):
        package_campaign(adaptive_campaign, tmp_path / "incomplete.tar.gz")


@pytest.mark.parametrize("mutation", ["memory", "outcome", "order", "seed", "chain"])
def test_adaptive_state_is_bound_to_published_chain(adaptive_campaign, mutation):
    state = json.loads((adaptive_campaign / "state.json").read_bytes())
    if mutation == "memory":
        state["memory_bytes_base64"] = state["memory_bytes_base64"][::-1]
    elif mutation == "outcome":
        state["prior_outcomes"][0]["summary"] = "stale summary"
    elif mutation == "order":
        state["completed_waves"].reverse()
    elif mutation == "seed":
        (adaptive_campaign / "initial-memory.bin").write_bytes(b"wrong seed")
    else:
        # Even an internally self-consistent second-wave publication cannot
        # replace its authoritative pre-wave snapshot with a stale one.
        plan = adaptive_campaign / "waves/wave-002/plan.json"
        value = json.loads(plan.read_bytes())
        value["wave"]["pre_wave_memory_sha256"] = "f" * 64
        plan.write_text(json.dumps(value))
    write(adaptive_campaign, "state.json", state)
    assert "invalid-adaptive-memory-chain" in {issue["kind"] for issue in inventory_campaign(adaptive_campaign)["issues"]}


def test_legacy_campaign_without_seed_reports_missing_seed_not_false_chain_mismatch(adaptive_campaign):
    (adaptive_campaign / "initial-memory.bin").unlink()
    kinds = {issue["kind"] for issue in inventory_campaign(adaptive_campaign)["issues"]}
    assert kinds == {"missing-initial-memory"}


def test_adaptive_archive_keeps_memory_only_objects_without_mutating_evidence(adaptive_campaign, tmp_path):
    before = {path: path.read_bytes() for path in adaptive_campaign.rglob("*") if path.is_file()}
    destination = tmp_path / "adaptive.tar.gz"
    assert package_campaign(adaptive_campaign, destination)["complete"]
    with tarfile.open(destination) as archive:
        for path, payload in before.items():
            if path.name == "allocation.lock":
                assert "campaign/allocation.lock" not in archive.getnames()
                continue
            assert archive.extractfile("campaign/" + path.relative_to(adaptive_campaign).as_posix()).read() == payload
    assert all(path.read_bytes() == payload for path, payload in before.items())


def test_consolidated_memory_cannot_disagree_with_ordered_deltas(campaign):
    path = campaign / "waves/wave-001/consolidated-memory.json"
    value = json.loads(path.read_bytes())
    value["deltas"][0]["memory_delta_hex"] = b"wrong delta".hex()
    path.write_text(json.dumps(value))
    publication = campaign / "waves/wave-001/published.json"
    value = json.loads(publication.read_bytes())
    value["consolidated_memory_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    publication.write_text(json.dumps(value))
    assert "invalid-wave-publication" in {issue["kind"] for issue in inventory_campaign(campaign)["issues"]}


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


@pytest.mark.parametrize("direct_digest", [True, False])
def test_raw_report_descriptor_and_verified_package(campaign, tmp_path, direct_digest):
    store = tmp_path / "reports"
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "summary.csv").write_text("metric,value\npipe,3\n")
    digest = report_digest(raw)
    record = {"evidence": {"profile": "registry-target", "remote_report": "/runtime/report"},
              "result": ({"report_sha256": digest} if direct_digest else
                         {"summary": {"report_sha256": digest}})}
    write(campaign, "measurements/run/record.json", record)
    manifest = inventory_campaign(campaign, artifact_store=store)
    report = manifest["remote_reports"][0]
    assert report["target"] == "registry-target"
    assert report["record_origin"] == "measurements/run/record.json"
    assert report["status"] == "missing"
    assert report["expected_sha256"] == digest
    assert report["digest_origin"] == "capture-time"
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


@pytest.mark.parametrize("field,name", [
    ("phase1_memory_sha256", "phase1-memory.jsonl"),
    ("phase1_terminal_sha256", "phase1-terminal.json"),
])
@pytest.mark.parametrize("mutation", ["remove", "corrupt"])
def test_native_and_standard_snapshot_explicit_digests(campaign, field, name, mutation):
    snapshot = campaign / "cells/cell/snapshot"
    memory = write(snapshot, "phase1-memory.jsonl", {"lesson": "retained"})
    terminal = write(snapshot, "phase1-terminal.json", {"status": "complete"})
    write(snapshot, "snapshot.json", {
        "phase1_memory_sha256": hashlib.sha256(memory.read_bytes()).hexdigest(),
        "phase1_terminal_sha256": hashlib.sha256(terminal.read_bytes()).hexdigest()})
    assert inventory_campaign(campaign)["complete"]
    if mutation == "remove":
        (snapshot / name).unlink()
    else:
        (snapshot / name).write_text("changed retained snapshot bytes")
    assert {"kind": "missing-or-corrupt-snapshot-file", "path": name} in inventory_campaign(campaign)["issues"]
    with pytest.raises(ValueError, match="incomplete"):
        package_campaign(campaign, campaign.parent / "corrupt-snapshot.tar.gz")


def test_report_digest_rejects_directory_symlinks(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "metric.csv").write_text("retained")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "omitted.csv").write_text("must not be silently omitted")
    (raw / "linked-directory").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        report_digest(raw)


@pytest.mark.parametrize("mutation", [None, "missing-marker", "corrupt-marker", "missing-evidence", "corrupt-evidence", "pending"])
def test_closed_failure_wave_is_complete_only_with_authenticated_evidence(campaign, tmp_path, mutation):
    from benchmarks.a5kernels.phase1_curriculum import WaveSpec
    from benchmarks.a5kernels.phase1_wave_runtime import LiveWaveCoordinator, WaveProjectResult
    from benchmarks.a5kernels.phase1_project import ExecutionEvidence, EvidenceClassification

    plan = json.loads((campaign / "waves/wave-001/plan.json").read_text())["wave"]
    plan["wave_id"] = "wave-002"
    wave = WaveSpec.from_mapping(plan)
    failed = campaign / "waves/wave-002"
    with pytest.raises(RuntimeError, match="not verified"):
        LiveWaveCoordinator(failed, wave, b"initial", lambda _: WaveProjectResult(
            b"retained failed experiment", ExecutionEvidence(EvidenceClassification.HOST_MISMATCH, "mismatch"),
            b'{"host":"mismatch"}'), 1).run()
    outcome = failed / "branches/project/outcome.json"
    marker = write(failed, "closed-failure.json", {"wave_fingerprint": wave.fingerprint,
        "outcome_sha256": hashlib.sha256(outcome.read_bytes()).hexdigest()})
    evidence = failed / "branches/project/execution-evidence.bin"
    if mutation == "missing-marker":
        marker.unlink()
    elif mutation == "corrupt-marker":
        write(failed, "closed-failure.json", {"wave_fingerprint": wave.fingerprint, "outcome_sha256": "bad"})
    elif mutation == "missing-evidence":
        evidence.unlink()
    elif mutation == "corrupt-evidence":
        evidence.write_bytes(b"changed")
    elif mutation == "pending":
        write(campaign, "pending.json", {"wave_id": "wave-002"})
    assert inventory_campaign(campaign)["complete"] is (mutation is None)
    target = tmp_path / "failure.tar.gz"
    if mutation:
        with pytest.raises(ValueError, match="incomplete"):
            package_campaign(campaign, target)
    else:
        assert package_campaign(campaign, target)["package_kind"] == "complete"


@pytest.mark.parametrize("adaptive_campaign", [True], indirect=True)
def test_partial_wave_inventory_and_portable_packaging_preserve_interruption(adaptive_campaign, tmp_path):
    branch = "waves/wave-001/branches/offline"
    assert not (adaptive_campaign / branch / "outcome.json").exists()
    retained = (adaptive_campaign / branch / "interruption.json").read_bytes()
    assert inventory_campaign(adaptive_campaign)["complete"]
    package = tmp_path / "partial.tar.gz"
    assert package_campaign(adaptive_campaign, package)["complete"]
    with tarfile.open(package) as archive:
        assert archive.extractfile("campaign/" + branch + "/interruption.json").read() == retained
        archive.extractall(tmp_path / "reloaded", filter="data")
    assert inventory_campaign(tmp_path / "reloaded/campaign")["complete"]
    (adaptive_campaign / branch / "interruption.json").write_text("{}")
    issues = inventory_campaign(adaptive_campaign)["issues"]
    assert "invalid-wave-publication" in {issue["kind"] for issue in issues}

def test_remediation_checkpoint_restore_keeps_budget_and_pending_retry(campaign, tmp_path):
    from benchmarks.ascend.automatic_remediation import FailureTimeBudget
    budget_path = 'failure-budgets/failure.json'
    remediation = dict(schema='automatic-remediation-v1', retry_feedback_ids=['previous'],
        failures={'failure': dict(retries=[dict(root='/retained/original-retries/failure/retry-1',
            feedback_ids=['previous'])], pending='retry-2', active_seconds=1200, remaining_seconds=600)})
    files = {'remediation.json': remediation,
        budget_path: dict(max_seconds=1800, elapsed_seconds=1200, active_since=None),
        'original-retries/failure/retry-2/state.json': dict(completed_waves=[], pending='native-handle'),
        'original-retries/failure/retry-2/plan.json': dict(original_project_id='unchanged')}
    for relative, body in files.items():
        write(campaign, relative, body)
    memory = campaign/'original-retries/failure/retry-2/retry-memory.bin'
    memory.write_bytes(b'frozen-updated-memory')
    destination = tmp_path/'remediation.tar.gz'
    assert package_campaign(campaign, destination)['complete']
    restored = tmp_path/'restored'
    with tarfile.open(destination) as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.startswith('campaign/'):
                output = restored/member.name.removeprefix('campaign/')
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(archive.extractfile(member).read())
    assert json.loads((restored/'remediation.json').read_bytes()) == remediation
    assert FailureTimeBudget(restored/budget_path, 1800).remaining_seconds == 600
    assert (restored/'original-retries/failure/retry-2/retry-memory.bin').read_bytes() == b'frozen-updated-memory'
    assert json.loads((restored/'original-retries/failure/retry-2/state.json').read_bytes())['pending'] == 'native-handle'
