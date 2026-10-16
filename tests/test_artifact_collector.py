import hashlib
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmarks.ascend.artifact_collector import ArtifactCollector, _SOURCE_DIGEST, report_digest
from benchmarks.ascend.training_annex import inventory_campaign


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from benchmarks.ascend import training_annex

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "memory.json").write_text('{"learned": true}')
    source = tmp_path / "source"
    source.mkdir()
    (source / "subdir").mkdir()
    (source / "subdir" / "raw.csv").write_text("raw profile")
    descriptor = dict(id="a" * 64, target="configured-target", remote_path="/reports/raw",
                      expected_sha256=report_digest(source), record_origin="closed-record.json",
                      kind="directory", digest_algorithm="relative-path-sha256-v1")
    discovered = []

    def inventory(*args, **kwargs):
        discovered.append(1)
        return {"remote_reports": [descriptor, descriptor]}

    monkeypatch.setattr(training_annex, "inventory_campaign", inventory)
    worker = ArtifactCollector(campaign, tmp_path / "store", tmp_path / "queue.json",
                               cpl_remote="fake-cpl", free_floor_bytes=0)
    calls = []
    responses = []

    def run(args, stdout=None, stderr=None, check=False, **kwargs):
        if args[2] == "capabilities":
            return SimpleNamespace(stdout=json.dumps({"backend": "remote-agent",
                                                     "capabilities": {"download": True}}))
        calls.append(args)
        response = responses.pop(0)
        staging = worker.store / (".partial-" + descriptor["id"])
        if response == "success":
            (staging / "subdir").mkdir(exist_ok=True)
            (staging / "subdir" / "raw.csv").write_text("raw profile")
            response = {"state": "completed"}
        elif response == "corrupt":
            (staging / "raw.csv").write_text("truncated")
            response = {"state": "completed"}
        elif response == "disconnect":
            stdout.write("not json")
            return SimpleNamespace(returncode=2)
        stdout.write(json.dumps(response))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    return worker, descriptor, responses, calls, discovered


def entry(worker):
    return next(iter(json.loads(worker.path.read_text())["entries"].values()))


def test_digest_and_publish_deduplicate(setup):
    worker, descriptor, responses, calls, discovered = setup
    responses.append("success")
    assert worker.collect_once(now=1)["counts"] == {"complete": 1}
    assert calls[0][1:4] == ["--json", "download", descriptor["target"]]
    assert calls[0][4] == "/reports/raw/"
    assert report_digest(worker.store / descriptor["id"]) == descriptor["expected_sha256"]
    worker.collect_once(now=2)
    worker.collect_once(now=301)
    assert len(calls) == 1
    assert len(discovered) == 2


def test_handle_disconnect_restart_and_fetch(setup):
    worker, descriptor, responses, calls, _ = setup
    handle = "remote:configured-target:transfer:123"
    responses.extend([{"state": "observation-unavailable", "handle": handle},
                      "disconnect", {"state": "completed", "handle": handle}, "success"])
    worker.collect_once(now=1)
    assert entry(worker)["handle"] == handle
    # Recreate worker as after process restart; never submit another download.
    restarted = ArtifactCollector(worker.root, worker.store, worker.path,
                                  cpl_remote="fake-cpl", free_floor_bytes=0)
    restarted.collect_once(now=121)
    restarted.collect_once(now=241)
    restarted.collect_once(now=242)
    assert [c[2] for c in calls] == ["download", "observe", "observe", "fetch"]
    assert all(c[3] == handle for c in calls[1:])
    assert entry(worker)["status"] == "complete"


def test_unknown_submission_is_held_without_duplicate(setup):
    worker, _, responses, calls, _ = setup
    responses.append("disconnect")
    worker.collect_once(now=1)
    worker.collect_once(now=10000)
    assert len(calls) == 1
    assert entry(worker)["status"] == "uncertain"


def test_native_partial_retry_and_defer(setup):
    worker, descriptor, responses, calls, _ = setup
    responses.extend([{"state": "failed"}] * 3 + ["success"])
    worker.collect_once(now=1)
    partial = worker.store / (".partial-" + descriptor["id"])
    (partial / "native-partial").mkdir()
    (partial / "native-partial" / "chunk").write_text("retained")
    worker.collect_once(now=120)
    assert len(calls) == 1
    worker.collect_once(now=121)
    worker.collect_once(now=241)
    assert entry(worker)["next_retry"] == 1141
    assert (partial / "native-partial" / "chunk").read_text() == "retained"
    worker.collect_once(now=1140)
    assert len(calls) == 3
    # Native transfer normally clears its completed partial directory itself.
    (partial / "native-partial" / "chunk").unlink()
    (partial / "native-partial").rmdir()
    worker.collect_once(now=1141)
    assert entry(worker)["status"] == "complete"


def test_corruption_never_published(setup):
    worker, descriptor, responses, _, _ = setup
    responses.append("corrupt")
    status = worker.collect_once(now=1)
    assert status["counts"] == {"retry": 1}
    assert not (worker.store / descriptor["id"]).exists()
    assert entry(worker)["error"] == "digest mismatch"
    assert list(worker.store.glob(".corrupt-*"))
    responses.append("success")
    worker.collect_once(now=121)
    assert entry(worker)["status"] == "complete"


def test_low_disk_pauses_and_campaign_unchanged(setup, monkeypatch):
    worker, _, responses, calls, _ = setup
    before = (worker.root / "memory.json").read_bytes()
    worker.floor = 10
    monkeypatch.setattr("shutil.disk_usage", lambda path: SimpleNamespace(free=1))
    assert worker.collect_once(now=1)["paused_low_disk"]
    assert calls == []
    worker.floor = 0
    responses.append({"state": "failed"})
    worker.collect_once(now=2)
    assert (worker.root / "memory.json").read_bytes() == before
    assert list(worker.root.iterdir()) == [worker.root / "memory.json"]


def test_inflight_journal_recovery(setup):
    worker, _, responses, calls, _ = setup
    worker.floor = 10**30
    worker.collect_once(now=1)
    state = json.loads(worker.path.read_text())
    item = next(iter(state["entries"].values()))
    item.update(status="inflight", operation="download")
    worker._save(state)
    handle = "remote:configured-target:transfer:retained"
    journal = worker.path.parent / (worker.path.name + ".transport.json")
    journal.write_text(json.dumps({"state": "running", "handle": handle}))
    worker.collect_once(now=2)
    assert entry(worker)["handle"] == handle
    assert calls == []


def test_ssh_interrupted_worker_resumes_same_partial_tree(setup):
    worker, descriptor, responses, calls, _ = setup
    worker.floor = 10**30
    worker.collect_once(now=1)
    state = json.loads(worker.path.read_text())
    item = next(iter(state["entries"].values()))
    item.update(status="inflight", operation="download", backend="ssh")
    worker._save(state)
    partial = worker.store / (".partial-" + descriptor["id"])
    partial.mkdir()
    (partial / "subdir").mkdir()
    (partial / "subdir" / "raw.csv").write_text("partial")
    worker.collect_once(now=2)
    assert entry(worker)["status"] == "retry"
    worker.floor = 0
    responses.append("success")
    worker.collect_once(now=122)
    assert calls[0][5] == str(partial)
    assert entry(worker)["status"] == "complete"


def test_campaign_locations_rejected(tmp_path):
    with pytest.raises(ValueError, match="outside campaign"):
        ArtifactCollector(tmp_path, tmp_path / "store", tmp_path / "state.json")


def test_child_holds_worker_lock_after_parent_closes_it(setup, monkeypatch):
    worker, _, _, calls, _ = setup
    # Use a real child for the inherited-descriptor lifetime, without transport.
    monkeypatch.undo()
    lock = worker.path.with_suffix(worker.path.suffix + ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    child = subprocess.Popen([sys.executable, "-c", "input()"], stdin=subprocess.PIPE,
                             pass_fds=(lock.fileno(),))
    lock.close()
    try:
        assert worker.collect_once(now=1)["counts"] == {"worker_busy": 1}
    finally:
        child.communicate(b"release\n", timeout=5)
    assert calls == []


def test_digest_matches_relative_path_contract(tmp_path):
    (tmp_path / "x").write_bytes(b"payload")
    expected = hashlib.sha256(b"x\0" + hashlib.sha256(b"payload").digest()).hexdigest()
    assert report_digest(tmp_path) == expected


def test_historical_a5_source_hash_handle_recovery_and_receipt(setup):
    worker, descriptor, responses, calls, _ = setup
    source_digest = descriptor["expected_sha256"]
    descriptor["expected_sha256"] = None
    job = "remote:configured-target:job:source-digest"
    responses.extend([{"state": "running", "handle": job}, "disconnect",
                      {"state": "completed", "handle": job}, "disconnect",
                      {"state": "completed", "handle": job,
                       "content": json.dumps({"source_sha256": source_digest})}, "success"])
    worker.collect_once(now=1)
    assert entry(worker)["source_hash_handle"] == job
    # Submission runs only a stdlib hash script, never a device workload.
    script = Path(calls[0][-1]).read_text()
    assert "import hashlib, json" in script
    assert "Path('/reports/raw')" in script
    restarted = ArtifactCollector(worker.root, worker.store, worker.path,
                                  cpl_remote="fake-cpl", free_floor_bytes=0)
    restarted.collect_once(now=121)
    restarted.collect_once(now=241)
    restarted.collect_once(now=242)
    restarted.collect_once(now=362)
    restarted.collect_once(now=363)
    assert [c[2] for c in calls] == ["run", "observe", "observe", "logs", "logs", "download"]
    assert all(c[3] == job for c in calls[1:5])
    assert entry(worker)["status"] == "complete"
    receipt = json.loads((worker.store / (descriptor["id"] + ".receipt.json")).read_text())
    assert receipt == {"descriptor_id": descriptor["id"], "source_sha256": source_digest,
                       "downloaded_sha256": source_digest, "digest_origin": "collection-time"}
    # Capture records remain unmodified: hydration belongs only to queue/receipt.
    assert descriptor["expected_sha256"] is None


def test_source_hash_uncertain_submission_never_resubmits_even_on_ssh(setup):
    worker, descriptor, responses, calls, _ = setup
    descriptor["expected_sha256"] = None
    worker.floor = 10**30
    worker.collect_once(now=1)
    state = json.loads(worker.path.read_text())
    item = next(iter(state["entries"].values()))
    item["backend"] = "ssh"
    worker._save(state)
    worker.floor = 0
    responses.append("disconnect")
    worker.collect_once(now=2)
    worker.collect_once(now=10000)
    assert [c[2] for c in calls] == ["run"]
    assert entry(worker)["status"] == "uncertain"


def test_source_hash_inflight_journal_retains_job_handle(setup):
    worker, descriptor, _, calls, _ = setup
    descriptor["expected_sha256"] = None
    worker.floor = 10**30
    worker.collect_once(now=1)
    state = json.loads(worker.path.read_text())
    item = next(iter(state["entries"].values()))
    item.update(status="inflight", operation="hash-submit")
    worker._save(state)
    job = "remote:configured-target:job:hash-recovered"
    journal = worker.path.parent / (worker.path.name + ".transport.json")
    journal.write_text(json.dumps({"state": "running", "handle": job}))
    worker.collect_once(now=2)
    assert entry(worker)["status"] == "hash-observe"
    assert entry(worker)["source_hash_handle"] == job
    assert calls == []


def test_readonly_hash_payload_matches_report_digest(tmp_path, capsys):
    source = tmp_path / "a 'quoted' report"
    source.mkdir()
    (source / "raw.csv").write_bytes(b"payload")
    exec(compile(_SOURCE_DIGEST.replace("SOURCE_PATH", repr(str(source))), "hash-payload", "exec"))
    assert json.loads(capsys.readouterr().out)["source_sha256"] == report_digest(source)
    assert list(source.iterdir()) == [source / "raw.csv"]


def test_historical_a5_receipt_recovers_without_hash_roundtrip(setup):
    worker, descriptor, _, calls, _ = setup
    digest = descriptor["expected_sha256"]
    descriptor.update(expected_sha256=None, resolved_sha256=digest,
                      digest_origin="collection-time")
    destination = worker.store / descriptor["id"]
    (destination / "subdir").mkdir(parents=True)
    (destination / "subdir" / "raw.csv").write_text("raw profile")
    assert worker.collect_once(now=1)["counts"] == {"complete": 1}
    assert calls == []
    assert json.loads((worker.store / (descriptor["id"] + ".receipt.json")).read_text())["source_sha256"] == digest


def test_historical_a5_collector_receipt_accepted_by_annex(setup, monkeypatch):
    worker, descriptor, responses, _, _ = setup
    source_digest = descriptor["expected_sha256"]
    (worker.root / "campaign-identity.json").write_text(json.dumps({"execution_profile": "configured-target"}))
    retained = worker.root / "measurements" / "record.json"
    retained.parent.mkdir()
    retained.write_text(json.dumps({"capture": {"evidence": {
        "remote_tree": "/reports/raw", "remote_tree_retained": True,
        "archive_sha256": "b" * 64}}}))
    monkeypatch.setattr("benchmarks.ascend.training_annex.inventory_campaign", inventory_campaign)
    actual_descriptor = inventory_campaign(worker.root)["remote_reports"][0]
    descriptor.clear()
    descriptor.update(actual_descriptor)
    job = "remote:configured-target:job:source-digest"
    responses.extend([{"state": "completed", "handle": job},
                      {"state": "completed", "handle": job,
                       "content": json.dumps({"source_sha256": source_digest})}, "success"])
    worker.collect_once(now=1)
    worker.collect_once(now=2)
    worker.collect_once(now=3)
    report = inventory_campaign(worker.root, artifact_store=worker.store)["remote_reports"][0]
    assert report["id"] == descriptor["id"]
    assert report["status"] == "available"
    assert report["expected_sha256"] is None
    assert report["resolved_sha256"] == source_digest
    assert report["digest_origin"] == "collection-time"


@pytest.mark.parametrize("persisted", [False, True])
def test_incomplete_descriptor_does_not_starve_valid_queue(setup, monkeypatch, persisted):
    worker, descriptor, responses, calls, _ = setup
    incomplete = dict(descriptor, id="b" * 64, target=None, status="incomplete-descriptor")
    monkeypatch.setattr("benchmarks.ascend.training_annex.inventory_campaign",
                        lambda *args, **kwargs: {"remote_reports": [incomplete, descriptor]})
    if persisted:
        # Repair queues created before incomplete descriptors were deferred.
        worker._save({"entries": {incomplete["id"]: dict(incomplete, status="pending")}})
    responses.append("success")
    status = worker.collect_once(now=1)
    assert status["counts"] == {"incomplete-descriptor": 1, "complete": 1}
    assert len(calls) == 1
    assert calls[0][3] == descriptor["target"]
    state = json.loads(worker.path.read_text())
    assert state["entries"][incomplete["id"]]["status"] == "incomplete-descriptor"
