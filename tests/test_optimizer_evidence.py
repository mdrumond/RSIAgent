import json

import pytest

from benchmarks.ascend.optimizer_evidence import freeze_campaign, verify_campaign
from benchmarks.ascend.training_annex import inventory_campaign, package_campaign
from optimizer.jobs import JobStore, canonical, sha
from optimizer.memory import verify_snapshot


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def campaign(root, *, job_seed=None):
    root.mkdir(parents=True, exist_ok=True)
    seed = {"schema": "optimizer-memory-snapshot-v1", "source_identity": {"target": "a3"},
            "memory_entries": [{"lesson": "historical learning"}], "files": {}}
    seed["snapshot_id"] = sha(canonical(seed))
    write(root / "seed-memory.json", seed)
    task = {"task_id": "matmul-residual", "rationale": "Compose matmul and residual",
            "submission": {"source": "def f(x): return x", "entrypoint": "f",
                           "signatures": [{"args": [{"shape": [4, 4], "dtype": "float32"}]}]}, "citations": []}
    store = JobStore(root / "tasks/matmul-residual/jobs")
    job = store.create("def f(x): return x", "f", {"signatures": task["submission"]["signatures"]},
                       seed if job_seed is None else job_seed)
    key = job["job_id"]
    source = "// real candidate for this attempt"
    candidate = store.put_artifact(key, "candidate.cpp", canonical(source))
    baseline = store.put_artifact(key, "baseline", canonical({"suite_sha256": "suite", "cases": [
        {"case": "case-0", "timing": {"torch": {"latency_us": 10}}}]}))
    result = {"attempt_id": "attempt-001", "task_id": task["task_id"], "job_id": key,
              "verified": True, "classification": "host-verified", "summary": "Target passed",
              "source_sha256": sha(source.encode()), "candidate": candidate, "baseline": baseline,
              "speedup": 2, "profiling": None, "handles": [],
              "result": {"status": "verified", "candidate_sha256": sha(source.encode()),
                         "suite_sha256": "suite", "cases": [{"case": "case-0",
                            "timing": {"ascendc": {"latency_us": 5}}}]}}
    result["result_artifact"] = store.put_artifact(key, "phase2-result", canonical(result))
    store.add_event(key, "phase2-attempt-result", result)
    row = {"attempt_id": "attempt-001", "kind": "target", "task_id": task["task_id"],
           "task": task, "result": result, "verified": True,
           "execution_directory": "tasks/matmul-residual"}
    write(root / "attempts/attempt-001.json", row)
    state = {"schema": "ascend-optimizer-phase2-v1", "phase": "optimizer-phase2",
             "snapshot_id": seed["snapshot_id"], "seed_file": "seed-memory.json", "target": task,
             "attempts": [row], "terminal": {"status": "READY", "reason": "Target verified"},
             "memory_entries": [{"evidence_sha256": sha(canonical(row)), "interpretation": "Native works"}]}
    write(root / "state.json", state)
    return state, store, key


def test_completed_snapshot_retains_seed_sources_and_evidence(tmp_path):
    root = tmp_path / "campaign"
    state, _, _ = campaign(root)
    frozen = freeze_campaign(root)
    verify_snapshot(frozen)
    assert frozen["phase"] == "optimizer-phase2"
    assert len(frozen["memory_entries"]) == 2
    assert "seed-memory.json" in frozen["files"]
    assert any(p.startswith("tasks/") and "/artifacts/" in p for p in frozen["files"])
    assert verify_campaign(root)[0] == state
    inventory = inventory_campaign(root)
    assert inventory["complete"], inventory
    package_campaign(root, tmp_path / "bundle.tar.gz")


@pytest.mark.parametrize("change", ["pending", "practice-final", "row", "seed", "candidate", "memory"])
def test_untrusted_ready_or_missing_evidence_not_exported(tmp_path, change):
    root = tmp_path / "campaign"
    state, store, key = campaign(root)
    if change == "pending":
        write(root / "pending.json", {})
    elif change == "practice-final":
        state["attempts"][-1]["kind"] = "practice"
        write(root / "attempts/attempt-001.json", state["attempts"][-1])
    elif change == "row":
        state["attempts"][-1]["result"]["summary"] = "forged"
    elif change == "seed":
        seed = json.loads((root / "seed-memory.json").read_bytes())
        seed["memory_entries"].append({"fake": "lesson"})
        write(root / "seed-memory.json", seed)
    elif change == "candidate":
        store.artifact_path(key, state["attempts"][0]["result"]["candidate"]["sha256"]).unlink()
    else:
        state["memory_entries"][0]["evidence_sha256"] = "unpublished"
    write(root / "state.json", state)
    with pytest.raises((OSError, ValueError)):
        freeze_campaign(root)


def test_incomplete_campaign_can_be_inventoried_but_not_exported(tmp_path):
    root = tmp_path / "campaign"
    state, _, _ = campaign(root)
    state["terminal"] = None
    write(root / "state.json", state)
    verify_campaign(root)
    with pytest.raises(ValueError, match="final target"):
        freeze_campaign(root)


@pytest.mark.parametrize("authenticated", [False, True])
def test_task_job_must_use_authenticated_reviewed_seed(tmp_path, authenticated):
    other = {"schema": "optimizer-memory-snapshot-v1", "source_identity": {"target": "a3"},
             "memory_entries": [{"lesson": "a different campaign"}], "files": {}}
    other["snapshot_id"] = sha(canonical(other)) if authenticated else "not-authenticated"
    root = tmp_path / "campaign"
    campaign(root, job_seed=other)
    with pytest.raises(ValueError, match="snapshot digest|reviewed Phase 2 seed"):
        freeze_campaign(root)


def test_snapshot_reads_each_retained_file_once(tmp_path, monkeypatch):
    from benchmarks.ascend import training_annex
    root = tmp_path / "campaign"
    campaign(root)
    calls = []

    class Retained:
        def read_bytes(self):
            calls.append(True)
            return b"single consistent payload"

    monkeypatch.setattr(training_annex, "_retained", lambda root: [("capture.bin", Retained())])
    frozen = freeze_campaign(root)
    assert len(calls) == 1
    verify_snapshot(frozen)


def test_snapshot_revalidates_if_evidence_changes_while_reading(tmp_path, monkeypatch):
    from benchmarks.ascend import training_annex
    root = tmp_path / "campaign"
    state, store, key = campaign(root)
    original = training_annex._retained

    def changed(directory):
        yield from original(directory)
        store.artifact_path(key, state["attempts"][0]["result"]["candidate"]["sha256"]).write_bytes(b"changed")

    monkeypatch.setattr(training_annex, "_retained", changed)
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        freeze_campaign(root)


def test_result_candidate_suite_binding_is_checked(tmp_path):
    root = tmp_path / "campaign"
    state, store, key = campaign(root)
    row = state["attempts"][0]
    row["result"]["result"]["suite_sha256"] = "other-suite"
    result = row["result"]
    result["result_artifact"] = store.put_artifact(key, "phase2-result", canonical(
        {k: v for k, v in result.items() if k != "result_artifact"}))
    store.add_event(key, "phase2-attempt-result", result)
    state["memory_entries"] = []
    write(root / "attempts/attempt-001.json", row)
    write(root / "state.json", state)
    with pytest.raises(ValueError, match="frozen suite"):
        freeze_campaign(root)


def test_real_execution_bridge_results_are_consumable(tmp_path):
    bridge = pytest.importorskip("benchmarks.ascend.optimizer_execution")
    root = tmp_path / "campaign"
    state, _, _ = campaign(root)
    # Replace the hand-built publication with the public execution bridge's output.
    seed = json.loads((root / "seed-memory.json").read_bytes())
    task = state["target"]
    actions = iter([{"action": "cases", "signatures": [], "summary": "Representative"},
        {"action": "candidate", "source": "native source", "block_count": 1, "summary": "Native"}])

    class Agent:
        identity = {"model": "fixture"}

        def complete(self, stage, context, record):
            return next(actions)

    class Backend:
        def prepare(self, spec, directory):
            return {"suite_sha256": "suite", "cases": [{"case": "case-0",
                "timing": {"torch": {"latency_us": 10}}}]}

        def evaluate(self, source, directory, *, block_count):
            return {"status": "verified", "candidate_sha256": sha(source.encode()),
                "suite_sha256": "suite", "cases": [{"case": "case-0",
                    "timing": {"ascendc": {"latency_us": 5}}}]}

        def profile(self, source, directory, *, block_count):
            return {"status": "unavailable", "reason": "fixture"}

    directory = "tasks/actual-execution"
    execution = bridge.OptimizerTaskExecutor(root / directory, task, seed,
        {"target": "bz-a3-1", "physical_device": 14,
         "validation_wrapper": str(tmp_path / "wrapper.sh"), "max_cases": 1},
        agent=Agent(), backend=Backend())
    result = execution.attempt("attempt-001")
    row = {"attempt_id": "attempt-001", "kind": "target", "task_id": task["task_id"],
           "task": task, "result": result, "verified": result["verified"],
           "execution_directory": directory}
    state.update(attempts=[row], memory_entries=[])
    write(root / "attempts/attempt-001.json", row)
    write(root / "state.json", state)
    assert verify_campaign(root, require_ready=True)[0] == state
    verify_snapshot(freeze_campaign(root))
