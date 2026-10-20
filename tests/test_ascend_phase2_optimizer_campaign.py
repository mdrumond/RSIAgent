import json

import pytest

from benchmarks.ascend.phase2_optimizer_campaign import OptimizerPhase2Coordinator
from optimizer.control import ObservationUnavailable
from test_ascend_phase2_optimizer_tasks import seed, proposal


def setup(tmp_path, *, verdicts=(True,), decisions=("ready",), budget=3):
    calls, prepared, tasks, responses = [], [], [], iter(decisions)
    outcomes = iter(verdicts)
    cache = {}

    def selector(**kwargs):
        value = proposal()
        if kwargs["kind"] == "practice":
            value["task_id"] = "focused-practice-" + str(len(tasks))
        tasks.append(kwargs)
        return value

    class Executor:
        def __init__(self, path, task, memory):
            self.path, self.task = str(path), task

        def prepare(self):
            prepared.append(self.path)
            return {"suite_sha256": "a" * 64}

        def attempt(self, key, **kwargs):
            if (self.path, key) not in cache:
                value = next(outcomes)
                if isinstance(value, Exception):
                    raise value
                cache[self.path, key] = {"verified": value, "summary": "explicit model interpretation"}
                calls.append((key, self.task["task_id"], kwargs))
            return cache[self.path, key]

    def complete(messages):
        return json.dumps(dict(action=next(responses), reason="host evidence"))

    def build():
        return OptimizerPhase2Coordinator(tmp_path, "new composition", seed(), complete,
            Executor, selector=selector, max_target_attempts=budget)

    return build, calls, prepared, tasks


def test_target_reference_and_ready_without_projectspec(tmp_path):
    build, calls, prepared, tasks = setup(tmp_path)
    assert build().resume()["status"] == "READY"
    state = json.loads((tmp_path / "state.json").read_bytes())
    assert state["phase"] == "optimizer-phase2"
    assert state["target"]["submission"] == proposal()["submission"]
    assert len(set(prepared)) == 1
    assert state["attempts"][0]["verified"] is True
    assert state["memory_entries"][0]["interpretation"] == "explicit model interpretation"
    assert build().resume()["status"] == "READY"
    assert len(calls) == 1


def test_failure_practice_and_retry_original_frozen_target(tmp_path):
    build, calls, prepared, tasks = setup(tmp_path, verdicts=(False, True, True),
                                        decisions=("practice", "ready", "ready"))
    assert build().resume()["status"] == "READY"
    assert [row[1] for row in calls] == ["matmul-residual-relu", "focused-practice-1", "matmul-residual-relu"]
    state = json.loads((tmp_path / "state.json").read_bytes())
    assert [row["kind"] for row in state["attempts"]] == ["target", "practice", "target"]
    assert state["target_attempts"] == 2
    assert len(state["memory_entries"]) == 2  # Failed execution retained as feedback, not successful learning.
    assert calls[2][2]["memory_snapshot"]["memory_entries"]
    assert calls[2][2]["feedback"][-1]["kind"] == "practice"
    assert len(set(prepared)) == 2


def test_practice_budget_forces_retry(tmp_path):
    build, calls, _, _ = setup(tmp_path, verdicts=(False, False, False, True),
                               decisions=("practice", "practice", "practice", "ready"))
    assert build().resume()["status"] == "READY"
    assert calls[-1][1] == "matmul-residual-relu"
    assert len(calls) == 4


def test_failed_last_target_cannot_be_ready(tmp_path):
    build, calls, _, _ = setup(tmp_path, verdicts=(False,), decisions=("ready",), budget=1)
    assert build().resume()["status"] == "BUDGET_EXHAUSTED"
    assert len(calls) == 1


def test_infrastructure_keeps_pending_without_semantic_charge(tmp_path):
    build, calls, _, _ = setup(tmp_path, verdicts=(ObservationUnavailable("lost response"), True))
    with pytest.raises(ObservationUnavailable):
        build().resume()
    state = json.loads((tmp_path / "state.json").read_bytes())
    assert state["target_attempts"] == 0
    pending = (tmp_path / "pending.json").read_bytes()
    assert build().resume()["status"] == "READY"
    assert calls[0][0] == json.loads(pending)["attempt_id"]


def test_prepare_stop_imports_feedback_before_propagating(tmp_path):
    from benchmarks.ascend.phase2_feedback import FeedbackStore
    from optimizer.control import WorkerStopped
    from optimizer.jobs import JobStore
    prepared = []
    class Executor:
        def __init__(self, directory, task, memory):
            self.store = JobStore(directory / 'jobs')
            self.job = self.store.create('def f(x): return x', 'f', {}, {}, idempotency_key='prepare')
        def prepare(self):
            prepared.append(True)
            if len(prepared) > 1:
                self.store.transition(self.job['job_id'], 'stopped')
                raise WorkerStopped('reference preparation stopped')
            return {}
        def attempt(self, *args, **kwargs):
            raise AssertionError('stopped preparation must not attempt')
    coordinator = OptimizerPhase2Coordinator(tmp_path, 'objective', seed(), lambda _: '', Executor,
        selector=lambda **kwargs: proposal())
    with pytest.raises(WorkerStopped, match='reference preparation'):
        coordinator.run_attempt()
    records = FeedbackStore(tmp_path / 'feedback').records()
    assert len(records) == 1
    assert records[0]['outcome']['completed'] is False
    assert records[0]['outcome']['job_status']['status'] == 'stopped'


def test_initial_qualification_stop_is_retained_and_historically_importable(tmp_path):
    from benchmarks.ascend.phase2_feedback import FeedbackStore
    from optimizer.control import WorkerStopped
    from optimizer.jobs import JobStore
    root = tmp_path / 'campaign'
    class Executor:
        def __init__(self, directory, task, memory):
            self.jobs = JobStore(directory / 'jobs')
            self.job = self.jobs.create('def f(x): return x', 'f', {}, {}, idempotency_key='qualification')
        def prepare(self):
            self.jobs.transition(self.job['job_id'], 'stopped')
            raise WorkerStopped('initial qualification stopped')
    coordinator = OptimizerPhase2Coordinator(root, 'objective', seed(), lambda _: '', Executor,
        selector=lambda **kwargs: proposal())
    with pytest.raises(WorkerStopped, match='initial qualification'):
        coordinator.run_attempt()
    assert not (root / 'state.json').exists()
    record = FeedbackStore(root / 'feedback').records()[0]
    assert record['outcome']['kind'] == 'qualification'
    assert record['outcome']['completed'] is False
    historical = FeedbackStore(tmp_path / 'historical-feedback')
    assert historical.import_campaign(root)['imported'] == 1
    assert historical.records()[0]['outcome'] == record['outcome']


@pytest.mark.parametrize('status', ['stopped', 'failed'])
def test_practice_qualification_failure_is_retained_before_pending(tmp_path, status):
    from benchmarks.ascend.phase2_feedback import FeedbackStore
    from optimizer.control import WorkerStopped
    from optimizer.jobs import JobStore
    root = tmp_path / 'campaign'
    class Executor:
        def __init__(self, directory, task, memory):
            self.task = task
            self.jobs = JobStore(directory / 'jobs')
            self.job = self.jobs.create('def f(x): return x', 'f', {}, {}, idempotency_key=task['task_id'])
        def prepare(self):
            if self.task['task_id'] == 'practice-copy':
                self.jobs.transition(self.job['job_id'], status)
                error = WorkerStopped if status == 'stopped' else ObservationUnavailable
                raise error('practice reference interrupted')
        def attempt(self, key, **kwargs):
            return dict(verified=False, job_id=self.job['job_id'])
    def selector(**kwargs):
        task = proposal()
        if kwargs['kind'] == 'practice':
            task['task_id'] = 'practice-copy'
        return task
    coordinator = OptimizerPhase2Coordinator(root, 'objective', seed(),
        lambda _: json.dumps(dict(action='practice', reason='isolate failure')), Executor, selector=selector)
    with pytest.raises((WorkerStopped, ObservationUnavailable), match='practice reference'):
        coordinator.resume()
    assert not (root / 'pending.json').exists()
    state = json.loads((root / 'state.json').read_bytes())
    assert len(state['attempts']) == 1 and state['target_attempts'] == 1
    record = next(row for row in FeedbackStore(root / 'feedback').records()
                  if row['outcome']['kind'] == 'practice-qualification')
    assert record['outcome']['completed'] is False
    assert record['outcome']['job_status']['status'] == status
    historical = FeedbackStore(tmp_path / 'historical')
    historical.import_campaign(root)
    assert any(row['outcome'] == record['outcome'] for row in historical.records())


def test_resume_after_state_commit_before_pending_cleanup(tmp_path):
    build, calls, _, _ = setup(tmp_path)
    assert build().run_attempt()["status"] == "RUNNING"
    state = json.loads((tmp_path / "state.json").read_bytes())
    row = state["attempts"][0]
    (tmp_path / "pending.json").write_text(json.dumps(dict(
        attempt_id=row["attempt_id"], kind=row["kind"], task=row["task"])))
    assert build().resume()["status"] == "READY"
    assert len(calls) == 1
    assert not (tmp_path / "pending.json").exists()


def test_seed_and_budget_changes_rejected(tmp_path):
    build, _, _, _ = setup(tmp_path)
    build().run_attempt()
    with pytest.raises(ValueError, match="identity changed"):
        OptimizerPhase2Coordinator(tmp_path, "changed objective", seed(), lambda _: "", lambda *a: None).resume()


def test_reference_rejection_returns_feedback_and_reproposes(tmp_path):
    tasks = iter([proposal("def forward(x): raise ValueError('bad shapes')"), proposal()])
    messages = []
    def complete(prompt):
        messages.append(prompt)
        return json.dumps(next(tasks))
    class Executor:
        def __init__(self, directory, task, memory):
            self.task = task
        def prepare(self):
            if "raise" in self.task["submission"]["source"]:
                raise ValueError("representative failed: bad shapes")
            return {}
        def attempt(self, key, **kwargs):
            return {"verified": True}
    coordinator = OptimizerPhase2Coordinator(tmp_path, "new", seed(), complete, Executor)
    assert coordinator.run_attempt()["target_attempts"] == 1
    assert "bad shapes" in messages[1][-1]["content"]


def test_stalled_is_explicit_not_ready(tmp_path):
    build, _, _, _ = setup(tmp_path, verdicts=(False,), decisions=("stalled",))
    assert build().resume()["status"] == "STALLED"


def test_completed_optimizer_phase2_is_discoverable(tmp_path):
    from test_optimizer_phase2_artifacts import campaign
    from optimizer.memory import discover_sources, freeze_source
    campaign(tmp_path)
    assert freeze_source(tmp_path)["phase"] == "optimizer-phase2"
    assert discover_sources([tmp_path])[0]["available"] is True


def test_retry_memory_includes_practice_source_and_frozen_evidence(tmp_path):
    from test_optimizer_phase2_artifacts import campaign
    state, store, key = campaign(tmp_path)
    coordinator = object.__new__(OptimizerPhase2Coordinator)
    coordinator.root = tmp_path
    coordinator.seed = json.loads((tmp_path / "seed-memory.json").read_bytes())
    memory = coordinator._memory(state)
    hashes = {record["sha256"] for record in memory["files"].values()}
    result = state["attempts"][0]["result"]
    assert result["candidate"]["sha256"] in hashes
    assert result["result_artifact"]["sha256"] in hashes


def test_cli_composition_pins_worker_config_and_reviewed_treatment(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from benchmarks.ascend import campaign_cli
    from optimizer import memory
    from optimizer.jobs import sha, canonical
    guide = tmp_path / "guide.md"
    guide.write_text("Ascend C programming")
    config = dict(profile="bz-a3-1", target="bz-a3-1", runtime="py311-torch", physical_device=10,
                  arch="dav-2201", cpl_remote="/configured/transport", validation_wrapper="/configured/adapter",
                  remote_workspace="/configured/workspace", guide=str(guide))
    path = tmp_path / "worker.json"
    path.write_bytes(canonical(config))
    job = dict(root=str(tmp_path / "campaign"), objective="new", model="openai/gpt-5.6-sol",
               optimizer_worker_config_sha256=sha(path.read_bytes()),
               target_attempt_budget=3, practice_attempt_budget=2)
    native = SimpleNamespace(execution_profile="bz-a3-1", arch="dav-2201",
                             guide_sha256=sha(guide.read_bytes()), collection=None)
    composed = SimpleNamespace(curriculum=SimpleNamespace(_complete=lambda _: ""),
                               execute_project=SimpleNamespace(executor=SimpleNamespace(runtime_identity={"device": "10"})))
    monkeypatch.setattr(memory, "freeze_reviewed_phase1", lambda _: seed())
    controller = campaign_cli._optimizer_controller(job, {"optimizer_worker_config": str(path)}, None, composed, native)
    assert isinstance(controller, OptimizerPhase2Coordinator)
    assert controller.target_budget == 3
    native.collection = ["selected-kdb"]
    with pytest.raises(ValueError, match="KDB treatment"):
        campaign_cli._optimizer_controller(job, {"optimizer_worker_config": str(path)}, None, composed, native)
    path.write_text("{}")
    with pytest.raises(ValueError, match="configuration changed"):
        campaign_cli._optimizer_controller(job, {"optimizer_worker_config": str(path)}, None, composed, native)


@pytest.mark.parametrize("rejection", ["bad-reference", "duplicate-id"])
def test_rejected_practice_is_corrected_before_pending_and_target_retry(tmp_path, rejection):
    goal, bad, practice = proposal(), proposal(), proposal()
    bad["task_id"] = goal["task_id"] if rejection == "duplicate-id" else "invalid-practice"
    bad["submission"]["source"] = "def forward(a,b,r): raise ValueError('shape disagreement')"
    practice["task_id"] = "corrected-practice"
    responses = iter([goal, dict(action="practice", reason="repair residual layout"), bad, practice,
                      dict(action="ready", reason="practice passed"), dict(action="ready", reason="target passed")])
    messages, attempts = [], []
    def complete(prompt):
        messages.append(prompt)
        return json.dumps(next(responses))
    class Executor:
        def __init__(self, directory, task, memory):
            self.task = task
        def prepare(self):
            if "raise" in self.task["submission"]["source"]:
                assert not (tmp_path / "pending.json").exists()
                raise ValueError("representative shape disagreement")
            return {}
        def attempt(self, key, **kwargs):
            attempts.append(self.task["task_id"])
            return {"verified": len(attempts) > 1}
    result = OptimizerPhase2Coordinator(tmp_path, "new", seed(), complete, Executor).resume()
    assert result["status"] == "READY"
    assert attempts == [goal["task_id"], practice["task_id"], goal["task_id"]]
    assert json.loads((tmp_path / "curriculum/practice-001/selection.json").read_bytes())["proposal_count"] == 2


def test_curriculum_sees_optimizer_learning_after_large_phase1_index(tmp_path):
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.ascend.agent_presentation import memory_presentation
    import base64
    entries = [{"project_spec": {"project_id": f"seed-{i}"},
                "agent_lessons": [{"claim": "historical lesson " * 100}]} for i in range(64)]
    entries.append({"task_id": "practice-residual", "statement": "Frozen suite passed",
                    "interpretation": "Use physical row strides for residual reads"})
    raw = json.dumps(dict(initial_memory_base64=base64.b64encode(json.dumps(
        {"phase1_memory": entries}).encode()).decode(), project_memory_base64=[])).encode()
    view = memory_presentation(raw, ContentAddressedArtifacts(tmp_path / "cas"))
    latest = view["projects"][-1]
    assert latest["project_id"] == "practice-residual"
    assert latest["host_fact"] == "Frozen suite passed"
    assert "physical row strides" in latest["interpretation"]


def test_export_authenticates_job_once_not_once_per_profile_artifact(tmp_path, monkeypatch):
    from test_optimizer_phase2_artifacts import campaign
    from benchmarks.ascend.optimizer_phase2_artifacts import verify_campaign
    from optimizer.jobs import JobStore
    state, store, key = campaign(tmp_path)
    for i in range(20):
        store.put_artifact(key, f"compact-{i}.csv", f"value,{i}".encode())
    calls = []
    original = JobStore.get
    def counted(self, job_id):
        calls.append(job_id)
        return original(self, job_id)
    monkeypatch.setattr(JobStore, "get", counted)
    verify_campaign(tmp_path, require_ready=True)
    assert len(calls) <= 3
