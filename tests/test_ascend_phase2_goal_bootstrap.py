"""Autonomous selection is not correctness authority or learner training data."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import EvidenceLedger
from benchmarks.a5kernels.phase1_adaptive_actor import disabled_knowledge_action
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
from benchmarks.ascend.phase2_goal_bootstrap import bootstrap_goal, executable_identity
from benchmarks.ascend.targets import a3_curriculum_policy
from test_a5_phase1_project_executor import FakeBackend, _compile, _runtime
from test_ascend_phase2_curriculum import snapshot, project
from test_ascend_qualified_targets import RUNTIME


def setup(tmp_path, responses, *, failure=None, query=None, clock=None, max_seconds=7200):
    calls, messages = [], []
    class Backend(FakeBackend):
        def run(self, request, compilation):
            self.run_requests.append((request, compilation))
            return _runtime(request.expected.storage)
    backend = Backend(_compile())
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "learner-ledger"), runtime_identity=RUNTIME)
    class Actor:
        trace_recorder_factory = "learner-traces"
        def __call__(self, request):
            calls.append(request)
            assert self.trace_recorder_factory is None
            if getattr(self, "check_private_routes", False):
                for name in ("request_profile", "request_timing", "request_diagnostic"):
                    assert getattr(self, name)({}, None).observation["available"] is False
            if failure:
                failure(len(calls))
            digest = ContentAddressedArtifacts(request.branch_root / "artifacts").put_bytes(b"PRIVATE SOURCE")
            return SimpleNamespace(evidence_payload=json.dumps({"source_sha256": digest}).encode(),
                                   evidence=SimpleNamespace(verified=True))
    actor = Actor()
    actor.executor, actor.query_knowledge = executor, query or disabled_knowledge_action
    def complete(prompt):
        messages.append(deepcopy(prompt))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response(prompt) if callable(response) else response
    coordinator = SimpleNamespace(snapshot=snapshot(), execute_project=actor,
        curriculum=CurriculumAgent(complete, target_policy=a3_curriculum_policy()))
    root = tmp_path / "campaign"
    root.mkdir()
    coordinator.root = root
    def run():
        return bootstrap_goal(coordinator, root=root, private_root=tmp_path / "control" / "bootstrap",
                              objective="Optimize a new task", available_tools=getattr(run, "available_tools", ()),
                              max_seconds=max_seconds, clock=clock)
    run.coordinator = coordinator
    return run, root, calls, messages, actor, backend


def proposal(ordinal=1):
    return json.dumps({"project": project(ordinal), "rationale": "A fresh controlled kernel task"})


def test_reference_is_qualified_once_and_private_on_resume(tmp_path):
    run, root, calls, messages, actor, backend = setup(tmp_path, [proposal()])
    target, store = run()
    target.verify(store, RUNTIME, a3_curriculum_policy())
    assert len(backend.compile_requests) == len(backend.run_requests) == 1
    assert actor.trace_recorder_factory == "learner-traces"
    assert not (tmp_path / "learner-ledger").exists()
    assert len(calls) == 1
    assert b"PRIVATE SOURCE" not in calls[0].pre_wave_memory
    public = json.loads((root / "goal-selection.json").read_bytes())
    assert public["stage"] == "qualified" and public["qualified_target_id"] == target.target_id
    assert "PRIVATE SOURCE" not in json.dumps(public)
    assert not (root / "state.json").exists()
    assert run()[0] == target
    assert len(calls) == len(messages) == 1


def test_renamed_phase1_and_invalid_proposal_are_corrected(tmp_path):
    seed = deepcopy(snapshot().memory_entries()[0]["project_spec"])
    seed["project_id"] = "renamed-seed"
    seed["hypothesis"] = "A different marketing name"
    from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
    assert executable_identity(ProjectSpec.from_mapping(seed)) == executable_identity(
        ProjectSpec.from_mapping(snapshot().memory_entries()[0]["project_spec"]))
    run, root, calls, messages, _, _ = setup(tmp_path, ["not JSON", json.dumps(
        {"project": seed, "rationale": "new name"}), proposal()])
    run()
    assert len(calls) == 1
    assert "duplicates a Phase1" in messages[2][-1]["content"]
    assert json.loads((root / "goal-selection.json").read_bytes())["proposal_count"] == 3


def test_exhaustion_never_starts_learner(tmp_path):
    now = [0.0]
    def bad(_):
        now[0] += 1
        return 'bad'
    run, root, calls, messages, _, _ = setup(tmp_path, [bad] * 5,
        clock=lambda: now[0], max_seconds=5)
    with pytest.raises(TimeoutError, match="elapsed-time"):
        run()
    assert not calls and len(messages) == 5
    assert json.loads((root / "goal-selection.json").read_bytes())["stage"] == "failed"
    with pytest.raises(TimeoutError, match="elapsed-time"):
        run()
    assert len(messages) == 5


def test_reference_failure_feeds_next_proposal_but_infrastructure_preserves_budget(tmp_path):
    def fail(ordinal):
        if ordinal == 1:
            raise OSError("transport unavailable; retained handle")
        if ordinal == 2:
            raise ValueError("reference compiler rejected candidate")
    run, root, calls, messages, _, _ = setup(tmp_path, [proposal(), proposal(2)], failure=fail)
    with pytest.raises(OSError):
        run()
    state = json.loads((root / "goal-selection.json").read_bytes())
    assert state["stage"] == "qualification" and state["proposal_count"] == 1
    run()
    assert len(messages) == 2 and len(calls) == 3
    assert "reference compiler rejected" in messages[1][-1]["content"]


def test_gated_kdb_queries_are_retained_replayed_and_cited(tmp_path):
    queries = []
    citation = dict(collection="kernels", path="add.cpp", start_line=1, end_line=9, chunk_hash="a" * 64)
    def query(payload, context):
        queries.append(payload)
        assert context.request.branch_root.name == "bootstrap"
        return SimpleNamespace(observation={"available": True,
            "results": [{"citation": citation, "text": "REFERENCE EXAMPLE"}]})
    run, root, _, messages, _, _ = setup(tmp_path,
        [json.dumps({"query": "vector kernels"}), OSError("provider transient"), proposal()], query=query)
    with pytest.raises(OSError):
        run()
    run()
    assert len(queries) == 1
    assert "REFERENCE EXAMPLE" in messages[-1][-1]["content"]
    assert json.loads((root / "goal-selection.json").read_bytes())["candidate"]["citations"] == [citation]


def test_64_large_memories_and_tool_reads_fit_provider_budget_without_losing_seed(tmp_path):
    import base64
    from benchmarks.ascend.agent_presentation import memory_presentation
    def page(messages):
        index = json.loads(messages[1]["content"])["phase1_memory"]
        assert len(index["projects"]) == 64
        return json.dumps({"read_artifact": {"sha256": index["projects"][0]["sha256"], "limit": 8192}})
    def query(payload, context):
        return SimpleNamespace(observation={"results": [{"citation": {"path": "kernel.cpp"},
            "text": "huge reference code " * 2000}] * 3})
    run, _, calls, messages, _, _ = setup(tmp_path,
        [json.dumps({"query": "vector code"})] * 7 + [page] * 8 + [proposal(100)], query=query)
    memories = []
    for index in range(64):
        memory = deepcopy(snapshot().memory_entries()[0])
        memory["project_spec"] = project(index)
        memory["agent_lessons"] = [{"statement": "Useful measured lesson " * 100}] * (4 if index < 26 else 3)
        memories.append(memory)
    run.coordinator.snapshot.memory_entries = lambda: memories
    run()
    assert len(messages) == 16
    assert max(len(json.dumps(request).encode()) for request in messages) < 192 * 1024
    envelope = json.loads(calls[0].pre_wave_memory)
    assert json.loads(base64.b64decode(envelope["initial_memory_base64"]))["phase1_memory"] == memories
    assert calls[0].memory_root == run.coordinator.root
    readable = memory_presentation(calls[0].pre_wave_memory, ContentAddressedArtifacts(tmp_path / "presentation"))
    assert len(readable["projects"]) == 64 and "preview" not in readable


def test_reference_never_invokes_measurement_routes_bound_to_learner(tmp_path):
    run, root, _, _, actor, _ = setup(tmp_path, [proposal()])
    def learner_route(payload, context):
        (root / "LEAKED-REFERENCE-SOURCE").write_text("must not happen")
        raise AssertionError("private reference invoked learner measurement route")
    actor.request_profile = actor.request_timing = actor.request_diagnostic = learner_route
    actor.check_private_routes = True
    run()
    assert not (root / "LEAKED-REFERENCE-SOURCE").exists()
    assert actor.request_profile is actor.request_timing is actor.request_diagnostic is learner_route


def test_goal_can_follow_memory_hashes_to_historical_source_and_host_evidence_on_resume(tmp_path):
    from benchmarks.a5kernels.phase2_adaptive_memory import admit_adaptive_snapshot
    from tests.test_a5_phase2_adaptive_memory import _source
    _, source, review, memory = _source(tmp_path / "history")
    source_read = json.dumps({"read_artifact": {"sha256": memory["source_revision"]}})
    evidence_read = json.dumps({"read_artifact": {"sha256": memory["host_facts"][0]["evidence_sha256"]}})
    run, root, _, messages, _, _ = setup(tmp_path,
        [source_read, OSError("provider unavailable"), evidence_read, proposal()])
    reviewed = admit_adaptive_snapshot(tmp_path / "history/adaptive", b"", review, source, root)
    run.coordinator.snapshot = reviewed
    with pytest.raises(OSError):
        run()
    run()
    assert "actual candidate source" in json.dumps(messages[-1])
    assert json.loads(json.loads(messages[-1][-1]["content"])["text"])["host"] == "verified"
    assert json.loads((root / "goal-selection.json").read_bytes())["proposal_count"] == 1


def test_actual_reference_actor_can_submit_profile_timing_goal_without_measurements(tmp_path):
    from dataclasses import replace
    from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActorProjectExecutor
    from benchmarks.a5kernels.phase1_memory import parse_agent_distillation
    from benchmarks.a5kernels.phase2_adaptive_memory import admit_adaptive_snapshot
    from benchmarks.a5kernels.trial import ActorOutcome
    from tests.test_a5_phase2_adaptive_memory import _source
    class Driver:
        def __call__(self, instruction, *, turn_parser, action_executor, **kwargs):
            for ordinal, text in enumerate(('{"write":{"slot":"kernel","content":"fixed source"}}',
                                             '{"run":{}}', '{"submit":{}}'), 1):
                outcome = action_executor(turn_parser(text))
            assert outcome.terminal and outcome.status == "host-verified"
            return ActorOutcome(outcome.status, ordinal, 10)
        def distill(self, outcome, *, admitted_artifacts, **kwargs):
            text = '{"distillation":{"no_new_knowledge":true,"lessons":[]}}'
            return replace(outcome, terminal_response=text,
                           distillation=parse_agent_distillation(text, admitted_artifacts))
    value = project(1)
    value.update(capabilities=[*value["capabilities"], "profiling", "timing-sweep", "failure-repair"],
                 profile_metric="pipe-utilization", timing_lengths=[257, 258], failure_mode="compile",
                 learning_focus=["sync-performance"])
    run, root, _, _, actor, backend = setup(tmp_path, [json.dumps({"project": value, "rationale": "new timed goal"})])
    _, source, review, _ = _source(tmp_path / "history")
    run.coordinator.snapshot = admit_adaptive_snapshot(tmp_path / "history/adaptive", b"", review, source, root)
    def forbidden(*args):
        raise AssertionError("reference measurement leaked to learner")
    run.coordinator.execute_project = AdaptiveActorProjectExecutor(actor.executor, Driver(), object(),
        request_profile=forbidden, request_timing=forbidden, request_diagnostic=forbidden)
    run.available_tools = ("request-profile", "request-timing")
    target, _ = run()
    assert target.spec.as_dict() == value
    assert backend.compile_requests[-1].spec == target.spec
    assert backend.compile_requests[0].spec.timing_lengths == ()


def test_autonomous_goal_and_reference_share_campaign_time_allocation(tmp_path):
    from benchmarks.ascend.campaign_time import CampaignTimeBudget
    now = [0.0]
    def select(_):
        now[0] += 2
        return proposal()
    def reference(_):
        now[0] += 3
    run, root, calls, _, _, backend = setup(tmp_path, [select], failure=reference)
    original_run = backend.run
    def native_run(request, compilation):
        now[0] += 1
        return original_run(request, compilation)
    backend.run = native_run
    budget = CampaignTimeBudget(root, 10, 2, clock=lambda: now[0])
    with budget.active():
        bootstrap_goal(run.coordinator, root=root, private_root=tmp_path / 'control/bootstrap',
            objective='goal', available_tools=(), remaining_seconds=lambda: budget.remaining)
    assert calls[0].remaining_seconds == 8
    # The learner opens the same campaign ledger after goal qualification.
    assert CampaignTimeBudget(root, 10, 2, clock=lambda: now[0]).remaining == 4


def test_autonomous_goal_can_correct_more_than_three_proposals(tmp_path):
    run, _, calls, messages, _, _ = setup(tmp_path, ['bad'] * 5 + [proposal()])
    run()
    assert len(messages) == 6 and len(calls) == 1
