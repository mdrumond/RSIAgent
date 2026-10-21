"""Fresh-project adaptation and durable continuation across both phases."""
import base64
import json
import hashlib
from dataclasses import replace
from copy import deepcopy
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
from benchmarks.a5kernels.phase1_memory import HostFact
from benchmarks.a5kernels.phase2_adaptive_memory import _memory
from benchmarks.ascend.phase2_curriculum import AdaptivePhase2Coordinator
from test_a5_phase1_adaptive import SEED, _memory as project_memory, _project, _result, _wave


def snapshot(identity="reviewed-seed"):
    seed = _project("phase1-copy", "gm-ub-movement")
    return SimpleNamespace(snapshot_id=identity, verify=lambda: None,
                           memory_entries=lambda: [project_memory(ProjectSpec.from_mapping(seed)).as_dict()])


def project(ordinal):
    value = _project(f"phase2-{ordinal}", "simd-vector")
    value["shape"] = [256 + ordinal]
    for tensor in value["graph"]["inputs"]:
        tensor["shape"] = value["shape"]
    return value


def test_failed_project_focus_stays_in_selector_but_does_not_close_coverage_gaps(tmp_path):
    contexts = []
    failed = project(1)
    failed["learning_focus"] = ["tool-msprof", "pipe-vector"]

    def select(messages):
        contexts.append(json.loads(messages[-1]["content"].rsplit("\n", 1)[-1]))
        if len(contexts) == 1:
            return _wave(messages[-1]["content"], [failed])
        return 'DECISION: STALLED\n{"reason":"host mismatch requires another experiment"}'

    coordinator = AdaptivePhase2Coordinator(
        tmp_path, "Learn tools", snapshot(), CurriculumAgent(select),
        lambda req: _result(req.spec, False, "host mismatch; profiler not run"), project_target=1)
    report = coordinator.run()
    assert report["failed_projects"] == 1
    assert report["successful_projects"] == 0
    failed_wave = tmp_path / "waves" / "wave-001"
    plan = json.loads((failed_wave / "plan.json").read_bytes())
    raw_outcome = (failed_wave / "branches" / "phase2-1" / "outcome.json").read_bytes()
    assert json.loads((failed_wave / "closed-failure.json").read_bytes()) == {
        "wave_fingerprint": plan["wave_fingerprint"],
        "outcome_sha256": hashlib.sha256(raw_outcome).hexdigest(),
    }
    assert json.loads(raw_outcome)["verified"] is False
    assert json.loads(raw_outcome)["classification"] == "host-mismatch"
    assert not (failed_wave / "published.json").exists()
    assert not (tmp_path / "pending.json").exists()
    assert {"tool-msprof", "pipe-vector"} <= set(report["coverage_gaps"])
    prior = contexts[1]["prior_outcomes"][-1]
    assert not prior["verified"]
    assert set(prior["information"]["learning_focus"]) == {"tool-msprof", "pipe-vector"}


@pytest.mark.parametrize("observed", [False, True])
def test_reviewed_seed_tool_coverage_requires_observed_host_facts(tmp_path, observed):
    seed = _project("seed-with-requested-tools", "gm-ub-movement")
    seed["learning_focus"] = ["tool-msprof", "tool-msdebug", "debug-print"]
    memory = project_memory(ProjectSpec.from_mapping(seed))
    if observed:
        evidence = memory.host_facts[0].evidence_sha256
        memory = replace(memory, host_facts=(*memory.host_facts,
            HostFact("profiling", "profiler trace observed", evidence),
            HostFact("diagnostic", 'tool-observation:{"tool":"msdebug","available":true,"execution":{}}', evidence),
            HostFact("diagnostic", "verified-debug-print:observed values", evidence)))
    reviewed = SimpleNamespace(snapshot_id="reviewed-tools", verify=lambda: None,
                               memory_entries=lambda: [memory.as_dict()])
    coordinator = AdaptivePhase2Coordinator(tmp_path, "Learn tools", reviewed,
        CurriculumAgent(lambda _: 'DECISION: SATURATED\n{"reason":"reviewed seed checked"}'),
        lambda _: None, project_target=1)
    state = deepcopy(coordinator._load_state())
    # Simulate an older retained campaign that stored requested seed focus.
    state["seed_outcomes"][0]["spec"]["learning_focus"] = seed["learning_focus"]
    coordinator._atomic_json(tmp_path / "state.json", state)
    gaps = coordinator.report()["coverage_gaps"]
    for focus in seed["learning_focus"]:
        assert (focus not in gaps) is observed


def test_adapts_to_semantic_failure_and_prior_verified_learning(tmp_path):
    contexts, requests, events = [], [], []

    def select(messages):
        context = json.loads(messages[-1]["content"].rsplit("\n", 1)[-1])
        contexts.append(context)
        assert context["max_wave_projects"] == 1
        return _wave(messages[-1]["content"], [project(len(contexts))])

    def execute(request):
        requests.append(request)
        return _result(request.spec, len(requests) != 1, "semantic host result")

    coordinator = AdaptivePhase2Coordinator(tmp_path, "Explore new projects", snapshot(),
                                           CurriculumAgent(select), execute,
                                           project_target=2, trace_hook=events.append)
    result = coordinator.run()
    assert result["status"] == "TARGET_REACHED"
    assert result["successful_projects"] == 2
    assert result["failed_projects"] == 1
    assert [row["verified"] for row in contexts[1]["prior_outcomes"]] == [True, False]
    assert b"reviewed-seed" in requests[0].pre_wave_memory or b"reviewed-seed" in base64.b64decode(
        json.loads(requests[0].pre_wave_memory)["initial_memory_base64"])
    assert json.loads(requests[1].pre_wave_memory)["project_memory_base64"] == []
    assert len(json.loads(requests[2].pre_wave_memory)["project_memory_base64"]) == 1
    assert [event["verified"] for event in events] == [False, True, True]
    coordinator.resume()
    assert len(requests) == 3
    with pytest.raises(ValueError, match="initial memory"):
        AdaptivePhase2Coordinator(tmp_path, "Explore new projects", snapshot("new-review"),
                                  CurriculumAgent(select), execute).resume()


def test_infrastructure_exception_keeps_exact_pending_project(tmp_path):
    selections, executions = [], []

    def select(messages):
        selections.append(messages)
        return _wave(messages[-1]["content"], [project(1)])

    def execute(request):
        executions.append(request.spec.project_id)
        if len(executions) == 1:
            raise OSError("transport interrupted")
        return _result(request.spec, True, "host pass")

    coordinator = AdaptivePhase2Coordinator(tmp_path, "Learn", snapshot(),
                                           CurriculumAgent(select), execute, project_target=1)
    with pytest.raises(RuntimeError):
        coordinator.run()
    assert coordinator.report()["status"] == "RECOVERY_REQUIRED"
    assert coordinator.report()["failed_projects"] == 0
    assert not list((tmp_path / "waves").glob("*/closed-failure.json"))
    assert coordinator.resume()["successful_projects"] == 1
    assert not list((tmp_path / "waves").glob("*/closed-failure.json"))
    assert executions == ["phase2-1", "phase2-1"]
    assert len(selections) == 1


def test_recovers_empty_state_after_interrupted_phase2_initialization(tmp_path):
    coordinator = AdaptivePhase2Coordinator(
        tmp_path, "Learn", snapshot(),
        CurriculumAgent(lambda messages: _wave(messages[-1]["content"], [project(1)])),
        lambda req: _result(req.spec, True, "pass"), project_target=1)
    intermediate = AdaptivePhase1Coordinator._load_state(coordinator)
    assert "phase" not in intermediate
    assert coordinator.resume()["successful_projects"] == 1
    assert json.loads((tmp_path / "state.json").read_bytes())["phase"] == "adaptive-phase2"


def test_does_not_upgrade_nonempty_state_without_phase2_identity(tmp_path):
    coordinator = AdaptivePhase2Coordinator(
        tmp_path, "Learn", snapshot(), CurriculumAgent(lambda _: None), lambda _: None)
    state = AdaptivePhase1Coordinator._load_state(coordinator)
    state["session"] = {"messages": [{"role": "system", "content": "started"}], "turns": 0}
    coordinator._atomic_json(tmp_path / "state.json", state)
    with pytest.raises(ValueError, match="not an adaptive Phase 2 lineage"):
        coordinator.resume()


@pytest.mark.parametrize("phase2", [False, True])
@pytest.mark.parametrize("terminal", ["SATURATED", "STALLED"])
def test_explicit_continuation_preserves_publications_and_terminal_history(tmp_path, phase2, terminal):
    calls = []

    def select(messages):
        calls.append(messages)
        if len(calls) == 2:
            return f'DECISION: {terminal}\n{{"reason":"objective covered"}}'
        return _wave(messages[-1]["content"], [project(len(calls))])

    args = (tmp_path, "Learn", snapshot() if phase2 else SEED,
            CurriculumAgent(select), lambda req: _result(req.spec, True, "pass"))
    cls = AdaptivePhase2Coordinator if phase2 else AdaptivePhase1Coordinator
    coordinator = cls(*args, project_target=2, **({} if phase2 else {"max_parallel": 1}))
    assert coordinator.run()["status"] == terminal
    assert coordinator.resume()["successful_projects"] == 1
    with pytest.raises(ValueError, match="explicit reopen"):
        coordinator.continue_campaign(3)
    coordinator.continue_campaign(3, reopen_terminal=True)
    assert coordinator.run()["successful_projects"] == 3
    state = json.loads((tmp_path / "state.json").read_bytes())
    assert state["continuations"][0]["terminal"]["status"] == terminal
    coordinator.continue_campaign(4)
    assert coordinator.resume()["successful_projects"] == 4


def test_target_complete_phase1_memory_is_reviewable(tmp_path):
    coordinator = AdaptivePhase1Coordinator(
        tmp_path, "Learn", SEED,
        CurriculumAgent(lambda msgs: _wave(msgs[-1]["content"], [project(1)])),
        lambda req: _result(req.spec, True, "pass"), max_parallel=1, project_target=1)
    assert coordinator.run()["status"] == "TARGET_REACHED"
    state = json.loads((tmp_path / "state.json").read_bytes())
    assert len(_memory(state, SEED)[0]) == 1
    state["project_target"] = 2
    with pytest.raises(ValueError, match="must be complete"):
        _memory(state, SEED)


def test_automatic_phase2_rejected_selection_retains_transcript(tmp_path):
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts

    calls, responses = [], []
    now = [0.0]

    def complete(messages):
        calls.append(messages)
        now[0] += 1
        proposal = json.loads(_wave(messages[1]["content"], [project(1)]).split("\n", 1)[1])
        proposal["projects"][0]["requires"] = ["cube-tiling"]
        raw = "DECISION: WAVE\n" + json.dumps(proposal)
        responses.append(raw)
        return raw

    def execute(_):
        pytest.fail("invalid selection must never dispatch a project")

    coordinator = AdaptivePhase2Coordinator(
        tmp_path, "Learn", snapshot(), CurriculumAgent(complete), execute, project_target=2)
    coordinator.clock = coordinator.curriculum.clock = lambda: now[0]
    coordinator.max_seconds = 3
    with pytest.raises(ValueError, match="rejected selection artifact:") as failure:
        coordinator.run()
    assert len(calls) == 3
    digest = str(failure.value).rsplit(": ", 1)[1]
    artifact = ContentAddressedArtifacts(tmp_path / "artifacts").read_json(digest)
    assert [row["content"] for row in artifact["messages"]
            if row["role"] == "assistant"] == responses
    feedback = [row["content"] for row in artifact["messages"] if row["role"] == "user"][1:]
    assert len(feedback) == 3
    assert all("project phase2-1: unsupported requires ['cube-tiling']" in row
               and "allowed pre-wave coverage prerequisites: ['gm-ub-movement']" in row
               for row in feedback)
    assert not (tmp_path / "pending.json").exists()
    state = json.loads((tmp_path / "state.json").read_bytes())
    assert state["prior_outcomes"] == []
    assert state["completed_waves"] == []
