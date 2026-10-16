"""Fresh-project adaptation and durable continuation across both phases."""
import base64
import json
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
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
    assert coordinator.resume()["successful_projects"] == 1
    assert executions == ["phase2-1", "phase2-1"]
    assert len(selections) == 1


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
