"""Completed failures inform selection, without becoming verified knowledge."""
from dataclasses import replace
import json
import base64

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import canonical_bytes
from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
from benchmarks.a5kernels.phase1_memory import HostFact
from benchmarks.a5kernels.phase1_wave_runtime import SemanticProjectFailure
from benchmarks.a5kernels.phase2_adaptive_memory import _published_evidence, _memory
from benchmarks.ascend.training_annex import inventory_campaign, package_campaign
from test_a5_phase1_adaptive import _project, _result, _wave, SEED


def project(identifier, length):
    value = _project(identifier, "gm-ub-movement")
    value["shape"] = [length]
    value["graph"]["inputs"][0]["shape"] = [length]
    return value


def verified(request, *, low_yield=False):
    objects = ContentAddressedArtifacts(request.branch_root / "artifacts")
    source = objects.put_bytes(b"candidate:" + request.spec.project_id.encode())
    host = objects.put_bytes(b"verified:" + request.spec.project_id.encode())
    result = _result(request.spec, True, "host verified")
    memory = replace(result.memory, source_revision=source,
                     host_facts=(HostFact("host-verification", "host passed", host),),
                     lessons=() if low_yield else tuple(replace(lesson, evidence_artifact_sha256=(host,))
                                                       for lesson in result.memory.lessons),
                     no_new_knowledge=low_yield, admitted_artifact_sha256=(host,),
                     artifact_store="artifacts")
    return replace(result, memory=memory, evidence_payload=canonical_bytes({
        "classification": "host-verified", "source_sha256": source,
        "artifacts": {"classification": host}}))


@pytest.mark.parametrize("crash_point", ["before-state", "before-pending-removal", None])
def test_mixed_wave_preserves_seven_verified_siblings_and_closes_failure(tmp_path, monkeypatch, crash_point):
    contexts, calls = [], []
    projects = [project(f"verified-{i}", 32 + i) for i in range(7)]
    projects.append(project("failed-large", 4096))
    (tmp_path / "initial-memory.bin").write_bytes(SEED)

    def select(messages):
        prompt = messages[-1]["content"]
        contexts.append(json.loads(prompt.rsplit("\n", 1)[-1]))
        if len(contexts) == 1:
            return _wave(prompt, projects)
        # Revisit the failed operation at a smaller shape under a fresh ID.
        return _wave(prompt, [project("smaller-probe", 16)])

    def execute(request):
        calls.append(request.spec.project_id)
        if request.spec.project_id == "failed-large":
            raise SemanticProjectFailure("Actor exhausted 40 turns without host verification", {
                "iterations": 40, "tokens": None, "source_sha256": None,
                "actor_status": "turn-budget-exhausted"})
        return verified(request, low_yield=request.spec.project_id == "verified-0")

    def make():
        return AdaptivePhase1Coordinator(tmp_path, "Learn reusable kernels", SEED,
            CurriculumAgent(select), execute, max_parallel=8, project_target=8)

    controller = make()
    atomic = controller._atomic_json
    unlink = type(tmp_path).unlink
    if crash_point == "before-state":
        def crash(path, value):
            if path.name == "state.json" and value.get("wave_index") == 2:
                raise RuntimeError("simulated controller crash")
            atomic(path, value)
        monkeypatch.setattr(controller, "_atomic_json", crash)
    elif crash_point == "before-pending-removal":
        def crash_unlink(path, *args, **kwargs):
            if path == tmp_path / "pending.json":
                raise RuntimeError("simulated controller crash")
            return unlink(path, *args, **kwargs)
        monkeypatch.setattr(type(tmp_path), "unlink", crash_unlink)
    if crash_point:
        with pytest.raises(RuntimeError, match="simulated"):
            controller.run()
        retained = {str(path.relative_to(tmp_path)): path.read_bytes()
                    for path in (tmp_path / "waves/wave-001").rglob("*") if path.is_file()}
        monkeypatch.undo()
        report = make().resume()
        assert all((tmp_path / name).read_bytes() == payload for name, payload in retained.items())
    else:
        report = controller.run()
    assert report["status"] == "TARGET_REACHED"
    assert report["successful_projects"] == 8
    assert report["failed_projects"] == 1
    assert len(calls) == len(set(calls)) == 9
    assert "failed-large" not in report["published_projects"]
    assert all(memory["project_spec"]["project_id"] != "failed-large" for memory in report["memory"])
    feedback = {row["project_id"]: row for row in contexts[-1]["curriculum_memory"]}
    assert feedback["verified-0"]["yield"] == "no_new_knowledge"
    assert feedback["failed-large"]["yield"] == "failed"
    assert feedback["failed-large"]["evidence"]["iterations"] == 40
    assert feedback["failed-large"]["evidence"]["tokens"] is None
    assert contexts[-1]["remaining_projects"] == 1
    assert len([row for row in contexts[-1]["prior_outcomes"] if not row["verified"]]) == 1
    assert inventory_campaign(tmp_path)["complete"]
    package = package_campaign(tmp_path, tmp_path.parent / (tmp_path.name + "-annex.tar.gz"))
    assert package["complete"]
    state = json.loads((tmp_path / "state.json").read_bytes())
    _memory(state, SEED)
    files = _published_evidence(tmp_path, state, SEED)
    assert "waves/wave-001/branches/failed-large/execution-evidence.bin" in files
    assert len(state["memory_bytes_base64"]) == 8
    # Curriculum feedback is retained independently of successful memory.
    assert make().report()["curriculum_memory"] == report["curriculum_memory"]


def test_all_failed_wave_selects_again_and_failed_id_is_rejected(tmp_path):
    contexts, calls = [], []
    failed = project("failed", 4096)

    def select(messages):
        prompt = messages[-1]["content"]
        contexts.append(prompt)
        if len(contexts) == 1:
            return _wave(prompt, [failed])
        if len(contexts) == 2:
            return _wave(prompt, [failed])  # selection correction, not execution
        return _wave(messages[1]["content"], [project("fresh-smaller", 16)])

    def execute(request):
        calls.append(request.spec.project_id)
        if request.spec.project_id == "failed":
            return _result(request.spec, False, "observed host mismatch")
        return verified(request)

    report = AdaptivePhase1Coordinator(tmp_path, "Learn", SEED,
        CurriculumAgent(select), execute, max_parallel=1, project_target=1).run()
    assert calls == ["failed", "fresh-smaller"]
    assert report["successful_projects"] == report["failed_projects"] == 1
    assert "prior project ID" in contexts[-1]
    published = json.loads((tmp_path / "waves/wave-001/published.json").read_bytes())
    assert published["commits"] == []
    assert published["skipped_projects"] == ["failed"]


@pytest.mark.parametrize("failure", [RuntimeError("transport unavailable"), ValueError("bad provider")])
def test_infrastructure_or_programming_exception_does_not_close_attempt(tmp_path, failure):
    def execute(request):
        raise failure
    controller = AdaptivePhase1Coordinator(tmp_path, "Learn", SEED,
        CurriculumAgent(lambda messages: _wave(messages[-1]["content"], [project("unfinished", 32)])),
        execute, max_parallel=1, project_target=1)
    with pytest.raises(RuntimeError, match="failed before verification"):
        controller.run()
    report = controller.report()
    assert report["status"] == "RECOVERY_REQUIRED"
    assert report["failed_projects"] == report["successful_projects"] == 0
    assert not list(tmp_path.glob("waves/*/published.json"))


def test_real_actor_research_budget_exhaustion_is_not_a_kernel_failure(tmp_path):
    from benchmarks.a5kernels.evidence import EvidenceLedger
    from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActorProjectExecutor
    from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
    from test_a5_phase1_adaptive_actor import SourceBackend, ScriptedActor, _request
    executor = ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path / "evidence.jsonl"))
    actor = ScriptedActor(['{"query-knowledge":{"query":"cube API","limit":1}}'] * 40)
    request = _request(tmp_path, executor)
    with pytest.raises(SemanticProjectFailure) as failed:
        AdaptiveActorProjectExecutor(executor, actor, object(), max_actions=40,
                                     retain_semantic_failure=True)(request)
    detail = failed.value.evidence
    assert detail["actor_status"] == "turn-budget-exhausted"
    assert detail["iterations"] == 40
    assert detail["action_counts"] == {"query-knowledge": 40, "write": 0, "compile": 0, "run": 0}
    assert all(detail["action_counts"].get(kind, 0) == 0 for kind in ("write", "compile", "run"))
    assert detail["source_sha256"] is None
    assert actor.distill_calls == []
    assert "turn-budget-exhausted" in failed.value.summary


def test_64_project_feedback_is_durable_but_wire_detail_is_bounded(tmp_path):
    from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.phase1_provider import DirectCompletion, DirectCompletionProvider, direct_profile
    from test_a5_phase1_adaptive import _memory as memory_for
    controller = AdaptivePhase1Coordinator(tmp_path, "Learn", SEED,
        CurriculumAgent(lambda _: None), lambda _: None, max_parallel=1, project_target=128)
    state = controller._load_state()
    for index in range(64):
        spec = ProjectSpec.from_mapping(project(f"prior-{index}", index + 1))
        memory = memory_for(spec)
        memory = replace(memory, lessons=() if index in {0, 63} else tuple(
            replace(lesson, claim="A reusable observed finding. " * 60) for lesson in memory.lessons),
            no_new_knowledge=index in {0, 63})
        state["prior_outcomes"].append({"project_id": spec.project_id, "spec": spec.as_dict(),
                                        "verified": True, "summary": "host verified"})
        state["memory_bytes_base64"].append(base64.b64encode(canonical_bytes(memory.as_dict())).decode())
    state["failed_outcomes"] = [{"project_id": f"failed-{index}", "verified": False,
        "spec": project(f"failed-{index}", index + 100), "summary": "Actor budget exhausted; no kernel executed",
        "evidence": {"path": f"waves/old/branches/failed-{index}"}} for index in range(12)]
    before = canonical_bytes(state)
    context = controller._context(state, controller._memory_snapshot(state))
    assert len(context.curriculum_memory) == 8
    assert "no_new_knowledge" in context.prior_outcomes[0].summary
    assert len(controller._curriculum_memory(state)) == 76
    assert any(row["project_id"] == "prior-63" and row["yield"] == "no_new_knowledge"
               for row in context.curriculum_memory)
    assert sum(not row["verified"] for row in context.curriculum_memory) == 4
    for identity in MODEL_IDENTITIES:
        calls = []
        def transport(**kwargs):
            calls.append(kwargs)
            return DirectCompletion('DECISION: STALLED\n{"reason":"review retained results"}', 1)
        provider = DirectCompletionProvider({"OPENAI_API_KEY": "test", "DEEPSEEK_API_KEY": "test"},
                                            transport=transport)
        CurriculumAgent(lambda messages: provider.complete(direct_profile(identity), messages).text).select(context)
        assert len(calls) == 1
    assert canonical_bytes(state) == before
