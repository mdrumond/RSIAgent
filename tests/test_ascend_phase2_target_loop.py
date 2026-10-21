"""Target attempts remain separate from practice and preserve retained execution."""
import json
from dataclasses import replace

import pytest

from benchmarks.ascend.phase2_target_loop import TargetDrivenPhase2Coordinator
from benchmarks.ascend.phase2_target_selector import TargetCurriculumAgent
from test_ascend_phase2_targets import qualified, RUNTIME
from test_ascend_phase2_curriculum import snapshot
from test_a5_phase1_adaptive import _result


def build(tmp_path, decisions, execute=None, **kwargs):
    now = [0.0]
    target, store, _ = qualified(tmp_path / "reference")
    contexts, requests = [], []
    responses = iter(decisions)

    def select(messages):
        contexts.append(json.loads(messages[-1]["content"].rsplit("\n", 1)[-1]))
        return next(responses)

    def run(request):
        now[0] += 1
        requests.append(request)
        return execute(request, len(requests)) if execute else _result(request.spec, True, "host pass")

    kwargs.setdefault("max_seconds", 100)
    kwargs.setdefault("max_practice_seconds", 2)
    kwargs.setdefault("clock", lambda: now[0])
    coordinator = TargetDrivenPhase2Coordinator(tmp_path / "campaign", "fixed copy", snapshot(),
        TargetCurriculumAgent(select), run, target=target, artifacts=store, runtime=RUNTIME, **kwargs)
    return coordinator, contexts, requests


def wire(action, project=None):
    value = {"reason": "diagnose target behavior"}
    if project is not None:
        value.update(project=project.as_dict(), weakness="tail mismatch", control="isolate padded copy",
                     evidence="host output comparison", retry_plan="apply tail repair to target")
    return f"DECISION: {action}\n" + json.dumps(value)


def practice(coordinator, suffix="tail"):
    return replace(coordinator.target.spec, project_id=f"practice-{suffix}")


def test_target_first_failure_practice_learning_and_unchanged_retry(tmp_path):
    decisions = []
    c, contexts, requests = build(tmp_path, decisions,
        execute=lambda r, n: _result(r.spec, n != 1, "tail mismatch" if n == 1 else "fixed tail"))
    # Supply responses lazily after constructing the fixed contract.
    c.curriculum = TargetCurriculumAgent(lambda messages: (
        contexts.append(json.loads(messages[-1]["content"].rsplit("\n", 1)[-1])) or
        [wire("PRACTICE", practice(c)), wire("RETRY_TARGET"), wire("READY")][len(contexts)-1]))
    assert c.run()["status"] == "READY"
    assert [r.spec.project_id for r in requests] == [c.target.spec.project_id, "practice-tail", c.target.spec.project_id]
    assert [r.remaining_seconds for r in requests] == [100, 2, 98]
    assert contexts[0]["latest_target_attempt"]["verified"] is False
    assert len(json.loads(requests[-1].pre_wave_memory)["project_memory_base64"]) == 1
    assert c.resume()["target_attempts"] == 2
    assert len(requests) == 3


def test_practice_success_after_passing_target_requires_fresh_target(tmp_path):
    c, _, requests = build(tmp_path, [])
    responses = iter([wire("PRACTICE", practice(c)), wire("READY"), wire("READY")])
    c.curriculum = TargetCurriculumAgent(lambda _: next(responses))
    report = c.run()
    assert report["status"] == "READY"
    assert report["target_attempts"] == 2
    assert report["practice_attempts"] == 1
    assert requests[-1].spec == c.target.spec


def test_failed_practice_counts_and_forces_retry_after_two_attempts(tmp_path):
    c, _, requests = build(tmp_path, [], execute=lambda r, n: _result(r.spec, n == 4, "host verdict"))
    responses = iter([wire("PRACTICE", practice(c, "one")), wire("PRACTICE", practice(c, "two")), wire("READY")])
    c.curriculum = TargetCurriculumAgent(lambda _: next(responses))
    report = c.run()
    assert report["status"] == "READY" and report["practice_attempts"] == 2
    assert [r.spec == c.target.spec for r in requests] == [True, False, False, True]


def test_budget_and_stalled_are_not_success(tmp_path):
    c, _, _ = build(tmp_path, [], max_seconds=1,
                    execute=lambda r, n: _result(r.spec, False, "mismatch"))
    assert c.run()["status"] == "BUDGET_EXHAUSTED"
    other, _, _ = build(tmp_path / "other", [wire("STALLED")])
    assert other.run()["status"] == "STALLED"


def test_infrastructure_resume_reuses_exact_pending_branch_without_selection(tmp_path):
    def execute(request, count):
        retained = request.branch_root / "transport-handle.txt"
        if count == 1:
            retained.write_text("retained-handle")
            raise OSError("observation unavailable")
        assert retained.read_text() == "retained-handle"
        return _result(request.spec, True, "observed same operation")

    c, contexts, requests = build(tmp_path, [wire("READY")], execute=execute)
    with pytest.raises(RuntimeError):
        c.run()
    assert not contexts
    assert c.report()["status"] == "RECOVERY_REQUIRED"
    assert c.report()["target_attempts"] == 0
    assert c.resume()["status"] == "READY"
    assert requests[0].branch_root == requests[1].branch_root


def test_resume_rejects_changed_target_budget_runtime_and_legacy_conversion(tmp_path):
    c, _, _ = build(tmp_path, [])
    c.run_attempt()
    c.time.identity["max_seconds"] += 1
    with pytest.raises(ValueError, match="budgets changed"):
        c.report()
    c.time.identity["max_seconds"] -= 1
    c.runtime = {**RUNTIME, "compiler": "different"}
    with pytest.raises(ValueError, match="runtime changed"):
        c.report()
    c.runtime = RUNTIME
    path = c.root / "state.json"
    state = json.loads(path.read_text())
    state["phase"] = "adaptive-phase2"
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="budgets changed"):
        c.report()


def test_completed_outcome_is_reused_after_crash_before_state_publication(tmp_path, monkeypatch):
    c, _, requests = build(tmp_path, [wire("READY")])
    publish = c._publish
    monkeypatch.setattr(c, "_publish", lambda *args: (_ for _ in ()).throw(OSError("disk interrupted")))
    with pytest.raises(OSError, match="disk interrupted"):
        c.run()
    assert c.report()["status"] == "RECOVERY_REQUIRED"
    monkeypatch.setattr(c, "_publish", publish)
    assert c.resume()["status"] == "READY"
    assert len(requests) == 1


def test_no_practice_novelty_gate_on_repeated_target_spec(tmp_path):
    c, _, requests = build(tmp_path, [wire("RETRY_TARGET"), wire("READY")])
    assert c.run()["status"] == "READY"
    assert requests[0].spec == requests[1].spec == c.target.spec
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.ascend.agent_presentation import memory_presentation
    view = memory_presentation(requests[1].pre_wave_memory, ContentAddressedArtifacts(tmp_path / "view"))
    assert view["phase2_feedback"][0]["kind"] == "target"
    assert view["phase2_feedback"][0]["host_feedback"]


def test_host_diagnostics_reach_curriculum_and_private_reference_does_not(tmp_path):
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.a5kernels.evidence import canonical_bytes

    def execute(request, count):
        store = ContentAddressedArtifacts(request.branch_root / "artifacts")
        result = _result(request.spec, True, "host verified")
        observed = store.put_json({"compile_diagnostic": "actual compiler observation",
                                  "runtime_diagnostic": "actual runtime observation",
                                  "output": [0., 2.]})
        expected = store.put_json({"storage": [0., 1.]})
        return replace(result, evidence_payload=canonical_bytes({"classification": "host-verified",
            "detail": "actual host detail", "artifacts": {"observed": observed, "expected": expected}}))

    c, contexts, _ = build(tmp_path, [wire("READY")], execute=execute)
    c.run()
    actual = contexts[0]["latest_target_attempt"]["host_feedback"]
    assert actual["compile_diagnostic"] == "actual compiler observation"
    assert actual["runtime_diagnostic"] == "actual runtime observation"
    assert actual["detail"] == "actual host detail"
    assert actual["numerical_summary"]
    assert "private working reference" not in json.dumps(contexts)


def test_archive_memory_chain_uses_reviewed_seed_not_launcher_input(tmp_path, monkeypatch):
    from benchmarks.ascend import phase2_curriculum
    from benchmarks.ascend.training_annex import _verify_adaptive_state
    c, _, _ = build(tmp_path, [wire("RETRY_TARGET"), wire("READY")],
        execute=lambda r, n: _result(r.spec, n != 1, "actual host verdict"))
    c.run()
    (c.root / "initial-memory.bin").write_bytes(b"original Phase 1 launcher input")
    monkeypatch.setattr(phase2_curriculum, "reviewed_seed", lambda root, identity: (c.initial_memory, None))
    _verify_adaptive_state(c.root, json.loads((c.root / "state.json").read_bytes()))


def test_semantic_actor_failure_preserves_actual_summary_and_host_facts(tmp_path):
    from benchmarks.a5kernels.phase1_wave_runtime import SemanticProjectFailure
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.ascend.agent_presentation import memory_presentation
    def execute(request, ordinal):
        if ordinal == 1:
            raise SemanticProjectFailure("Actor stopped after compiler rejected missing barrier", {
                "actor_status": "completed-unverified", "host_facts": [
                    {"category": "diagnostic", "statement": "compiler: expected event before Vector load",
                     "evidence_sha256": "a" * 64}]})
        return _result(request.spec, True, "host pass")
    c, contexts, requests = build(tmp_path, [wire("RETRY_TARGET"), wire("READY")], execute=execute)
    assert c.run()["status"] == "READY"
    actual = contexts[0]["latest_target_attempt"]["host_feedback"]
    assert actual["summary"] == "Actor stopped after compiler rejected missing barrier"
    assert actual["host_facts"][0]["statement"] == "compiler: expected event before Vector load"
    presented = memory_presentation(requests[-1].pre_wave_memory,
                                   ContentAddressedArtifacts(tmp_path / "view"))
    assert presented["phase2_feedback"][0]["host_feedback"] == actual


@pytest.mark.parametrize("field", ["kind", "verified", "spec", "host_feedback"])
def test_terminal_attempt_metadata_must_match_retained_branch(tmp_path, monkeypatch, field):
    from copy import deepcopy
    from benchmarks.ascend import phase2_curriculum
    from benchmarks.ascend.training_annex import _verify_adaptive_state
    c, _, _ = build(tmp_path, [wire("READY")])
    c.run()
    monkeypatch.setattr(phase2_curriculum, "reviewed_seed", lambda root, identity: (c.initial_memory, None))
    state = json.loads((c.root / "state.json").read_bytes())
    _verify_adaptive_state(c.root, state)
    changed = deepcopy(state)
    row = changed["attempts"][-1]
    if field == "kind":
        row[field] = "practice"
    elif field == "verified":
        row[field] = False
    elif field == "spec":
        row[field]["hypothesis"] = "changed after completion"
    else:
        row[field]["detail"] = "different observed diagnostic"
    with pytest.raises(ValueError, match="retained host outcome"):
        _verify_adaptive_state(c.root, changed)


def test_seed_and_failed_practice_ids_are_corrected_before_execution(tmp_path):
    c, _, requests = build(tmp_path, [], execute=lambda r, n: _result(r.spec, n != 2, "host verdict"))
    c.run_attempt()
    first = practice(c, "first")
    replies = iter([wire("PRACTICE", replace(first, project_id=c.seed_outcomes[0]["project_id"])),
                    wire("PRACTICE", first)])
    calls = []
    c.curriculum = TargetCurriculumAgent(lambda messages: (calls.append(messages), next(replies))[1])
    assert c.run_attempt()["practice_attempts"] == 1
    initial = json.loads(calls[0][-1]["content"].rsplit("\n", 1)[-1])
    assert c.seed_outcomes[0]["project_id"] in initial["used_project_ids"]
    assert "fresh project ID" in calls[1][-1]["content"]
    # A failed practice also reserves its ID after reloading retained state.
    calls.clear()
    replies = iter([wire("PRACTICE", first), wire("PRACTICE", practice(c, "second")), wire("READY")])
    assert c.resume()["status"] == "READY"
    initial = json.loads(calls[0][-1]["content"].rsplit("\n", 1)[-1])
    assert first.project_id in initial["used_project_ids"]
    assert "fresh project ID" in calls[1][-1]["content"]
    assert [r.spec.project_id for r in requests] == [c.target.spec.project_id,
        first.project_id, "practice-second", c.target.spec.project_id]


def test_duplicate_practice_correction_is_bounded_without_consuming_attempts(tmp_path):
    c, _, requests = build(tmp_path, [])
    c.run_attempt()
    calls = []
    duplicate = replace(practice(c), project_id=c.seed_outcomes[0]["project_id"])
    c.curriculum = TargetCurriculumAgent(lambda messages: (
        calls.append(messages), wire("PRACTICE", duplicate))[1])
    with pytest.raises(ValueError, match="after 3 attempts.*fresh project ID"):
        c.resume()
    assert len(calls) == 3
    assert len(requests) == 1
    assert not (c.root / "pending.json").exists()
    assert c.report()["practice_attempts"] == 0
    c.curriculum = TargetCurriculumAgent(lambda _: wire("READY"))
    assert c.resume()["status"] == "READY"
    assert len(requests) == 1


@pytest.mark.parametrize("field", ["target_attempts", "practice_since_target"])
def test_archive_rejects_budget_counters_that_differ_from_completed_attempts(tmp_path, monkeypatch, field):
    from benchmarks.ascend import phase2_curriculum
    from benchmarks.ascend.training_annex import _verify_adaptive_state, inventory_campaign
    c, _, _ = build(tmp_path, [], execute=lambda r, n: _result(r.spec, n != 2, "host verdict"))
    c.curriculum = TargetCurriculumAgent(lambda _: wire("PRACTICE", practice(c)))
    c.run_attempt()
    c.run_attempt()
    monkeypatch.setattr(phase2_curriculum, "reviewed_seed", lambda root, identity: (c.initial_memory, None))
    state = json.loads((c.root / "state.json").read_bytes())
    assert (state["target_attempts"], state["practice_since_target"]) == (1, 1)
    _verify_adaptive_state(c.root, state)
    state[field] = 0
    with pytest.raises(ValueError, match="budget counters differ"):
        _verify_adaptive_state(c.root, state)
    c._atomic_json(c.root / "state.json", state)
    assert {"kind": "invalid-adaptive-memory-chain"} in inventory_campaign(c.root)["issues"]


def test_archive_budget_counts_failed_practice_and_resets_on_each_target_retry(tmp_path, monkeypatch):
    from benchmarks.ascend import phase2_curriculum
    from benchmarks.ascend.training_annex import _verify_adaptive_state
    c, _, _ = build(tmp_path, [], execute=lambda r, n: _result(r.spec, n > 2, "host verdict"))
    replies = iter([wire("PRACTICE", practice(c, "one")), wire("RETRY_TARGET"),
                    wire("PRACTICE", practice(c, "two")), wire("RETRY_TARGET"), wire("READY")])
    c.curriculum = TargetCurriculumAgent(lambda _: next(replies))
    monkeypatch.setattr(phase2_curriculum, "reviewed_seed", lambda root, identity: (c.initial_memory, None))
    _verify_adaptive_state(c.root, c._load_state())
    for counters in [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0)]:
        c.run_attempt()
        state = json.loads((c.root / "state.json").read_bytes())
        assert (state["target_attempts"], state["practice_since_target"]) == counters
        _verify_adaptive_state(c.root, state)
    assert c.resume()["status"] == "READY"
    _verify_adaptive_state(c.root, c._load_state())


def test_archive_budget_counters_exclude_infrastructure_pending_attempts(tmp_path, monkeypatch):
    from benchmarks.ascend import phase2_curriculum
    from benchmarks.ascend.training_annex import _verify_adaptive_state
    def execute(request, ordinal):
        if ordinal in (1, 3):
            raise OSError("transport interrupted")
        return _result(request.spec, True, "host verdict")
    c, _, requests = build(tmp_path, [], execute=execute)
    replies = iter([wire("PRACTICE", practice(c)), wire("RETRY_TARGET"), wire("READY")])
    c.curriculum = TargetCurriculumAgent(lambda _: next(replies))
    monkeypatch.setattr(phase2_curriculum, "reviewed_seed", lambda root, identity: (c.initial_memory, None))
    for counters in [(0, 0), (1, 0)]:
        with pytest.raises(RuntimeError):
            c.run_attempt()
        state = c._load_state()
        assert (c.root / "pending.json").exists()
        assert (state["target_attempts"], state["practice_since_target"]) == counters
        _verify_adaptive_state(c.root, state)
        c.run_attempt()
        _verify_adaptive_state(c.root, c._load_state())
    assert c.resume()["status"] == "READY"
    assert len(requests) == 6  # Interrupted practice time also consumes its window.
    _verify_adaptive_state(c.root, c._load_state())


@pytest.mark.parametrize("boundary", ["empty", "failed-target", "practice-after-passing-target"])
def test_archive_ready_requires_final_verified_target(tmp_path, monkeypatch, boundary):
    from benchmarks.ascend import phase2_curriculum
    from benchmarks.ascend.training_annex import _verify_adaptive_state, inventory_campaign
    c, _, _ = build(tmp_path, [], execute=lambda r, n: _result(
        r.spec, boundary != "failed-target", "actual host verdict"))
    if boundary != "empty":
        c.run_attempt()
    if boundary == "practice-after-passing-target":
        c.curriculum = TargetCurriculumAgent(lambda _: wire("PRACTICE", practice(c)))
        c.run_attempt()
    monkeypatch.setattr(phase2_curriculum, "reviewed_seed", lambda root, identity: (c.initial_memory, None))
    state = c._load_state()
    _verify_adaptive_state(c.root, state)
    state["terminal"] = {"status": "READY", "reason": "unsupported readiness claim"}
    with pytest.raises(ValueError, match="READY requires a final host-verified target attempt"):
        _verify_adaptive_state(c.root, state)
    c._atomic_json(c.root / "state.json", state)
    assert {"kind": "invalid-adaptive-memory-chain"} in inventory_campaign(c.root)["issues"]


@pytest.mark.parametrize("boundary", ["initial", "completed", "pending-target", "pending-practice"])
def test_archive_next_kind_matches_completed_history_including_pending_work(tmp_path, monkeypatch, boundary):
    from benchmarks.ascend import phase2_curriculum
    from benchmarks.ascend.training_annex import _verify_adaptive_state, inventory_campaign
    def execute(request, ordinal):
        if (boundary == "pending-target" and ordinal == 1
                or boundary == "pending-practice" and ordinal == 2):
            raise OSError("transport interrupted")
        return _result(request.spec, True, "actual host verdict")
    c, _, _ = build(tmp_path, [], execute=execute)
    c.curriculum = TargetCurriculumAgent(lambda _: wire("PRACTICE", practice(c)))
    if boundary in {"completed", "pending-practice"}:
        c.run_attempt()
    if boundary.startswith("pending-"):
        with pytest.raises(RuntimeError):
            c.run_attempt()
        assert (c.root / "pending.json").exists()
    state = c._load_state()
    monkeypatch.setattr(phase2_curriculum, "reviewed_seed", lambda root, identity: (c.initial_memory, None))
    _verify_adaptive_state(c.root, state)
    state["next_kind"] = "review" if state["next_kind"] == "target" else "target"
    with pytest.raises(ValueError, match="next attempt kind differs"):
        _verify_adaptive_state(c.root, state)
    c._atomic_json(c.root / "state.json", state)
    assert {"kind": "invalid-adaptive-memory-chain"} in inventory_campaign(c.root)["issues"]
