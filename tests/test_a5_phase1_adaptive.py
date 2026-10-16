from __future__ import annotations

import base64
import json
import threading

import pytest

import run_a5_phase1_adaptive
from benchmarks.a5kernels.evidence import canonical_bytes, canonical_digest
from benchmarks.a5kernels.phase1_adaptive import (
    AdaptivePhase1Coordinator,
    AdaptiveProjectResult,
)
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
from benchmarks.a5kernels.phase1_memory import (
    AgentLesson,
    HostFact,
    Phase1ProjectMemory,
)
from benchmarks.a5kernels.phase1_project import (
    EvidenceClassification,
    ExecutionEvidence,
    evaluate_reference,
)


SEED = b'{"existing":"published knowledge"}'


def _project(project_id, capability):
    operation = "add" if capability == "simd-vector" else "copy"
    shape = [1025] if capability == "padded-tail" else [257]
    inputs = [{"name": "lhs", "shape": shape}]
    operands = ["lhs"]
    if operation == "add":
        inputs.append({"name": "rhs", "shape": shape})
        operands.append("rhs")
    return {
        "project_id": project_id,
        "capabilities": [capability],
        "shape": shape,
        "dtype": "float32",
        "block_count": 1,
        "padding_multiple": 64,
        "failure_mode": "none",
        "timing_lengths": [],
        "profile_metric": None,
        "hypothesis": f"Host evidence tests {project_id}.",
        "graph": {
            "inputs": inputs,
            "nodes": [{
                "name": "output", "op": operation,
                "operands": operands, "axis": None,
            }],
            "output": "output",
        },
    }


def _wave(prompt, projects):
    context = json.loads(prompt.rsplit("\n", 1)[-1])
    return "DECISION: WAVE\n" + json.dumps({
        "wave_id": context["next_wave_id"],
        "pre_wave_memory_sha256": context["pre_wave_memory_sha256"],
        "pre_wave_coverage": context["pre_wave_coverage"],
        "projects": [{"spec": project, "requires": []} for project in projects],
    })


def _memory(spec):
    evidence = canonical_digest({"project": spec.project_id})
    lesson = AgentLesson(
        "data-movement", f"lesson {spec.project_id}", "A5 Catlass projects",
        "Host verification passes", "Only the measured shape", (evidence,),
        None, None, "high",
    )
    return Phase1ProjectMemory(
        spec, f"source-{spec.project_id}", ("run",),
        (HostFact("host-verification", "host PASS", evidence),),
        lessons=(lesson,), no_new_knowledge=False,
        admitted_artifact_sha256=(evidence,),
    )


def _result(spec, verified, summary):
    classification = (
        EvidenceClassification.HOST_VERIFIED
        if verified else EvidenceClassification.HOST_MISMATCH
    )
    evidence = ExecutionEvidence(classification, summary)
    payload = canonical_bytes({
        "classification": classification.value,
        "project_id": spec.project_id,
        "summary": summary,
    })
    return AdaptiveProjectResult(_memory(spec), evidence, payload, summary)


def test_invalid_curriculum_selection_is_corrected_before_execution_and_publication(tmp_path):
    calls = []
    executed = []

    def complete(messages):
        calls.append(messages)
        if len(calls) == 1:
            invalid = _project("invalid-metadata", "simd-vector")
            invalid["graph"]["nodes"][0]["op"] = "copy"
            invalid["graph"]["nodes"][0]["operands"] = ["lhs"]
            return _wave(messages[-1]["content"], [invalid])
        if len(calls) == 2:
            assert "simd-vector" in messages[-1]["content"]
            return _wave(messages[1]["content"], [
                _project("valid-copy", "gm-ub-movement"),
            ])
        return 'DECISION: SATURATED\n{"reason":"requested coverage complete"}'

    def execute(request):
        executed.append(request.spec.project_id)
        return _result(request.spec, True, "verified corrected project")

    result = AdaptivePhase1Coordinator(
        tmp_path / "correction", "Learn bounded skills.", SEED,
        CurriculumAgent(complete), execute, max_parallel=1,
    ).run()
    assert result["status"] == "SATURATED"
    assert executed == ["valid-copy"]
    assert result["published_projects"] == ["valid-copy"]
    assert len(calls) == 3


def test_exhausted_selection_preserves_published_state_for_resume(tmp_path):
    calls = []
    stop_after_first = False

    def complete(messages):
        calls.append(messages)
        if len(calls) == 1:
            return _wave(messages[-1]["content"], [
                _project("published", "gm-ub-movement"),
            ])
        if not stop_after_first:
            return "DECISION: WAVE\n{broken"
        return 'DECISION: SATURATED\n{"reason":"requested coverage complete"}'

    executions = []

    def execute(request):
        executions.append(request.spec.project_id)
        return _result(request.spec, True, "verified before selection failure")

    root = tmp_path / "exhaustion"
    coordinator = AdaptivePhase1Coordinator(
        root, "Learn bounded skills.", SEED,
        CurriculumAgent(complete), execute, max_parallel=1,
    )
    with pytest.raises(ValueError, match="3 attempts"):
        coordinator.run()
    before = json.loads((root / "state.json").read_text())
    assert coordinator.report()["published_projects"] == ["published"]
    assert len(calls) == 4
    stop_after_first = True
    resumed = AdaptivePhase1Coordinator(
        root, "Learn bounded skills.", SEED,
        CurriculumAgent(complete), execute, max_parallel=1,
    ).resume()
    assert resumed["status"] == "SATURATED"
    assert resumed["published_projects"] == ["published"]
    assert executions == ["published"]
    after = json.loads((root / "state.json").read_text())
    assert after["memory_bytes_base64"] == before["memory_bytes_base64"]


def test_reverse_completion_failure_resume_and_authored_atomic_publication(tmp_path):
    selection_prompts = []

    def complete(messages):
        prompt = messages[-1]["content"]
        selection_prompts.append(prompt)
        if len(selection_prompts) == 1:
            return _wave(prompt, [
                _project("first", "gm-ub-movement"),
                _project("second", "simd-vector"),
                _project("third", "padded-tail"),
            ])
        return "DECISION: SATURATED\n" + json.dumps({
            "reason": "the requested reusable coverage is complete",
        })

    calls = []
    snapshots = []
    completion_order = []
    observed_oracles = []
    fail_second = True
    lock = threading.Lock()
    first_attempt = threading.Barrier(3)
    third_done = threading.Event()
    second_done = threading.Event()

    def execute(request):
        nonlocal fail_second
        with lock:
            calls.append(request.spec.project_id)
            snapshots.append(request.pre_wave_memory)
            observed_oracles.append(request.oracle)
            attempt = calls.count(request.spec.project_id)
        assert request.oracle == evaluate_reference(request.spec, request.inputs)
        if attempt == 1:
            first_attempt.wait()
            if request.spec.project_id == "first":
                second_done.wait()
            elif request.spec.project_id == "second":
                third_done.wait()
        with lock:
            completion_order.append(request.spec.project_id)
        if attempt == 1 and request.spec.project_id == "third":
            third_done.set()
        elif attempt == 1 and request.spec.project_id == "second":
            second_done.set()
        verified = not (request.spec.project_id == "second" and fail_second)
        return _result(
            request.spec, verified,
            f"{request.spec.project_id} host verification {'passed' if verified else 'failed'}",
        )

    root = tmp_path / "adaptive"
    first = AdaptivePhase1Coordinator(
        root, "Learn reusable Catlass DSL programming.", SEED,
        CurriculumAgent(complete), execute, max_parallel=3,
    )
    with pytest.raises(RuntimeError, match="not verified"):
        first.run()

    assert set(calls) == {"first", "second", "third"}
    assert len(calls) == 3
    assert completion_order == ["third", "second", "first"]
    assert len(set(snapshots)) == 1
    assert len(observed_oracles) == 3
    failed_report = first.report()
    assert failed_report["status"] == "RECOVERY_REQUIRED"
    assert failed_report["published_projects"] == []
    assert failed_report["pending_wave"] == {
        "wave_id": "wave-001",
        "projects": [
            {"project_id": "first", "status": "verified"},
            {"project_id": "second", "status": "unverified"},
            {"project_id": "third", "status": "verified"},
        ],
    }
    assert len(selection_prompts) == 1

    fail_second = False
    resumed = AdaptivePhase1Coordinator(
        root, "Learn reusable Catlass DSL programming.", SEED,
        CurriculumAgent(complete), execute, max_parallel=2,
    )
    result = resumed.resume()

    assert calls.count("first") == calls.count("third") == 1
    assert calls.count("second") == 2
    assert len(selection_prompts) == 2
    assert result["status"] == "SATURATED"
    assert result["coverage"] == [
        "gm-ub-movement", "simd-vector", "padded-tail",
    ]
    assert result["published_projects"] == ["first", "second", "third"]
    assert [row["source_revision"] for row in result["memory"]] == [
        "source-first", "source-second", "source-third",
    ]
    assert [row["project_spec"]["project_id"] for row in result["memory"]] == [
        "first", "second", "third",
    ]
    state = json.loads((root / "state.json").read_text())
    assert [base64.b64decode(value) for value in state["memory_bytes_base64"]] == [
        canonical_bytes(memory) for memory in result["memory"]
    ]
    prompt = json.loads(selection_prompts[1].rsplit("\n", 1)[-1])
    assert prompt["pre_wave_coverage"] == result["coverage"]
    assert prompt["pre_wave_memory_sha256"] == result["memory_sha256"]
    assert [row["information"]["operations"] for row in prompt["prior_outcomes"]] == [
        ["copy"], ["add"], ["copy"],
    ]
    assert not (root / "pending.json").exists()


@pytest.mark.parametrize("crash_point", ["before-state", "before-pending-removal"])
def test_resume_across_adaptive_publication_crash_boundaries(
    tmp_path, monkeypatch, crash_point,
):
    selections = []
    executions = []

    def complete(messages):
        selections.append("select")
        if len(selections) == 1:
            return _wave(messages[-1]["content"], [
                _project("first", "gm-ub-movement"),
            ])
        return 'DECISION: SATURATED\n{"reason":"coverage complete"}'

    def execute(request):
        executions.append(request.spec.project_id)
        return _result(request.spec, True, "first host verification passed")

    root = tmp_path / crash_point
    coordinator = AdaptivePhase1Coordinator(
        root, "Learn bounded skills.", SEED,
        CurriculumAgent(complete), execute, max_parallel=1,
    )
    if crash_point == "before-state":
        atomic_json = coordinator._atomic_json

        def fail_before_state(path, value):
            if path.name == "state.json" and value.get("wave_index") == 2:
                raise RuntimeError("injected before adaptive state publication")
            atomic_json(path, value)

        monkeypatch.setattr(coordinator, "_atomic_json", fail_before_state)
    else:
        unlink = type(root).unlink

        def fail_before_pending_removal(path, *args, **kwargs):
            if path == root / "pending.json":
                raise RuntimeError("injected before pending removal")
            return unlink(path, *args, **kwargs)

        monkeypatch.setattr(type(root), "unlink", fail_before_pending_removal)

    with pytest.raises(RuntimeError, match="injected"):
        coordinator.run()
    monkeypatch.undo()

    resumed = AdaptivePhase1Coordinator(
        root, "Learn bounded skills.", SEED,
        CurriculumAgent(complete), execute, max_parallel=1,
    ).resume()
    assert resumed["status"] == "SATURATED"
    assert resumed["published_projects"] == ["first"]
    assert [row["project_spec"]["project_id"] for row in resumed["memory"]] == ["first"]
    assert executions == ["first"]


@pytest.mark.parametrize("terminal", ["SATURATED", "STALLED"])
def test_terminal_decisions_are_durable_after_verified_project(tmp_path, terminal):
    selection_calls = []
    project_calls = []

    def complete(messages):
        selection_calls.append("select")
        if len(selection_calls) == 1:
            return _wave(messages[-1]["content"], [
                _project("first", "gm-ub-movement"),
            ])
        return f'DECISION: {terminal}\n{{"reason":"bounded stop"}}'

    def execute(request):
        project_calls.append(request.spec.project_id)
        return _result(request.spec, True, "first host verification passed")

    coordinator = AdaptivePhase1Coordinator(
        tmp_path, "Learn bounded skills.", SEED,
        CurriculumAgent(complete), execute, max_parallel=1,
    )
    assert coordinator.run()["status"] == terminal
    assert coordinator.resume()["status"] == terminal
    assert selection_calls == ["select", "select"]
    assert project_calls == ["first"]


def test_initial_memory_identity_is_checked_on_resume(tmp_path):
    agent = CurriculumAgent(
        lambda _: 'DECISION: STALLED\n{"reason":"bounded stop"}'
    )
    AdaptivePhase1Coordinator(
        tmp_path, "Learn bounded skills.", SEED, agent,
        lambda _: None, max_parallel=1,
    ).report()

    with pytest.raises(ValueError, match="initial memory"):
        AdaptivePhase1Coordinator(
            tmp_path, "Learn bounded skills.", b"different", agent,
            lambda _: None, max_parallel=1,
        ).report()


def test_cli_uses_explicit_production_factory_and_reports_json(tmp_path, capsys):
    coordinator = AdaptivePhase1Coordinator(
        tmp_path, "Learn bounded skills.", SEED,
        CurriculumAgent(lambda _: 'DECISION: STALLED\n{"reason":"bounded stop"}'),
        lambda _: None, max_parallel=1,
    )
    seen = []

    def loader(reference):
        seen.append(reference)
        return lambda args: coordinator if args.root == tmp_path else None

    assert run_a5_phase1_adaptive.main(
        ["report", "--root", str(tmp_path), "--factory", "binding:build"],
        factory_loader=loader,
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "RUNNING"
    assert seen == ["binding:build"]


def test_target_stops_and_resume_with_larger_target_preserves_memory(tmp_path):
    counter = 0
    def complete(messages):
        nonlocal counter
        counter += 1
        item = _project(f"project-{counter}", "gm-ub-movement")
        item["learning_focus"] = ["pipe-mte2" if counter == 1 else "pipe-mte3"]
        return _wave(messages[-1]["content"], [item])
    def execute(request):
        return _result(request.spec, True, "verified")
    def campaign(target):
        return AdaptivePhase1Coordinator(tmp_path, "Learn pipes", SEED,
            CurriculumAgent(complete), execute, max_parallel=1, project_target=target)
    first = campaign(1).run()
    assert first["status"] == "TARGET_REACHED"
    with pytest.raises(ValueError, match="continue_campaign"):
        campaign(2).resume()
    continued = campaign(1)
    continued.continue_campaign(2)
    second = continued.resume()
    assert second["successful_projects"] == 2
    assert second["published_projects"][0] == first["published_projects"][0]
    assert "pipe-mte2" not in second["coverage_gaps"]
    assert "pipe-mte3" not in second["coverage_gaps"]


def test_omitted_target_preserves_retained_override_and_new_default(tmp_path):
    def create(root, target=None):
        return AdaptivePhase1Coordinator(root, "Learn pipes", SEED,
            CurriculumAgent(lambda _: 'DECISION: STALLED\n{"reason":"unneeded"}'),
            lambda _: None, max_parallel=1, project_target=target)
    assert create(tmp_path / "new").project_target == 64
    retained = create(tmp_path / "retained", 128)
    state = retained._load_state()
    state["project_target"] = 128
    retained._atomic_json(retained.root / "state.json", state)
    assert create(retained.root).project_target == 128
    assert create(retained.root, 256).project_target == 256


@pytest.mark.parametrize("observed", [False, True])
def test_tool_coverage_uses_observed_host_facts_not_requested_labels(observed):
    spec = _project("tools", "gm-ub-movement")
    spec["learning_focus"] = ["tool-msprof", "tool-msdebug", "tool-mssanitizer", "debug-print"]
    facts = [{"category": "host-verification", "statement": "verified"}]
    if observed:
        facts += [{"category": "profiling", "statement": "profiled"},
                  {"category": "runtime", "statement": 'tool-observation:{"available":true,"tool":"msdebug","execution":{}}'},
                  {"category": "runtime", "statement": 'tool-observation:{"available":true,"tool":"memcheck","execution":{}}'},
                  {"category": "runtime", "statement": "verified-debug-print:source"}]
    else:
        facts += [{"category": "runtime", "statement": 'tool-observation:{"available":false,"tool":"memcheck"}'}]
    state = {"prior_outcomes": [{"spec": spec, "verified": True}], "memory_bytes_base64": [
        base64.b64encode(json.dumps({"host_facts": facts}).encode()).decode()]}
    effective = AdaptivePhase1Coordinator._effective_outcomes(state)[0]["spec"]
    assert effective.get("learning_focus", []) == (spec["learning_focus"] if observed else [])
    gaps = AdaptivePhase1Coordinator._coverage_gaps(state)
    assert all((focus not in gaps) == observed for focus in spec["learning_focus"])
