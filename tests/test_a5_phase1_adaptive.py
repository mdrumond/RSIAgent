from __future__ import annotations

import json
import threading
import time

import pytest

import run_a5_phase1_adaptive
from benchmarks.a5kernels.evidence import canonical_digest
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
from benchmarks.a5kernels.phase1_registry import DEFAULT_PROPOSALS


SEED = b'{"existing":"published knowledge"}'


def _project(project_id, capability):
    return {
        "project_id": project_id,
        "capabilities": [capability],
        "shape": [257],
        "dtype": "float32",
        "block_count": 1,
        "padding_multiple": 64,
        "failure_mode": "none",
        "timing_lengths": [],
        "profile_metric": None,
        "hypothesis": f"Host evidence tests {project_id}.",
    }


def _wave(prompt, projects):
    context = json.loads(prompt.rsplit("\n", 1)[-1])
    return "DECISION: WAVE\n" + json.dumps({
        "wave_id": context["next_wave_id"],
        "pre_wave_memory_sha256": context["pre_wave_memory_sha256"],
        "pre_wave_coverage": context["pre_wave_coverage"],
        "projects": [{"spec": project, "requires": []} for project in projects],
    })


def _memory(ordinal, project_id):
    evidence = canonical_digest({"project": project_id})
    lesson = AgentLesson(
        "data-movement", f"lesson {project_id}", "A5 Catlass projects",
        "Host verification passes", "Only the measured shape", (evidence,),
        None, None, "high",
    )
    return Phase1ProjectMemory(
        DEFAULT_PROPOSALS[ordinal - 1], f"source-{project_id}", ("run",),
        (HostFact("host-verification", "host PASS", evidence),),
        lessons=(lesson,), no_new_knowledge=False,
        admitted_artifact_sha256=(evidence,),
    )


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
    fail_second = True
    lock = threading.Lock()

    def execute(request):
        nonlocal fail_second
        with lock:
            calls.append(request.spec.project_id)
            snapshots.append(request.pre_wave_memory)
        time.sleep({"first": .04, "second": .02, "third": 0}[request.spec.project_id])
        with lock:
            completion_order.append(request.spec.project_id)
        verified = not (request.spec.project_id == "second" and fail_second)
        ordinal = {"first": 1, "second": 2, "third": 3}[request.spec.project_id]
        return AdaptiveProjectResult(
            _memory(ordinal, request.spec.project_id), verified,
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
    assert first.report()["published_projects"] == []
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
    prompt = json.loads(selection_prompts[1].rsplit("\n", 1)[-1])
    assert prompt["pre_wave_coverage"] == result["coverage"]
    assert prompt["pre_wave_memory_sha256"] == result["memory_sha256"]
    assert not (root / "pending.json").exists()


@pytest.mark.parametrize("terminal", ["SATURATED", "STALLED"])
def test_terminal_decisions_are_durable_without_project_execution(tmp_path, terminal):
    calls = []

    def complete(_messages):
        calls.append("select")
        return f'DECISION: {terminal}\n{{"reason":"bounded stop"}}'

    def execute(_request):
        pytest.fail("a terminal selector decision must not execute a project")

    coordinator = AdaptivePhase1Coordinator(
        tmp_path, "Learn bounded skills.", SEED,
        CurriculumAgent(complete), execute, max_parallel=1,
    )
    assert coordinator.run()["status"] == terminal
    assert coordinator.resume()["status"] == terminal
    assert calls == ["select"]


def test_initial_memory_identity_is_checked_on_resume(tmp_path):
    agent = CurriculumAgent(
        lambda _: 'DECISION: STALLED\n{"reason":"bounded stop"}'
    )
    AdaptivePhase1Coordinator(
        tmp_path, "Learn bounded skills.", SEED, agent,
        lambda _: None, max_parallel=1,
    ).run()

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
        ["run", "--root", str(tmp_path), "--factory", "binding:build"],
        factory_loader=loader,
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "STALLED"
    assert seen == ["binding:build"]
