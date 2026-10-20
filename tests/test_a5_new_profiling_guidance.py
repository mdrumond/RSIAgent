import hashlib
import json

import pytest

from benchmarks.a5kernels.phase1_experiments import (
    ProfilingGuidance, PendingTreatmentError, admit_runnable_cell, build_cells,
)
from benchmarks.a5kernels.profiling_guidance import (
    admit_profiling_guidance, verify_profiling_guidance,
)
from test_a5_phase1_adaptive import _project, _result, _wave
from test_a5_phase1_adaptive_production import GUIDE_BYTES, FakeCommandExecutor, _guide, _paths
from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator, AdaptiveProjectRequest
from benchmarks.a5kernels.phase1_adaptive_production import build_adaptive_phase1
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
from benchmarks.a5kernels.phase1_provider import DirectCompletion


def _guidance(tmp_path):
    for name in ("SKILL.md", "references/a5-metric-semantics.md", "references/a5-timeline-semantics.md"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen {name}: missing is not zero; profiled time is diagnostic")
    return admit_profiling_guidance(tmp_path, cpl_skills_revision="c" * 40)


def test_admission_hashes_all_references_and_preserves_legacy_cells(tmp_path):
    identity, text = _guidance(tmp_path)
    assert identity.guidance_sha256 == hashlib.sha256(text.encode()).hexdigest()
    assert "request-timing" in text and "request-profile" in text
    assert "BasicInfo" in text and "a5-timeline-semantics.md" in text
    verify_profiling_guidance(identity, text)
    with pytest.raises(ValueError, match="does not match"):
        verify_profiling_guidance(identity, text + " changed")
    old = build_cells(_guide())
    new = build_cells(_guide(), profiling_guide=identity)
    for before, after in zip(old, new, strict=True):
        if before.profiling is ProfilingGuidance.NEW:
            with pytest.raises(PendingTreatmentError):
                admit_runnable_cell(before)
            assert admit_runnable_cell(after) is after
            assert before.cell_id != after.cell_id
            assert after.identity_values()["profiling_guide"] == identity.as_dict()
        else:
            assert before.as_dict() == after.as_dict()


def test_guidance_reaches_actual_curriculum_provider_and_actor_context(tmp_path):
    identity, text = _guidance(tmp_path / "skill")
    cell = next(c for c in build_cells(_guide(), profiling_guide=identity)
                if c.profiling is ProfilingGuidance.NEW and c.model.provider == "OpenAI"
                and c.knowledge.value == "without-kdb")
    calls = []

    def transport(**kwargs):
        calls.append(kwargs["request"])
        return DirectCompletion('DECISION: WAVE\n{}', 1)

    campaign = build_adaptive_phase1(
        cell=cell, paths=_paths(tmp_path), objective="Learn reusable profiling.",
        initial_memory=b"", guide_bytes=GUIDE_BYTES,
        profiling_guidance_text=text, environment={"OPENAI_API_KEY": "test", "DEEPSEEK_API_KEY": "test"},
        provider_transport=transport, command_executor=FakeCommandExecutor(),
    )
    assert json.loads(campaign.project_actor.immutable_context)["profiling_guidance"] == text
    with pytest.raises(ValueError, match="3 attempts"):
        campaign.run()
    assert text in calls[0]["messages"][1]["content"]
    spec = _project("context-copy", "gm-ub-movement")
    from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
    prepared = campaign.project_actor.executor.prepare(ProjectSpec.from_mapping(spec))
    campaign.actor.max_turns = 1
    with pytest.raises(RuntimeError, match="without host-verified"):
        campaign.project_actor(AdaptiveProjectRequest(
            prepared.spec, 1, b"", tmp_path / "actor-proof", prepared.inputs, prepared.expected,
        ))
    actor_request = calls[-1]
    actor_context = json.loads(actor_request["messages"][1]["content"].split(
        "Production context (immutable, identical for every project): ", 1,
    )[1])
    assert actor_context["profiling_guidance"] == text
    retained = list((tmp_path / "actor-proof" / "artifacts" / "objects").rglob("*"))
    assert any(path.is_file() and path.read_bytes() == production_request_bytes({"request": actor_request})
               for path in retained)
    with pytest.raises(ValueError, match="does not match"):
        build_adaptive_phase1(
            cell=cell, paths=_paths(tmp_path / "bad"), objective="Learn.",
            initial_memory=b"", guide_bytes=GUIDE_BYTES,
            profiling_guidance_text=text + " changed", command_executor=FakeCommandExecutor(),
        )


def production_request_bytes(value):
    from benchmarks.a5kernels.evidence import canonical_bytes
    return canonical_bytes(value)


def test_project_cap_stops_before_next_selection_and_survives_resume(tmp_path):
    calls = []

    def complete(messages):
        calls.append(messages)
        return _wave(messages[-1]["content"], [_project("only", "gm-ub-movement")])

    coordinator = AdaptivePhase1Coordinator(
        tmp_path, "Learn.", b"", CurriculumAgent(complete),
        lambda request: _result(request.spec, True, "verified"),
        max_parallel=1, max_projects=1,
    )
    report = coordinator.run()
    assert report["status"] == "PROJECT_LIMIT_REACHED"
    assert report["published_projects"] == ["only"]
    assert len(calls) == 1
    before = (tmp_path / "state.json").read_bytes()
    assert coordinator.resume() == report
    assert (tmp_path / "state.json").read_bytes() == before
    with pytest.raises(ValueError, match="project limit"):
        AdaptivePhase1Coordinator(tmp_path, "Learn.", b"", CurriculumAgent(complete),
                                  lambda _: None, max_parallel=1, max_projects=2).resume()


def test_over_budget_wave_is_corrected_before_any_execution(tmp_path):
    calls = []

    def complete(messages):
        calls.append(messages)
        original = messages[1]["content"]
        if len(calls) == 1:
            return _wave(original, [_project("one", "gm-ub-movement"), _project("two", "simd-vector")])
        assert "remaining project budget" in messages[-1]["content"]
        return _wave(original, [_project("one", "gm-ub-movement")])

    executed = []

    def execute(request):
        executed.append(request.spec.project_id)
        return _result(request.spec, True, "verified")

    report = AdaptivePhase1Coordinator(tmp_path, "Learn.", b"", CurriculumAgent(complete),
                                     execute, max_parallel=1, max_projects=1).run()
    assert report["status"] == "PROJECT_LIMIT_REACHED"
    assert executed == ["one"]


pytestmark = pytest.mark.usefixtures("scripted_pi")
