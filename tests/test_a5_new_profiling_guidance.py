import hashlib
import json

import pytest

from canonical_kernel_fixtures import canonical_kernel
from test_rsi_production_domain_bindings import CanonicalSession

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


def test_guidance_reaches_actual_curriculum_provider_and_actor_context(tmp_path, canonical_kernel):
    identity, text = _guidance(tmp_path / "skill")
    cell = next(c for c in build_cells(_guide(), profiling_guide=identity)
                if c.profiling is ProfilingGuidance.NEW and c.model.provider == "DeepSeek"
                and c.knowledge.value == "without-kdb")
    calls = []
    now = [0.0]

    def transport(**kwargs):
        calls.append(kwargs["request"])
        now[0] += 1
        return DirectCompletion('DECISION: WAVE\n{}', 1)

    campaign = build_adaptive_phase1(
        cell=cell, paths=_paths(tmp_path), objective="Learn reusable profiling.",
        initial_memory=b"", guide_bytes=GUIDE_BYTES,
        profiling_guidance_text=text, environment={"OPENAI_API_KEY": "test", "DEEPSEEK_API_KEY": "test"},
        provider_transport=transport, command_executor=FakeCommandExecutor(),
    )
    assert json.loads(campaign.project_actor.immutable_context)["profiling_guidance"] == text
    ticks = iter(range(0, 1000000, 2000))
    campaign.coordinator.curriculum.clock = lambda: next(ticks)
    with pytest.raises(ValueError, match="time budget exhausted"):
        campaign.run()
    assert text in calls[0]["messages"][1]["content"]
    spec = _project("context-copy", "gm-ub-movement")
    from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
    prepared = campaign.project_actor.executor.prepare(ProjectSpec.from_mapping(spec))
    from rsi.domain_bindings import kernel_adoption
    from rsi.contracts import BudgetLimits
    case = canonical_kernel
    campaign.actor.canonical = kernel_adoption(baseline_query=case.baseline_query)
    campaign.actor.canonical_budgets = BudgetLimits(actor_seconds=120, wall_seconds=120)
    case.enable_native()
    offsets, instruction_refs = [], []
    def select(context, allowed):
        instruction = context['task']['domain']['instruction']
        ref = hashlib.sha256(instruction.encode()).hexdigest()
        if not instruction_refs:
            instruction_refs.append(ref)
        offset = len(offsets) * 3072
        if offset < len(instruction.encode()):
            offsets.append(offset)
            return 'read_context', dict(record=ref, offset=offset, limit=3072)
        return CanonicalSession.next_action(case.session, context, allowed)
    case.select = select
    case.after_select = lambda context, selected: setattr(case, 'now', case.now + 121) if selected[0] == 'perform' else None
    # Keep production composition and treatment binding; only model decisions
    # are scripted through the actual canonical Pi SDK gateway.
    with pytest.raises(RuntimeError, match="without host-verified"):
        campaign.project_actor(AdaptiveProjectRequest(
            prepared.spec, 1, b"", tmp_path / "actor-proof", prepared.inputs, prepared.expected,
        ))
    pages = {}
    def visit(value):
        if isinstance(value, dict):
            if value.get('record') == instruction_refs[0] and 'text' in value:
                pages[value['offset']] = value['text']
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            try:
                decoded = json.loads(value)
            except (ValueError, TypeError):
                return
            if not isinstance(decoded, str):
                visit(decoded)
    visit(case.wires)
    delivered = ''.join(pages[offset] for offset in sorted(pages))
    actor_context = json.loads(delivered.split(
        "Production context (immutable, identical for every project): ", 1)[1])
    assert actor_context['profiling_guidance'] == text
    retained = [row['request'] for row in case.runner.events if row['type'] == 'pi-provider-request']
    assert retained == case.wires and retained
    dispatches = [case.runner.store.read_json(row['reference'])
                  for row in case.runner.events if row['type'] == 'domain-dispatch']
    assert not any(row['category'] in ('compilation', 'execution', 'profiling') for row in dispatches)
    with pytest.raises(ValueError, match="does not match"):
        build_adaptive_phase1(
            cell=cell, paths=_paths(tmp_path / "bad"), objective="Learn.",
            initial_memory=b"", guide_bytes=GUIDE_BYTES,
            profiling_guidance_text=text + " changed", command_executor=FakeCommandExecutor(),
        )


def test_time_allocation_stops_before_next_selection_and_survives_resume(tmp_path):
    calls, now = [], [0.0]
    def complete(messages):
        calls.append(messages)
        return _wave(messages[-1]["content"], [_project("only", "gm-ub-movement")])
    def execute(request):
        assert request.remaining_seconds == 5
        now[0] += 5
        return _result(request.spec, True, "verified")
    coordinator = AdaptivePhase1Coordinator(tmp_path, "Learn.", b"", CurriculumAgent(complete),
        execute, max_parallel=1, max_seconds=5, clock=lambda: now[0])
    report = coordinator.run()
    assert report["status"] == "TIME_BUDGET_EXHAUSTED"
    assert report["elapsed_seconds"] == 5 and report["remaining_seconds"] == 0
    assert report["published_projects"] == ["only"]
    assert len(calls) == 1
    before = (tmp_path / "state.json").read_bytes()
    assert coordinator.resume() == report
    assert (tmp_path / "state.json").read_bytes() == before
    with pytest.raises(ValueError, match="time allocation"):
        AdaptivePhase1Coordinator(tmp_path, "Learn.", b"", CurriculumAgent(complete),
                                  lambda _: None, max_parallel=1, max_seconds=6).resume()


def test_slow_selection_cannot_admit_execution_after_deadline(tmp_path):
    now, executed = [0.0], []
    def complete(messages):
        now[0] += 6
        return _wave(messages[-1]["content"], [_project("one", "gm-ub-movement")])
    report = AdaptivePhase1Coordinator(tmp_path, "Learn.", b"", CurriculumAgent(complete),
        lambda request: executed.append(request), max_parallel=1,
        max_seconds=5, clock=lambda: now[0]).run()
    assert report["status"] == "TIME_BUDGET_EXHAUSTED"
    assert report["elapsed_seconds"] == 6
    assert executed == []


pytestmark = pytest.mark.usefixtures("scripted_pi")
