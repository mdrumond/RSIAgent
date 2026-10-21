import json
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
from benchmarks.ascend.memory_usefulness import UsefulnessVerifier, paired_outcome
from test_a5_phase1_adaptive import _memory, _project, _result, _wave


@pytest.mark.parametrize('control,treatment,expected', [
    (False, True, 'demonstrated-benefit'), (True, True, 'both-succeeded'),
    (False, False, 'neither-succeeded'), (True, False, 'treatment-regressed'),
    (None, True, 'inconclusive')])
def test_host_outcomes(control, treatment, expected):
    assert paired_outcome({'verified': control}, {'verified': treatment}) == expected


def test_production_curriculum_pair_isolated_frozen_and_idempotent(tmp_path):
    requests, prompts = [], []
    def author(messages):
        prompts.append(messages[-1]['content'])
        return _wave(prompts[-1], [_project('ab-copy', 'gm-ub-movement')])
    def execute(request):
        requests.append(request)
        return _result(request.spec, True, 'host pass')
    verifier = UsefulnessVerifier(tmp_path, CurriculumAgent(author), execute,
        lambda stage, evidence: stage, enabled=True)
    memory = _memory(ProjectSpec.from_mapping(_project('source', 'gm-ub-movement')))
    prior = b'{"initial_memory_base64":"e30=","project_memory_base64":[]}'
    result = verifier.assess_memory(memory, prior, 'phase1/source')
    assert result[0]['outcome'] == 'both-succeeded'
    assert 'SMALL' in prompts[0] and '600' in prompts[0]
    assert requests[0].spec == requests[1].spec
    assert requests[0].inputs == requests[1].inputs and requests[0].oracle == requests[1].oracle
    assert requests[0].branch_root != requests[1].branch_root
    assert all(0 < r.remaining_seconds <= 300 and not r.publish_learning for r in requests)
    assert requests[0].remaining_seconds == requests[1].remaining_seconds
    baseline, treatment = map(lambda r: json.loads(r.pre_wave_memory), requests)
    assert treatment.pop('usefulness_lesson')['claim'] == memory.lessons[0].claim
    assert baseline == treatment
    assert verifier.assess_memory(memory, prior, 'phase1/source') == result
    assert len(requests) == 2


def test_disabled_and_infrastructure_failure_do_not_gate_memory(tmp_path):
    memory = _memory(ProjectSpec.from_mapping(_project('source', 'gm-ub-movement')))
    verifier = UsefulnessVerifier(tmp_path, None, None, None)
    assert verifier.assess_memory(memory, b'{}', 'source') == []
    assert not tmp_path.exists() or not list(tmp_path.iterdir())
    verifier.enabled = True
    assert verifier.assess_memory(memory, b'{}', 'source')[0]['outcome'] == 'inconclusive'
    assert memory.lessons


def test_committed_phase1_lessons_assessed_on_resume_without_duplicates(tmp_path):
    from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
    from test_a5_phase1_adaptive import SEED
    def author(messages):
        return _wave(messages[-1]['content'], [_project('committed', 'gm-ub-movement')])
    coordinator = AdaptivePhase1Coordinator(tmp_path, 'learn', SEED, CurriculumAgent(author),
        lambda req: _result(req.spec, True, 'pass'), project_target=1, max_parallel=1)
    calls = []
    coordinator.usefulness_verifier = SimpleNamespace(enabled=True,
        assess_memory=lambda memory, prior, source: calls.append((memory, prior, source)))
    coordinator.run()
    assert len(calls) == 1
    assert calls[0][2] == 'wave-001/committed'
    assert json.loads(calls[0][1])['project_memory_base64'] == []
    coordinator.resume()
    # The engine deduplicates repeat reconciliation calls by stable experiment ID.
    assert len(calls) == 2


def test_experiment_actor_skips_distillation_and_returns_no_lessons(tmp_path):
    from dataclasses import replace
    from benchmarks.a5kernels.evidence import EvidenceLedger
    from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActorProjectExecutor
    from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
    from test_a5_phase1_adaptive_actor import SourceBackend, ScriptedActor, _request
    executor = ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path / 'evidence.jsonl'))
    actor = ScriptedActor(('{"write":{"slot":"kernel","content":"fixed source"}}',
                           '{"run":{}}', '{"submit":{}}'))
    result = AdaptiveActorProjectExecutor(executor, actor, object())(
        replace(_request(tmp_path, executor), publish_learning=False))
    assert result.evidence.verified
    assert actor.distill_calls == []
    assert result.memory.no_new_knowledge and result.memory.lessons == ()


@pytest.mark.parametrize('exhaust_stage', ['inspect', 'curriculum', 'control'])
def test_parent_deadline_blocks_further_calls_and_caps_arms(tmp_path, exhaust_stage):
    now = [0.0]
    stages, requests = [], []
    def verify(stage, evidence):
        stages.append(stage)
        if stage == exhaust_stage:
            now[0] = 10
        return stage
    def author(messages):
        if exhaust_stage == 'curriculum':
            now[0] = 10
        return _wave(messages[-1]['content'], [_project('deadline-copy', 'gm-ub-movement')])
    def execute(request):
        requests.append(request)
        assert request.remaining_seconds <= 5
        if exhaust_stage == 'control':
            now[0] = 10
        return _result(request.spec, True, 'host pass')
    verifier = UsefulnessVerifier(tmp_path, CurriculumAgent(author), execute, verify,
        enabled=True, remaining_seconds=lambda: max(0, 10-now[0]), clock=lambda: now[0])
    memory = _memory(ProjectSpec.from_mapping(_project('source', 'gm-ub-movement')))
    result = verifier.assess_memory(memory, b'{"initial_memory_base64":"e30=","project_memory_base64":[]}', 'parent')
    assert result[0]['outcome'] == 'inconclusive'
    assert 'interpret' not in stages
    assert len(requests) == (1 if exhaust_stage == 'control' else 0)


def test_expired_parent_makes_no_model_or_actor_calls(tmp_path):
    def unexpected(*args):
        pytest.fail('expired parent must not dispatch')
    verifier = UsefulnessVerifier(tmp_path, None, unexpected, unexpected,
        enabled=True, remaining_seconds=lambda: 0)
    assert verifier.assess_lesson({'claim': 'new'}, b'{}', 'source')['outcome'] == 'inconclusive'
