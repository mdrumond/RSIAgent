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


def test_factory_requests_follow_current_rebound_parent_allowance(tmp_path):
    from benchmarks.ascend.memory_usefulness import build_usefulness_verifier
    requests = []
    class Provider:
        def complete(self, profile, messages, *, max_seconds=None):
            requests.append(max_seconds)
            return SimpleNamespace(text='response')
    verifier = build_usefulness_verifier(root=tmp_path, provider=Provider(), profile=object(),
        curriculum=CurriculumAgent(lambda _: pytest.fail('use bounded provider binding')),
        project_actor=None, enabled=True, remaining_seconds=lambda: 2)
    verifier.verify('inspect', {})
    verifier.curriculum._complete([])
    verifier.remaining_seconds = lambda: 5
    verifier.verify('interpret', {})
    verifier.curriculum._complete([])
    assert requests == [2, 2, 5, 5]
    verifier.remaining_seconds = lambda: 0
    with pytest.raises(TimeoutError):
        verifier.verify('inspect', {})
    with pytest.raises(TimeoutError):
        verifier.curriculum._complete([])
    assert len(requests) == 4


@pytest.mark.parametrize('seconds', [float('nan'), float('inf'), float('-inf'), 0, -1, True, '600'])
def test_pair_allowance_requires_finite_positive_number(tmp_path, seconds):
    with pytest.raises(ValueError, match='finite and positive'):
        UsefulnessVerifier(tmp_path, None, None, None, pair_seconds=seconds)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('inspection_seconds', [3, 11])
def test_factory_caps_every_provider_call_by_local_pair_and_parent(tmp_path, inspection_seconds):
    from benchmarks.ascend.memory_usefulness import build_usefulness_verifier
    now, calls, arms = [0.0], [], []
    class Provider:
        def complete(self, profile, messages, *, max_seconds):
            calls.append(max_seconds)
            if 'memory usefulness verifier' in messages[0]['content']:
                stage = json.loads(messages[-1]['content'])['stage']
                now[0] += inspection_seconds if stage == 'inspect' else 0
                return SimpleNamespace(text=stage)
            now[0] += 2
            return SimpleNamespace(text=_wave(messages[-1]['content'],
                [_project('timed-copy', 'gm-ub-movement')]))
    def execute(request):
        arms.append(request)
        now[0] += 1
        return _result(request.spec, True, 'pass')
    verifier = build_usefulness_verifier(root=tmp_path, provider=Provider(), profile=object(),
        curriculum=CurriculumAgent(lambda _: pytest.fail('production binding required')),
        project_actor=execute, enabled=True, pair_seconds=10, remaining_seconds=lambda: 100)
    verifier.clock = lambda: now[0]
    result = verifier.assess_lesson({'claim': 'copy'},
        b'{"initial_memory_base64":"e30=","project_memory_base64":[]}', 'source')
    if inspection_seconds > 10:
        assert result['outcome'] == 'inconclusive'
        assert calls == [10] and arms == []
    else:
        assert result['outcome'] == 'both-succeeded'
        assert calls == [10, 7, 3]
        assert [request.remaining_seconds for request in arms] == [2.5, 2.5]
    assert verifier._experiment_remaining is None


def test_semantic_failure_arms_retain_linkable_evidence(tmp_path):
    import hashlib
    from benchmarks.a5kernels.evidence import canonical_bytes
    from benchmarks.a5kernels.phase1_wave_runtime import SemanticProjectFailure
    evidence = {'actor_status': 'completed-unverified', 'source_sha256': 'a' * 64}
    def execute(request):
        raise SemanticProjectFailure('Host verification failed', evidence)
    verifier = UsefulnessVerifier(tmp_path, CurriculumAgent(lambda messages:
        _wave(messages[-1]['content'], [_project('failed-copy', 'gm-ub-movement')])),
        execute, lambda stage, body: stage, enabled=True)
    assessment = verifier.assess_lesson({'claim': 'copy'},
        b'{"initial_memory_base64":"e30=","project_memory_base64":[]}', 'failed-source')
    assert assessment['outcome'] == 'neither-succeeded'
    expected = canonical_bytes({'verified': False, 'detail': 'Host verification failed', 'evidence': evidence})
    for arm in ('control', 'treatment'):
        path = tmp_path / assessment['experiment_id'] / arm / 'evidence.json'
        assert path.read_bytes() == expected
        assert assessment['arms'][arm]['evidence_sha256'] == hashlib.sha256(expected).hexdigest()


def test_shared_actor_contract_withholds_teaching_fields_without_changing_host_task(tmp_path):
    from benchmarks.a5kernels.phase1_project import materialize_inputs, evaluate_reference
    from benchmarks.ascend.memory_usefulness import actor_project
    authored = _project('reuse-private-lesson', 'gm-ub-movement')
    authored['hypothesis'] = 'Expected lesson: private optimization guidance'
    authored['graph']['inputs'][0]['name'] = 'privatelesson'
    authored['graph']['nodes'][0]['operands'][0] = 'privatelesson'
    spec = ProjectSpec.from_mapping(authored)
    hidden = actor_project(spec)
    original_inputs = materialize_inputs(spec)
    assert evaluate_reference(spec, original_inputs).storage == evaluate_reference(hidden, original_inputs).storage
    requests = []
    def execute(request):
        requests.append(request)
        return _result(request.spec, True, 'host pass')
    verifier = UsefulnessVerifier(tmp_path, CurriculumAgent(lambda messages: _wave(messages[-1]['content'], [authored])),
        execute, lambda stage, evidence: stage, enabled=True)
    lesson = {'claim': 'private optimization guidance'}
    result = verifier.assess_lesson(lesson,
        b'{"initial_memory_base64":"e30=","project_memory_base64":[]}', 'source')
    assert result['outcome'] == 'both-succeeded'
    assert requests[0].spec == requests[1].spec == hidden
    assert requests[0].inputs == requests[1].inputs and requests[0].oracle == requests[1].oracle
    assert requests[0].oracle.storage == evaluate_reference(spec, requests[0].inputs).storage
    control = json.loads(requests[0].pre_wave_memory)
    treatment = json.loads(requests[1].pre_wave_memory)
    assert 'usefulness_lesson' not in control
    assert treatment.pop('usefulness_lesson') == lesson
    assert treatment == control


@pytest.mark.parametrize('partial', [b'', b'{"kind": "arm-started",', b'null'])
def test_partial_retained_event_does_not_block_resume_or_repeat_execution(tmp_path, partial):
    def unexpected(*args):
        pytest.fail('incomplete retained experiment must not dispatch work')
    verifier = UsefulnessVerifier(tmp_path, None, unexpected, unexpected, enabled=True)
    lesson, prior = {'claim': 'new'}, b'{}'
    from benchmarks.ascend.memory_usefulness import digest
    import hashlib
    identifier = digest(dict(lesson_id=digest(lesson), source_id='retained',
        prior_memory_sha256=hashlib.sha256(prior).hexdigest(), pair_seconds=600))
    directory = tmp_path / identifier
    directory.mkdir()
    # Preserve a gap and malformed last event: next append must not overwrite either.
    (directory/'0000.json').write_text(json.dumps(dict(kind='started')))
    (directory/'0002.json').write_bytes(partial)
    result = verifier.assess_lesson(lesson, prior, 'retained')
    assert result['outcome'] == 'inconclusive'
    assert result['unreadable_events'] == ['0002.json']
    assert (directory/'0002.json').read_bytes() == partial
    assert (directory/'0003.json').is_file()
    assert verifier.assess_lesson(lesson, prior, 'retained') == result
    assert len(list(directory.glob('[0-9]*.json'))) == 3


@pytest.mark.parametrize('interpretation_failure', ['deadline', 'provider'])
def test_completed_host_pair_survives_optional_interpretation_failure(tmp_path, interpretation_failure):
    import hashlib
    now, requests, stages = [0.0], [], []
    def execute(request):
        requests.append(request)
        if len(requests) == 2 and interpretation_failure == 'deadline':
            now[0] = 10
        return _result(request.spec, len(requests) == 2, 'retained host verdict')
    def verify(stage, evidence):
        stages.append(stage)
        if stage == 'interpret':
            raise RuntimeError('provider unavailable')
        return stage
    verifier = UsefulnessVerifier(tmp_path, CurriculumAgent(lambda messages:
        _wave(messages[-1]['content'], [_project('host-pair', 'gm-ub-movement')])),
        execute, verify, enabled=True, clock=lambda: now[0], remaining_seconds=lambda: max(0, 10-now[0]))
    result = verifier.assess_lesson({'claim': 'new'},
        b'{"initial_memory_base64":"e30=","project_memory_base64":[]}', 'host-pair')
    assert result['outcome'] == 'demonstrated-benefit'
    assert result['arms']['control']['verified'] is False
    assert result['arms']['treatment']['verified'] is True
    assert result['interpretation_error']['error_type'] == ('TimeoutError' if interpretation_failure == 'deadline' else 'RuntimeError')
    assert stages == (['inspect'] if interpretation_failure == 'deadline' else ['inspect', 'interpret'])
    for arm in ('control', 'treatment'):
        data = (tmp_path/result['experiment_id']/arm/'evidence.json').read_bytes()
        assert hashlib.sha256(data).hexdigest() == result['arms'][arm]['evidence_sha256']
