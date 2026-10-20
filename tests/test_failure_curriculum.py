import base64
import json
from dataclasses import replace

import pytest

from benchmarks.ascend.failure_curriculum import FailureCurriculum, select_feedback, retry_spec, link_retry
from benchmarks.ascend.phase2_feedback import FeedbackStore
from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent, CurriculumContext
from optimizer.jobs import canonical, sha
from test_phase2_feedback import campaign, write
from test_a5_phase1_adaptive import _project, _wave, _result


def setup(tmp_path):
    campaign(tmp_path / 'failed')
    campaign(tmp_path / 'success', result={'verified': True, 'speedup': 2})
    store = FeedbackStore(tmp_path / 'feedback')
    store.import_campaign(tmp_path / 'failed')
    store.import_campaign(tmp_path / 'success')
    selection = select_feedback(store.root, 'a3')
    root = tmp_path / 'phase1'
    root.mkdir()
    return store, selection, FailureCurriculum(root, store.root, selection)


def investigation(selection, **changes):
    return dict(feedback_id=selection['selected'][0]['feedback_id'], status='investigating',
        classification='learnable', reason='isolate data movement', project_id='probe',
        control_project_id='control', hypothesis='tail changes behavior',
        observation='host oracle compares aligned and tail', **changes)


def test_selection_controls_and_compatibility(tmp_path):
    store, selection, feedback = setup(tmp_path)
    assert len(selection['selected']) == len(selection['controls']) == 1
    assert selection['controls'][0]['outcome']['verified']
    with pytest.raises(ValueError, match='incompatible'):
        select_feedback(store.root, 'a5')
    with pytest.raises(ValueError, match='unknown'):
        select_feedback(store.root, 'a3', ['missing'])
    context = feedback.context({})
    task = selection['selected'][0]['outcome']['task']
    text = feedback.read(sha(canonical(task)), 0, 8192)['text']
    assert json.loads(text) == task
    assert context[0]['task_sha256'] == sha(canonical(task))
    with pytest.raises(ValueError, match='not in selected'):
        feedback.read('a'*64)


def test_wave_lineage_and_host_evidenced_retry_recommendation(tmp_path):
    store, selection, feedback = setup(tmp_path)
    calls = []
    def complete(messages):
        calls.append(messages)
        if len(calls) == 1:
            prompt = messages[-1]['content']
            assert 'budget' in prompt and 'NOT verified' in prompt
            payload = json.loads(_wave(prompt, [_project('probe', 'padded-tail'),
                _project('control', 'gm-ub-movement')]).split('\n', 1)[1])
            payload['failure_investigations'] = [investigation(selection)]
            return 'DECISION: WAVE\n' + json.dumps(payload)
        row = investigation(selection)
        row.update(status='ready-to-retry', reason='host controls isolated the tail behavior')
        return 'DECISION: SATURATED\n' + json.dumps(dict(reason='focused investigation complete',
            failure_investigations=[row]))
    curriculum = CurriculumAgent(complete)
    curriculum.failure_feedback = feedback
    runner = AdaptivePhase1Coordinator(feedback.root, 'Isolate a tail.', b'{}', curriculum,
        lambda request: _result(request.spec, True, 'host passed'), max_parallel=2, project_target=3)
    runner.failure_feedback = feedback
    result = runner.resume()
    assert result['status'] == 'SATURATED'
    original = next(r for r in store.records() if r['feedback_id'] in feedback.ids)
    assert original['investigation']['status'] == 'ready-to-retry'
    assert original['outcome']['verified'] is False
    assert original['investigation']['evidence']
    assert original['investigation']['projects'][0]['control_project_id'] == 'control'
    assert runner.resume() == result


def test_successful_new_probes_do_not_automatically_resolve_or_ready(tmp_path):
    store, selection, feedback = setup(tmp_path)
    def complete(messages):
        value = json.loads(_wave(messages[-1]['content'], [_project('probe', 'padded-tail'),
            _project('control', 'gm-ub-movement')]).split('\n', 1)[1])
        value['failure_investigations'] = [investigation(selection)]
        return 'DECISION: WAVE\n' + json.dumps(value)
    curriculum = CurriculumAgent(complete)
    curriculum.failure_feedback = feedback
    runner = AdaptivePhase1Coordinator(feedback.root, 'Isolate a tail.', b'{}', curriculum,
        lambda request: _result(request.spec, True, 'passed'), max_parallel=2, project_target=2)
    runner.failure_feedback = feedback
    runner.resume()
    assert next(r for r in store.records() if r['feedback_id'] in feedback.ids)['investigation']['status'] == 'investigating'


def test_budget_only_failure_is_not_a_host_probe(tmp_path):
    _, selection, feedback = setup(tmp_path)
    row = investigation(selection)
    state = dict(prior_outcomes=[{'project_id': 'control'}], memory_bytes_base64=[base64.b64encode(
        canonical(dict(source_revision='a'*64, host_facts=[{'evidence_sha256':'b'*64}]))).decode()],
        failed_outcomes=[dict(project_id='probe', evidence=dict(sha256='c'*64, source_sha256=None,
            classification='host-unverified', action_counts={}))])
    assert not feedback._supported(row, state)
    state['failed_outcomes'][0]['evidence'].update(source_sha256='d'*64, classification='host-mismatch')
    assert feedback._supported(row, state)


def test_no_control_and_unexecuted_retry_rejected(tmp_path):
    _, selection, feedback = setup(tmp_path)
    feedback.context({})
    context = CurriculumContext('learn', 1, 'a'*64, (), ())
    row = investigation(selection)
    row.update(status='ready-to-retry')
    with pytest.raises(ValueError, match='working control'):
        feedback.admit([row], None, context)
    deferred = dict(feedback_id=row['feedback_id'], status='deferred', classification='provider',
                    reason='host retained provider transport failure; no kernel was executed')
    assert feedback.admit([deferred], None, context)
    feedback.record([deferred])
    assert feedback.store.records()[0]['investigation']['status'] in {'resolved', 'deferred'}


def test_feedback_artifact_reads_preserve_selection_turn_accounting(tmp_path):
    _, selection, feedback = setup(tmp_path)
    task = selection['selected'][0]['outcome']['task']
    calls = []
    def complete(messages):
        calls.append(messages)
        if len(calls) == 1:
            return json.dumps({'read_artifact': {'sha256': sha(canonical(task)), 'offset': 0, 'limit': 8192}})
        assert 'reference' in messages[-1]['content']
        return 'DECISION: SATURATED\n' + json.dumps(dict(reason='provider evidence deferred',
            failure_investigations=[dict(feedback_id=selection['selected'][0]['feedback_id'],
                status='deferred', classification='provider', reason='retained provider error')]))
    agent = CurriculumAgent(complete)
    agent.failure_feedback = feedback
    from benchmarks.a5kernels.phase1_curriculum_selector import PriorOutcome, ProjectInformation
    prior = PriorOutcome('working', ProjectInformation.from_project(_project('working', 'gm-ub-movement')), True, 'passed')
    context = CurriculumContext('learn', 1, 'a'*64, (), (prior,), phase2_feedback=feedback.context({}))
    _, session = agent.select(context)
    assert session.turns == 2


@pytest.mark.parametrize('speedup,resolved', [(None, False), (.5, False), (2, True)])
def test_slow_original_requires_successful_retry_not_only_correctness(tmp_path, speedup, resolved):
    campaign(tmp_path / 'original', result=dict(verified=True, speedup=.4))
    store = FeedbackStore(tmp_path / 'feedback')
    store.import_campaign(tmp_path / 'original')
    original = store.records()[0]
    campaign(tmp_path / 'retry', task=original['outcome']['task'], result=dict(verified=True, speedup=speedup))
    link_retry(dict(root=str(tmp_path / 'retry'), feedback_root=str(store.root),
                    retry_feedback_id=original['feedback_id']))
    result = next(r for r in store.records() if r['feedback_id'] == original['feedback_id'])
    assert (result['investigation']['status'] == 'resolved') is resolved
    assert len(result['investigation']['retries']) == 1


def test_fixed_task_retry_skips_goal_selection_and_qualifies_original(tmp_path):
    from benchmarks.ascend.phase2_optimizer_campaign import OptimizerPhase2Coordinator
    from test_ascend_phase2_optimizer_tasks import seed, proposal
    task, prepared = proposal(), []
    class Executor:
        def __init__(self, directory, value, memory):
            assert value == task
        def prepare(self):
            prepared.append(task)
        def attempt(self, key, **kwargs):
            return dict(verified=True, summary='host PASS')
    runner = OptimizerPhase2Coordinator(tmp_path, 'retry', seed(),
        lambda _: '{"action":"ready","reason":"verified"}', Executor,
        selector=lambda **kw: pytest.fail('must not select a new goal'), fixed_task=task)
    assert runner.resume()['status'] == 'READY'
    assert prepared


def test_retry_freezes_original_expanded_cases_and_rejects_changed_suite(tmp_path):
    from test_phase2_optimizer_execution import executor
    execution, agent, backend = executor(tmp_path, [])
    frozen = dict(execution.task['submission'])
    expected = dict(frozen_submission=frozen, suite_sha256='suite', policy={'name':'frozen'})
    original = backend.prepare
    backend.prepare = lambda spec, directory: {**original(spec, directory), 'manifest': {'policy': expected['policy']}}
    execution.expected_baseline = expected
    execution.prepare()
    assert not agent.contexts  # No fresh Actor case proposal.
    execution.expected_baseline = dict(expected, suite_sha256='other')
    with pytest.raises(ValueError, match='suite'):
        execution.prepare()
