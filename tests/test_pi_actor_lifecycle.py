"""Structured Pi selection shares the host's admission and recovery lifecycle."""
import json

import pytest

from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
from benchmarks.a5kernels.phase1_provider import DirectActorDriver, DirectCompletion, DirectCompletionProvider, direct_profile
from benchmarks.a5kernels.trial import ActionOutcome, parse_trial_action
from benchmarks.ascend.actor_control import ActorControl
import coding.pi.actor as pi_module


def pi_driver(monkeypatch, select, **options):
    monkeypatch.setattr(pi_module, 'runtime_preflight', lambda: None)
    class Session:
        completion_tokens = 0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def next_action(self, context, allowed, **kwargs): return select(context, allowed)
    monkeypatch.setattr(pi_module.PiActorDriver, '_new_session', lambda *args: Session())
    return pi_module.PiActorDriver(DirectCompletionProvider({'OPENAI_API_KEY': 'fake', 'DEEPSEEK_API_KEY': 'fake'}), **options)


def test_pi_dispatches_structured_actions_and_retains_invalid_feedback(monkeypatch, tmp_path):
    requests, executed = [], []
    def select(context, allowed):
        requests.append(context)
        if len(requests) == 1:
            return 'read-artifact', {'sha256': 'invalid digest'}
        if len(requests) == 2:
            assert context['latest_host_observation']['host']['event'] == 'invalid-action'
            return 'query-knowledge', {'query': '  tail copies  '}
        assert context['latest_host_observation'] == {'host': {'event': 'query-complete'}}
        # The newest observation has a single owner in the model-facing view.
        assert all(row['content'] != json.dumps(context['latest_host_observation'])
                   for row in context['recent_exchanges'])
        return 'submit', {}
    actor = pi_driver(monkeypatch, select)
    def execute(action):
        executed.append(action)
        return ActionOutcome(json.dumps({'host': {'event': 'query-complete'}}), action.kind == 'submit')
    result = actor('task', profile=direct_profile(MODEL_IDENTITIES[0]), workspace=tmp_path,
        context=None, turn_parser=lambda _: pytest.fail('Pi must not use the legacy text parser'),
        action_executor=execute)
    assert result.status == 'done'
    assert [(a.kind, dict(a.payload)) for a in executed] == [
        ('query-knowledge', {'query': 'tail copies', 'limit': 5}), ('submit', {})]
    retained = ActorControl(tmp_path).status()['checkpoint']['working']['recent']
    assert len([row for row in retained if row['role'] == 'user']) == 3


def test_pi_resume_uses_retained_conversation_and_requires_fresh_host_evidence(monkeypatch, tmp_path):
    control, requests = ActorControl(tmp_path), []
    class Executor:
        source = 'retained source'
        def restore_source(self, source):
            self.source = source
        def __call__(self, action):
            if action.kind == 'compile':
                control.stop()
                return ActionOutcome('{"host":{"event":"compiled","passed":true}}')
            return ActionOutcome('{}', True)
    executor = Executor()
    def select(context, allowed):
        requests.append(context)
        return ('compile' if len(requests) == 1 else 'submit'), {}
    actor = pi_driver(monkeypatch, select)
    options = dict(profile=direct_profile(MODEL_IDENTITIES[0]), workspace=tmp_path,
        context=None, turn_parser=parse_trial_action, action_executor=executor)
    assert actor('task', **options).status == 'user-stopped'
    control.stop(False)
    assert actor('task', **options).status == 'done'
    assert requests[1]['source'] == 'retained source'
    assert requests[1]['latest_host_observation'] == (
        'The actor resumed from retained source and conversation. '
        'Host verification and measurements must be repeated before submission.')
    assert any(json.loads(row['content']).get('host', {}).get('event') == 'compiled'
               for row in requests[1]['recent_exchanges'] if row['role'] == 'user')


@pytest.mark.parametrize('backend', ['direct', 'pi'])
def test_late_selection_never_executes_host_action(monkeypatch, tmp_path, backend):
    now = [0]
    def select(*args):
        now[0] = 11
        return 'submit', {}
    if backend == 'pi':
        actor = pi_driver(monkeypatch, select, clock=lambda: now[0], max_seconds=10)
    else:
        def transport(**options):
            select()
            return DirectCompletion('{"submit":{}}', 2)
        actor = DirectActorDriver(DirectCompletionProvider({'OPENAI_API_KEY': 'fake', 'DEEPSEEK_API_KEY': 'fake'},
            transport=transport), clock=lambda: now[0], max_seconds=10)
    result = actor('task', profile=direct_profile(MODEL_IDENTITIES[0]), workspace=tmp_path,
        context=None, turn_parser=parse_trial_action,
        action_executor=lambda _: pytest.fail('expired selection executed'))
    assert result.status == 'time-budget-exhausted'


def test_direct_terminal_distillation_separates_host_result_from_user_feedback(tmp_path):
    requests = []
    feedback = 'Inspect the padded tail before completing the task.'
    terminal = json.dumps({'host': {'event': 'submitted', 'passed': True,
                                   'source_sha256': 'a' * 64}})
    ActorControl(tmp_path).send('tail-feedback', feedback)
    def transport(**options):
        requests.append(options['request'])
        return DirectCompletion('{"submit":{}}' if len(requests) == 1 else
            '{"distillation":{"no_new_knowledge":true,"lessons":[]}}', 1)
    actor = DirectActorDriver(DirectCompletionProvider(
        {'OPENAI_API_KEY': 'fake', 'DEEPSEEK_API_KEY': 'fake'}, transport=transport))
    outcome = actor('Implement and verify the kernel.', profile=direct_profile(MODEL_IDENTITIES[0]),
        workspace=tmp_path, context=None, turn_parser=parse_trial_action,
        action_executor=lambda action: ActionOutcome(terminal, True, 'host-verified'))
    assert outcome.continuation.messages[-1] == {'role': 'user', 'content': terminal}
    distilled = actor.distill(outcome)
    assert distilled.distillation is not None
    messages = requests[1]['messages']
    distillation_context = json.loads(messages[-1]['content'].split('\n')[-1])
    assert distillation_context['terminal_host_observation'] == terminal
    assert {'role': 'user', 'content': feedback} in messages
    assert distillation_context['terminal_host_observation'] != feedback
