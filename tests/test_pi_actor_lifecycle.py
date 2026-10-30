"""Default Direct/Pi drivers share canonical host admission and recovery."""
import json

import pytest

from benchmarks.a5kernels.phase1_provider import DirectCompletion, DISTILLATION_PROMPT
from benchmarks.ascend.actor_control import ActorControl
from canonical_kernel_fixtures import canonical_kernel


def test_pi_dispatches_structured_actions_and_retains_invalid_feedback(canonical_kernel):
    case = canonical_kernel
    case.invalid = True
    result = case.run(case.make_driver('pi'))
    assert result.status == 'host-verified'
    assert any(context['observation'] and context['observation'].get('event') == 'invalid-action'
               for context in case.contexts)
    assert case.actions.latest.evidence.verified
    assert case.calls.count('run') >= 2  # A real mismatch is repaired numerically.
    assert case.actions.source == case.session.repaired
    assert any(event['type'] == 'actor-invalid-action' for event in case.runner.events)


def test_pi_resume_uses_retained_conversation_and_requires_fresh_host_evidence(canonical_kernel):
    case = canonical_kernel
    control = ActorControl(case.workspace)
    def stop_after_test(kind):
        if kind == 'run':
            control.stop()
            case.after_host = None
    case.after_host = stop_after_test
    actor = case.make_driver('pi')
    assert case.run(actor).status == 'user-stopped'
    retained_source = case.actions.source
    retained_runs = case.calls.count('run')
    control.stop(False)
    # Reconstruct the public driver; retained workflow and host checkpoint own recovery.
    result = case.run(case.make_driver('pi'))
    assert result.status == 'host-verified'
    assert case.calls.count('run') > retained_runs
    assert any(context['source']['text'] == retained_source for context in case.contexts)
    assert case.actions.latest.evidence.verified
    assert case.actions.source == case.session.repaired


@pytest.mark.parametrize('backend', ['direct', 'pi'])
def test_late_selection_never_executes_host_action(canonical_kernel, backend):
    case = canonical_kernel
    def expire(context, selected):
        if selected[0] == 'perform':
            case.now += 121
    case.after_select = expire
    result = case.run(case.make_driver(backend))
    assert result.status == 'budget_exhausted'
    assert case.calls == []
    assert not case.actions.latest
    assert not case.actions.admitted


def test_direct_terminal_distillation_separates_host_result_from_user_feedback(canonical_kernel):
    case = canonical_kernel
    feedback = 'Inspect the padded tail before completing the task.'
    ActorControl(case.workspace).send('tail-feedback', feedback)
    requests = []
    def transport(**options):
        requests.append(options['request'])
        return DirectCompletion('{"distillation":{"no_new_knowledge":true,"lessons":[]}}', 1)
    case.provider._transport = transport
    outcome = case.run()
    assert outcome.status == 'host-verified'
    terminal_history = outcome.continuation.runner.workflow.state['history']
    assert terminal_history[-1]['passed'] is True
    distilled = case.driver.distill(outcome)
    assert distilled.distillation is not None
    terminal_message = next(message for message in requests[-1]['messages']
                            if message['content'].startswith(DISTILLATION_PROMPT))
    context = json.loads(terminal_message['content'].split('\n')[-1])
    assert context['workflow_history'][-1] == terminal_history[-1]
    assert feedback in json.dumps(requests[-1]['messages'])
    assert context['workflow_history'] != feedback
