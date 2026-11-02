"""Default Direct/Pi drivers share canonical host admission and recovery."""
import json
from pathlib import Path

import pytest

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
    context_boundary = len(case.contexts)
    control.stop(False)
    # Reconstruct the public driver; retained workflow and host checkpoint own recovery.
    result = case.run(case.make_driver('pi'))
    assert result.status == 'host-verified'
    assert case.calls.count('run') > retained_runs
    resumed_contexts = case.contexts[context_boundary:]
    assert resumed_contexts
    assert resumed_contexts[0]['source']['text'] == retained_source
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


def test_native_terminal_distillation_retains_host_result_and_current_feedback(canonical_kernel, monkeypatch, scripted_provider):
    from test_native_pi_programming_roles import redirect, registered_profile
    from test_pi_runtime import _native_wire_objects
    from test_native_pi_workflow import text_blocks
    case = canonical_kernel
    case.profile = registered_profile()
    url, calls, requests = scripted_provider
    redirect(monkeypatch, url)
    feedback = 'Inspect the padded tail before completing the task.'
    ActorControl(case.workspace).send('tail-feedback', feedback)
    outcome = case.run()
    assert outcome.status == 'host-verified'
    terminal_history = outcome.continuation.runner.workflow.state['history']
    assert terminal_history[-1]['passed'] is True
    # Also deliver an instruction arriving after the final programming stage.
    late = ActorControl(case.workspace).send('terminal-feedback', 'Keep the new lesson specific to this host receipt.')
    read = []
    def choose(wire):
        pointer = next(value for role,value in _native_wire_objects([wire]) if role=='user' and 'context_path' in value)
        if not read:
            read.append(pointer['context_path'])
            return 'read', dict(path=read[0])
        return 'submit_curriculum_proposal', dict(value=dict(distillation=dict(no_new_knowledge=True, lessons=[])))
    calls.responder = choose
    effects = list(case.calls)
    distilled = case.driver.distill(outcome)
    assert distilled.distillation is not None and distilled.distillation_failure is None
    runner = outcome.continuation.runner
    context = json.loads(json.loads(Path(read[0]).read_text())[-1]['content'])
    archive = context['historical_workflow']
    assert archive['scope'] == 'origin-task-archive'
    assert runner.store.read_json(archive['reference']) == terminal_history
    assert context['host_outcome'] == 'host-verified'
    assert [row['content'] for row in context['feedback']] == [feedback, late['text']]
    assert case.calls == effects
    tool_observations = [text for wire in requests for message in wire['messages']
                         if message['role']=='tool' for text in text_blocks(message['content'])]
    assert any(feedback in text and late['text'] in text for text in tool_observations)
