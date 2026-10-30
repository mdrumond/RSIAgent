import json
from types import SimpleNamespace

import pytest

from benchmarks.ascend.actor_control import ActorControl, ActorConversation, ActorStopped, grant_elapsed
from benchmarks.ascend.agent_context import WorkingContext
from benchmarks.ascend.time_budget import AttemptTimeBudget
from canonical_kernel_fixtures import canonical_kernel


def test_feedback_order_idempotency_and_atomic_receipts(tmp_path):
    inbox = ActorControl(tmp_path)
    assert inbox.send('one', 'change direction')['id'] == 1
    inbox.send('two', 'retain the old evidence')
    assert inbox.send('one', 'change direction')['id'] == 1
    with pytest.raises(ValueError):
        inbox.send('one', 'a different direction')
    feedback = []
    checkpoint = {'feedback': feedback}
    inbox.boundary(checkpoint, feedback)
    assert [row['feedback_id'] for row in feedback] == [1, 2]
    assert all(row['consumed'] is not None for row in inbox.status()['messages'])
    assert inbox.status()['checkpoint']['feedback'] == feedback
    inbox.boundary(checkpoint, feedback)
    assert len(feedback) == 2


def test_stop_preserves_checkpoint_without_consuming_feedback(tmp_path):
    inbox = ActorControl(tmp_path)
    inbox.send('one', 'redirect')
    inbox.stop()
    with pytest.raises(ActorStopped):
        inbox.boundary({'source': 'old source'}, [])
    assert inbox.status()['checkpoint']['source'] == 'old source'
    assert inbox.status()['messages'][0]['consumed'] is None
    inbox.stop(False)
    feedback = []
    inbox.boundary({'feedback': feedback}, feedback)
    assert feedback[0]['content'] == 'redirect'


def test_checkpoint_restores_source_work_log_and_feedback_without_host_authority(tmp_path):
    fixed = [{'role': 'system', 'content': 'contract'}, {'role': 'user', 'content': 'task'}]
    working = WorkingContext(fixed)
    actions = SimpleNamespace(source='saved source')
    conversation = ActorConversation(tmp_path, 'task', working, actions)
    ActorControl(tmp_path).send('redirect', 'inspect masks')
    conversation.boundary()
    working.work_log = 'Earlier compile failed; inspect retained log.'
    working.host_state = {'host-verified': {'passed': True}}
    conversation.save(4)
    restored = WorkingContext(fixed)
    sources = []
    resumed = ActorConversation(tmp_path, 'task', restored,
                                SimpleNamespace(restore_source=sources.append, source=None))
    assert resumed.turn == 4 and sources == ['saved source']
    assert restored.work_log == working.work_log
    assert restored.host_state == {}
    assert restored.pinned == [{'role': 'user', 'content': 'inspect masks'}]
    with pytest.raises(ValueError):
        ActorConversation(tmp_path, 'changed task', WorkingContext(fixed))


def test_explicit_grant_preserves_elapsed_across_original_identity_resume(tmp_path):
    path, now = tmp_path / 'elapsed.json', [0.0]
    budget = AttemptTimeBudget(path, 10, clock=lambda: now[0])
    with pytest.raises(ValueError):
        grant_elapsed(path, 'grant', 5)
    now[0] = 7
    budget.close()
    assert grant_elapsed(path, 'grant', 5) == 15
    assert grant_elapsed(path, 'grant', 5) == 15
    with pytest.raises(ValueError):
        grant_elapsed(path, 'grant', 7)
    resumed = AttemptTimeBudget(path, 10, clock=lambda: now[0])
    assert resumed.deadline - now[0] == 8
    resumed.close()
    assert json.loads(path.read_text())['elapsed_seconds'] == 7
    assert json.loads(path.read_text())['grants'] == {'grant': 5}


def test_direct_actor_receives_feedback_and_resumes_without_repeating_action(canonical_kernel):
    case = canonical_kernel
    control = ActorControl(case.workspace)
    control.send('first', 'inspect masks')
    def stop_after_compile(kind):
        if kind == 'compile':
            control.stop()
            case.after_host = None
    case.after_host = stop_after_compile
    assert case.run().status == 'user-stopped'
    def compile_dispatches():
        dispatches = [(row['reference'], case.runner.store.read_json(row['reference']))
                      for row in case.runner.events if row['type'] == 'domain-dispatch']
        return [(ref, row) for ref, row in dispatches if row['name'] == 'compile']
    def identity(row):
        return row['operation_id'], row['stage'], row['source_hash'], row['name']
    completed = [case.runner.store.read_json(row['reference'])
                 for row in case.runner.events if row['type'] == 'domain-result']
    settled_refs = {row['dispatch_ref'] for row in completed
                    if row['observed_terminal'] and row['status'] == 'complete'}
    settled = {identity(row) for ref, row in compile_dispatches() if ref in settled_refs}
    assert settled
    control.stop(False)
    control.send('second', 'submit after review')
    assert case.run(case.make_driver()).status == 'host-verified'
    # Repaired source or later stages may compile; the settled operation cannot replay.
    after = [identity(row) for _, row in compile_dispatches()]
    assert all(after.count(key) == 1 for key in settled)
    assert any(key not in settled for key in after)
    domain_views = [context['task']['domain'] for context in case.contexts]
    feedback = [row['content'] for view in domain_views for row in view['feedback']]
    assert 'inspect masks' in feedback and 'submit after review' in feedback
    assert all(row['consumed'] is not None for row in control.status()['messages'])
    assert case.actions.latest.evidence.verified


def test_optimizer_delivers_feedback_in_model_context(tmp_path):
    from optimizer.jobs import JobStore, canonical, sha
    from optimizer.loop import OptimizationLoop
    body = {'files': {}, 'memory_entries': []}
    snapshot = {**body, 'snapshot_id': sha(canonical(body))}
    store = JobStore(tmp_path / 'jobs')
    job = store.create('def forward(x): return x', 'forward',
        {'args': [{'shape': [5], 'dtype': 'float32'}]}, snapshot, max_seconds=100)
    control = ActorControl(store.root / job['job_id'])
    control.send('redirect', 'prefer vector work')
    contexts = []
    class Agent:
        identity = {}
        def complete(self, stage, context, record):
            contexts.append(context)
            return {'action': 'finish', 'summary': 'done'}
    store.transition(job['job_id'], 'running')
    loop = OptimizationLoop(store, job['job_id'], Agent(), SimpleNamespace())
    loop.context, loop.single_evaluation, loop.prepare_only = {}, False, False
    loop._model('optimize', {})
    assert contexts[0]['user_feedback'][0]['content'] == 'prefer vector work'
    assert control.status()['messages'][0]['consumed'] is not None


@pytest.mark.parametrize('engine', ['direct', 'pi'])
def test_feedback_arriving_during_model_request_supersedes_unexecuted_action(canonical_kernel, engine):
    case = canonical_kernel
    inbox = ActorControl(case.workspace)
    sent = []
    def feedback_during_selection(context, selected):
        if selected[0] == 'perform' and context['stage'] == 'test' and not sent:
            inbox.send('arrived-inflight', 'review direction before executing')
            sent.append(len(case.contexts))
    host_feedback = []
    def observe_dispatch(kind):
        consumed = inbox.status()['messages'][0]['consumed'] is not None
        observed = any(any(row['content'] == 'review direction before executing'
                           for row in context['task']['domain']['feedback'])
                       for context in case.contexts)
        host_feedback.append(consumed and observed)
    case.after_host = observe_dispatch
    case.after_select = feedback_during_selection
    assert case.run(case.make_driver(engine)).status == 'host-verified'
    assert sent
    # The selected operation must yield to feedback before executing a host command.
    received = [context for context in case.contexts[sent[0]:]
                if any(row['content'] == 'review direction before executing'
                       for row in context['task']['domain']['feedback'])]
    assert received
    assert host_feedback and all(host_feedback)
    assert inbox.status()['messages'][0]['consumed'] is not None
    assert case.actions.latest.evidence.verified
