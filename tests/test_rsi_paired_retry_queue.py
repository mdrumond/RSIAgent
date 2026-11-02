"""The candidate in an existing pair is the retry; no standalone third execution."""
import copy
from dataclasses import replace
import json

import pytest

from core.artifacts import ContentAddressedArtifacts
from rsi.contracts import CycleSpec, Mode, ProjectSpec
from rsi.cycles import CycleScheduler, HostTaskResult, SnapshotUpdate
from rsi.recovery import CycleController
from test_rsi_cycles import isolation
from test_rsi_recovery import LearningStub


def scenario(tmp_path, outcomes, *, limit=5, mode=Mode.PHASE2, recover=True):
    trace, results = [], []
    store = ContentAddressedArtifacts(tmp_path/'artifacts')
    project = ProjectSpec('root', 'Keep the full original oracle', ('python', 'verify.py'))
    def original(spec, memory, skills, context):
        trace.append(('original', context['wave']))
        return HostTaskResult('failed', store.put_json({'original_failure': True}), 1.)
    def memory(previous, rows, context):
        trace.append(('memory', context['wave'], rows[0]['operation_id']))
        return SnapshotUpdate(previous+'+', .1)
    def writer(skills, memory, row, context):
        trace.append(('writer', context['wave'], memory, row['operation_id']))
        return SnapshotUpdate('candidate-'+str(len(trace)), .1)
    scheduler = CycleScheduler(tmp_path/'cycle.json',
        CycleSpec('paired', mode, (project,), 'memory', 'incumbent'), original, memory,
        skill_update=writer, independent_projects=True, recover=recover,
        max_failed_recovery_attempts=limit, tuning_every=None,
        isolate=(lambda m, s, c: isolation(tmp_path/'fork', m, s, c['namespace'])) if mode == Mode.INFERENCE else None)
    def pair(event, operation, remaining):
        trace.append(('pair', scheduler.state['wave'], event['memory'], event.get('paired_checkpoint')))
        selected = outcomes[min(len(results), len(outcomes)-1)]
        arms = {}
        for name, outcome in zip(('incumbent', 'candidate'), selected):
            arms[name] = dict(outcome=outcome, workflow=outcome=='accepted', correctness=outcome=='accepted',
                task_failure=outcome=='failed', actor_seconds=1., classification=outcome)
            if outcome == 'harness':
                arms[name].update(outcome='failed', classification='harness_failure',
                    terminal_failure_ref='host-terminal-proof', audit_ref='host-audit', task_failure=False)
        result = dict(arms=arms, decision='inconclusive', actor_seconds=2.,
                      frozen='frozen-'+operation, evidence_ref=store.put_json(arms))
        results.append((event, operation, result))
        return type('CompletedPair', (), {'result': lambda self: result})()
    def feedback(event, operation, result):
        accepted = next((name for name, row in result['arms'].items() if row['outcome']=='accepted'), 'candidate')
        return dict(project=project.to_dict(), original_id='root', operation_id=operation+'-'+accepted,
            outcome=result['arms'][accepted]['outcome'], actor_seconds=1., artifact_ref=result['evidence_ref'],
            paired_checkpoint=dict(operation_id=operation, frozen=result['frozen'], arm=accepted),
            retry_checkpoint=dict(operation_id=operation, frozen=result['frozen'], arm='candidate'))
    controller = CycleController(tmp_path/'controller.json', scheduler, LearningStub('incumbent'),
        artifacts=store, pair_factory=pair)
    controller.pair_feedback = feedback
    return scheduler, controller, trace, results


def drain(scheduler, controller):
    for _ in range(80):
        pending = [e for e in scheduler.extension_events() if e['kind']=='paired_recovery_due']
        if scheduler.state['phase']=='complete' and not pending:
            return
        controller.step()
    pytest.fail('paired queue did not reach its finite terminal outcome')


@pytest.mark.parametrize('limit', [2, 5])
def test_only_failed_candidate_counts_and_queue_resumes_without_recounting(tmp_path, limit):
    scheduler, controller, trace, results = scenario(tmp_path, [('failed', 'failed')], limit=limit)
    try:
        drain(scheduler, controller)
        item = scheduler.state['paired_retry_queue']['root']
        assert item['status']=='unrecoverable' and item['failed_attempts']==limit
        assert len(results)==limit
        assert [r[1] for r in trace if r[0]=='pair']==list(range(1, limit+1))
        assert [r[0] for r in trace].count('original')==1
        assert scheduler.state['originals'][0]['outcome']=='failed'
        assert len([r for r in trace if r[0]=='memory'])==limit+1
        assert item['candidate_actor_seconds']==limit  # Incumbent control cost is separate.
        event, operation, result = results[-1]
        before = copy.deepcopy(scheduler.state)
        receipt = scheduler.state['paired_retry_receipts'][event['id']]
        scheduler.complete_paired_retry(event, result, receipt, item['result'])
        scheduler.resume()
        controller.step()
        assert scheduler.state==before
        assert all(r[3] is not None for r in trace if r[0]=='pair' and r[1]>1)
    finally:
        scheduler.close()


@pytest.mark.parametrize('outcomes', [('accepted', 'failed'), ('failed', 'accepted'), ('accepted', 'accepted')])
def test_any_accepted_arm_resolves_original_even_when_comparison_is_inconclusive(tmp_path, outcomes):
    scheduler, controller, trace, results = scenario(tmp_path, [outcomes], limit=1)
    try:
        drain(scheduler, controller)
        item = scheduler.state['paired_retry_queue']['root']
        assert item['status']=='accepted'
        assert len(results)==1
        assert scheduler.state['originals'][0]['outcome']=='failed'
        assert trace[-1][0]=='memory'  # The accepted arm reaches Memory too.
        assert item['failed_attempts']==int(outcomes[1]=='failed')
    finally:
        scheduler.close()


def test_pair_failure_updates_memory_before_the_next_writer_and_pair(tmp_path):
    scheduler, controller, trace, results = scenario(tmp_path,
        [('failed', 'failed'), ('failed', 'accepted')])
    try:
        drain(scheduler, controller)
        assert [r[0] for r in trace]==[
            'original', 'memory', 'writer', 'pair', 'memory', 'writer', 'pair', 'memory']
        assert trace[4][1]==trace[5][1]==trace[6][1]==2
        assert trace[3][2]!=trace[6][2]
        assert scheduler.state['paired_retry_queue']['root']['failed_attempts']==1
    finally:
        scheduler.close()


def test_host_authenticated_harness_fault_abandons_without_failure_count_or_replay(tmp_path):
    scheduler, controller, trace, results = scenario(tmp_path, [('failed', 'harness')])
    try:
        drain(scheduler, controller)
        item = scheduler.state['paired_retry_queue']['root']
        assert item['status']=='abandoned' and item['failed_attempts']==0
        assert len(results)==1 and trace[-1][0]=='memory'
        assert not scheduler.extension_events() or not any(
            e['kind']=='paired_recovery_due' for e in scheduler.extension_events())
    finally:
        scheduler.close()


def test_disabled_inference_recovery_never_creates_a_pair(tmp_path):
    scheduler, controller, trace, results = scenario(tmp_path, [('failed', 'failed')],
        mode=Mode.INFERENCE, recover=False)
    try:
        drain(scheduler, controller)
        assert not results and not scheduler.state.get('paired_retry_queue')
        assert [r[0] for r in trace]==['original', 'memory', 'writer']
    finally:
        scheduler.close()


def test_pending_pair_never_increments_or_reexecutes_unknown_candidate(tmp_path):
    from rsi.runtime import WorkflowError
    scheduler, controller, trace, results = scenario(tmp_path, [('failed', 'failed')])
    dispatched = []
    class UnknownPair:
        def result(self):
            return None
        def step(self):
            dispatched.append('one-unknown-dispatch')
            raise ConnectionError('retained candidate return is unknown')
    controller.pair_factory = lambda *args: UnknownPair()
    try:
        controller.step(); controller.step(); controller.step()
        with pytest.raises(ConnectionError):
            controller.step()
        identity = controller.state['pending']['operation_id']
        with pytest.raises(WorkflowError):
            controller.reconcile(dispatch=False)
        assert dispatched==['one-unknown-dispatch']
        assert controller.state['pending']['operation_id']==identity
        assert not scheduler.state.get('paired_retry_queue')
    finally:
        scheduler.close()


def test_reopen_preserves_frozen_counter_without_another_pair(tmp_path):
    scheduler, controller, _, _ = scenario(tmp_path, [('failed', 'failed')], limit=2)
    try:
        drain(scheduler, controller)
        before = copy.deepcopy(scheduler.state['paired_retry_queue'])
    finally:
        scheduler.close()
    reopened, controller, trace, results = scenario(tmp_path, [('failed', 'failed')], limit=2)
    try:
        reopened.resume()
        drain(reopened, controller)
        assert reopened.state['paired_retry_queue']==before and not results and not trace
    finally:
        reopened.close()


from rsi.configuration import set_configuration, show_configuration
from rsi.services import TestBindings as Bindings
from test_rsi_services import instance, Decisions, Roles
from test_rsi_service_wave import wave_query


class AnswerDecision(Decisions):
    def __init__(self, repair, broken=False):
        self.repair, self.broken, self.edited = repair, broken, False
    def next_action(self, context, allowed, **kwargs):
        if any(name not in {row['skill'] for row in context['read_skills']} for name in context['skills']):
            return super().next_action(context, allowed, **kwargs)
        if (self.repair or self.broken and not self.edited) and context['stage']=='debug' and any(
                row['classification']=='numeric_failure' for row in context['task']['feedback']):
            source = context['source']
            self.edited = True
            return 'perform', dict(proposal='Correct the observed frozen-oracle mismatch', edit=dict(
                path=source['path'], sha256=source['sha256'],
                source='answer = 42\n' if self.repair else 'answer = 40\n'))
        return super().next_action(context, allowed, **kwargs)


def test_real_cpu_pair_findings_feed_memory_and_next_pair_uses_latest_candidate(instance):
    root = instance.root
    (root/'source/solution.py').write_text('answer = 41\n')
    settings = show_configuration(root)['payload']
    settings['phase2_projects'][0]['budgets'].update(actor_seconds=2.5,
        design_iterations=None, implementation_iterations=None, execution_tries=None)
    settings['recovery'] = dict(enabled=True, max_failed_attempts=2, budgets=dict(
        actor_seconds=2.5, wall_seconds=60, design_iterations=None,
        implementation_iterations=None, execution_tries=None))
    settings['tuning'] = dict(at_wave_end=False)
    set_configuration(root, settings)
    created = []
    def sessions():
        created.append(len(created))
        # Original + first pair fail; only the second pair's candidate repairs.
        return AnswerDecision(repair=len(created)==5, broken=len(created)==3)
    instance.test = Bindings(wave_query, sessions, Roles, 'explicit-scripted-CPU-fixture')
    cycle = instance.create_cycle('real-paired-queue', 'phase2')
    try:
        for _ in range(30):
            if cycle.scheduler.state['phase']=='complete' and not any(
                    e['kind']=='paired_recovery_due' for e in cycle.scheduler.extension_events()):
                break
            cycle.controller.step()
        state = cycle.scheduler.state
        assert state['phase']=='complete'
        item = state['paired_retry_queue']['answer']
        assert item['status']=='accepted' and item['failed_attempts']==1
        assert len(created)==5  # One original and exactly two existing two-arm pairs.
        assert state['originals'][0]['outcome']!='accepted'
        assert (root/'source/solution.py').read_text()=='answer = 41\n'
        assert len(list((cycle.root/'pairs').glob('*.json')))==2
        assert len(cycle.memory.records())>=3
        pairs = [json.loads(p.read_text()) for p in (cycle.root/'pairs').glob('*.json')]
        assert all(all(row['status']=='complete' for row in pair['arms'].values()) for pair in pairs)
        latest_pair, _, latest = cycle.paired_checkpoint(item['checkpoint'])
        assert cycle.store.read_bytes(latest_pair.frozen['files']['solution.py'])==b'answer = 40\n'
        assert (latest.backend.project/'solution.py').read_bytes()==b'answer = 42\n'
        packet = cycle.store.read_json(item['result']['artifact_ref'])
        assert packet['host_outcome']=='accepted'
        assert cycle.load_project(item['result'])['outcome']=='accepted'
    finally:
        cycle.close()



def test_reconciled_completed_arm_without_outer_return_reaches_canonical_memory(instance):
    cycle = instance.create_cycle('lost-return-feedback', 'phase2')
    try:
        assert cycle.controller.step().outcome == 'accepted'
        original = cycle.scheduler.state['originals'][0]
        event = dict(original=original, memory=cycle.scheduler.state['memory'],
            incumbent=cycle.scheduler.state['skills'], candidate=cycle.scheduler.state['skills'])
        pair = cycle.pair(event, 'lost-return-pair', cycle.scheduler._allocation_remaining())
        factory, created = pair.factory, []
        def disconnected(request, *, reopen):
            runner = factory(request, reopen=reopen)
            if not reopen:
                created.append(request.arm)
                if request.arm == 'incumbent':
                    execute = runner.run
                    def lost_return():
                        execute()
                        raise ConnectionError('canonical arm completed before outer return')
                    runner.run = lost_return
            return runner
        pair.factory = disconnected
        with pytest.raises(ConnectionError):
            pair.step()
        observed = pair.reconcile('incumbent')  # Observe only, never execute again.
        assert observed['outcome'] == 'accepted'
        assert 'run_result' not in pair.state['arms']['incumbent']
        pair.step()
        feedback = cycle.paired_feedback(event, 'lost-return-pair', pair.result())
        assert feedback['paired_checkpoint']['arm'] == 'incumbent'
        assert created == ['incumbent', 'candidate']
        assert cycle.load_project(feedback)['outcome'] == 'accepted'
        update = cycle.learning.memory_update(cycle.scheduler.state['memory'], [feedback],
            cycle.scheduler._context(dict(id='lost-return-memory', kind='memory')))
        assert update.reference and cycle.memory.records()
        assert created == ['incumbent', 'candidate']
    finally:
        cycle.close()
