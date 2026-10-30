"""CPU stub hosts; real canonical stages, skill receipts and KDB evidence."""
from hashlib import sha256

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from rsi.contracts import BudgetLimits, ProjectSpec, Stage
from rsi.domain_execution import (CanonicalAdoption, CanonicalConfigurationError,
                                  DomainObservation, DomainRunner, require_adoption)
from rsi.runtime import WorkflowError
from rsi.skills import builtin_bundle


class Clock:
    value = 1000.
    def __call__(self):
        return self.value


class Host:
    editable_files = ('kernel.py',)
    identity = dict(target='CPU TEST', provider='TEST', kdb='baseline-1',
                    memory='snapshot-1', verifier='independent-number-oracle-1')

    def __init__(self, root, clock):
        self.store = ContentAddressedArtifacts(root / 'artifacts')
        self.clock, self.source = clock, 'unstable'
        self.calls, self.verified = [], None
        self.manifest = {'original': 'preserved'}

    def activity_observed_terminal(self, name, value):
        return True  # This fixture performs synchronous local CPU work only.

    def accepted_checkpoint_valid(self):
        return self.verified is True

    def checkpoint(self):
        return dict(source=self.source, verified=self.verified, calls=self.calls)

    def restore_checkpoint(self, snapshot, sources):
        if self.source != snapshot['source'] and not getattr(self, 'allow_restore', False):
            raise WorkflowError('CPU host source differs from observed checkpoint')
        self.source, self.verified, self.calls = snapshot['source'], snapshot['verified'], snapshot['calls']

    def project_hash(self):
        return sha256(self.source.encode()).hexdigest()

    def read(self, path):
        assert path == 'kernel.py'
        return dict(content=self.source, project_hash=self.project_hash())

    def edit(self, path, text, expected_file_hash, expected_project_hash):
        assert path == 'kernel.py'
        assert expected_project_hash == expected_file_hash == self.project_hash()
        self.source, self.verified = text, None
        return self.project_hash()

    def preflight(self, context):
        self.clock.value += 100  # actual setup, outside actor allocation

    def execute(self, stage, args, operation, activity):
        self.calls.append(stage)
        if stage in (Stage.TEST, Stage.VALIDATION):
            activity.run('copy-input', 'transfer', lambda _: self.advance(20))
            def verify(seconds):
                self.advance(3)
                self.verified = self.source == 'stable'
            activity.run('oracle', 'execution', verify, execution=True)
            ref = self.store.put_json(dict(source=self.project_hash(), numeric=self.verified))
            return DomainObservation(self.verified, (ref,), correctness=self.verified)
        if stage == Stage.ACCEPTANCE:
            return DomainObservation(self.verified is True)
        return DomainObservation(True)

    def advance(self, seconds):
        self.clock.value += seconds

    def context(self):
        return dict(last_numeric=self.verified)


class Session:
    def __init__(self, clock, *, skip_query=False):
        self.clock, self.skip_query, self.calls = clock, skip_query, 0

    def prepare(self, timeout, *, schemas):
        self.clock.value += 50

    def next_action(self, context, allowed, **kwargs):
        self.calls += 1
        self.clock.value += 1
        for name in context['unread_skills']:
            return 'read_skill', dict(name=name)
        evidence = {row['name']: row['result'] for row in context['evidence']}
        stage = Stage(context['stage'])
        if stage in (Stage.DESIGN, Stage.OPTIMISATION) and not self.skip_query:
            if 'query_kdb' not in evidence:
                return 'query_kdb', dict(query='avoid unstable accumulation')
            if 'retrospect' not in evidence:
                return 'retrospect', dict(supplement='Use prior failed oracle results')
            if 'select_strategy' not in evidence:
                return 'select_strategy', dict(proposal='stable',
                    query_ref=evidence['query_kdb']['reference'],
                    retrospective_ref=evidence['retrospect']['reference'],
                    skill_refs=[r['receipt_id'] for r in context['read_skills']],
                    decisions=[dict(group='negative_records', index=0, action='rejected',
                                    reason='Prior numerical failure')])
            if stage == Stage.OPTIMISATION and 'optimisation_decision' not in evidence:
                return 'optimisation_decision', dict(action='noop', reason='Preserve verified source')
        args = dict(proposal='stable')
        if stage == Stage.DEBUG:
            args['edit'] = dict(path='kernel.py', sha256=context['source']['sha256'], source='stable')
        return 'perform', args

    def close(self):
        pass


def make(tmp_path, *, skip_query=False, cap=20, performance_goal=None):
    clock = Clock()
    host, queries = Host(tmp_path, clock), []
    session = Session(clock, skip_query=skip_query)
    def query(payload):
        queries.append(payload)
        return dict(results=[], negative_records=[dict(reason='TEST retained numeric failure')],
                    provenance={'baseline': 'TEST baseline-1'})
    injections = dict(query=query, backend_identity='TEST baseline-1',
        session_factory=lambda: session, test_backend=True, clock=clock)
    spec = ProjectSpec('domain', 'Correct numeric result', ('CPU-test-host',),
        budgets=BudgetLimits(execution_tries=cap, actor_seconds=500), performance_goal=performance_goal)
    runner = DomainRunner.create(tmp_path / 'task', spec, builtin_bundle(), host, **injections)
    return runner, host, session, queries, injections


def test_canonical_domain_numeric_repair_and_receipts(tmp_path):
    runner, host, _, queries, _ = make(tmp_path)
    result = runner.run()
    assert result['status'] == 'accepted', result
    assert result['correctness'] is True
    history = result['history']
    assert {row['evidence']['stage'] for row in history} == {s.value for s in Stage}
    assert any(row['evidence']['stage'] == 'test' and row['passed'] is False for row in history)
    assert host.calls.count(Stage.TEST) == 2
    assert len(queries) == 2
    assert runner.workflow.state['executions'] == 3
    assert runner.workflow.accounting.seconds('execution') == 9
    assert runner.workflow.accounting.seconds('transfer') == 60
    assert runner.workflow.accounting.seconds('setup') == 150
    assert host.manifest == {'original': 'preserved'}
    strategy = [runner.store.read_json(e['reference']) for e in runner.events
                if e['type'] == 'strategy-evidence' and e['kind'] == 'retrospective']
    assert strategy[-1]['payload']['failures']
    assert all(row['evidence']['skill_refs'] for row in history)


def test_missing_retrieval_blocks_host_design(tmp_path):
    runner, host, _, queries, _ = make(tmp_path, skip_query=True)
    result = runner.run()
    assert result['status'] == 'suspended'
    assert result['stage'] == 'design' and not queries
    assert Stage.DESIGN not in host.calls


def test_execution_cap_prevents_another_launch(tmp_path):
    runner, host, _, _, _ = make(tmp_path, cap=1)
    result = runner.run()
    assert result['status'] == 'budget_exhausted'
    assert runner.workflow.state['executions'] == 1
    assert host.calls.count(Stage.TEST) <= 2
    assert runner.workflow.accounting.seconds('execution') == 3


def test_changed_frozen_baseline_and_out_of_band_source_refused(tmp_path):
    runner, host, _, _, injections = make(tmp_path)
    host.identity = dict(host.identity, memory='changed')
    with pytest.raises(CanonicalConfigurationError, match='changed on resume'):
        DomainRunner(runner.root, host, **injections)
    host.identity = dict(host.identity, memory='snapshot-1')
    host.source = 'outside change'
    result = runner.run()
    assert result['status'] == 'suspended' and 'outside workflow' in result['error']
    assert not host.calls


def test_pending_host_operation_never_replayed(tmp_path):
    runner, host, session, _, injections = make(tmp_path)
    runner.workflow.begin()
    runner.workflow.dispatched(runner.workflow.pending['id'], 'retained-remote-handle')
    reopened = DomainRunner(runner.root, host, **injections)
    result = reopened.run()
    assert result['status'] == 'suspended' and result['pending']['handle'] == 'retained-remote-handle'
    assert not host.calls and session.calls == 0


def test_missing_configuration_has_actionable_failure():
    with pytest.raises(CanonicalConfigurationError, match='baseline KDB/memory'):
        require_adoption(None)


@pytest.mark.parametrize('entrypoint', ['a3', 'a5', 'core', 'optimizer'])
def test_active_entrypoint_uses_domain_runner_without_legacy_loop(tmp_path, entrypoint):
    runner, _, _, _, _ = make(tmp_path)
    adoption = CanonicalAdoption(lambda owner, **inputs: runner,
                                lambda owner, runner, result, inputs: result)
    if entrypoint == 'a3':
        from benchmarks.a3kernels.trial import A3TrialLoop
        owner = object.__new__(A3TrialLoop)
        owner.canonical = adoption
        owner._run = lambda: pytest.fail('old A3 loop dispatched')
        result = owner.run()
    elif entrypoint == 'optimizer':
        from optimizer.loop import OptimizationLoop
        owner = object.__new__(OptimizationLoop)
        owner.canonical = adoption
        owner.agent = type('Agent', (), {'close': lambda _: None})()
        owner._run_owned = lambda **_: pytest.fail('old optimizer loop dispatched')
        result = owner.run()
    else:
        if entrypoint == 'core':
            from benchmarks.a5kernels.trial import CoreAttemptDriver
            owner = CoreAttemptDriver(None, None, None, canonical=adoption)
        else:
            from benchmarks.a5kernels.phase1_provider import DirectActorDriver
            owner = object.__new__(DirectActorDriver)
            owner.canonical = adoption
            owner._select_action = lambda **_: pytest.fail('old direct/Pi loop dispatched')
        result = owner('task', profile=None, workspace=tmp_path / 'legacy',
                       context=None, turn_parser=None, action_executor=None)
    assert result['status'] == 'accepted', result


def test_interrupted_execution_retains_unknown_timing_and_never_replays(tmp_path):
    runner, host, _, _, injections = make(tmp_path)
    original = host.execute
    def interrupt(stage, args, operation, activity):
        if stage == Stage.TEST:
            def callback(seconds):
                host.advance(4)
                raise OSError('lost observation of remote operation')
            activity.run('owned-command', 'execution', callback, execution=True)
        return original(stage, args, operation, activity)
    host.execute = interrupt
    result = runner.run()
    assert result['status'] == 'suspended'
    assert result['pending']
    assert runner.workflow.state['executions'] == 1
    assert any(row[2] is None for row in runner.workflow.state['intervals'].values())
    with pytest.raises(WorkflowError, match='active or uncertain'):
        runner.workflow.accounting.seconds()
    reopened = DomainRunner(runner.root, host, **injections)
    before = tuple(host.calls)
    assert reopened.run()['status'] == 'suspended'
    assert tuple(host.calls) == before


def test_required_performance_cannot_be_accepted_without_measurement(tmp_path):
    runner, _, _, _, _ = make(tmp_path, performance_goal='frozen target latency')
    result = runner.run()
    assert result['status'] != 'accepted'
    assert not runner.workflow.outcome
    assert result['correctness'] is True and result['performance'] == 'not_measured'


def test_failed_native_packet_and_shared_export_do_not_require_python_manifest(tmp_path):
    from rsi.evidence_io import export_task
    runner, host, session, _, _ = make(tmp_path)
    original = session.next_action
    def malformed(context, allowed, **options):
        if context['stage'] == 'debug':
            raise ValueError('CPU fixture observed ordinary invalid host proposal')
        return original(context, allowed, **options)
    session.next_action = malformed
    result = runner.run()
    assert result['status'] == 'suspended'
    pending = dict(runner.workflow.pending)
    proof = runner.terminal_failure_proof()
    assert proof['failed_stage'] == 'debug'
    assert proof['pending'] == pending and proof['replay_allowed'] is False
    assert proof['command_receipts'] and proof['actor_seconds_total'] > 0
    assert not (runner.root/'manifest.json').exists()
    manifest = export_task(runner.root, tmp_path/'native-export')
    assert manifest['task_contract'] == runner.contract_ref
    assert runner.workflow.pending == pending
    records = [__import__('json').loads(row) for row in (tmp_path/'native-export'/'evidence.jsonl').read_text().splitlines()]
    assert any(row['kind']=='event' and row['data']['type']=='domain-result' for row in records)
    assert any(row['kind']=='artifact' and row['id']==proof['accounting_snapshot'] for row in records)


def test_fresh_native_wrapper_rehydrates_accepted_state_without_any_actor_or_host_call(tmp_path):
    runner, host, _, _, injections = make(tmp_path)
    assert runner.run()['status'] == 'accepted'
    prior_seconds = runner.workflow.accounting.seconds()
    prior_counts = runner.workflow.state['executions']
    fresh = Host(tmp_path, host.clock)
    fresh.allow_restore = True
    def forbidden(*_args, **_kwargs):
        pytest.fail('accepted checkpoint reran actor/target work')
    fresh.execute = forbidden
    resumed = DomainRunner(runner.root, fresh, **dict(injections, session_factory=forbidden))
    assert fresh.verified is True and fresh.source == 'stable'
    assert resumed.run()['status'] == 'accepted'
    assert resumed.workflow.state['executions'] == prior_counts
    assert resumed.workflow.accounting.seconds() == prior_seconds


def test_unobserved_native_activity_cannot_produce_failure_proof_or_export(tmp_path):
    from rsi.evidence_io import export_task
    runner, host, _, _, _ = make(tmp_path)
    original = host.execute
    def lost(stage, args, operation, activity):
        if stage == Stage.TEST:
            activity.run('unknown-host-job','execution',lambda _: {'status':'running','handle':'CPU-UNKNOWN'},execution=True)
            raise ValueError('observation lost')
        return original(stage,args,operation,activity)
    host.execute = lost
    assert runner.run()['status'] == 'suspended'
    with pytest.raises(WorkflowError):
        runner.terminal_failure_proof()
    with pytest.raises(WorkflowError):
        export_task(runner.root,tmp_path/'unsafe-export')
    assert not (tmp_path/'unsafe-export').exists()


def test_corrupt_native_receipt_or_identity_cannot_gain_terminal_authority(tmp_path):
    runner, host, session, _, _ = make(tmp_path)
    original = session.next_action
    def fail(context, allowed, **kwargs):
        if context['stage'] == 'debug':
            raise ValueError('terminal local proposal error')
        return original(context,allowed,**kwargs)
    session.next_action = fail
    runner.run()
    assert runner.terminal_failure_proof()['outcome'] == 'failed'
    host.identity = dict(host.identity, target='different CPU identity')
    with pytest.raises(WorkflowError):
        runner.terminal_failure_proof()


@pytest.mark.parametrize('failed_stage', [Stage.PREFLIGHT, Stage.TASK_CONSTRUCTION])
def test_failed_admission_is_terminal_even_with_zero_clock_and_all_caps_disabled(tmp_path, failed_stage):
    clock = Clock()
    host = Host(tmp_path, clock)
    calls = []
    def preflight(context):
        calls.append(Stage.PREFLIGHT)
        return failed_stage != Stage.PREFLIGHT
    host.preflight = preflight
    original = host.execute
    def perform(stage,args,operation,activity):
        calls.append(stage)
        if stage == failed_stage:
            return DomainObservation(False)
        return original(stage,args,operation,activity)
    host.execute = perform
    class ZeroClockSession(Session):
        def prepare(self, timeout, *, schemas):
            self.schemas = schemas
        def next_action(self,*args,**kwargs):
            before = clock.value
            result = super().next_action(*args,**kwargs)
            clock.value = before
            return result
    session = ZeroClockSession(clock)
    factories=[]
    def factory():
        factories.append(True)
        return session
    spec = ProjectSpec('admission','frozen admission',('CPU-host',),budgets=BudgetLimits(
        design_iterations=None,implementation_iterations=None,execution_tries=None,actor_seconds=None,wall_seconds=None))
    runner=DomainRunner.create(tmp_path/'task',spec,builtin_bundle(),host,
        query=lambda _: pytest.fail('admission queried development knowledge'),backend_identity='TEST-admission',
        session_factory=factory,test_backend=True,clock=clock)
    result=runner.run()
    assert result['status']=='rejected' and result['stage']==failed_stage.value
    assert calls.count(failed_stage)==1 and runner.workflow.pending is None
    assert runner.workflow.accounting.seconds()==0
    assert not any(row['evidence']['stage']=='acceptance' for row in result['history'])
    failure=runner.workflow.state['failure']
    assert failure['stage']==failed_stage.value and failure['reason'] and failure['artifact_refs']
    assert all(row['evidence']['skill_refs'] for row in result['history'])
    if failed_stage==Stage.PREFLIGHT:
        assert factories==[] and session.calls==0
    else:
        assert len(factories)==1 and session.calls==len(builtin_bundle().skills_for(failed_stage))+1


def test_nested_native_receipts_keep_closed_own_intervals_and_union_elapsed(tmp_path):
    from rsi.domain_execution import DomainActivity
    runner,host,_,_,_=make(tmp_path)
    operation=runner.workflow.begin()
    runner.workflow.dispatched(operation,'CPU-local-host')
    activity=DomainActivity(runner,operation)
    def outer(seconds):
        host.advance(3)
        activity.run('inner','execution',lambda _:host.advance(2),execution=True)
        host.advance(1)
    activity.run('outer','compilation',outer,execution=True)
    proof=runner._completed_effects()
    assert len(proof['command_receipts'])==2
    assert proof['actor_seconds_total']==6
    assert runner.workflow.state['executions']==2


def test_domain_refresh_supersedes_inflight_host_action_after_actual_feedback_receipt(tmp_path):
    from copy import deepcopy
    from benchmarks.ascend.actor_control import ActorControl
    from rsi.domain_bindings import SourceHost
    runner, host, session, _, _ = make(tmp_path)
    host.control_root, host.feedback, host.observations = tmp_path/'control', [], []
    host.instruction = 'Use the existing frozen CPU oracle'
    host.workflow = runner.workflow
    host.context = SourceHost.context.__get__(host)
    host.context_refresh = SourceHost.context_refresh.__get__(host)
    inbox = ActorControl(host.control_root)
    decisions, sent = [], []
    original_decision = session.next_action
    def decide(context, allowed, **kwargs):
        decisions.append(deepcopy(context))
        selected = original_decision(context, allowed, **kwargs)
        if context['stage'] == 'test' and selected[0] == 'perform' and not sent:
            sent.append(inbox.send('during-selection', 'Review the numeric direction before host work'))
        return selected
    session.next_action = decide
    original_execute = host.execute
    def execute(stage, args, operation, activity):
        if stage == Stage.TEST:
            assert inbox.status()['messages'][0]['consumed'] is not None
            assert any(sent[0]['id'] in [row['feedback_id'] for row in
                context['task']['domain']['feedback']] for context in decisions
                if context['operation_id'] == operation)
        return original_execute(stage, args, operation, activity)
    host.execute = execute
    result = runner.run()
    assert result['status'] == 'accepted' and result['correctness'] is True
    superseded = [event for event in runner.events if event['type'] == 'actor-action-superseded']
    assert len(superseded) == 1 and superseded[0]['executed'] is False
    operation = superseded[0]['operation_id']
    contexts = [row for row in decisions if row['operation_id'] == operation]
    assert contexts[0]['task']['domain']['feedback'] == []
    assert contexts[-1]['task']['domain']['feedback'][0]['feedback_id'] == sent[0]['id']
    assert host.calls.count(Stage.TEST) == 2  # Original numeric failure plus genuine repair.
    assert len([row for row in result['history'] if row.get('evidence', {}).get('operation_id') == operation]) == 1


def test_context_budget_error_retains_typed_abort_and_durable_closure_before_transport_or_host(tmp_path):
    from benchmarks.ascend.agent_context import ContextBudget, ContextBudgetError
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider, direct_profile
    runner,host,session,_,_=make(tmp_path)
    profile=direct_profile(MODEL_IDENTITIES[0])
    transports=[]
    def transport(**kwargs):
        transports.append(kwargs)
        pytest.fail('oversized request reached transport')
    provider=DirectCompletionProvider({'OPENAI_API_KEY':'CPU-only','DEEPSEEK_API_KEY':'CPU-only'},
        transport=transport,context_budgets={profile.identity.backend_model.value:ContextBudget(5000,3500,6000)})
    def decision(context,allowed,**kwargs):
        provider.complete(profile,[dict(role='user',content='oversized frozen input '*1000)])
    session.next_action=decision
    with pytest.raises(ContextBudgetError):runner.run()
    assert transports==[] and host.calls==[]
    assert runner.workflow.pending['stage']=='task_construction'
    assert any(row['type']=='project-failure' and row['classification']=='context_budget_exhausted'
               for row in runner.events)
    outcome=runner.store.read_json(runner.latest('project-run-outcome')['reference'])
    assert outcome['result']['classification']=='context_budget_exhausted'
    assert outcome['closure_ref']
    assert runner.store.read_json(outcome['closure_ref'])['observation']['provider_settled'] is True
