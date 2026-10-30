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
