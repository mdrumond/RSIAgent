"""Trusted injection, phase accounting and durable unknown-work behavior."""
import json
import sys

import pytest

from rsi.contracts import BudgetLimits
from rsi.python_execution import CommandResult, PythonExecution
from rsi.runtime import Workflow, WorkflowError


@pytest.fixture
def target(tmp_path, monkeypatch):
    now = [100.]
    # Deliberately distinct clock domains: monotonic is never an epoch.
    monkeypatch.setattr('rsi.python_execution.time.monotonic', lambda: now[0] + 900.)
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'solution.py').write_text('pass\n')
    verifier = source / 'verify.py'
    verifier.write_text('pass\n')
    snapshot = PythonExecution.snapshot(tmp_path / 'task', source, verifier,
        executor_identity='test-profile@frozen', commands={
            'compile': [sys.executable, 'solution.py'],
            'test': [sys.executable, '{verifier}'],
            'profile': [sys.executable, 'solution.py']})
    workflow = Workflow(tmp_path / 'state.json', lambda *a: True,
        source_hash=snapshot.project_hash(), clock=lambda: now[0],
        limits=BudgetLimits(actor_seconds=20, wall_seconds=100))

    def bind(executor):
        identities = []
        runner = PythonExecution(snapshot.root, workflow,
            executor_factory=lambda identity: identities.append(identity) or executor)
        assert identities == ['test-profile@frozen']
        return runner

    return snapshot, workflow, now, bind


@pytest.mark.parametrize('command,profile,category', [
    ('compile', False, 'compilation'), ('test', False, 'execution'),
    ('profile', True, 'profiling')])
def test_phase_accounting_timeout_and_elapsed_union(target, command, profile, category):
    snapshot, workflow, now, bind = target
    thinking = workflow.accounting.start('thinking')
    now[0] += 3
    workflow.accounting.end(thinking)
    calls = []

    class Executor:
        def execute(self, request, *, observe, remaining):
            calls.append(request)
            assert request.project == snapshot.project
            assert request.verifier == snapshot.verifier
            assert request.verifier_hash == snapshot.contract['verifier']
            assert request.source_hash == snapshot.project_hash()
            assert request.category == category
            assert not (request.project / 'verify.py').exists()
            if command == 'test':
                assert str(request.verifier) in request.argv
            if profile:
                request.profile_path.write_bytes(b'profile output')
            observe('setup', runtime='named-test-runtime')
            now[0] += 7
            assert remaining() == 10
            observe('command', handle='remote:test:job:actual-42', physical_device=2,
                    candidate_case='case-1')
            now[0] += 2
            assert remaining() == 8
            observe('transfer')
            now[0] += 8
            assert remaining() == 8
            observe('command')
            now[0] += 3
            assert remaining() == 5
            return CommandResult(1, stdout=b'out', stderr=b'failed')

    runner = bind(Executor())
    result = runner.run('op', command, profile=profile, timeout=10)
    assert result['elapsed'] == 5
    assert result['observed_elapsed'] == 20
    assert workflow.accounting.seconds(category) == 5
    assert workflow.accounting.seconds('setup') == 7
    assert workflow.accounting.seconds('transfer') == 8
    assert workflow.accounting.seconds() == 8  # Thinking plus command, once each.
    assert workflow.remaining() == {'actor_seconds': 12, 'wall_seconds': 77}
    assert all(row[2] is not None for row in workflow.accounting.records.values())
    assert result['returncode'] == 1
    assert runner.store.read_bytes(result['stderr']) == b'failed'
    assert result['observations'][1]['handle'] == 'remote:test:job:actual-42'
    if profile:
        assert runner.store.read_bytes(result['profile']) == b'profile output'
    assert bind(Executor()).run('op', command, profile=profile) == result
    assert len(calls) == 1


def test_unknown_dispatch_blocks_replay_and_budget_after_reopen(target):
    snapshot, workflow, now, bind = target
    calls = []

    class Executor:
        def execute(self, request, *, observe, remaining):
            calls.append(request.operation_id)
            observe('command', handle='remote:test:job:actual-99')
            now[0] += 2
            raise ConnectionError('terminal state unknown')

    runner = bind(Executor())
    with pytest.raises(ConnectionError):
        runner.run('unknown', 'test')
    pending = json.loads(runner._receipt_path('unknown').read_text())
    assert pending['status'] == 'pending'
    assert pending['observations'][0]['handle'] == 'remote:test:job:actual-99'
    reopened_workflow = Workflow(workflow.path, lambda *a: True,
        source_hash=snapshot.project_hash(), clock=lambda: now[0])
    reopened = PythonExecution(snapshot.root, reopened_workflow, executor_factory=lambda _: Executor())
    with pytest.raises(WorkflowError, match='never replayed'):
        reopened.run('unknown', 'test')
    with pytest.raises(WorkflowError, match='uncertain'):
        reopened.run('different', 'test')
    assert calls == ['unknown']
    assert reopened_workflow.state['executions'] == 1


def test_frozen_binding_required_and_verifier_changes_stay_pending(target):
    snapshot, workflow, now, bind = target
    with pytest.raises(ValueError, match='trusted executor factory'):
        PythonExecution(snapshot.root, workflow)
    with pytest.raises(ValueError, match='implement execute'):
        PythonExecution(snapshot.root, workflow, executor_factory=lambda _: None)

    class Executor:
        def execute(self, request, *, observe, remaining):
            observe('command')
            request.verifier.write_text('changed\n')
            return CommandResult(0)

    runner = bind(Executor())
    with pytest.raises(WorkflowError, match='verifier changed during command'):
        runner.run('verifier-change', 'test')
    with pytest.raises(WorkflowError, match='never replayed'):
        runner.run('verifier-change', 'test')


def test_thinking_overlaps_setup_without_refunding_actor_time(target):
    snapshot, workflow, now, bind = target
    thinking = workflow.accounting.start('thinking')

    class Executor:
        def execute(self, request, *, observe, remaining):
            observe('setup')
            now[0] += 7
            assert remaining() == 13
            observe('command')
            now[0] += 2
            assert remaining() == 11
            return CommandResult(0)

    result = bind(Executor()).run('overlap', 'compile')
    workflow.accounting.end(thinking)
    assert result['elapsed'] == 2
    assert workflow.accounting.seconds() == 9
    assert workflow.accounting.seconds('compilation') == 2


def test_timeout_exhaustion_and_stop_are_propagated(target):
    snapshot, workflow, now, bind = target

    class Executor:
        def execute(self, request, *, observe, remaining):
            observe('setup')
            now[0] += 7
            assert remaining() == 2
            observe('command')
            now[0] += 2
            assert remaining() == 0
            workflow.state['stopped'] = True
            assert remaining() == 0
            return CommandResult(-9, timed_out=True)

    result = bind(Executor()).run('timeout', 'compile', timeout=2)
    assert result['classification'] == 'timeout'
    assert workflow.accounting.seconds('compilation') == 2
    assert result['observed_elapsed'] == 9
