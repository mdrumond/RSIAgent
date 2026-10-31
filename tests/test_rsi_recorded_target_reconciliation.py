"""Retained controller phases settle lost receipts without producer-time placement."""
import json

import pytest

from rsi.python_execution import CommandResult, atomic_json, reconcile_recorded
from rsi.runtime import WorkflowError
from test_rsi_trusted_executor import target


@pytest.fixture
def recorded(target):
    snapshot, workflow, now, bind = target
    workflow.begin()
    calls = []
    state = dict(closed=True)

    class Executor:
        def execute(self, request, *, observe, remaining):
            calls.append(('execute', request))
            observe('setup')
            now[0] += 1
            observe('command', handle='remote:test:job:original')
            now[0] += 2
            observe('transfer', handle='remote:test:transfer:original')
            now[0] += 5
            observe('command', handle='remote:test:job:original')
            now[0] += 3
            raise ConnectionError('download acknowledgement lost')

        def reconcile(self, request, *, observe, remaining):
            calls.append(('reconcile', request))
            assert remaining() > 0
            now[0] += .5
            observe('transfer', handle='remote:test:transfer:original')
            now[0] += 2
            observe('command', handle='remote:test:job:original')
            now[0] += .25
            if not state['closed']:
                raise WorkflowError('target producer remains unknown')
            return CommandResult(0, stdout=b'original result')

    backend = bind(Executor())
    with pytest.raises(ConnectionError):
        backend.numeric('original')
    path = backend._receipt_path('original')
    original = json.loads(path.read_text())
    timing = dict(operation_id='original', source_hash=snapshot.project_hash(), controller_recorded=True)
    return backend, workflow, now, calls, state, path, original, timing


def test_recorded_fragmented_phases_reconcile_once_without_rebilling(recorded):
    backend, workflow, now, calls, state, path, original, timing = recorded
    clock = workflow.accounting
    ledger = {k: list(v) for k, v in clock.records.items()}
    pending = dict(workflow.pending)
    assert original['elapsed'] == 5 and original['observed_elapsed'] == 11
    now[0] += 20  # Controller downtime is not a new producer interval.
    identities = []
    def factory(identity):
        identities.append(identity)
        return backend.executor
    result = reconcile_recorded(backend.root, workflow, 'original', executor_factory=factory)
    assert identities == [backend.contract['executor_identity']]
    assert result['elapsed'] == 5 and result['observed_elapsed'] == 11
    assert result['classification'] == 'passed'
    assert backend.store.read_bytes(result['stdout']) == b'original result'
    assert all(clock.records[k] == v for k, v in ledger.items())
    assert clock.seconds() == 5.75
    assert clock.seconds('setup') == 1
    assert clock.seconds('transfer') == 7
    assert not clock.uncertainties and not clock.receipts
    assert 'receipt:original' not in clock.records
    assert workflow.pending == pending and workflow.outcome is None
    assert backend.project_hash() == original['source_hash']
    assert calls[0][1] == calls[1][1]
    after = {k: list(v) for k, v in clock.records.items()}
    assert backend.reconcile_target('original', timing) == result
    assert clock.records == after
    assert [action for action, _ in calls] == ['execute', 'reconcile']


@pytest.mark.parametrize('damage', [
    'missing_elapsed', 'missing_observed', 'invalid_elapsed', 'changed_elapsed',
    'open_interval', 'wrong_operation', 'wrong_source_claim', 'changed_source',
    'wrong_executor', 'wrong_command', 'missing_token', 'changed_phase',
    'unrelated_uncertainty', 'wrong_uncertainty', 'producer_claim', 'prior_timing',
])
def test_recorded_timing_requires_matching_measurements_and_closed_identity(recorded, damage):
    backend, workflow, now, calls, state, path, record, timing = recorded
    clock = workflow.accounting
    if damage == 'missing_elapsed':
        record.pop('elapsed')
    elif damage == 'missing_observed':
        record.pop('observed_elapsed')
    elif damage == 'invalid_elapsed':
        record['elapsed'] = -1
    elif damage == 'changed_elapsed':
        record['elapsed'] = 4
    elif damage == 'open_interval':
        clock.records[record['clock_token']][2] = None
    elif damage == 'wrong_operation':
        record['operation_id'] = 'another'
    elif damage == 'wrong_source_claim':
        timing['source_hash'] = 'another'
    elif damage == 'changed_source':
        (backend.project / 'solution.py').write_text('changed\n')
    elif damage == 'wrong_executor':
        record['executor_identity'] = 'another'
    elif damage == 'wrong_command':
        record['argv'] = ['different-command']
    elif damage == 'missing_token':
        record['clock_token'] = 'missing'
    elif damage == 'changed_phase':
        record['observations'][1]['phase'] = 'transfer'
    elif damage == 'unrelated_uncertainty':
        clock.uncertainties['actor:another'] = dict(status='unknown')
    elif damage == 'wrong_uncertainty':
        clock.uncertainties['executor:original']['receipt'] = 'another'
    elif damage == 'producer_claim':
        timing['producer'] = dict(elapsed=44.513)
    elif damage == 'prior_timing':
        clock.receipts['receipt:original'] = dict(elapsed=4)
    atomic_json(path, record)
    with pytest.raises(WorkflowError):
        backend.reconcile_target('original', timing)
    assert [action for action, _ in calls] == ['execute']
    assert 'executor:original' in clock.uncertainties
    assert json.loads(path.read_text())['status'] == 'pending'
    assert workflow.pending is not None


def test_unknown_producer_retains_uncertainty_and_retry_observes_same_request(recorded):
    backend, workflow, now, calls, state, path, original, timing = recorded
    ledger = {k: list(v) for k, v in workflow.accounting.records.items()}
    state['closed'] = False
    with pytest.raises(WorkflowError):
        backend.reconcile_target('original', timing)
    assert 'executor:original' in workflow.accounting.uncertainties
    assert json.loads(path.read_text())['status'] == 'pending'
    state['closed'] = True
    result = backend.reconcile_target('original', timing)
    assert result['elapsed'] == 5
    assert workflow.accounting.seconds() == 6.5
    assert workflow.accounting.seconds('transfer') == 9
    assert all(workflow.accounting.records[k] == v for k, v in ledger.items())
    assert [action for action, _ in calls] == ['execute', 'reconcile', 'reconcile']
    assert calls[0][1] == calls[1][1] == calls[2][1]


def test_command_only_lost_ack_retains_measured_prelude_outside_observed_span(target, monkeypatch):
    snapshot, workflow, now, bind = target
    workflow.begin()
    calls = []

    class Executor:
        def execute(self, request, *, observe, remaining):
            calls.append(('execute', request))
            now[0] += 2
            raise ConnectionError('command acknowledgement lost')

        def reconcile(self, request, *, observe, remaining):
            calls.append(('reconcile', request))
            now[0] += .25
            return CommandResult(0)

    backend = bind(Executor())
    save = backend._save
    delayed = []
    def save_with_prelude(operation, record):
        save(operation, record)
        if record.get('clock_token') and not delayed:
            delayed.append(True)
            now[0] += .125  # Durable receipt I/O before _execute_target starts.
    monkeypatch.setattr(backend, '_save', save_with_prelude)
    with pytest.raises(ConnectionError):
        backend.run('command-only', 'compile')
    original = json.loads(backend._receipt_path('command-only').read_text())
    assert original['elapsed'] == 2.125
    assert original['observed_elapsed'] == 2
    ledger = {k: list(v) for k, v in workflow.accounting.records.items()}
    result = backend.reconcile_target('command-only', dict(operation_id='command-only',
        source_hash=snapshot.project_hash(), controller_recorded=True))
    assert result['elapsed'] == 2.125 and result['observed_elapsed'] == 2
    assert workflow.accounting.seconds() == 2.375
    assert all(workflow.accounting.records[k] == v for k, v in ledger.items())
    assert not workflow.accounting.receipts and not workflow.accounting.uncertainties
    assert calls[0][1] == calls[1][1]
    assert workflow.pending is not None and workflow.outcome is None
