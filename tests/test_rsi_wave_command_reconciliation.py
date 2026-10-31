"""Observe retained target commands without replay or invented clock placement."""
import pytest

from rsi.python_execution import CommandResult
from rsi.runtime import WorkflowError
from test_rsi_trusted_executor import target


def test_original_target_handle_finalizes_once_with_authoritative_elapsed(target):
    snapshot, workflow, now, bind = target
    workflow.begin()
    calls = []

    class Executor:
        def execute(self, request, *, observe, remaining):
            calls.append(('execute', request.operation_id))
            observe('command', handle='remote:test:job:original')
            now[0] += 2
            raise ConnectionError('retained target still running')

        def reconcile(self, request, *, observe, remaining):
            calls.append(('observe', request.operation_id))
            observe('transfer', handle='remote:test:job:original')
            now[0] += 1
            return CommandResult(1, stderr=b'actual failed verifier')

    backend = bind(Executor())
    with pytest.raises(ConnectionError):
        backend.run('original', 'test', numeric=True)
    now[0] += 3
    timing = dict(operation_id='original', source_hash=snapshot.project_hash(),
        producer={'elapsed': 5.}, observed_start=100., observed_end=105.,
        execution_start=100., execution_end=105.)
    result = backend.reconcile_target('original', timing)
    assert result['classification'] == 'numeric_failure'
    assert backend.store.read_bytes(result['stderr']) == b'actual failed verifier'
    assert workflow.accounting.seconds() == 5.
    assert workflow.accounting.seconds('transfer') == 1.
    assert backend.reconcile_target('original', timing) == result
    assert calls == [('execute', 'original'), ('observe', 'original')]
    assert workflow.pending is not None  # Command closure never manufactures stage acceptance.


def test_missing_observation_support_does_not_execute_again(target):
    snapshot, workflow, now, bind = target
    workflow.begin()
    calls = []

    class Executor:
        def execute(self, request, *, observe, remaining):
            calls.append(request.operation_id)
            raise ConnectionError('unknown command')

    backend = bind(Executor())
    with pytest.raises(ConnectionError):
        backend.run('unknown', 'test')
    with pytest.raises(WorkflowError):
        backend.reconcile_target('unknown', dict(operation_id='unknown', source_hash=snapshot.project_hash()))
    assert calls == ['unknown']
    with pytest.raises(WorkflowError):
        backend.receipt('unknown')
