"""Behavioral backend tests; run only under the coordinator's serial token."""
import json
from pathlib import Path
import sys

import pytest

from rsi.contracts import BudgetLimits, Stage
from rsi.runtime import Workflow, WorkflowError
from rsi.python_execution import PythonExecution, ingest_elapsed


def backend(tmp_path, *, seconds=10):
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'solution.py').write_text('def stable_sum(xs):\n    return sum(xs)\n')
    (source / 'helper.py').write_text('value = 1\n')
    verifier = Path(__file__).resolve().parents[1] / 'examples/cpu/verify.py'
    runner = PythonExecution.snapshot(tmp_path / 'task', source, verifier,
        commands={'compile': [sys.executable, '-m', 'compileall', '-q', '.'],
                  'test': [sys.executable, '{verifier}'],
                  'validation': [sys.executable, '{verifier}'],
                  'slow': [sys.executable, '-c', 'import time; time.sleep(5)']},
        performance_seconds=1)
    workflow = Workflow(tmp_path / 'workflow.json', lambda *args: True,
        source_hash=runner.project_hash(), limits=BudgetLimits(actor_seconds=seconds))
    runner.workflow = workflow
    return runner


def test_numeric_failure_repair_and_multiple_files(tmp_path):
    runner = backend(tmp_path)
    old = runner.project_hash()
    failure = runner.numeric('numeric-1')
    assert failure['classification'] == 'numeric_failure'
    assert failure['returncode'] != 0
    file = runner.read('solution.py')
    runner.edit('solution.py', 'import math\ndef stable_sum(xs):\n    return math.fsum(xs)\n',
                expected_file_hash=file['file_hash'], expected_project_hash=old)
    runner.edit('helper.py', 'value = 2\n', **{
        'expected_file_hash': runner.read('helper.py')['file_hash'],
        'expected_project_hash': runner.project_hash()})
    assert runner.numeric('numeric-2')['classification'] == 'passed'
    assert runner.project_hash() != old
    assert runner.workflow.source_hash == runner.project_hash()
    with pytest.raises(WorkflowError):
        runner.edit('solution.py', '', expected_file_hash=file['file_hash'], expected_project_hash=old)
    with pytest.raises(ValueError):
        runner.read('../verifier.py')


def test_compile_failure_timeout_and_no_replay(tmp_path):
    runner = backend(tmp_path, seconds=10)
    (runner.project / 'broken.py').write_text('def broken(:\n')
    failed = runner.run('compile-1', 'compile')
    assert failed['classification'] == 'compile_failure'
    assert runner.workflow.accounting.seconds('compilation') > 0
    timed = runner.run('slow-1', 'slow', timeout=.05)
    assert timed['timed_out']
    assert runner.workflow.accounting.seconds('execution') > 0
    assert runner.run('slow-1', 'slow', timeout=.05) == timed
    assert runner.store.read_bytes(timed['stderr']) is not None


def test_external_change_invalidates_validation_and_frozen_verifier(tmp_path):
    runner = backend(tmp_path)
    workflow = runner.workflow
    workflow.state.update(stage=Stage.ACCEPTANCE.value, tested=workflow.source_hash,
                          verified=workflow.source_hash)
    workflow._save()
    (runner.project / 'helper.py').write_text('value = 3\n')
    runner.sync_source()
    assert workflow.verified_source is None
    assert workflow.stage == Stage.TEST
    assert runner.workflow.source_hash == runner.project_hash()
    (runner.verifier).write_text('print("forged")')
    with pytest.raises(WorkflowError):
        runner.numeric('bad-verifier')


def test_pending_receipt_is_not_replayed(tmp_path):
    runner = backend(tmp_path)
    runner._save('pending', {'status': 'pending'})
    with pytest.raises(WorkflowError):
        runner.run('pending', 'compile')


def test_remote_durations_ignore_producer_epoch_and_union_overlap(tmp_path):
    now = [100.]
    workflow = Workflow(tmp_path / 'clock.json', lambda *args: True, source_hash='x',
                        clock=lambda: now[0])
    transfer = workflow.accounting.start('transfer')
    thinking = workflow.accounting.start('thinking')
    now[0] = 110.
    workflow.accounting.end(thinking)
    workflow.accounting.end(transfer)
    ingest_elapsed(workflow.accounting, 'remote:1', {'elapsed': 6., 'producer_start': 900000.},
                   observed_start=102., observed_end=110., category='execution')
    assert workflow.accounting.seconds() == 10.
    assert workflow.accounting.seconds('transfer') == 10.
    ingest_elapsed(workflow.accounting, 'remote:1', {'elapsed': 6.},
                   observed_start=102., observed_end=110., category='execution')
    assert workflow.accounting.seconds() == 10.
    for receipt in ({}, {'elapsed': -1}, {'elapsed': 20}, {'elapsed': float('nan')}):
        with pytest.raises(WorkflowError):
            ingest_elapsed(workflow.accounting, 'bad', receipt,
                           observed_start=102., observed_end=110., category='execution')


def test_profile_and_performance_goal_are_retained(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'main.py').write_text('import time; time.sleep(.02)\n')
    verifier = tmp_path / 'verify.py'
    verifier.write_text('print("verified")\n')
    runner = PythonExecution.snapshot(tmp_path / 'task', source, verifier,
        commands={'profile': [sys.executable, 'main.py']}, performance_seconds=.001)
    runner.workflow = Workflow(tmp_path / 'state.json', lambda *a: True,
                               source_hash=runner.project_hash())
    result = runner.run('profile-1', 'profile', profile=True)
    assert result['classification'] == 'performance_shortfall'
    assert runner.store.read_bytes(result['profile'])
    assert runner.workflow.accounting.seconds('profiling') > .02
    reopened = PythonExecution(runner.root, runner.workflow)
    assert reopened.receipt('profile-1') == result
    assert reopened.contract['performance_seconds'] == .001


def test_snapshot_preserves_input_and_refuses_dirty_repository(tmp_path):
    import subprocess
    source = tmp_path / 'repo'
    source.mkdir()
    verifier = tmp_path / 'verify.py'
    verifier.write_text('pass\n')
    subprocess.run(['git', 'init', str(source)], check=True, capture_output=True)
    (source / 'untracked.py').write_text('pass\n')
    with pytest.raises(ValueError):
        PythonExecution.snapshot(tmp_path / 'task', source, verifier, commands={})
    assert not (tmp_path / 'task').exists()
    assert (source / 'untracked.py').read_text() == 'pass\n'


def test_edit_in_mutation_operation_defers_source_commit(tmp_path):
    runner = backend(tmp_path)
    runner.workflow.state['stage'] = Stage.IMPLEMENTATION.value
    runner.workflow._save()
    runner.workflow.begin()
    original = runner.workflow.source_hash
    file = runner.read('helper.py')
    changed = runner.edit('helper.py', 'value = 4\n',
        expected_file_hash=file['file_hash'], expected_project_hash=file['project_hash'])
    assert changed != original
    assert runner.workflow.source_hash == original
    assert runner.workflow.pending is not None


@pytest.mark.parametrize('command,profile', [('compile', False), ('test', False), ('profile', True)])
def test_each_command_reserves_try_including_failures(tmp_path, command, profile):
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'broken.py').write_text('invalid syntax here\n')
    verifier = tmp_path / 'verify.py'
    verifier.write_text('raise AssertionError("numeric mismatch")\n')
    runner = PythonExecution.snapshot(tmp_path / 'task', source, verifier, commands={
        'compile': [sys.executable, '-m', 'compileall', '-q', '.'],
        'test': [sys.executable, '{verifier}'],
        'profile': [sys.executable, 'broken.py']})
    runner.workflow = Workflow(tmp_path / 'state.json', lambda *a: True,
        source_hash=runner.project_hash(), limits=BudgetLimits(execution_tries=1))
    result = runner.run('first', command, profile=profile)
    assert result['returncode'] != 0
    assert runner.run('first', command, profile=profile) == result
    assert runner.workflow.state['executions'] == 1
    with pytest.raises(WorkflowError):
        runner.run('second', command, profile=profile)
    assert runner.workflow.state['executions'] == 1


def test_legacy_execution_credit_and_unknown_reservation_survive_restart(tmp_path):
    runner = backend(tmp_path)
    runner.workflow.begin(execution=True)
    runner.run('legacy-command', 'test')
    assert runner.workflow.state['executions'] == 1
    runner.run('another-command', 'test')
    assert runner.workflow.state['executions'] == 2
    assert runner.workflow.reserve_execution('crashed-before-receipt')
    workflow = Workflow(runner.workflow.path, lambda *a: True, source_hash=runner.project_hash())
    runner.workflow = workflow
    with pytest.raises(WorkflowError):
        runner.run('crashed-before-receipt', 'test')
    assert workflow.state['executions'] == 3


def test_generated_cache_does_not_invalidate_but_declared_inputs_do(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'solution.py').write_text('value = 42\n')
    (source / 'input.json').write_text('{}')
    verifier = source / 'verify.py'
    verifier.write_text('from pathlib import Path\nfrom solution import value\n'
                        'assert value == 42\nPath(".pytest_cache").mkdir(exist_ok=True)\n'
                        'Path(".pytest_cache/log").write_text("generated")\n'
                        'Path("output.log").write_text("run output")\n')
    runner = PythonExecution.snapshot(tmp_path / 'task', source, verifier,
        commands={'test': [sys.executable, '{verifier}']})
    runner.workflow = Workflow(tmp_path / 'state.json', lambda *a: True,
                               source_hash=runner.project_hash())
    original = runner.project_hash()
    assert not (runner.project / 'verify.py').exists()
    with pytest.raises(ValueError):
        runner.read('verify.py')
    for operation in ('first', 'second'):
        assert runner.numeric(operation)['classification'] == 'passed'
        assert runner.project_hash() == original
    (runner.project / 'input.json').write_text('{"changed":true}')
    assert runner.project_hash() != original


def test_partial_overlap_cannot_invent_execution_placement(tmp_path):
    now = [100.]
    workflow = Workflow(tmp_path / 'state.json', lambda *a: True, source_hash='x',
                        clock=lambda: now[0])
    now[0] = 110.
    thinking = workflow.accounting.start('thinking')
    now[0] = 120.
    workflow.accounting.end(thinking)
    with pytest.raises(WorkflowError):
        ingest_elapsed(workflow.accounting, 'compile', {'elapsed': 10., 'producer_epoch': 999999.},
                       observed_start=100., observed_end=120., category='compilation')
    with pytest.raises(WorkflowError):
        workflow.remaining()
    workflow = Workflow(workflow.path, lambda *a: True, source_hash='x', clock=lambda: now[0])
    with pytest.raises(WorkflowError):
        workflow.reserve_execution('next')
    assert workflow.accounting.uncertainties['receipt:compile']['envelope'] == [100., 120.]
    ingest_elapsed(workflow.accounting, 'compile', {'elapsed': 10.},
                   observed_start=100., observed_end=120., category='compilation',
                   observed_execution_start=100., observed_execution_end=110.)
    assert workflow.accounting.seconds() == 20.
    assert workflow.accounting.receipts['receipt:compile']['placement'] == 'authoritative_host_interval'
