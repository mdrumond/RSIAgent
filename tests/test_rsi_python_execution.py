"""Behavioral backend tests; run only under the coordinator's serial token."""
import json
import math
from pathlib import Path
import sys

import pytest

from rsi.contracts import BudgetLimits, Stage
from rsi.runtime import Workflow, WorkflowError
from rsi.python_execution import PythonExecution, ingest_elapsed


def backend(tmp_path, *, seconds=10):
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'solution.py').write_text('def stable_sum(xs):\n    total = 0.0\n    for x in xs:\n        total += x\n    return total\n')
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


@pytest.mark.parametrize('elapsed', [.1, .01, .2])
@pytest.mark.parametrize('authoritative', [False, True])
def test_epoch_quantized_envelope_preserves_measured_duration(tmp_path, elapsed, authoritative):
    start = 1700000000.
    end = start + elapsed
    workflow = Workflow(tmp_path / 'clock.json', lambda *a: True, source_hash='x',
                        clock=lambda: end)
    bounds = (dict(observed_execution_start=start, observed_execution_end=end)
              if authoritative else {})
    ingest_elapsed(workflow.accounting, 'quantized', {'elapsed': elapsed},
                   observed_start=start, observed_end=end, category='execution', **bounds)
    reopened = Workflow(workflow.path, lambda *a: True, source_hash='x', clock=lambda: end)
    receipt = reopened.accounting.receipts['receipt:quantized']
    assert receipt['elapsed'] == elapsed
    assert receipt['envelope'] == [start, end]
    assert receipt['placement'] == ('authoritative_host_interval' if authoritative
                                    else 'synthetic_union_equivalent')
    assert reopened.accounting.records['receipt:quantized'] == ['execution', start, end]
    assert abs(reopened.accounting.seconds() - elapsed) <= math.ulp(start) + math.ulp(end)


@pytest.mark.parametrize('start,end,elapsed,now', [
    (1700000000., 1700000000.1, .100002, 1700000001.),  # Above epoch quantum.
    (1., 1.1, .100000001, 2.),  # Below 1us but above the endpoint quantum.
    (1e12, 1e12 + .1, .10001, 1e12 + 1.),  # Coarse ULP cannot bypass 1us cap.
    (1700000000., 1700000000.1, .1, 1700000000.),  # Future envelope.
    (1700000000.1, 1700000000., 0., 1700000001.),  # Backward envelope.
    (1700000000., 1700000000.1, -.1, 1700000001.),
    (1700000000., 1700000000.1, float('nan'), 1700000001.),
    (1700000000., 1700000000.1, float('inf'), 1700000001.),
    (1700000000., 1700000000.1, True, 1700000001.),
    (1700000000., 1700000000.1, '0.1', 1700000001.),
    (1700000000., float('inf'), .1, 1700000001.),
])
def test_envelope_tolerance_rejects_invalid_receipts_durably(tmp_path, start, end, elapsed, now):
    workflow = Workflow(tmp_path / 'clock.json', lambda *a: True, source_hash='x',
                        clock=lambda: now)
    with pytest.raises(WorkflowError):
        ingest_elapsed(workflow.accounting, 'invalid', {'elapsed': elapsed},
                       observed_start=start, observed_end=end, category='execution')
    reopened = Workflow(workflow.path, lambda *a: True, source_hash='x', clock=lambda: now)
    assert 'receipt:invalid' in reopened.accounting.uncertainties
    assert 'receipt:invalid' not in reopened.accounting.receipts
    assert 'receipt:invalid' not in reopened.accounting.records
    with pytest.raises(WorkflowError):
        reopened.accounting.seconds()


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
        commands={'test': [sys.executable, '{verifier}']}, generated_paths=['output.log'])
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


@pytest.mark.parametrize('stage', [Stage.IMPLEMENTATION, Stage.DEBUG, Stage.OPTIMISATION])
def test_pending_edit_compile_check_then_complete_preserves_stage_identity(tmp_path, stage):
    from rsi.contracts import StepEvidence
    runner = backend(tmp_path)
    flow = runner.workflow
    flow.state['stage'] = stage.value
    flow._save()
    operation = flow.begin()
    original = flow.source_hash
    file = runner.read('solution.py')
    changed = runner.edit('solution.py', 'import math\ndef stable_sum(xs):\n    return math.fsum(xs)\n',
                          expected_file_hash=file['file_hash'], expected_project_hash=original)
    for command in ('compile', 'test'):
        result = runner.run(command + '-during-stage', command)
        assert result['classification'] == 'passed'
        assert result['source_hash'] == changed
        assert result['stage_source_hash'] == original
        assert result['stage_operation'] == operation
        assert flow.source_hash == original
    flow.complete(StepEvidence(stage, operation, source_hash=original, skill_refs=('read',),
                               retrieval_refs=('query',), retrospective_ref='retro'),
                  new_source_hash=changed)
    assert flow.source_hash == changed
    assert flow.pending is None
    assert flow.verified_source is None


def test_noneditable_and_new_inputs_invalidate_receipts_and_validation(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    for name, text in {'main.py': 'pass\n', 'helper.py': 'value = 1\n',
                       'config.json': '{"value":1}'}.items():
        (source / name).write_text(text)
    verifier = tmp_path / 'verify.py'
    verifier.write_text('pass\n')
    runner = PythonExecution.snapshot(tmp_path / 'task', source, verifier,
        commands={'test': [sys.executable, '{verifier}']}, editable_files=['main.py'],
        generated_paths=['results'])
    flow = runner.workflow = Workflow(tmp_path / 'state.json', lambda *a: True,
                                      source_hash=runner.project_hash())
    for index, name in enumerate(('helper.py', 'config.json', 'new_input.data')):
        before = runner.project_hash()
        receipt = runner.run(str(index), 'test')
        assert receipt['classification'] == 'passed'
        flow.state.update(stage=Stage.ACCEPTANCE.value, tested=before, verified=before)
        flow._save()
        with pytest.raises(ValueError):
            runner.edit(name, 'changed', expected_file_hash=None, expected_project_hash=before)
        (runner.project / name).write_text('changed')
        assert runner.project_hash() != before
        with pytest.raises(WorkflowError):
            runner.run(str(index), 'test')
        runner.sync_source()
        assert flow.verified_source is None
        assert flow.stage == Stage.TEST
    before = runner.project_hash()
    (runner.project / 'results').mkdir()
    (runner.project / 'results' / 'output.json').write_text('{"timing":1}')
    assert runner.project_hash() == before


def test_interrupt_reaps_owned_child_and_retains_charged_receipt(tmp_path, monkeypatch):
    import os
    import subprocess
    import time
    runner = backend(tmp_path)
    original = subprocess.Popen
    owned = []

    class InterruptOnce(original):
        interrupted = False

        def communicate(self, *args, **kwargs):
            if not self.interrupted:
                self.interrupted = True
                # Let the real sleep process execute before cancellation.
                time.sleep(.08)
                owned.append(self)
                raise KeyboardInterrupt('user stop')
            return super().communicate(*args, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', InterruptOnce)
    with pytest.raises(KeyboardInterrupt):
        runner.run('cancelled', 'slow')
    assert len(owned) == 1
    assert owned[0].poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(owned[0].pid, 0)
    receipt = runner.receipt('cancelled')
    assert receipt['classification'] == 'interrupted'
    assert receipt['elapsed'] >= .08
    assert runner.workflow.accounting.seconds('execution') >= .08
    assert runner.run('cancelled', 'slow') == receipt
    assert runner.workflow.state['executions'] == 1
    assert len(owned) == 1


def test_minimal_installed_wheel_executes_cpu_backend(tmp_path):
    import subprocess
    import venv
    repository = Path(__file__).resolve().parents[1]
    wheels = tmp_path / 'wheels'
    build = subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps',
                            '--no-build-isolation', '--no-index', '--wheel-dir', str(wheels),
                            str(repository)], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    environment = tmp_path / 'venv'
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / 'bin/python'
    wheel, = wheels.glob('rsi-*.whl')
    site = subprocess.check_output([str(python), '-I', '-c',
        'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip()
    install = subprocess.run([str(python), '-m', 'pip', '--isolated', 'install',
        '--no-deps', '--no-index', '--ignore-installed', '--target', site, str(wheel)],
        capture_output=True, text=True)
    assert install.returncode == 0, install.stdout + install.stderr
    proof = r'''
import json, sys
from pathlib import Path
from rsi.python_execution import PythonExecution
from rsi.runtime import Workflow
try:
    import requests
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('minimal environment unexpectedly has requests')
root = Path.cwd()
source = root / 'input'
source.mkdir()
(source / 'solution.py').write_text('def stable_sum(xs):\n    total = 0.0\n    for x in xs:\n        total += x\n    return total\n')
verifier = root / 'verify.py'
verifier.write_text('from decimal import Decimal\nfrom solution import stable_sum\nxs = [1e16, 1., -1e16]\nexpected = float(sum(map(Decimal.from_float, xs), Decimal(0)))\nassert stable_sum(xs) == expected\n')
runner = PythonExecution.snapshot(root / 'task', source, verifier,
    commands={'test': [sys.executable, '{verifier}']})
runner.workflow = Workflow(root / 'workflow.json', lambda *a: True, source_hash=runner.project_hash())
failed = runner.numeric('minimal-wheel-bug')
assert failed['classification'] == 'numeric_failure', failed
file = runner.read('solution.py')
runner.edit('solution.py', 'import math\ndef stable_sum(xs):\n    return math.fsum(xs)\n',
            expected_file_hash=file['file_hash'], expected_project_hash=file['project_hash'])
result = runner.numeric('minimal-wheel-repair')
assert result['classification'] == 'passed', result
print(json.dumps({'classification': result['classification'], 'tries': runner.workflow.state['executions']}))
'''
    run = subprocess.run([str(python), '-I', '-c', proof], cwd=tmp_path,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout) == {'classification': 'passed', 'tries': 2}
