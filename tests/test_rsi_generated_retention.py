"""Selected generated evidence survives failures, replay and lossless export."""
import base64
import io
import json
import shutil
import sys
import zipfile

import pytest

from rsi.contracts import BudgetLimits, ProjectSpec
from rsi.evidence_io import _references, export_task
from rsi.project_runner import ProjectConfig, ProjectRunner
from rsi.python_execution import CommandResult, PythonExecution, atomic_json, digest
from rsi.runtime import Workflow, WorkflowError
from rsi.skills import builtin_bundle


def diagnostic_archive():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr(zipfile.ZipInfo('compiler.log', date_time=(2026, 1, 1, 0, 0, 0)),
                         b'\x00\xffcompiler diagnostic\x00')
    return stream.getvalue()


BINARY = diagnostic_archive()


def execution(tmp_path, *, retained_paths=('results',), generated_paths=('scratch',), exitcode=0, executor_identity=None):
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'solution.py').write_text(
        'from pathlib import Path\nimport sys\n'
        'Path("results").mkdir(exist_ok=True)\n'
        'Path("results/report.json").write_text(\'{"observed": 42}\')\n'
        f'Path("results/diagnostics.zip").write_bytes({BINARY!r})\n'
        'Path("results/__pycache__").mkdir(exist_ok=True)\n'
        'Path("results/__pycache__/ignored.pyc").write_bytes(b"cache bytes")\n'
        'Path("scratch").mkdir(exist_ok=True)\n'
        'with Path("scratch/calls").open("a") as stream: stream.write("call\\n")\n'
        f'sys.exit({exitcode})\n')
    verifier = source / 'verify.py'
    verifier.write_text('raise AssertionError("independent verifier")\n')
    runner = PythonExecution.snapshot(tmp_path / 'task', source, verifier,
        commands={'compile': [sys.executable, 'solution.py']},
        generated_paths=generated_paths, retained_paths=retained_paths, executor_identity=executor_identity)
    runner.workflow = Workflow(runner.root / 'workflow.json', lambda *a: True,
        source_hash=runner.project_hash(), limits=BudgetLimits(actor_seconds=20))
    # Canonical export envelope; no model or external service is needed.
    atomic_json(runner.root / 'runner.json', {'contract': runner.manifest['contract']})
    atomic_json(runner.root / 'events.json', [])
    return runner


@pytest.mark.parametrize('exitcode', [0, 7])
def test_success_and_failure_outputs_retain_reference_closure_after_deletion(tmp_path, exitcode):
    runner = execution(tmp_path, exitcode=exitcode)
    original = runner.project_hash()
    receipt = runner.run('command', 'compile')
    assert receipt['classification'] == ('passed' if exitcode == 0 else 'compile_failure')
    assert receipt['returncode'] == exitcode
    assert runner.project_hash() == original
    manifest_ref, = receipt['artifact_refs']
    manifest = runner.store.read_json(manifest_ref)
    files = {row['path']: row for row in manifest['files']}
    assert set(files) == {'results/report.json', 'results/diagnostics.zip'}
    assert manifest['operation_id'] == 'command'
    assert manifest['source_hash'] == original
    assert manifest['classification'] == receipt['classification']
    assert manifest['missing_paths'] == []
    assert runner.store.read_bytes(files['results/diagnostics.zip']['reference']) == BINARY
    assert files['results/diagnostics.zip']['bytes'] == len(BINARY)
    assert json.loads(runner.store.read_bytes(files['results/report.json']['reference'])) == {'observed': 42}
    assert set(_references(manifest)) == {row['reference'] for row in files.values()}
    assert manifest_ref in set(_references(receipt))

    shutil.rmtree(runner.project / 'results')
    reopened = PythonExecution(runner.root, runner.workflow)
    assert reopened.run('command', 'compile') == receipt
    assert (runner.project / 'scratch/calls').read_text().splitlines() == ['call']
    export = tmp_path / 'export'
    export_task(runner.root, export)
    rows = [json.loads(line) for line in (export / 'evidence.jsonl').read_text().splitlines()]
    objects = {row['id']: base64.b64decode(row['data']) for row in rows if row['kind'] == 'artifact'}
    assert objects[files['results/diagnostics.zip']['reference']] == BINARY
    with zipfile.ZipFile(io.BytesIO(objects[files['results/diagnostics.zip']['reference']])) as archive:
        assert archive.read('compiler.log') == b'\x00\xffcompiler diagnostic\x00'
    assert json.loads(objects[manifest_ref]) == manifest
    assert runner.verifier.read_bytes() not in objects.values()
    assert (runner.project / 'solution.py').read_bytes() not in objects.values()
    assert b'cache bytes' not in objects.values()


def test_specific_file_and_missing_selection_do_not_collect_siblings(tmp_path):
    runner = execution(tmp_path, retained_paths=('results/report.json', 'results/absent.json'),
                       generated_paths=('results', 'scratch'))
    receipt = runner.run('selected-file', 'compile')
    manifest = runner.store.read_json(receipt['artifact_refs'][0])
    assert [row['path'] for row in manifest['files']] == ['results/report.json']
    assert manifest['missing_paths'] == ['results/absent.json']
    assert not runner.store._path(digest(BINARY)).exists()


def test_empty_default_keeps_generated_cache_unretained(tmp_path):
    runner = execution(tmp_path, retained_paths=(), generated_paths=('results', 'scratch'))
    receipt = runner.run('default', 'compile')
    assert receipt['classification'] == 'passed'
    assert 'artifact_refs' not in receipt
    assert 'retained_paths' not in runner.contract
    assert not runner.store._path(digest(BINARY)).exists()


@pytest.mark.parametrize('selection', ['../frozen', '/tmp/report', '.', '__pycache__',
                                      'results/.pytest_cache', 'solution.py', 'verify.py'])
def test_snapshot_rejects_source_verifier_cache_and_escaping_selections(tmp_path, selection):
    with pytest.raises(ValueError):
        execution(tmp_path, retained_paths=(selection,))
    assert not (tmp_path / 'task').exists()


@pytest.mark.parametrize('directory_link', [False, True])
def test_symlink_output_does_not_leak_verifier_and_stays_pending(tmp_path, directory_link):
    runner = execution(tmp_path)
    # The command exits without writing outputs: an unsafe existing selection
    # must fail retention rather than becoming a normal completed receipt.
    (runner.project / 'solution.py').write_text('pass\n')
    outputs = runner.project / 'results'
    outputs.mkdir()
    if directory_link:
        (outputs / 'linked').symlink_to(runner.root / 'frozen', target_is_directory=True)
    else:
        (outputs / 'verifier.py').symlink_to(runner.verifier)
    with pytest.raises(ValueError, match='symlink'):
        runner.run('unsafe', 'compile')
    assert not runner.store._path(digest(runner.verifier.read_bytes())).exists()
    with pytest.raises(WorkflowError, match='never replayed'):
        runner.run('unsafe', 'compile')


def test_project_config_freezes_selection_and_actor_cannot_edit_output(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'solution.py').write_text('pass\n')
    verifier = tmp_path / 'verify.py'
    verifier.write_text('pass\n')
    config = ProjectConfig.from_dict(dict(source=str(source), verifier=str(verifier),
        commands={name: [sys.executable, '{verifier}'] for name in ('compile', 'test', 'validation')},
        retained_paths=['reports']))
    assert config.retained_paths == ('reports',)
    runner = ProjectRunner.create(tmp_path / 'task',
        ProjectSpec('retention', 'Retain diagnostic evidence', (sys.executable, 'solution.py')),
        builtin_bundle(), config, query=lambda *a: None, backend_identity='TEST-retention',
        session_factory=lambda: None, test_backend=True)
    assert runner.backend.contract['retained_paths'] == ['reports']
    frozen = runner.store.read_json(runner.contract_ref)
    assert frozen['config']['retained_paths'] == ['reports']
    reopened = ProjectRunner(runner.root, query=lambda *a: None, backend_identity='TEST-retention',
        session_factory=lambda: None, test_backend=True)
    assert reopened.config.retained_paths == ('reports',)
    with pytest.raises(ValueError, match='editable manifest'):
        reopened.backend.edit('reports/trace.txt', 'actor-created',
            expected_file_hash=None, expected_project_hash=reopened.backend.project_hash())


def test_output_paths_are_normalized_before_source_identity_exclusion(tmp_path):
    runner = execution(tmp_path, retained_paths=('./results//',))
    original = runner.project_hash()
    result = runner.run('normalized', 'compile')
    assert result['classification'] == 'passed'
    assert result['current_hash'] == original
    assert runner.contract['retained_paths'] == ['results']


@pytest.mark.parametrize('unknown', [False, True])
def test_target_outputs_are_collected_only_after_terminal_observation(tmp_path, unknown):
    runner = execution(tmp_path, executor_identity='test-output-executor')
    calls = []

    class Executor:
        def execute(self, request, *, observe, remaining):
            calls.append(request.operation_id)
            observe('command', handle='remote:test:job:retained-output')
            outputs = request.project / 'results'
            outputs.mkdir()
            (outputs / 'diagnostics.zip').write_bytes(BINARY)
            if unknown:
                raise ConnectionError('terminal state not observed')
            return CommandResult(3)

    runner = PythonExecution(runner.root, runner.workflow, executor_factory=lambda _: Executor())
    if unknown:
        with pytest.raises(ConnectionError):
            runner.run('target', 'compile')
        pending = json.loads(runner._receipt_path('target').read_text())
        assert pending['status'] == 'pending'
        assert 'artifact_refs' not in pending
        assert not runner.store._path(digest(BINARY)).exists()
        with pytest.raises(WorkflowError, match='never replayed'):
            runner.run('target', 'compile')
    else:
        receipt = runner.run('target', 'compile')
        assert receipt['classification'] == 'compile_failure'
        manifest = runner.store.read_json(receipt['artifact_refs'][0])
        assert runner.store.read_bytes(manifest['files'][0]['reference']) == BINARY
        assert receipt['observations'][0]['handle'] == 'remote:test:job:retained-output'
    assert calls == ['target']
    assert not (runner.project / 'scratch/calls').exists()
