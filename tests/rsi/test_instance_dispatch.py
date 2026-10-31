"""Prepared-runtime launches use the owned venv and pinned source, without network."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import pytest

from rsi import jobs, pinned_cli
from rsi.instance_dispatch import command_policy, launch_context
from rsi.instance_setup import machinery


@pytest.fixture
def ready(tmp_path, monkeypatch):
    root = tmp_path / 'instance'
    source = root / 'machinery/rsi'
    source.mkdir(parents=True)
    (root / 'config').mkdir()
    (root / 'runs').mkdir()
    (root / 'config/instance.json').write_text('{}')
    original = Path(jobs.__file__).parent
    for name in ('__init__.py', 'jobs.py', 'job_worker.py', 'configuration.py', 'contracts.py',
                 'instance_setup.py', 'instance_dispatch.py', 'pinned_cli.py'):
        shutil.copy2(original / name, source / name)
    observation = '''import importlib.util, json, shutil, subprocess, sys
from pathlib import Path

def observe():
    module = importlib.util.find_spec('_rsi_dispatch_only')
    dependency = None
    if module:
        import _rsi_dispatch_only
        dependency = _rsi_dispatch_only.value
    return dict(prefix=sys.prefix, executable=sys.executable, source=__file__,
                dependency=dependency, global_sdk=importlib.util.find_spec('openai') is not None,
                node=shutil.which('node'),
                node_version=subprocess.check_output(['node','--version'], text=True).strip())
'''
    (source / 'service.py').write_text(observation + '''
def run_job(root, request, configuration, *, should_stop, emit):
    row = observe()
    emit(row)
    return row
''')
    (source / 'services.py').write_text('# No production service needed for the launch fixture.\n')
    (source / 'cli.py').write_text(observation + '''
def main():
    from .pinned_cli import dispatch
    root = Path(sys.argv[sys.argv.index('--instance') + 1])
    result = dispatch(root, sys.argv[1:])
    if result is not None:
        return result
    row = dict(repair=True, prefix=sys.prefix) if sys.argv[-2:] == ['instance', 'setup'] else observe()
    (root / 'cli-observation.json').write_text(json.dumps(row))
    return 0
''')
    manifest = {'machinery': {'kind': 'test', 'revision': 'frozen'}, 'files': {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob('*') if p.is_file()}}
    (root / 'instance.json').write_text(json.dumps(manifest))
    environment = root / 'runtime/python'
    subprocess.run([sys.executable, '-I', '-m', 'venv', '--without-pip', str(environment)],
                   check=True, timeout=60)
    interpreter = environment / 'bin/python'
    purelib = subprocess.check_output([str(interpreter), '-I', '-c',
                                      'import sysconfig; print(sysconfig.get_path("purelib"))'],
                                     text=True, timeout=10).strip()
    # Install a tiny dependency available only in this venv, using a local wheel.
    assert importlib.util.find_spec('_rsi_dispatch_only') is None
    wheel = tmp_path / 'rsi_dispatch_fixture-1.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('_rsi_dispatch_only.py', 'value = 42\n')
        prefix = 'rsi_dispatch_fixture-1.0.dist-info/'
        archive.writestr(prefix + 'METADATA', 'Metadata-Version: 2.1\nName: rsi-dispatch-fixture\nVersion: 1.0\n')
        archive.writestr(prefix + 'WHEEL', 'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        archive.writestr(prefix + 'RECORD', '')
    clean = {k: v for k, v in os.environ.items() if not k.startswith(('PIP_', 'PYTHON'))}
    subprocess.run([sys.executable, '-I', '-m', 'pip', '--isolated', 'install', '--no-index',
                    '--no-deps', '--ignore-installed', '--target', purelib, str(wheel)],
                   check=True, env=clean, cwd=tmp_path, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, timeout=60)
    node = shutil.which('node')
    assert node, 'provide existing Node >=22 on PATH for this test'
    version = subprocess.check_output([node, '--version'], text=True, timeout=10).strip()
    assert int(version[1:].split('.')[0]) >= 22
    private = root / 'runtime/node-bin'
    private.mkdir()
    (private / 'node').symlink_to(Path(node).resolve())
    _, _, digest = machinery(root)
    receipt = dict(state='ready', machinery_hash=digest, python=str(interpreter),
                   node=str(private / 'node'), node_version=version)
    (root / 'runtime/setup.json').write_text(json.dumps(receipt))
    monkeypatch.setenv('PATH', str(tmp_path / 'empty-caller-path'))
    monkeypatch.chdir(tmp_path)
    return root, receipt


def assert_observation(root, receipt, row):
    assert row['prefix'] == str(root / 'runtime/python')
    assert row['executable'] == receipt['python']
    assert Path(row['source']).is_relative_to(root / 'machinery')
    assert row['dependency'] == 42 and not row['global_sdk']
    assert row['node'] == receipt['node'] and row['node_version'] == receipt['node_version']


def test_pinned_cli_reexecutes_once_in_owned_venv_and_receipt_node(ready):
    root, receipt = ready
    assert pinned_cli.dispatch(root, ['--instance', str(root), 'phase1', 'start']) == 0
    assert_observation(root, receipt, json.loads((root / 'cli-observation.json').read_text()))


def test_real_worker_uses_owned_runtime_and_pinned_source(ready):
    root, receipt = ready
    started = jobs.start(root, dict(action='start', mode='phase1', cycle_id='launch-proof'))
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            record = jobs.status(root, started['job_id'])
            if record['state'] in ('completed', 'failed'):
                break
            time.sleep(.02)
        assert record['state'] == 'completed', jobs.evidence(root, started['job_id'])
        assert_observation(root, receipt, record['result'])
    finally:
        jobs.stop(root, started['job_id'])
        os.waitpid(started['pid'], 0)


def test_changed_source_rejected_before_cli_or_worker_dispatch(ready, monkeypatch):
    root, _ = ready
    (root / 'machinery/rsi/service.py').write_text('changed')
    spawned = []
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: spawned.append(a))
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: spawned.append(a))
    with pytest.raises(ValueError, match='integrity'):
        pinned_cli.dispatch(root, ['--instance', str(root), 'phase1', 'start'])
    with pytest.raises(ValueError, match='integrity'):
        jobs.start(root, dict(action='start', mode='phase1'))
    assert not spawned


def test_copied_receipt_cannot_select_original_instance_runtime(ready):
    root, _ = ready
    copied = root.with_name('copied')
    shutil.copytree(root, copied, symlinks=True)
    with pytest.raises(ValueError, match='another instance'):
        pinned_cli.dispatch(copied, ['--instance', str(copied), 'phase1', 'start'])
    with pytest.raises(ValueError, match='another instance'):
        jobs.start(copied, dict(action='start', mode='phase1'))


@pytest.mark.parametrize('state', ['failed', 'installing'])
def test_failed_setup_blocks_execution_but_repair_can_use_caller(ready, state):
    root, receipt = ready
    receipt['state'] = state
    (root / 'runtime/setup.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='instance setup'):
        launch_context(root, required=True)
    _, _, executable, env = launch_context(root, required=True, repair=True)
    assert executable == sys.executable and env['PATH'] == os.environ['PATH']
    assert command_policy(['--instance', str(root), 'instance', 'setup']) == (False, True)


def test_legacy_read_config_works_without_ready_receipt(ready):
    root, _ = ready
    (root / 'runtime/setup.json').unlink()
    assert launch_context(root)[2] == sys.executable
    with pytest.raises(ValueError, match='instance setup'):
        launch_context(root, required=True)
    assert command_policy(['--instance=instance', 'config', 'show']) == (False, False)


def test_actual_pinned_cli_can_enter_setup_repair_with_failed_receipt(ready):
    root, receipt = ready
    receipt['state'] = 'failed'
    (root / 'runtime/setup.json').write_text(json.dumps(receipt))
    assert pinned_cli.dispatch(root, ['--instance', str(root), 'instance', 'setup']) == 0
    row = json.loads((root / 'cli-observation.json').read_text())
    assert row == {'repair': True, 'prefix': sys.prefix}


@pytest.mark.parametrize('shared_node_python_bin', [False, True])
def test_env_python_helpers_use_owned_dependencies_with_user_site_disabled(
        ready, tmp_path, monkeypatch, shared_node_python_bin):
    root, receipt = ready
    caller = tmp_path / 'caller-bin'
    caller.mkdir()
    for name in ('python', 'python3', 'node'):
        executable = caller / name
        executable.write_text('#!/bin/sh\nexit 97\n')
        executable.chmod(0o700)
    if shared_node_python_bin:
        for name in ('python', 'python3'):
            executable = Path(receipt['node']).parent / name
            executable.write_text('#!/bin/sh\nexit 98\n')
            executable.chmod(0o700)
    monkeypatch.setenv('PATH', str(caller))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path / 'foreign-packages'))
    monkeypatch.setenv('PYTHONHOME', str(tmp_path / 'foreign-home'))
    _, _, _, env = launch_context(root, required=True)
    helper = tmp_path / 'env-python-helper'
    helper.write_text('#!/usr/bin/env python3\n'
        'import json, site, sys, _rsi_dispatch_only\n'
        'print(json.dumps(dict(prefix=sys.prefix, dependency=_rsi_dispatch_only.value, '
        'user_site=site.ENABLE_USER_SITE)))\n')
    helper.chmod(0o700)
    row = json.loads(subprocess.check_output([str(helper)], env=env, text=True, timeout=10))
    assert row == dict(prefix=str(root / 'runtime/python'), dependency=42, user_site=False)
    assert shutil.which('node', path=env['PATH']) == receipt['node']
    assert subprocess.check_output(['node', '--version'], env=env, text=True,
                                   timeout=10).strip() == receipt['node_version']
    # Re-entering the launch context keeps owned executable precedence stable.
    monkeypatch.setenv('PATH', env['PATH'])
    assert launch_context(root, required=True)[3]['PATH'] == env['PATH']
