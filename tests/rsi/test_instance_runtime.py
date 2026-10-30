"""Runtime setup contracts and actual loopback binding, without network installs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import zipfile

import pytest

from rsi import instance_setup, dashboard_runtime, runtime_cli


def instance(tmp_path):
    root = tmp_path / 'instance'
    source = root / 'machinery'
    source.mkdir(parents=True)
    for directory in ('config', 'runtime', 'dashboard'):
        (root / directory).mkdir()
    (source / 'pyproject.toml').write_text('[project]\nname="rsi"\nversion="0.1"\n')
    (source / 'requirements-roles.txt').write_text('nooa==0.0.10\n')
    manifest = {'files': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in source.rglob('*') if p.is_file()}}
    (root / 'instance.json').write_text(json.dumps(manifest))
    (root / 'config/instance.json').write_text('{}')
    return root


def setup_spy(monkeypatch, root, fail=None):
    commands = []
    node = root / 'node'
    node.write_text('node')
    (root / 'npm').write_text('npm')
    def output(command, **kwargs):
        if command[1] == '--version':
            return 'v22.23.3\n'
        if 'import importlib.metadata' in command[-1]:
            return json.dumps({'python': '3.12', 'packages': {'rsi': '0.1.0'}})
        return str(Path(command[0]).parent.parent / 'lib/python3.12/site-packages') + '\n'
    monkeypatch.setattr(instance_setup.subprocess, 'check_output', output)
    def runner(command, **kwargs):
        commands.append((command, kwargs))
        if fail and fail(command):
            raise ValueError('install failed')
    return node, commands, runner


def test_isolated_install_targets_and_offline_caches(tmp_path, monkeypatch):
    root = instance(tmp_path)
    node, commands, runner = setup_spy(monkeypatch, root)
    monkeypatch.setenv('PIP_TARGET', '/wrong/location')
    monkeypatch.setenv('PYTHONPATH', '/ambient/source')
    receipt = instance_setup.setup(root, node=node, offline=True, pip_cache=tmp_path,
                                   npm_cache=tmp_path, runner=runner)
    assert receipt['state'] == 'ready' and receipt['actor_seconds'] == 0
    pip = [cmd for cmd, _ in commands if 'pip' in cmd]
    assert len(pip) == 2
    for cmd in pip:
        assert '--isolated' in cmd and '--ignore-installed' in cmd and '--no-index' in cmd
        target = Path(cmd[cmd.index('--target') + 1])
        assert target.is_relative_to(root / 'runtime')
    for _, call in commands:
        assert 'PIP_TARGET' not in call['env'] and 'PYTHONPATH' not in call['env']
        assert 0 < call['timeout'] <= 900
    npm = next(cmd for cmd, _ in commands if cmd[0].endswith('/npm'))
    assert '--offline' in npm and npm[npm.index('--cache') + 1] == str(tmp_path)


def test_failed_install_retains_failure_not_ready(tmp_path, monkeypatch):
    root = instance(tmp_path)
    node, _, runner = setup_spy(monkeypatch, root, lambda cmd: 'pip' in cmd)
    with pytest.raises(ValueError, match='install failed'):
        instance_setup.setup(root, node=node, runner=runner)
    receipt = json.loads((root / 'runtime/setup.json').read_text())
    assert receipt['state'] == 'failed' and receipt['actor_seconds'] == 0
    assert receipt['error_type'] == 'ValueError'


def test_changed_machinery_blocks_setup_before_execution(tmp_path, monkeypatch):
    root = instance(tmp_path)
    (root / 'machinery/pyproject.toml').write_text('changed')
    node, commands, runner = setup_spy(monkeypatch, root)
    with pytest.raises(ValueError, match='integrity'):
        instance_setup.setup(root, node=node, runner=runner)
    assert not commands


def test_installed_inventory_reconstructed_as_installable_wheel(tmp_path):
    source = tmp_path / 'installed'
    metadata = source / 'rsi-0.1.0.dist-info'
    metadata.mkdir(parents=True)
    (metadata / 'METADATA').write_text('Name: rsi\nVersion: 0.1.0\n')
    (metadata / 'WHEEL').write_text('Wheel-Version: 1.0\nTag: py3-none-any\n')
    (source / 'rsi').mkdir()
    (source / 'rsi/__init__.py').write_text('value = 42\n')
    wheel = instance_setup._wheel(source, tmp_path)
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read('rsi/__init__.py') == b'value = 42\n'
        assert 'rsi-0.1.0.dist-info/WHEEL' in archive.namelist()


def test_runtime_registration_dispatches_supplied_setup_options(monkeypatch, tmp_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=Path)
    commands = parser.add_subparsers(dest='command')
    actions = commands.add_parser('instance').add_subparsers(dest='action')
    runtime_cli.register(commands, actions)
    args = parser.parse_args(['--instance', str(tmp_path), 'instance', 'setup', '--offline',
                              '--no-roles', '--node', '/node', '--embeddings'])
    observed = {}
    monkeypatch.setattr(instance_setup, 'setup', lambda root, **kw: observed.update(root=root, **kw))
    args.runtime_handler(args)
    assert observed['root'] == tmp_path and observed['node'] == Path('/node')
    assert observed['offline'] and observed['embeddings'] and not observed['roles']


def test_loopback_binds_port_zero_and_retains_actual_endpoint(tmp_path, monkeypatch):
    from flask import Flask
    import dashboard.app
    import werkzeug.serving
    import urllib.request
    root = instance(tmp_path)
    directory = root / 'dashboard'
    (directory / 'process.json').write_text(json.dumps(
        {'launch_id': 'test', 'state': 'launching', 'requested_port': 0}))
    app = Flask(__name__)
    app.add_url_rule('/', view_func=lambda: 'instance dashboard')
    monkeypatch.setattr(dashboard.app, 'create_app', lambda *_: app)
    make_server = werkzeug.serving.make_server
    responses = []
    def server(host, port, application, **kwargs):
        real = make_server(host, port, application, **kwargs)
        def once(**_):
            import threading
            reader = threading.Thread(target=lambda: responses.append(
                urllib.request.urlopen(f'http://127.0.0.1:{real.server_port}', timeout=3).read()))
            reader.start()
            real.handle_request()
            reader.join(timeout=3)
        real.serve_forever = once
        return real
    monkeypatch.setattr(werkzeug.serving, 'make_server', server)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    previous = signal.getsignal(signal.SIGTERM)
    dashboard_runtime.serve(str(root), 'test', str(descriptor))
    receipt = json.loads((directory / 'process.json').read_text())
    assert receipt['state'] == 'stopped' and receipt['port'] > 0
    assert receipt['url'] == f'http://127.0.0.1:{receipt["port"]}'
    assert responses == [b'instance dashboard']
    assert signal.getsignal(signal.SIGTERM) == previous
