"""Install a pinned instance into its own environments; never charge actor time."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

from .configuration import _bytes, _locations, _publish, show_configuration, set_configuration


def machinery(instance):
    root, _ = _locations(instance)
    manifest = json.loads((root / 'instance.json').read_text())
    source = root / 'machinery'
    if source.resolve() != source:
        raise ValueError('machinery must be the pinned instance directory')
    inventory = {key: value for key, value in manifest['files'].items()
                 if key.startswith('machinery/')}
    if not inventory:
        raise ValueError('instance has no machinery inventory; bootstrap a current release')
    for relative, digest in inventory.items():
        path = root / relative
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'pinned machinery integrity failure: {relative}')
    digest = hashlib.sha256(_bytes(inventory)).hexdigest()
    return root, source, digest


def run(command, *, cwd, env, timeout):
    """Bound setup work and reap its entire owned group on interruption/timeout."""
    child = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    try:
        code = child.wait(timeout=timeout)
        if code:
            raise ValueError(f'setup command {Path(command[0]).name} failed (exit {code})')
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        raise


def _wheel(source, target):
    """Reconstitute the bootstrap's installed-wheel inventory without rebuilding."""
    metadata = list(source.glob('rsi-*.dist-info'))
    if len(metadata) != 1:
        raise ValueError('snapshot lacks source packaging or RSI wheel metadata')
    stem = metadata[0].name.removesuffix('.dist-info')
    wheel = target / f'{stem}-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(source))
    return wheel


def setup(instance, *, python=None, node=None, roles=True, embeddings=False,
          offline=False, pip_cache=None, npm_cache=None, timeout=900, runner=run):
    """Explicit reruns repair a partial setup; no provider credentials are recorded."""
    if timeout <= 0:
        raise ValueError('setup timeout must be positive')
    root, source, digest = machinery(instance)
    runtime = root / 'runtime'
    runtime.mkdir(exist_ok=True)
    if runtime.resolve() != runtime:
        raise ValueError('runtime directory escapes instance')
    lock = os.open(runtime, os.O_RDONLY | os.O_DIRECTORY)
    dashboard_lock = None
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            dashboard = root / 'dashboard'
            if dashboard.resolve() != dashboard:
                raise ValueError('dashboard directory escapes instance')
            dashboard_lock = os.open(dashboard, os.O_RDONLY | os.O_DIRECTORY)
            fcntl.flock(dashboard_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('setup or dashboard is active; stop it before changing dependencies') from None
        settings = show_configuration(root)['payload'].get('runtime', {})
        python = python or settings.get('python') or sys.executable
        node = Path(node or settings.get('node') or shutil.which('node') or '')
        if not node.is_file():
            raise ValueError('install Node >=22 externally and pass --node /absolute/path/node')
        node = node.resolve()
        version = subprocess.check_output([str(node), '--version'], text=True, timeout=10).strip()
        if not version.startswith('v') or int(version[1:].split('.')[0]) < 22:
            raise ValueError('Pi requires Node >=22; Node installation is operator-managed')
        npm = node.parent / 'npm'
        if not npm.is_file():
            raise ValueError('the supplied Node installation must include npm alongside node')
        env = dict(os.environ, PATH=str(node.parent) + os.pathsep + os.environ.get('PATH', ''),
                   PYTHONNOUSERSITE='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
        env = {key: value for key, value in env.items()
               if not key.startswith(('PIP_', 'PYTHON'))}
        env.update(PYTHONNOUSERSITE='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
        started = time.monotonic()
        receipt = {'state': 'installing', 'machinery_hash': digest, 'node': str(node),
                   'node_version': version, 'actor_seconds': 0, 'embeddings': embeddings,
                   'roles': roles, 'offline': offline}
        receipt_path = runtime / 'setup.json'
        _publish(receipt_path, _bytes(receipt))

        def invoke(argv):
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError('instance setup elapsed budget exhausted')
            runner([str(item) for item in argv], cwd=runtime, env=env, timeout=remaining)

        def pip(interpreter, *arguments):
            purelib = subprocess.check_output(
                [str(interpreter), '-I', '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'],
                text=True, env=env, cwd=runtime, timeout=10).strip()
            if not Path(purelib).resolve().is_relative_to(Path(interpreter).parent.parent.parent):
                raise ValueError('venv install target escapes owned environment')
            options = ['--ignore-installed', '--target', purelib, '--timeout', '30', '--retries', '1']
            if offline:
                options += ['--no-index']
                if pip_cache:
                    options += ['--find-links', str(Path(pip_cache).resolve())]
            elif pip_cache:
                options += ['--cache-dir', str(Path(pip_cache).resolve())]
            invoke([interpreter, '-I', '-m', 'pip', '--isolated', 'install', *options, *arguments])

        try:
            host = runtime / 'python' / 'bin' / 'python'
            invoke([python, '-m', 'venv', runtime / 'python'])
            release = source if (source / 'pyproject.toml').is_file() else _wheel(source, runtime)
            extras = 'legacy,embeddings' if embeddings else 'legacy'
            pip(host, str(release) + '[' + extras + ']')
            if roles:
                role_requirements = source / 'requirements-roles.txt'
                role_python = runtime / 'roles' / 'bin' / 'python'
                invoke([python, '-m', 'venv', runtime / 'roles'])
                invoke([role_python, '-c', 'import sys; assert (3,12) <= sys.version_info[:2] < (3,14), "NOOA needs Python 3.12 or 3.13"'])
                if role_requirements.is_file():
                    pip(role_python, '-r', role_requirements)
                else:
                    pip(role_python, str(release) + '[roles]')
                invoke([role_python, '-I', '-c', 'import nooa; from pathlib import Path; import sys; assert Path(nooa.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())'])
            invoke([npm, 'ci', '--prefix', source / 'integrations/pi', '--fetch-timeout=30000',
                    '--fetch-retries=1', *(['--offline'] if offline else []),
                    *(['--cache', str(Path(npm_cache).resolve())] if npm_cache else [])])
            invoke([host, '-I', '-c', 'import rsi, flask; import dashboard.app; from pathlib import Path; import sys; assert all(Path(m.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()) for m in (rsi, flask)); from importlib.metadata import version; print(version("rsi"))'])
            invoke([source / 'integrations/pi/node_modules/.bin/pi', '--version'])
            if embeddings:
                invoke([host, '-I', '-c', 'import torch, transformers; from pathlib import Path; import sys; assert all(Path(m.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()) for m in (torch, transformers))'])
            probe = 'import importlib.metadata as m, json, sys; print(json.dumps({"python": sys.version, "packages": {k:m.version(k) for k in ("rsi", "Flask", "Pygments", "openai")}}))'
            receipt['versions'] = json.loads(subprocess.check_output(
                [str(host), '-I', '-c', probe], cwd=runtime, env=env, text=True,
                timeout=min(30, max(.01, timeout - (time.monotonic() - started)))))
            if roles:
                current = show_configuration(root)
                if not current['payload'].get('roles_python'):
                    payload = dict(current['payload'], roles_python=str(runtime / 'roles/bin/python'))
                    revision = set_configuration(root, payload, expect_revision=current['revision'])
                    receipt['configuration_revision'] = revision['revision']
            receipt.update(state='ready', python=str(host),
                           roles_python=str(runtime / 'roles/bin/python') if roles else None)
        except BaseException as exc:
            receipt.update(state='failed', error_type=type(exc).__name__)
            raise
        finally:
            receipt['setup_seconds'] = time.monotonic() - started
            _publish(receipt_path, _bytes(receipt))
        return receipt
    finally:
        if dashboard_lock is not None:
            os.close(dashboard_lock)
        os.close(lock)
