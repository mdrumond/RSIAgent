"""Select an instance-owned runtime after verifying its pinned machinery."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def launch_context(instance, *, required=False, repair=False):
    """Return root, pinned source, interpreter, environment; never resolve venv Python.

    Read/configuration operations can inspect or repair an unprepared instance.
    Execution requires a ready receipt matching this exact instance and source.
    """
    from .instance_setup import machinery
    root, source, digest = machinery(instance)
    env = dict(os.environ)
    caller = (root, source, sys.executable, env)
    if repair:
        return caller
    path = root / 'runtime/setup.json'
    try:
        receipt = json.loads(path.read_text())
        if receipt.get('state') != 'ready':
            raise ValueError('setup receipt is not ready')
        if receipt.get('machinery_hash') != digest:
            raise ValueError('setup receipt does not match pinned machinery')
        interpreter = root / 'runtime/python/bin/python'
        if receipt.get('python') != str(interpreter):
            raise ValueError('setup receipt belongs to another instance or Python environment')
        if interpreter.parent.resolve() != interpreter.parent:
            raise ValueError('instance Python environment directory changed')
        if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            raise ValueError('instance Python environment is missing')
        node = Path(receipt['node'])
        if not node.is_absolute() or not node.is_file() or not os.access(node, os.X_OK):
            raise ValueError('receipt Node executable is missing')
        version = subprocess.check_output([str(node), '--version'], text=True,
                                          stderr=subprocess.DEVNULL, timeout=10).strip()
        if version != receipt.get('node_version') or not version.startswith('v') or int(version[1:].split('.')[0]) < 22:
            raise ValueError('receipt Node identity/version changed; Node >=22 is required')
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        if not required:
            return caller
        raise ValueError(f'instance runtime unavailable: {exc}; run instance setup to repair') from None
    # Do not resolve this executable: its lexical venv location selects sys.prefix.
    # CPL and other env-python helpers must select the same owned environment.
    # Keep the receipt's Node ahead of any executable in the Python bin directory.
    prefixes = [str(node.parent), str(interpreter.parent)]
    inherited = env.get('PATH', os.defpath).split(os.pathsep)
    env['PATH'] = os.pathsep.join(prefixes + [p for p in inherited if p not in prefixes])
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env['PYTHONNOUSERSITE'] = '1'
    return root, source, str(interpreter), env


def command_policy(argv):
    """Classify parsed CLI argv without confusing an instance pathname for a command."""
    words, index = [], 0
    while index < len(argv):
        word = argv[index]
        if word == '--instance':
            index += 2
            continue
        if not word.startswith('--instance='):
            words.append(word)
        index += 1
    command, action = (words + [None, None])[:2]
    repair = (command, action) == ('instance', 'setup')
    required = (command in ('phase1', 'phase2', 'inference') and action != 'stop') or command == 'kdb-add-repo' or (
        command == 'skills' and action == 'tune') or (command == 'dashboard' and action == 'start')
    return required, repair
