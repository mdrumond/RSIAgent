"""Instance input revisions and read-only node folder browsing.

A revision freezes configuration inputs, not the contents of referenced folders.
Cycle runners must snapshot the returned revision/payload at task boundaries.
Unknown JSON settings are retained for future components; known contracts are
validated here without importing providers or launching any execution.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from .contracts import BudgetLimits, ProjectSpec


_COMPAT = ('machinery', 'kdb', 'skills', 'skill_map', 'dashboard')


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def _snapshot(payload):
    data = _bytes(payload)
    return {'revision': hashlib.sha256(data).hexdigest(),
            'payload': json.loads(data)}


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be an object')


def _reference(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonblank reference')


def validate_configuration(payload):
    """Validate known sections, preserving advanced JSON settings unchanged."""
    _object(payload, 'configuration')
    for name in ('machinery', 'kdb', 'skills', 'skill_map', 'baseline_kdb',
                 'baseline_skills', 'skill_map_ref', 'roles_python'):
        if name in payload and not (name == 'roles_python' and payload[name] is None):
            _reference(payload[name], name)
    if 'budgets' in payload:
        BudgetLimits.from_dict(payload['budgets'])
    for name in ('phase1_projects', 'phase2_projects'):
        if name not in payload:
            continue
        projects = payload[name]
        if not isinstance(projects, list):
            raise ValueError(f'{name} must be an array')
        parsed = [ProjectSpec.from_dict(project) for project in projects]
        if len({project.project_id for project in parsed}) != len(parsed):
            raise ValueError(f'{name}: project IDs must be unique')
    for name in ('prompts', 'provider', 'roles', 'recovery', 'tuning', 'dashboard'):
        if name in payload:
            _object(payload[name], name)
    for name in ('curriculum', 'actor', 'waves', 'skillwriter'):
        prompts = payload.get('prompts', {})
        if name in prompts and not isinstance(prompts[name], str):
            raise ValueError(f'prompts.{name} must be text')
    for name in ('profile_id', 'model'):
        if name in payload.get('provider', {}):
            _reference(payload['provider'][name], f'provider.{name}')
    if 'browser_roots' in payload:
        roots = payload['browser_roots']
        if not isinstance(roots, list) or not roots:
            raise ValueError('browser_roots must be a nonempty array')
        for root in roots:
            _reference(root, 'browser_roots entry')
    _bytes(payload)  # Reject non-JSON values and nonfinite numbers, including advanced settings.


def _locations(instance):
    if instance is None:
        raise ValueError('--instance PATH is required')
    root = Path(instance).expanduser().resolve(strict=True)
    if not root.is_dir() or not (root / 'instance.json').is_file():
        raise ValueError('not an RSI instance')
    config = (root / 'config').resolve(strict=True)
    if not config.is_relative_to(root) or not config.is_dir():
        raise ValueError('configuration directory escapes instance')
    return root, config


def _inside(config, relative):
    path = (config / relative).resolve()
    if not path.is_relative_to(config):
        raise ValueError('configuration path escapes instance')
    return path


def _defaults(bootstrap):
    return {
        'phase1_projects': [], 'phase2_projects': [],
        'prompts': {'curriculum': '', 'actor': '', 'waves': '', 'skillwriter': ''},
        'provider': {'profile_id': 'deepseek-flash-native-v1', 'model': 'deepseek-flash'},
        'roles': {}, 'roles_python': None, 'budgets': BudgetLimits().to_dict(),
        'recovery': {'enabled': True},
        'tuning': {'every_projects': 5, 'at_wave_end': True,
                   'heldout_projects': 5, 'runs_per_arm': 2},
        'browser_roots': ['.'], **bootstrap,
    }


def _read(config):
    pointer = _inside(config, 'current.json')
    if not pointer.exists():
        payload = _defaults(json.loads(_inside(config, 'instance.json').read_text()))
        validate_configuration(payload)
        return _snapshot(payload)
    revision = json.loads(pointer.read_text())['revision']
    if not isinstance(revision, str) or not re.fullmatch('[0-9a-f]{64}', revision):
        raise ValueError('invalid configuration revision')
    payload = json.loads(_inside(config, f'revisions/{revision}.json').read_text())
    validate_configuration(payload)
    snapshot = _snapshot(payload)
    if snapshot['revision'] != revision:
        raise ValueError('configuration revision integrity failure')
    return snapshot


def show_configuration(instance):
    """Return detached revision+payload; never create state, locks or defaults."""
    _, config = _locations(instance)
    return _read(config)


def _publish(path, data, *, immutable=False):
    descriptor, temporary = tempfile.mkstemp(prefix='.config-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError('existing configuration revision differs')
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def set_configuration(instance, payload, *, expect_revision=None):
    """Publish full input settings, inheriting omitted bootstrap compatibility keys.

    Lock the existing directory inode: conflict/invalid writes need no lock file.
    Existing revisions are never rewritten. Readers see either complete pointer.
    """
    _, config = _locations(instance)
    validate_configuration(payload)
    lock = os.open(config, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = _read(config)
        if expect_revision is not None and previous['revision'] != expect_revision:
            raise ValueError('configuration revision conflict')
        merged = {key: previous['payload'][key] for key in _COMPAT
                  if key in previous['payload']}
        merged.update(payload)
        merged = _defaults(merged)
        validate_configuration(merged)
        current = _snapshot(merged)
        directory = _inside(config, 'revisions')
        directory.mkdir(exist_ok=True)
        for snapshot in (previous, current):
            path = _inside(config, f"revisions/{snapshot['revision']}.json")
            _publish(path, _bytes(snapshot['payload']), immutable=True)
        _publish(_inside(config, 'current.json'), _bytes({'revision': current['revision']}))
        return current
    finally:
        os.close(lock)


def list_folders(instance, path=None):
    """List directories only, with symlink targets confined to browser roots."""
    root, _ = _locations(instance)
    snapshot = show_configuration(root)
    def resolve(value):
        candidate = Path(value).expanduser()
        return (candidate if candidate.is_absolute() else root / candidate).resolve(strict=True)
    roots = [resolve(value) for value in snapshot['payload'].get('browser_roots', ['.'])]
    if any(not item.is_dir() for item in roots):
        raise ValueError('browser roots must be existing directories')
    selected = resolve(path) if path is not None else roots[0]
    def allowed(candidate):
        return any(candidate.is_relative_to(item) for item in roots)
    if not selected.is_dir() or not allowed(selected):
        raise ValueError('folder is outside browser roots or is not a directory')
    folders = []
    for child in sorted(selected.iterdir(), key=lambda item: item.name):
        try:
            target = child.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if target.is_dir() and allowed(target):
            folders.append({'name': child.name, 'path': str(target)})
    return {'revision': snapshot['revision'], 'path': str(selected),
            'roots': [str(item) for item in roots], 'folders': folders}
