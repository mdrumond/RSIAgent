"""Atomic, independent snapshots of RSI machinery and explicit baseline inputs.

Bootstrap requires Linux with libc renameat2(RENAME_NOREPLACE). Publication
fails cleanly when this operation is unavailable; no weaker rename is used.

From a Git checkout, machinery is copied from committed HEAD, including this
module. Commit machinery changes before bootstrapping; uncommitted and untracked
files are intentionally absent. Installed wheels use their release inventory.
No dashboard is started by bootstrap.
"""
from __future__ import annotations

import ctypes
import errno
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile


_BASELINE_EXCLUDED = {
    '.git', '.env', '.ssh', '.aws', '.venv', 'venv', '__pycache__',
    '.cache', '.pytest_cache', 'node_modules', 'credentials', 'credentials.json',
    '.credentials', 'secrets', 'secrets.json', 'secrets.yaml',
}
_RELEASE_EXCLUDED = _BASELINE_EXCLUDED | {
    '.agent-state', '.codex', '.agents', 'runs', 'artifacts', 'logs', 'history',
}


def _allowed(path: Path, *, baseline=False) -> bool:
    excluded = _BASELINE_EXCLUDED if baseline else _RELEASE_EXCLUDED
    return not any(part in excluded or part.startswith('.env.')
                   or part.endswith(('.pem', '.key', '.pyc')) for part in path.parts)


def _sqlite_sidecar(path: Path) -> bool:
    for suffix in ('-wal', '-shm', '-journal'):
        if path.name.endswith(suffix):
            database = path.with_name(path.name[:-len(suffix)])
            if database.is_file():
                with database.open('rb') as stream:
                    return stream.read(16) == b'SQLite format 3\x00'
    return False


def _git(root, *args):
    result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, check=True)
    return result.stdout


def _copy_file(source: Path, destination: Path):
    if source.is_symlink():
        raise ValueError(f'symlink baseline files are unsupported: {source.name}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as stream:
        sqlite = stream.read(16) == b'SQLite format 3\x00'
    if sqlite:
        with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as original:
            with sqlite3.connect(destination) as copied:
                original.backup(copied)
    else:
        shutil.copy2(source, destination)


def _copy_input(source: Path, destination: Path):
    if source.is_symlink():
        raise ValueError('baseline symlinks are unsupported')
    if source.is_file():
        _copy_file(source, destination)
        return
    destination.mkdir(parents=True)
    for path in sorted(source.rglob('*')):
        relative = path.relative_to(source)
        if not _allowed(relative, baseline=True):
            continue
        if path.is_symlink():
            raise ValueError(f'baseline symlink is unsupported: {relative}')
        if path.is_dir():
            (destination / relative).mkdir(parents=True, exist_ok=True)
        elif path.is_file() and not _sqlite_sidecar(path):
            _copy_file(path, destination / relative)


def _copy_machinery(destination: Path):
    root = Path(__file__).resolve().parent.parent
    if (root / '.git').exists():
        revision = _git(root, 'rev-parse', 'HEAD').decode().strip()
        # Read committed blobs, never mutable or untracked checkout contents.
        entries = _git(root, 'ls-tree', '-rz', '--full-tree', revision).split(b'\0')
        for entry in filter(None, entries):
            info, name = entry.split(b'\t', 1)
            mode, kind, oid = info.split()
            relative = Path(os.fsdecode(name))
            if kind != b'blob' or mode == b'120000' or not _allowed(relative):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_git(root, 'cat-file', 'blob', oid.decode()))
            target.chmod(0o755 if mode == b'100755' else 0o644)
        return {'kind': 'git', 'revision': revision}
    distribution = metadata.distribution('rsi')
    files = distribution.files
    if not files:
        raise ValueError('installed RSI distribution has no release file inventory')
    for item in files:
        relative = Path(str(item))
        if relative.is_absolute() or '..' in relative.parts or not _allowed(relative):
            continue
        source = Path(distribution.locate_file(item))
        if source.is_file():
            _copy_file(source, destination / relative)
    return {'kind': 'distribution', 'revision': distribution.version}


def _publish(staging: Path, destination: Path):
    # Linux no-replace rename is atomic even against another creator racing us.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        raise OSError('atomic no-replace directory publication requires renameat2')
    if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(code, os.strerror(code), str(destination))


def _clone(repository, revision, destination):
    local = Path(repository).expanduser()
    if local.exists():
        if _git(local, 'status', '--porcelain', '--untracked-files=all').strip():
            raise ValueError('local repository must be clean')
        revision = _git(local, 'rev-parse', '--verify', (revision or 'HEAD') + '^{commit}').decode().strip()
        source = str(local.resolve())
    else:
        source = str(repository)
        # Never persist credentials in clone configuration or the manifest.
        from urllib.parse import urlsplit
        parsed = urlsplit(source)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('repository URL must not contain credentials')
    subprocess.run(['git', 'clone', '--no-hardlinks', '--dissociate', '--no-checkout', '--', source,
                    str(destination)], check=True, capture_output=True)
    _git(destination, 'checkout', '--detach', revision or 'HEAD')
    pinned = _git(destination, 'rev-parse', 'HEAD').decode().strip()
    _git(destination, 'remote', 'remove', 'origin')
    return {'revision': pinned}


def create_instance(destination, *, baseline_kdb, baseline_skills, skill_map,
                    repository=None, repository_revision=None):
    """Create a snapshot; return its manifest. Never merge into an existing path."""
    destination = Path(destination).expanduser().absolute()
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    inputs = tuple(Path(p).expanduser().absolute()
                   for p in (baseline_kdb, baseline_skills, skill_map))
    if any(source.is_symlink() for source in inputs):
        raise ValueError('baseline symlinks are unsupported')
    destination = destination.resolve()
    kdb, skills, mapping = (source.resolve() for source in inputs)
    if not kdb.exists() or not skills.is_dir() or not mapping.is_file():
        raise ValueError('explicit existing KDB, skills directory and skill map are required')
    json.loads(mapping.read_text(encoding='utf-8'))
    if repository_revision and not repository:
        raise ValueError('repository revision requires a repository')
    for source in (kdb, skills, mapping):
        if destination == source or source in destination.parents:
            raise ValueError('instance destination cannot be inside a baseline input')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-', dir=destination.parent))
    try:
        machinery = _copy_machinery(staging / 'machinery')
        _copy_input(kdb, staging / 'memory' / kdb.name)
        _copy_input(skills, staging / 'skills')
        _copy_file(mapping, staging / 'config/skill-map.json')
        for name in ('runs', 'artifacts', 'dashboard'):
            (staging / name).mkdir()
        config = {'dashboard': {'host': '127.0.0.1', 'port': 0},
                  'kdb': f'memory/{kdb.name}', 'skills': 'skills',
                  'skill_map': 'config/skill-map.json', 'machinery': 'machinery'}
        (staging / 'config/instance.json').write_text(json.dumps(config, indent=2) + '\n')
        repo = _clone(repository, repository_revision, staging / 'repository') if repository else None
        # Git internals are independent but not part of content integrity inventory.
        hashes = {str(p.relative_to(staging)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(staging.rglob('*')) if p.is_file()
                  and '.git' not in p.relative_to(staging).parts}
        manifest = {'schema_version': 1, 'machinery': machinery, 'repository': repo,
                    'files': hashes}
        (staging / 'instance.json').write_text(json.dumps(manifest, indent=2) + '\n')
        _publish(staging, destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def inspect_instance(path):
    """Read a published instance manifest without mutating its state."""
    return json.loads((Path(path) / 'instance.json').read_text(encoding='utf-8'))
