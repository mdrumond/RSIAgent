"""Thin local Restic adapter. Callers quiesce writers and publish before finalize.

Restic owns compression, deduplication, indexes and interrupted-backup recovery.
Empty-password repositories provide filesystem privacy only, not encryption.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

BLOCK_BYTES = 64 * 1024


def _identity(path):
    st = os.stat(path)
    return {key: getattr(st, 'st_' + key) for key in
            ('dev', 'ino', 'size', 'mtime_ns', 'ctime_ns')}


def _stable(source, expected):
    if _identity(source) != expected:
        raise ValueError('source changed; quiesce writers before archiving')


def _sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_json(path, value):
    fd, name = tempfile.mkstemp(prefix='.manifest-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        _sync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def _repository(value):
    value = os.fspath(value)
    if ':' in value or '\x00' in value:
        raise ValueError('only local repository directories are supported')
    return Path(value).resolve()


@contextmanager
def _request(repository):
    repository.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(repository, 0o700)
    with (repository / '.rsi.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _command(repository, *args):
    binary = os.environ.get('RSI_RESTIC') or shutil.which('restic')
    if not binary:
        raise FileNotFoundError('Restic required; set RSI_RESTIC or install restic on PATH')
    env = {k: v for k, v in os.environ.items() if not k.startswith('RESTIC_')}
    env.setdefault('GOMAXPROCS', '1')
    env.setdefault('GOMEMLIMIT', '96MiB')
    return [binary, '--repo', str(repository), '--insecure-no-password',
            '--no-cache', '--pack-size', '16', *args], env


def _run(repository, *args):
    command, env = _command(repository, *args)
    return subprocess.run(command, env=env, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout


def archive_files(paths, destination_directory):
    """One native backup batch; return repository, snapshot and file manifests."""
    repository = _repository(destination_directory)
    files = []
    for path in dict.fromkeys(str(Path(p).resolve()) for p in paths):
        source = Path(path)
        if not source.is_file():
            raise ValueError('archive inputs must be regular files')
        if source.is_relative_to(repository):
            raise ValueError('source must be outside repository')
        identity = _identity(source)
        files.append(dict(source=path, path=path, source_identity=identity,
                          size=identity['size'], sha256=_sha(source)))
        _stable(source, identity)
    if not files:
        raise ValueError('archive batch is empty')
    with _request(repository):
        if not (repository / 'config').exists():
            _run(repository, 'init', '--quiet')
        _run(repository, 'unlock', '--quiet')  # stale locks only; never --remove-all
        with tempfile.NamedTemporaryFile(dir=repository, prefix='.files-') as listing:
            for item in files:
                listing.write(os.fsencode(item['source']) + b'\0')
            listing.flush()
            output = _run(repository, 'backup', '--json', '--quiet', '--read-concurrency',
                          '1', '--files-from-raw', listing.name)
        summary = next(row for row in map(json.loads, output.splitlines())
                       if row.get('message_type') == 'summary')
        snapshot = summary['snapshot_id']
        for item in files:
            _stable(item['source'], item['source_identity'])
            item.update(repository=str(repository), snapshot=snapshot, state='complete')
    return dict(repository=str(repository), snapshot=snapshot, files=files)


def compress_file(source, destination_directory):
    """Single-artifact wrapper with a durable JSON reference, never custom chunks."""
    manifest = archive_files([source], destination_directory)['files'][0]
    key = hashlib.sha256(os.fsencode(manifest['source'])).hexdigest()
    path = Path(manifest['repository']) / (key + '.manifest.json')
    manifest['manifest_path'] = str(path)
    _publish_json(path, manifest)
    return manifest


def iter_artifact_bytes(manifest):
    """Stream native dump in 64 KiB blocks; exhaust to verify SHA and size."""
    if manifest.get('state') != 'complete':
        raise ValueError('archive is incomplete')
    repository = _repository(manifest['repository'])
    digest, size = hashlib.sha256(), 0
    command, env = _command(repository, 'dump', manifest['snapshot'], manifest['path'])
    with _request(repository), tempfile.TemporaryFile() as errors:
        with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=errors) as process:
            try:
                while block := process.stdout.read(BLOCK_BYTES):
                    digest.update(block)
                    size += len(block)
                    yield block
                if process.wait():
                    errors.seek(0)
                    raise RuntimeError(errors.read().decode(errors='replace'))
            finally:
                if process.poll() is None:
                    process.terminate()
                process.stdout.close()
                process.wait()
    if size != manifest['size'] or digest.hexdigest() != manifest['sha256']:
        raise ValueError('archive integrity mismatch')


def restore_files(manifests, target):
    """One native restore for a shared snapshot; target maps manifest to filename.

    A directory target uses source basenames. Verify every staged file before
    publishing with atomic no-overwrite links; all destinations must share a FS.
    """
    manifests = list(manifests)
    if not manifests:
        return []
    repository = _repository(manifests[0]['repository'])
    snapshot = manifests[0]['snapshot']
    destinations = [Path(target(m) if callable(target) else Path(target) / Path(m['source']).name)
                    .absolute() for m in manifests]
    if len(set(destinations)) != len(destinations):
        raise ValueError('duplicate restore destination')
    for m, dest in zip(manifests, destinations):
        if m.get('state') != 'complete' or (_repository(m['repository']), m['snapshot']) != (repository, snapshot):
            raise ValueError('restore batch requires one complete repository snapshot')
        if os.path.lexists(dest):
            raise FileExistsError(dest)
    with _request(repository), tempfile.TemporaryDirectory(prefix='.restore-', dir=destinations[0].parent) as temp:
        args = ['restore', snapshot, '--target', temp]
        with tempfile.NamedTemporaryFile(mode='w', dir=temp) as selection:
            for m in manifests:
                pattern = ''.join('\\' + c if c in '\\*?[]' else c for c in m['path'])
                if '\n' in pattern or '\r' in pattern:
                    args.extend(['--include', pattern])
                else:
                    selection.write(pattern + '\n')
            selection.flush()
            _run(repository, *args, '--include-file', selection.name)
        staged = [Path(temp) / m['path'].lstrip('/') for m in manifests]
        for m, path in zip(manifests, staged):
            if path.stat().st_size != m['size'] or _sha(path) != m['sha256']:
                raise ValueError('archive integrity mismatch')
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
        published = []
        try:
            for path, dest in zip(staged, destinations):
                os.link(path, dest)
                published.append(dest)
            for parent in {d.parent for d in destinations}:
                _sync_directory(parent)
        except BaseException:
            for dest in published:
                dest.unlink()
            raise
    return [str(dest) for dest in destinations]


def restore_file(manifest, destination):
    return restore_files([manifest], lambda _: destination)[0]


def finalize_source(manifest):
    """After ledger publication: verify repository and unchanged source, then unlink.

    Caller must keep writers quiescent throughout; stat cannot lock other writers.
    """
    source = Path(manifest['source'])
    _stable(source, manifest['source_identity'])
    for _ in iter_artifact_bytes(manifest):
        pass
    if _sha(source) != manifest['sha256']:
        raise ValueError('source integrity mismatch')
    _stable(source, manifest['source_identity'])
    source.unlink()
    _sync_directory(source.parent)
