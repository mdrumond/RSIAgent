"""Sequential, bounded-memory gzip archives. Callers must quiesce source writers.

Packaging already includes ascend_tools.*; no extra dependency or daemon is needed.
Persist the returned JSON manifest, pass it to iter_artifact_bytes/restore_file,
and call finalize_source only after the caller has published its own reference.
"""
from contextlib import ExitStack
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zlib

MAX_CHUNK_BYTES = 32 * 1024 * 1024
BLOCK_BYTES = 64 * 1024


def _identity(path):
    st = os.stat(path)
    return {key: getattr(st, 'st_' + key) for key in
            ('dev', 'ino', 'size', 'mtime_ns', 'ctime_ns')}


def _stable(source, expected):
    if _identity(source) != expected:
        raise ValueError('source changed; quiesce writers before archiving')


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


def _chunk_bytes(manifest, chunk):
    digest = hashlib.sha256()
    size = 0
    with gzip.open(Path(manifest['directory']) / chunk['file'], 'rb') as stream:
        while block := stream.read(BLOCK_BYTES):
            size += len(block)
            if size > chunk['size']:
                raise ValueError('chunk exceeds declared size')
            digest.update(block)
            yield block
    if size != chunk['size'] or digest.hexdigest() != chunk['sha256']:
        raise ValueError('chunk integrity mismatch')


def iter_artifact_bytes(manifest):
    """Yield verified archive bytes; exhaust iterator to verify the full digest."""
    if manifest.get('state') != 'complete':
        raise ValueError('archive is incomplete')
    digest = hashlib.sha256()
    size = 0
    for chunk in manifest['chunks']:
        for block in _chunk_bytes(manifest, chunk):
            digest.update(block)
            size += len(block)
            yield block
    if size != manifest['source_identity']['size'] or digest.hexdigest() != manifest['sha256']:
        raise ValueError('archive integrity mismatch')


def compress_file(source, destination_directory, chunk_bytes=MAX_CHUNK_BYTES, *,
                  expected_identity=None, on_chunk=None):
    """Archive a quiescent file, resuming durable chunks after interruption.

    on_chunk(manifest) runs after each durable chunk checkpoint (useful for
    progress/cancellation). expected_identity is a prior source_identity dict.
    Source deletion is always a separate explicit finalize_source operation.
    """
    if not 0 < chunk_bytes <= MAX_CHUNK_BYTES:
        raise ValueError('chunk_bytes must be between 1 and 32 MiB')
    source = Path(source).resolve()
    directory = Path(destination_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # One worker per directory, including across processes.
    with (directory / '.archive.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        identity = _identity(source)
        if expected_identity is not None:
            _stable(source, expected_identity)
        key = hashlib.sha256(os.fsencode(source)).hexdigest()
        progress = directory / (key + '.progress.json')
        published = directory / (key + '.manifest.json')
        manifest = dict(version=1, state='incomplete', source=str(source),
                        source_identity=identity, directory=str(directory),
                        chunk_bytes=chunk_bytes, chunks=[], manifest_path=str(published))
        if progress.exists():
            manifest = json.loads(progress.read_text())
            if (manifest['source_identity'] != identity or
                    manifest['chunk_bytes'] != chunk_bytes):
                raise ValueError('resume source or chunk size changed')
        full_hash = hashlib.sha256()
        chunks = []
        with source.open('rb') as stream:
            if _identity(source) != identity or os.fstat(stream.fileno()).st_ino != identity['ino']:
                raise ValueError('source changed before read')
            for index, offset in enumerate(range(0, identity['size'], chunk_bytes)):
                _stable(source, identity)
                size = min(chunk_bytes, identity['size'] - offset)
                old = manifest['chunks'][index] if index < len(manifest['chunks']) else None
                if old:
                    try:
                        for _ in _chunk_bytes(manifest, old):
                            pass
                    except (OSError, EOFError, ValueError, zlib.error):
                        old = None
                temporary = None
                try:
                    digest = hashlib.sha256()
                    with ExitStack() as stack:
                        out = None
                        if not old:
                            fd, temporary = tempfile.mkstemp(prefix='.chunk-', dir=directory)
                            raw = stack.enter_context(os.fdopen(fd, 'wb'))
                            out = stack.enter_context(gzip.GzipFile(
                                fileobj=raw, mode='wb', mtime=0, filename=''))
                        remaining = size
                        while remaining:
                            block = stream.read(min(BLOCK_BYTES, remaining))
                            if not block:
                                raise ValueError('source truncated')
                            remaining -= len(block)
                            digest.update(block)
                            full_hash.update(block)
                            if out:
                                out.write(block)
                        if out:
                            out.close()
                            raw.flush()
                            os.fsync(raw.fileno())
                    sha = digest.hexdigest()
                    chunk = dict(file=sha + '.gz', sha256=sha, size=size)
                    if old and old != chunk:
                        raise ValueError('source bytes changed')
                    if not old:
                        # Equal content in later chunks or other archives is reusable.
                        try:
                            for _ in _chunk_bytes(manifest, chunk):
                                pass
                        except (OSError, EOFError, ValueError, zlib.error):
                            os.replace(temporary, directory / chunk['file'])
                            _sync_directory(directory)
                    chunks.append(chunk)
                    _stable(source, identity)
                    manifest['chunks'] = chunks + manifest['chunks'][index + 1:]
                    _publish_json(progress, manifest)
                    if on_chunk:
                        on_chunk(json.loads(json.dumps(manifest)))
                finally:
                    if temporary is not None:
                        Path(temporary).unlink(missing_ok=True)
        _stable(source, identity)
        manifest.update(chunks=chunks, sha256=full_hash.hexdigest(), state='complete')
        for _ in iter_artifact_bytes(manifest):
            pass
        _stable(source, identity)
        _publish_json(published, manifest)
        _publish_json(progress, manifest)
        return manifest


def restore_file(manifest, destination):
    """Verify into a sibling temporary file; atomically publish without overwrite."""
    destination = Path(destination)
    fd, temporary = tempfile.mkstemp(prefix='.restore-', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            for block in iter_artifact_bytes(manifest):
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, destination)  # fails atomically if destination exists
        _sync_directory(destination.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return str(destination)


def finalize_source(manifest):
    """Explicitly remove original after archive verification and caller publication.

    Caller must keep writers stopped through this operation (stat cannot lock an
    uncooperative writer). A changed source is never deliberately removed.
    """
    source = Path(manifest['source'])
    _stable(source, manifest['source_identity'])
    for _ in iter_artifact_bytes(manifest):
        pass
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        while block := stream.read(BLOCK_BYTES):
            digest.update(block)
    _stable(source, manifest['source_identity'])
    if digest.hexdigest() != manifest['sha256']:
        raise ValueError('source integrity mismatch')
    source.unlink()
    _sync_directory(source.parent)
