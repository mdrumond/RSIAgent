"""Committed append-only CAS references with legacy JSON-array checkpoints.

One controller owns a journal. The head publishes only fsynced whole records;
unpublished tails are ignored by readers and removed by the next writer.
"""
import json
import os
from pathlib import Path
import re

from rsi.python_execution import atomic_json


WIDTH = 65


def event_files(root):
    root = Path(root)
    return [root / name for name in ('events.json', 'events.head.json', 'events.refs')
            if (root / name).exists()]


def _head(root):
    path = root / 'events.head.json'
    return json.loads(path.read_bytes()) if path.exists() else None


def read_event_refs(root):
    """Read every published reference, including events since the checkpoint."""
    root = Path(root)
    head = _head(root)
    path = root / 'events.json'
    refs = json.loads(path.read_bytes()) if path.exists() else []
    if not isinstance(refs, list):
        raise ValueError('invalid canonical event checkpoint')
    if head is None:
        return refs
    if not path.exists():
        raise ValueError('canonical event checkpoint is missing')
    base, count, offset = (head[key] for key in ('base', 'count', 'offset'))
    if (head.get('version') != 1 or any(type(n) is not int for n in (base, count, offset))
            or not 0 <= base <= count or offset != (count - base) * WIDTH):
        raise ValueError('invalid canonical event journal head')
    with (root / 'events.refs').open('rb') as stream:
        data = stream.read(offset)
    # Appends and newer checkpoints cannot modify this committed prefix. A
    # reader may return its captured head even while the writer advances it.
    if len(data) != offset or len(refs) < base or data and not data.endswith(b'\n'):
        raise ValueError('canonical event journal prefix is missing')
    records = data.splitlines()
    if any(not re.fullmatch(rb'[0-9a-f]{64}', row) for row in records):
        raise ValueError('invalid canonical event journal record')
    added = [row.decode('ascii') for row in records]
    if len(added) != count - base or data != b''.join(row + b'\n' for row in records):
        raise ValueError('invalid canonical event journal framing')
    overlap = min(len(refs), count) - base
    if refs[base:base + overlap] != added[:overlap]:
        raise ValueError('event checkpoint differs from committed journal')
    return refs[:base] + added


class EventJournal:
    def __init__(self, root):
        self.root = Path(root)
        self.refs = read_event_refs(root)
        self.head = _head(self.root)
        path = self.root / 'events.json'
        self.checkpoint_count = len(json.loads(path.read_bytes())) if path.exists() else 0
        self.failed = False

    def checkpoint(self):
        """Materialize the legacy index at a quiescent boundary, without truncation."""
        if self.failed:
            raise RuntimeError('event publication uncertain; reopen journal before writing')
        path = self.root / 'events.json'
        if not path.exists() or len(self.refs) != self.checkpoint_count:
            atomic_json(path, self.refs)
            self.checkpoint_count = len(self.refs)

    def append(self, reference):
        if self.failed:
            raise RuntimeError('event publication uncertain; reopen journal before writing')
        if not isinstance(reference, str) or not re.fullmatch(r'[0-9a-f]{64}', reference):
            raise ValueError('event reference must be a CAS digest')
        # Geometric checkpoints bound total index bytes, rather than one rewrite
        # per stream delta. Checkpoint only the already-published prefix.
        if len(self.refs) >= max(256, 2 * self.checkpoint_count):
            self.checkpoint()
        if not (self.root / 'events.json').exists():
            self.checkpoint()
        head = self.head or dict(version=1, base=len(self.refs), count=len(self.refs), offset=0)
        try:
            if _head(self.root) != self.head:
                raise RuntimeError('event journal has another writer; reopen before appending')
            fd = os.open(self.root / 'events.refs', os.O_RDWR | os.O_CREAT, 0o600)
            try:
                os.ftruncate(fd, head['offset'])  # Only an unpublished tail can be discarded.
                os.lseek(fd, head['offset'], os.SEEK_SET)
                pending = (reference + '\n').encode('ascii')
                while pending:
                    written = os.write(fd, pending)
                    if written <= 0:
                        raise OSError('event journal write made no progress')
                    pending = pending[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
            published = dict(head, count=head['count'] + 1, offset=head['offset'] + WIDTH)
            atomic_json(self.root / 'events.head.json', published)
        except BaseException:
            self.failed = True
            raise
        self.head = published
        self.refs.append(reference)
