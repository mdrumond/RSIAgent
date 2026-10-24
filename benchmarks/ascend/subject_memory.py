"""Versioned accepted memory; drafts never appear in wave snapshots."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import re
from pathlib import Path

from optimizer.jobs import atomic, canonical


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


class StaleMemory(ValueError):
    pass


class SubjectMemory:
    def __init__(self, root, initial_memory=b'', *, index=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.initial = initial_memory.decode('utf-8')
        self.index = index
        with self.transaction() as state:
            identity = hashlib.sha256(initial_memory).hexdigest()
            if not state:
                state.update(schema='subject-memory-v1', initial_sha256=identity, active={}, admissions={})
            elif state['initial_sha256'] != identity:
                raise ValueError('initial memory changed')

    @contextmanager
    def transaction(self):
        with (self.root / '.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / 'active.json'
            state = json.loads(path.read_bytes()) if path.exists() else {}
            yield state
            atomic(path, canonical(state))

    def records(self):
        with self.transaction() as state:
            return [self.read(version) for version in state['active'].values()]

    def read(self, version):
        record = json.loads((self.root / 'versions' / (version + '.json')).read_bytes())
        if digest(record) != version:
            raise ValueError('memory version content changed')
        return record

    def snapshot(self):
        return canonical(dict(schema='subject-memory-snapshot-v1', initial_memory=self.initial,
                              memories=self.records()))

    def bases_current(self, proposal):
        with self.transaction() as state:
            return all(state['active'].get(row['memory_id']) == row['version']
                       for row in proposal['base_versions'])

    def accept(self, review_id, proposal, provenance):
        """Index first, then switch accepted pointers in one locked publication."""
        with self.transaction() as state:
            if review_id in state['admissions']:
                return state['admissions'][review_id]
            bases = proposal['base_versions']
            if any(state['active'].get(row['memory_id']) != row['version'] for row in bases):
                raise StaleMemory('memory replacement base changed; reconcile and reevaluate')
            memory_id = bases[0]['memory_id'] if bases else review_id
            record = dict(memory_id=memory_id, proposal=proposal, provenance=provenance)
            version = digest(record)
            directory = self.root / 'versions'
            directory.mkdir(exist_ok=True)
            atomic(directory / (version + '.json'), canonical(record))
            if self.index is not None:
                self.index(version, record)
            for row in bases:
                state['active'].pop(row['memory_id'])
            state['active'][memory_id] = version
            state['admissions'][review_id] = dict(memory_id=memory_id, version=version)
            return state['admissions'][review_id]

    def search(self, query):
        """Return exact accepted version identities alongside reference-KDB hits."""
        words = set(re.findall(r'\w+', query.casefold()))
        records = []
        for record in self.records():
            if words & set(re.findall(r'\w+', json.dumps(record).casefold())):
                records.append(dict(record, version=digest(record)))
        return records
