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
    def __init__(self, root, initial_memory=b'', *, index=None, ledger=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.initial = initial_memory.decode('utf-8')
        self.index, self.ledger = index, ledger
        with self.transaction() as state:
            identity = hashlib.sha256(initial_memory).hexdigest()
            if not state:
                state.update(schema='subject-memory-v1', initial_sha256=identity, active={}, admissions={})
            elif state['initial_sha256'] != identity:
                raise ValueError('initial memory changed')
            if self.ledger is not None:
                for version in state['active'].values():
                    self.ledger.bootstrap(self.read(version))

    @contextmanager
    def transaction(self):
        with (self.root / '.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / 'active.json'
            state = json.loads(path.read_bytes()) if path.exists() else {}
            yield state
            atomic(path, canonical(state))

    def records(self):
        if self.ledger is not None:
            return self.ledger.records()
        with self.transaction() as state:
            return [self.read(version) for version in state['active'].values()
                    if self.ledger is None or self.ledger.available(version)]

    def read(self, version):
        path = self.root / 'versions' / (version + '.json')
        if not path.exists() and self.ledger is not None:
            return self.ledger.record(version)
        record = json.loads(path.read_bytes())
        if digest(record) != version:
            raise ValueError('memory version content changed')
        return record

    def snapshot(self):
        return canonical(dict(schema='subject-memory-snapshot-v1', initial_memory=self.initial,
                              memories=self.records()))

    def bases_current(self, proposal):
        if self.ledger is not None:
            return self.ledger.bases_current(proposal['base_versions'])
        with self.transaction() as state:
            return all(state['active'].get(row['memory_id']) == row['version']
                       for row in proposal['base_versions'])

    def accept(self, review_id, proposal, provenance, *, check_remaining=None):
        """Index first, then switch accepted pointers in one locked publication."""
        with self.transaction() as state:
            if review_id in state['admissions']:
                return state['admissions'][review_id]
            bases = proposal['base_versions']
            if len({row['memory_id'] for row in bases}) != len(bases):
                raise ValueError('memory bases must identify distinct memories')
            memory_id = bases[0]['memory_id'] if bases else review_id
            record = dict(memory_id=memory_id, proposal=proposal, provenance=provenance)
            version = digest(record)
            if self.ledger is not None:
                try:
                    published = self.ledger.record(version)
                except KeyError:
                    published = None
                if published is not None:
                    if published != record:
                        raise ValueError('published memory receipt differs from proposed content')
                    # The shared transaction can survive a failed local receipt write.
                    # Recover only its receipt, never republish an old global head.
                    state['admissions'][review_id] = dict(memory_id=memory_id, version=version)
                    return state['admissions'][review_id]
            if not (self.ledger.bases_current(bases) if self.ledger is not None else all(
                    state['active'].get(row['memory_id']) == row['version'] for row in bases)):
                raise StaleMemory('memory replacement base changed; reconcile and reevaluate')
            expected_statuses = ({base['version']: self.ledger.status(base['version']) for base in bases}
                                 if self.ledger is not None else None)
            directory = self.root / 'versions'
            directory.mkdir(exist_ok=True)
            atomic(directory / (version + '.json'), canonical(record))
            if self.index is not None:
                self.index(version, record)
            if check_remaining is not None:
                check_remaining()
            if self.ledger is not None:
                self.ledger.publish(record, expected_statuses=expected_statuses)
            for row in bases:
                state['active'].pop(row['memory_id'], None)
            state['active'][memory_id] = version
            state['admissions'][review_id] = dict(memory_id=memory_id, version=version)
            return state['admissions'][review_id]

    def search(self, query):
        """Return exact accepted version identities alongside reference-KDB hits."""
        words = set(re.findall(r'\w+', query.casefold()))
        records = []
        for record in self.records():
            if words & set(re.findall(r'\w+', json.dumps(record).casefold())):
                artifacts = dict(record['provenance'].get('request', {}).get('artifacts', {}))
                for role in ('before', 'after'):
                    if record['proposal'].get(role):
                        artifacts[role] = record['proposal'][role]['sha256']
                artifacts.update({f'evidence:{i}': sha for i, sha in enumerate(record['proposal']['evidence'])})
                records.append(dict(record, version=digest(record), artifacts=artifacts))
        return records
