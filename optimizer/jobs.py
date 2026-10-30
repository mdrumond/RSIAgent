"""Atomic job records; callers own execution and remote-operation observation."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, data):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


class JobStore:
    """One locked manifest per job, immutable hashed submission and event chain."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _directory(self, job_id):
        if not re.fullmatch(r'[0-9a-f]{32}', job_id):
            raise ValueError('invalid job identity')
        return self.root / job_id

    @contextmanager
    def _lock(self, directory):
        with (directory / '.lock').open('a+b') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def create(self, source, entrypoint, signature, memory_snapshot, *, name='',
               init_args=None, init_kwargs=None, idempotency_key=None,
               max_attempts=25, max_seconds=7200):
        if not isinstance(source, str) or not source.strip() or not entrypoint:
            raise ValueError('source and entrypoint are required')
        self._budget(max_attempts, max_seconds)
        submission = dict(source=source, entrypoint=entrypoint, signature=signature,
                          memory_snapshot=memory_snapshot, name=name,
                          init_args=init_args or [], init_kwargs=init_kwargs or {},
                          max_attempts=max_attempts, max_seconds=max_seconds)
        data = canonical(submission)
        with self._lock(self.root):
            job_id = sha(str(idempotency_key).encode())[:32] if idempotency_key else uuid.uuid4().hex
            directory = self._directory(job_id)
            if directory.exists():
                existing = self.get(job_id)
                if existing['submission_sha256'] != sha(data):
                    raise ValueError('idempotency key reused with different submission')
                return existing
            # Publish only after the complete initial directory is durable.
            staging = self.root / ('.creating-' + uuid.uuid4().hex)
            staging.mkdir()
            atomic(staging / 'submission.json', data)
            manifest = dict(schema='optimizer-job-v1', budget_policy='elapsed-time-v1', job_id=job_id, name=name,
                            submission_sha256=sha(data), created_at=time.time(),
                            status='queued', stage='queued', attempts=0, elapsed_seconds=0,
                            max_attempts=max_attempts, max_seconds=max_seconds,
                            stop_requested=False, handles=[])
            atomic(staging / 'manifest.json', canonical(manifest))
            self._event(staging, 'created', {'submission_sha256': sha(data)})
            staging.rename(directory)
            descriptor = os.open(self.root, os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return manifest

    @staticmethod
    def _budget(attempts, seconds):
        if type(seconds) is not int or seconds < 1:
            raise ValueError('time budget must be a positive integer')

    def get(self, job_id):
        directory = self._directory(job_id)
        manifest = json.loads((directory / 'manifest.json').read_bytes())
        # A retained event is the commit boundary if a crash preceded the cache replace.
        for event in reversed(self._events(directory)):
            if 'manifest' in event['payload']:
                manifest = event['payload']['manifest']
                break
        if sha((directory / 'submission.json').read_bytes()) != manifest['submission_sha256']:
            raise ValueError('submission digest mismatch')
        return manifest

    def submission(self, job_id):
        self.get(job_id)
        return json.loads((self._directory(job_id) / 'submission.json').read_bytes())

    def list(self):
        return sorted((self.get(path.name) for path in self.root.iterdir()
                       if re.fullmatch(r'[0-9a-f]{32}', path.name)),
                      key=lambda row: row['created_at'], reverse=True)

    def events(self, job_id):
        return self._events(self._directory(job_id))

    @staticmethod
    def _events(directory):
        path = directory / 'events.jsonl'
        rows = [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []
        previous = '0' * 64
        for index, row in enumerate(rows):
            if (row.get('sequence') != index or row.get('previous') != previous
                    or sha(canonical({k: v for k, v in row.items() if k != 'sha256'})) != row.get('sha256')):
                raise ValueError('event chain digest mismatch')
            previous = row['sha256']
        return rows

    def _event(self, directory, kind, payload):
        return self._event_batch(directory, [(kind, payload)])[0]

    def _event_batch(self, directory, entries):
        """Commit a group without repeatedly reading and rewriting the chain."""
        rows = self._events(directory)
        added = []
        previous = rows[-1]['sha256'] if rows else '0' * 64
        for kind, payload in entries:
            row = dict(sequence=len(rows) + len(added), previous=previous,
                       timestamp=time.time(), kind=kind, payload=payload)
            row['sha256'] = sha(canonical(row))
            added.append(row)
            previous = row['sha256']
        if added:
            atomic(directory / 'events.jsonl', b''.join(canonical(item) + b'\n' for item in [*rows, *added]))
        return added

    def add_event(self, job_id, kind, payload):
        directory = self._directory(job_id)
        with self._lock(directory):
            self.get(job_id)
            return self._event(directory, kind, payload)

    def _update(self, job_id, kind, change):
        directory = self._directory(job_id)
        with self._lock(directory):
            manifest = self.get(job_id)
            change(manifest)
            # Event precedes the replace: an interrupted mutation remains visible.
            self._event(directory, kind, {'manifest': manifest})
            atomic(directory / 'manifest.json', canonical(manifest))
            return manifest

    def transition(self, job_id, status, *, stage=None):
        allowed = {'queued': {'running', 'stopped', 'failed'},
                   'running': {'running', 'completed', 'failed', 'stopped', 'budget-exhausted'},
                   'budget-exhausted': {'budget-exhausted'}}
        def change(row):
            if status not in allowed.get(row['status'], set()):
                raise ValueError('invalid job transition')
            row.update(status=status, stage=stage or status)
        return self._update(job_id, 'transition', change)

    def record_attempt(self, job_id, *, candidate_sha256, result):
        def change(row):
            if row['status'] != 'running' or row['stop_requested']:
                raise ValueError('job is not running')
            row['attempts'] += 1
            row['latest_attempt'] = dict(candidate_sha256=candidate_sha256, result=result)
            self._exhaust(row)
        return self._update(job_id, 'attempt', change)

    @staticmethod
    def _exhaust(row):
        if row['elapsed_seconds'] >= row['max_seconds']:
            row['status'] = 'budget-exhausted'

    def account_time(self, job_id, seconds):
        if not isinstance(seconds, (int, float)) or seconds < 0:
            raise ValueError('elapsed time must be nonnegative')
        def change(row):
            row['elapsed_seconds'] += seconds
            if row['status'] == 'running':
                self._exhaust(row)
        return self._update(job_id, 'elapsed', change)

    def account_canonical_elapsed(self, job_id, project, total):
        """Idempotent cumulative controller receipt, including failed work."""
        if not isinstance(total, (int, float)) or not 0 <= total < float('inf'):
            raise ValueError('canonical elapsed time must be finite and nonnegative')
        def change(row):
            receipts = row.setdefault('canonical_elapsed', {})
            previous = receipts.get(project, 0)
            if total < previous:
                raise ValueError('canonical elapsed receipt moved backwards')
            row['elapsed_seconds'] += total - previous
            receipts[project] = total
            if row['status'] == 'running':
                self._exhaust(row)
        return self._update(job_id, 'canonical-elapsed', change)

    def stop(self, job_id):
        def change(row):
            row['stop_requested'] = True
            if row['status'] == 'queued':
                row.update(status='stopped', stage='stopped')
        return self._update(job_id, 'stop-requested', change)

    def recover(self, job_id):
        """Requeue failed checkpoint recovery without granting more runtime."""
        def change(row):
            if row['status'] != 'failed' or row['stop_requested']:
                raise ValueError('only failed jobs without a stop request can recover')
            status = ('queued' if row['elapsed_seconds'] < row['max_seconds']
                      else 'budget-exhausted')
            row.update(status=status, stage=status)
        return self._update(job_id, 'recovered', change)

    def resume(self, job_id, *, additional_seconds, additional_attempts=0):
        self._budget(additional_attempts, additional_seconds)
        def change(row):
            if row['status'] not in {'stopped', 'budget-exhausted', 'failed'}:
                raise ValueError('job must be stopped before resume')
            row.update(max_attempts=row['max_attempts'] + additional_attempts,
                       max_seconds=row['max_seconds'] + additional_seconds,
                       stop_requested=False, status='queued', stage='queued')
        return self._update(job_id, 'resumed', change)

    def continue_with_feedback(self, job_id, *, additional_seconds, key):
        """A campaign lease owner reopens a settled job without changing inputs."""
        self._budget(0, additional_seconds)
        def change(row):
            grants = row.setdefault('manual_grants', {})
            if key in grants:
                if grants[key] != additional_seconds:
                    raise ValueError('manual grant key reused')
                return
            grants[key] = additional_seconds
            row.update(max_seconds=row['max_seconds'] + additional_seconds,
                       stop_requested=False, status='queued', stage='queued')
        return self._update(job_id, 'manual-continuation', change)

    def retain_handle(self, job_id, handle, *, operation):
        def change(row):
            record = dict(handle=handle, operation=operation)
            if record not in row['handles']:
                row['handles'].append(record)
        return self._update(job_id, 'handle', change)

    def artifact_path(self, job_id, digest):
        if not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('invalid artifact digest')
        return self._directory(job_id) / 'artifacts' / digest

    def put_artifact(self, job_id, name, data):
        return self.put_artifacts(job_id, [(name, data)])[0]

    def put_artifacts(self, job_id, entries, *, deduplicate=False):
        """Publish an iterable of (name, bytes) as one atomic inventory commit.

        Content blobs precede the event commit. A failed iterator or interrupted
        write may leave unreferenced blobs, never a partially published batch.
        Existing artifact events and readers retain exactly the same schema.
        Compact-operation retries may deduplicate exact name/content records.
        """
        directory = self._directory(job_id)
        with self._lock(directory):
            self.get(job_id)
            known = {(r['name'], r['sha256'], r['size']) for r in self.artifacts(job_id)} if deduplicate else set()
            records = []
            for name, data in entries:
                digest = sha(data)
                path = self.artifact_path(job_id, digest)
                path.parent.mkdir(exist_ok=True)
                atomic(path, data)
                records.append(dict(name=name, sha256=digest, size=len(data)))
            additions = [('artifact', r) for r in records
                         if (r['name'], r['sha256'], r['size']) not in known]
            if additions:
                self._event_batch(directory, additions)
            return records

    def read_artifact(self, job_id, digest):
        self.get(job_id)
        data = self.artifact_path(job_id, digest).read_bytes()
        if sha(data) != digest:
            raise ValueError('artifact digest mismatch')
        return data

    def artifacts(self, job_id):
        return [event['payload'] for event in self.events(job_id) if event['kind'] == 'artifact']
