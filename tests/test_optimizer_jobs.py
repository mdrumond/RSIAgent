import pytest

from optimizer.jobs import JobStore


def create(store, **kwargs):
    return store.create('def forward(x): return x', 'forward', {'args': []},
                        {'snapshot_id': 'test'}, **kwargs)


def test_submission_is_immutable_idempotent_and_artifacts_verified(tmp_path):
    store = JobStore(tmp_path)
    job = create(store, idempotency_key='click')
    assert create(store, idempotency_key='click')['job_id'] == job['job_id']
    with pytest.raises(ValueError, match='idempotency'):
        create(store, idempotency_key='click', name='different')
    assert JobStore(tmp_path).get(job['job_id'])['max_attempts'] == 25
    ref = store.put_artifact(job['job_id'], 'source', b'kernel')
    assert store.read_artifact(job['job_id'], ref['sha256']) == b'kernel'
    store.artifact_path(job['job_id'], ref['sha256']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='digest'):
        store.read_artifact(job['job_id'], ref['sha256'])


def test_budget_stop_resume_and_retained_handles(tmp_path):
    store = JobStore(tmp_path)
    job = create(store, max_attempts=1)
    key = job['job_id']
    store.transition(key, 'running', stage='compile')
    store.retain_handle(key, 'bz-a3-1:one', operation='compile')
    assert store.get(key)['handles'][0]['handle'] == 'bz-a3-1:one'
    store.record_attempt(key, candidate_sha256='a' * 64, result='compile-failed')
    assert store.get(key)['status'] == 'budget-exhausted'
    with pytest.raises(ValueError):
        store.record_attempt(key, candidate_sha256='b' * 64, result='passed')
    store.resume(key, additional_attempts=1, additional_seconds=60)
    assert store.get(key)['status'] == 'queued'
    store.stop(key)
    assert store.get(key)['stop_requested']
    assert len(store.events(key)) >= 5


def test_events_chain_detects_tampering_and_time_budget(tmp_path):
    store = JobStore(tmp_path)
    key = create(store, max_seconds=1)['job_id']
    store.transition(key, 'running')
    store.account_time(key, 2)
    assert store.get(key)['status'] == 'budget-exhausted'
    events = tmp_path / key / 'events.jsonl'
    events.write_text(events.read_text().replace('created', 'changed'))
    with pytest.raises(ValueError, match='event'):
        store.events(key)


def test_manifest_recovers_from_committed_event_and_queued_stop(tmp_path):
    import json
    store = JobStore(tmp_path)
    original = create(store)
    key = original['job_id']
    store.retain_handle(key, 'bz-a3-1:retained', operation='run')
    (tmp_path / key / 'manifest.json').write_text(json.dumps(original))
    assert store.get(key)['handles'][0]['handle'] == 'bz-a3-1:retained'
    assert store.stop(key)['status'] == 'stopped'
    assert store.resume(key, additional_attempts=1, additional_seconds=1)['status'] == 'queued'


@pytest.mark.parametrize('terminal', ['completed', 'failed', 'stopped'])
def test_late_accounting_preserves_terminal_status(tmp_path, terminal):
    store = JobStore(tmp_path)
    key = create(store, max_seconds=1)['job_id']
    store.transition(key, 'running')
    store.transition(key, terminal)
    store.account_time(key, 2)
    assert store.get(key)['status'] == terminal


def test_published_job_parent_is_synced(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    synced = []
    original = os.fsync
    def track(descriptor):
        synced.append(Path(os.readlink('/proc/self/fd/' + str(descriptor))))
        original(descriptor)
    monkeypatch.setattr(os, 'fsync', track)
    create(JobStore(tmp_path))
    assert synced[-1] == tmp_path


def test_submission_digest_reused_but_expected_digest_always_checked(tmp_path, monkeypatch):
    import json
    store = JobStore(tmp_path)
    job = create(store)
    key = job['job_id']
    calls = []
    original = store._hash_submission
    def count(stream):
        calls.append(stream.name)
        return original(stream)
    monkeypatch.setattr(store, '_hash_submission', count)
    store.get(key)
    store.get(key)
    store.put_artifact(key, 'small-result', b'{}')
    assert len(calls) == 1
    manifest = tmp_path / key / 'manifest.json'
    manifest.write_text(json.dumps({**job, 'submission_sha256': '0' * 64}))
    with pytest.raises(ValueError, match='submission digest mismatch'):
        store.get(key)
    assert len(calls) == 1
    # Cache is per store, not shared globally between worker processes.
    with pytest.raises(ValueError, match='submission digest mismatch'):
        JobStore(tmp_path).get(key)


@pytest.mark.parametrize('replacement', [False, True])
def test_submission_digest_invalidated_by_same_size_tamper(tmp_path, replacement):
    import os
    store = JobStore(tmp_path)
    key = create(store)['job_id']
    store.get(key)
    path = tmp_path / key / 'submission.json'
    before = path.stat()
    changed = path.read_bytes().replace(b'snapshot_id', b'snapshot_XX')
    assert len(changed) == before.st_size
    if replacement:
        other = path.with_suffix('.replacement')
        other.write_bytes(changed)
        other.replace(path)
    else:
        path.write_bytes(changed)
    # Even restoring mtime cannot hide ordinary mutation: ctime/inode changes.
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match='submission digest mismatch'):
        store.get(key)


def test_submission_modified_during_hash_is_not_cached(tmp_path, monkeypatch):
    store = JobStore(tmp_path)
    key = create(store)['job_id']
    path = tmp_path / key / 'submission.json'
    original = store._hash_submission
    def mutate(stream):
        digest = original(stream)
        path.write_bytes(path.read_bytes().replace(b'snapshot_id', b'snapshot_XX'))
        return digest
    monkeypatch.setattr(store, '_hash_submission', mutate)
    with pytest.raises(ValueError, match='changed during digest verification'):
        store.get(key)
    assert path not in store._submission_digests
