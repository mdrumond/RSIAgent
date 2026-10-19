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


def test_large_artifact_batch_authenticates_and_rewrites_once(tmp_path, monkeypatch):
    import time
    import optimizer.jobs as jobs
    store = JobStore(tmp_path)
    # Representative Phase 2 memory size and observed profiling-file count.
    key = store.create('def forward(x): return x', 'forward', {},
                       {'memory': 'x' * (56 * 1024 * 1024)})['job_id']
    counts = dict(get=0, chain_write=0)
    real_get, real_atomic = store.get, jobs.atomic
    def get(job):
        counts['get'] += 1
        return real_get(job)
    def atomic(path, data):
        if path.name == 'events.jsonl':
            counts['chain_write'] += 1
        return real_atomic(path, data)
    monkeypatch.setattr(store, 'get', get)
    monkeypatch.setattr(jobs, 'atomic', atomic)
    started = time.monotonic()
    records = store.put_artifacts(key, ((f'profile/{i}.csv', f'value,{i}\n'.encode())
                                        for i in range(10000)))
    print(f'10000 artifacts / 56 MiB submission: {time.monotonic() - started:.3f}s')
    assert counts == dict(get=1, chain_write=1)
    assert len(records) == 10000
    # Existing consumers see individual artifact records, not a new batch schema.
    assert JobStore(tmp_path).artifacts(key) == records
    assert store.read_artifact(key, records[-1]['sha256']) == b'value,9999\n'
    assert len(store.events(key)) == 10001


def test_artifact_batch_failure_never_publishes_partial_inventory(tmp_path, monkeypatch):
    import optimizer.jobs as jobs
    store = JobStore(tmp_path)
    key = create(store)['job_id']
    first = store.put_artifact(key, 'old', b'old')
    def incomplete():
        yield 'new', b'new'
        raise ValueError('archive truncated')
    with pytest.raises(ValueError, match='truncated'):
        store.put_artifacts(key, incomplete())
    assert store.artifacts(key) == [first]
    original = jobs.atomic
    def interrupted(path, data):
        if path.name == 'events.jsonl':
            raise OSError('disk full')
        return original(path, data)
    monkeypatch.setattr(jobs, 'atomic', interrupted)
    with pytest.raises(OSError, match='disk full'):
        store.put_artifacts(key, [('new', b'new')])
    assert store.artifacts(key) == [first]
    monkeypatch.setattr(jobs, 'atomic', original)
    store.put_artifacts(key, [('new', b'new')])
    assert [record['name'] for record in store.artifacts(key)] == ['old', 'new']
