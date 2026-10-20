import json
from pathlib import Path

import pytest

from benchmarks.ascend.phase2_feedback import FeedbackStore
from optimizer.jobs import canonical, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def campaign(root, *, result=None, task=None, attempt='attempt-001', baseline=True):
    task = task or dict(task_id='relu', submission={'source': 'reference'}, rationale='learn')
    row = dict(attempt_id=attempt, kind='target', task=task,
               result=result or dict(verified=False, summary='agent hypothesis'))
    if baseline and not row['result'].get('job_id'):
        from optimizer.jobs import JobStore
        jobs = JobStore(root / 'execution' / 'jobs')
        job = jobs.create('def forward(x): return x', 'forward', {}, {}, idempotency_key=str(root))
        value = baseline if isinstance(baseline, dict) else dict(suite_sha256='1' * 64,
            manifest=dict(policy=dict(name='host-policy', rtol=.01)))
        ref = jobs.put_artifact(job['job_id'], 'baseline.json', canonical(value))
        row.update(execution_directory='execution')
        row['result'].update(job_id=job['job_id'], baseline=ref)
    write(root / 'state.json', dict(phase='optimizer-phase2', target=task, attempts=[row]))
    write(root / 'seed-memory.json', dict(source_identity=dict(target='a3', language='ascend-c')))
    return row


@pytest.mark.parametrize('result,category', [
    ({'verified': True, 'speedup': .4}, 'verified-slow'),
    ({'verified': True, 'speedup': 1.1}, 'verified-success'),
    ({'verified': True}, 'verified-timing-unknown'),
    ({'verified': True, 'speedup': 0}, 'verified-timing-unknown'),
    ({'verified': False, 'source_sha256': 'abc'}, 'incorrect'),
    ({'verified': False}, 'no-candidate'),
    ({'verified': False, 'result': {'status': 'compile-error'}}, 'compile-failure'),
    ({'verified': False, 'classification': 'budget-exhausted'}, 'budget-exhausted'),
])
def test_outcomes_are_observations(tmp_path, result, category):
    source = tmp_path / 'source'
    campaign(source, result=result)
    store = FeedbackStore(tmp_path / 'feedback')
    assert store.import_campaign(source)['imported'] == 1
    record = store.records()[0]
    assert record['outcome']['category'] == category
    assert record['outcome']['target'] == 'a3'
    assert record['outcome']['language'] == 'ascend-c'
    assert record['outcome']['completed'] is True
    assert record['investigation']['status'] == ('resolved' if category in
        {'verified-success', 'verified-timing-unknown'} else 'open')


def test_numeric_goal_preserved_and_dedup_does_not_reset_investigation(tmp_path):
    source = tmp_path / 'source'
    task = dict(task_id='relu', performance_goal={'speedup': 2})
    campaign(source, task=task, result={'verified': True, 'speedup': 1.5})
    store = FeedbackStore(tmp_path / 'feedback')
    store.import_campaign(source)
    record = store.records()[0]
    assert record['outcome']['category'] == 'verified-slow'
    store.update_investigation(record['feedback_id'], status='investigating', projects=['probe-1'])
    assert store.import_campaign(source)['existing'] == 1
    assert store.records()[0]['investigation']['projects'] == ['probe-1']
    assert store.records()[0]['outcome'] == record['outcome']


def test_resolution_requires_successful_original_task_retry(tmp_path):
    store = FeedbackStore(tmp_path / 'feedback')
    task = {'task_id': 'relu', 'submission': {'source': 'original'}}
    campaign(tmp_path / 'failed', task=task, result={'verified': False})
    store.import_campaign(tmp_path / 'failed')
    failed = store.records()[0]['feedback_id']
    with pytest.raises(ValueError, match='evidence'):
        store.update_investigation(failed, status='ready-to-retry')
    with pytest.raises(ValueError, match='reason'):
        store.update_investigation(failed, status='deferred')
    store.update_investigation(failed, status='ready-to-retry', evidence=['host-probe-report'])
    with pytest.raises(ValueError, match='retry'):
        store.update_investigation(failed, status='resolved')
    campaign(tmp_path / 'retry', task=task, result={'verified': True, 'speedup': 1.2},
             baseline=dict(suite_sha256='1' * 64, policy=dict(name='host-policy', rtol=.01)))
    store.import_campaign(tmp_path / 'retry')
    retry = next(r['feedback_id'] for r in store.records() if r['outcome']['verified'])
    store.update_investigation(failed, status='resolved', retries=[retry])
    assert next(r for r in store.records() if r['feedback_id'] == failed)['investigation']['status'] == 'resolved'


def test_pending_job_import_is_not_completed_attempt(tmp_path):
    from optimizer.jobs import JobStore
    source = tmp_path / 'source'
    row = campaign(source)
    write(source / 'state.json', dict(phase='optimizer-phase2', attempts=[]))
    write(source / 'pending.json', dict(attempt_id=row['attempt_id'], kind='target', task=row['task']))
    relative = 'tasks/relu-' + sha(canonical(row['task']))[:12]
    jobs = JobStore(source / relative / 'jobs')
    job = jobs.create('def f(x): return x', 'f', {}, {}, idempotency_key='pending')
    jobs.retain_handle(job['job_id'], 'remote:exact', operation='evaluate')
    jobs.transition(job['job_id'], 'stopped')
    store = FeedbackStore(tmp_path / 'feedback')
    store.import_campaign(source)
    outcome = store.records()[0]['outcome']
    assert outcome['completed'] is False
    assert outcome['category'] == 'interrupted'
    assert outcome['result']['handles'] == [{'handle': 'remote:exact', 'operation': 'evaluate'}]


@pytest.mark.parametrize('status', ['failed', 'stopped', 'budget-exhausted'])
def test_pending_import_replays_terminal_event_over_stale_manifest(tmp_path, status):
    from optimizer.jobs import JobStore
    source = tmp_path / 'source'
    row = campaign(source)
    write(source / 'state.json', dict(phase='optimizer-phase2', attempts=[]))
    write(source / 'pending.json', dict(attempt_id=row['attempt_id'], kind='target', task=row['task']))
    relative = 'tasks/relu-' + sha(canonical(row['task']))[:12]
    jobs = JobStore(source / relative / 'jobs')
    job = jobs.create('def f(x): return x', 'f', {}, {}, idempotency_key='pending')
    jobs.transition(job['job_id'], 'running')
    path = jobs.root / job['job_id'] / 'manifest.json'
    stale = path.read_bytes()
    jobs.transition(job['job_id'], status)
    path.write_bytes(stale)  # Event committed, process crashed before cache replacement.
    store = FeedbackStore(tmp_path / 'feedback')
    assert store.import_campaign(source)['imported'] == 1
    outcome = store.records()[0]['outcome']
    assert outcome['completed'] is False
    assert outcome['job_status']['status'] == status
    assert outcome['category'] == ('budget-exhausted' if status == 'budget-exhausted' else 'interrupted')


def test_missing_artifact_unknown_target_and_unsupported_layout(tmp_path):
    source = tmp_path / 'source'
    row = campaign(source, result=dict(verified=False, job_id='a' * 32,
                                     candidate={'sha256': 'b' * 64, 'name': 'candidate'}))
    (source / 'seed-memory.json').unlink()
    store = FeedbackStore(tmp_path / 'feedback')
    report = store.import_campaign(source)
    assert any('missing candidate evidence' in message for message in report['warnings'])
    assert store.records()[0]['outcome']['target'] == 'unknown'
    assert store.records()[0]['outcome']['evidence']['candidate']['sha256'] == 'b' * 64
    assert not (source / 'jobs').exists()  # Backfill never mutates the source tree.
    write(tmp_path / 'legacy' / 'state.json', dict(phase='a5-phase2'))
    assert 'legacy' in store.import_campaign(tmp_path / 'legacy')['warnings'][0]


def test_backfill_nested_roots_cli_and_recovery_publication(tmp_path, capsys):
    from benchmarks.ascend.campaign_cli import main
    source = tmp_path / 'source'
    row = campaign(source / 'goal-0001')
    write(source / 'goal-0001' / 'state.json', dict(phase='optimizer-phase2', attempts=[]))
    write(source / 'goal-0001' / 'attempts' / 'attempt-001.json', row)
    config = tmp_path / 'dashboard.json'
    write(config, dict(campaigns=[dict(root=str(source), phase=2),
                                 dict(root='/ignore-phase1', phase=1)]))
    main(['feedback-phase2', '--feedback-root', str(tmp_path / 'feedback'), '--dashboard-config', str(config)])
    assert json.loads(capsys.readouterr().out)['reports'][0]['imported'] == 1
    main(['feedback-phase2', '--feedback-root', str(tmp_path / 'feedback'), '--source-root', str(source)])
    assert json.loads(capsys.readouterr().out)['reports'][0]['existing'] == 1


def test_coordinator_automatically_retains_failures(tmp_path):
    from test_ascend_phase2_optimizer_campaign import setup
    build, _, _, _ = setup(tmp_path, verdicts=(False, True), decisions=('retry-target', 'ready'))
    assert build().resume()['status'] == 'READY'
    records = FeedbackStore(tmp_path / 'feedback').records()
    assert len(records) == 2
    assert {r['outcome']['verified'] for r in records} == {False, True}


def test_correctness_retry_needs_no_timing_but_performance_retry_meets_goal(tmp_path):
    store = FeedbackStore(tmp_path / 'feedback')
    task = dict(task_id='relu', performance_goal={'speedup': 2})
    for name, result in [('incorrect', {'verified': False}),
                         ('slow', {'verified': True, 'speedup': 1.5}),
                         ('unknown', {'verified': True}),
                         ('fast', {'verified': True, 'speedup': 2})]:
        campaign(tmp_path / name, task=task, result=result)
        store.import_campaign(tmp_path / name)
    records = {Path(r['outcome']['source_root']).name: r for r in store.records()}
    store.update_investigation(records['incorrect']['feedback_id'], status='resolved',
                               retries=[records['unknown']['feedback_id']])
    with pytest.raises(ValueError, match='successful'):
        store.update_investigation(records['slow']['feedback_id'], status='resolved',
                                   retries=[records['unknown']['feedback_id']])
    store.update_investigation(records['slow']['feedback_id'], status='resolved',
                               retries=[records['fast']['feedback_id']])


def test_runtime_and_candidate_provenance_from_retained_job(tmp_path):
    from optimizer.jobs import JobStore
    source = tmp_path / 'source'
    jobs = JobStore(source / 'tasks' / 'retained' / 'jobs')
    job = jobs.create('def f(x): return x', 'f', {}, {}, idempotency_key='test')
    jobs.add_event(job['job_id'], 'phase2-task', dict(runtime=dict(target='bz-a3-1', arch='actual')))
    write(source / 'campaign-run.json', dict(target='a3'))
    ref = jobs.put_artifact(job['job_id'], 'candidate.cpp', b'actual candidate source')
    row = campaign(source, result=dict(verified=False, job_id=job['job_id'], candidate=ref))
    row['execution_directory'] = 'tasks/retained'
    write(source / 'state.json', dict(phase='optimizer-phase2', attempts=[row]))
    store = FeedbackStore(tmp_path / 'feedback')
    assert store.import_campaign(source)['warnings'] == []
    outcome = store.records()[0]['outcome']
    assert outcome['runtime']['arch'] == 'actual'
    assert outcome['target'] == 'a3'
    assert outcome['evidence']['candidate']['sha256'] == sha(b'actual candidate source')
    page = store.read_evidence(ref['sha256'], limit=7)
    assert page['text'] == 'actual '
    assert page['next_offset'] == 7
    assert store.read_evidence(ref['sha256'], offset=7)['text'] == 'candidate source'
    jobs.artifact_path(job['job_id'], ref['sha256']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='digest mismatch'):
        store.read_evidence(ref['sha256'])


def test_changed_source_is_reported_without_overwriting_observation(tmp_path):
    root = tmp_path / 'source'
    campaign(root, result={'verified': False})
    store = FeedbackStore(tmp_path / 'feedback')
    store.import_campaign(root)
    campaign(root, result={'verified': True})
    assert 'source outcome changed' in store.import_campaign(root)['warnings'][0]
    assert store.records()[0]['outcome']['verified'] is False


def test_compiler_diagnostics_classify_native_incorrect_report(tmp_path):
    root = tmp_path / 'source'
    campaign(root, result=dict(verified=False, result=dict(status='incorrect', cases=[
        dict(error='CalledProcessError: bisheng exited 1', compiler_diagnostics='invalid syntax')
    ])))
    store = FeedbackStore(tmp_path / 'feedback')
    store.import_campaign(root)
    assert store.records()[0]['outcome']['category'] == 'compile-failure'


@pytest.mark.parametrize('unknown', ['target', 'language'])
def test_unknown_identity_cannot_resolve_even_when_equal(tmp_path, unknown):
    store = FeedbackStore(tmp_path / 'feedback')
    for name, verified in [('failure', False), ('retry', True)]:
        root = tmp_path / name
        campaign(root, result={'verified': verified})
        identity = dict(target='a3', language='ascend-c')
        identity[unknown] = 'unknown'
        write(root / 'seed-memory.json', dict(source_identity=identity))
        store.import_campaign(root)
    records = {r['outcome']['verified']: r for r in store.records()}
    with pytest.raises(ValueError, match='known target and language'):
        store.update_investigation(records[False]['feedback_id'], status='resolved',
                                   retries=[records[True]['feedback_id']])


@pytest.mark.parametrize('mutation', ['suite', 'policy', 'missing', 'corrupt'])
def test_resolution_requires_same_authenticated_frozen_suite_policy(tmp_path, mutation):
    store = FeedbackStore(tmp_path / 'feedback')
    campaign(tmp_path / 'failure', result={'verified': False})
    baseline = dict(suite_sha256='1' * 64, policy=dict(name='host-policy', rtol=.01))
    if mutation == 'suite':
        baseline['suite_sha256'] = '2' * 64
    if mutation == 'policy':
        baseline['policy']['rtol'] = .5
    row = campaign(tmp_path / 'retry', result={'verified': True}, baseline=baseline)
    ref = row['result']['baseline']
    path = tmp_path / 'retry' / 'execution' / 'jobs' / row['result']['job_id'] / 'artifacts' / ref['sha256']
    if mutation == 'missing':
        path.unlink()
    if mutation == 'corrupt':
        path.write_bytes(b'corrupt')
    store.import_campaign(tmp_path / 'failure')
    store.import_campaign(tmp_path / 'retry')
    records = {r['outcome']['verified']: r for r in store.records()}
    with pytest.raises(ValueError, match='successful original-task retry'):
        store.update_investigation(records[False]['feedback_id'], status='resolved',
                                   retries=[records[True]['feedback_id']])
    assert records[False]['investigation']['status'] == 'open'


def test_resolution_rejects_missing_original_baseline_but_keeps_observation(tmp_path):
    store = FeedbackStore(tmp_path / 'feedback')
    campaign(tmp_path / 'failure', result={'verified': False}, baseline=False)
    campaign(tmp_path / 'retry', result={'verified': True})
    store.import_campaign(tmp_path / 'failure')
    store.import_campaign(tmp_path / 'retry')
    records = {r['outcome']['verified']: r for r in store.records()}
    with pytest.raises(ValueError, match='authenticated frozen suite'):
        store.update_investigation(records[False]['feedback_id'], status='resolved',
                                   retries=[records[True]['feedback_id']])


def test_resolution_requires_retry_record_in_same_store(tmp_path):
    store = FeedbackStore(tmp_path / 'feedback')
    campaign(tmp_path / 'failure', result={'verified': False})
    campaign(tmp_path / 'retry', result={'verified': True})
    store.import_campaign(tmp_path / 'failure')
    elsewhere = FeedbackStore(tmp_path / 'another-store')
    elsewhere.import_campaign(tmp_path / 'retry')
    failure = store.records()[0]['feedback_id']
    retry = elsewhere.records()[0]['feedback_id']
    with pytest.raises(ValueError, match='successful original-task retry'):
        store.update_investigation(failure, status='resolved', retries=[retry])


def test_restored_campaign_preserves_old_feedback_id_and_investigation(tmp_path):
    import shutil
    root = tmp_path / 'original'
    campaign(root, result={'verified': False})
    store = FeedbackStore(root / 'feedback')
    store.import_campaign(root)
    record = store.records()[0]
    # Simulate a pre-migration ID generated from the absolute source path.
    old_id = sha(canonical({key: record['outcome'][key]
                           for key in ('source_root', 'attempt_id', 'completed')}))
    (store.root / (record['feedback_id'] + '.json')).unlink()
    record['feedback_id'] = old_id
    record['investigation'].update(status='deferred', reason='needs isolated probe', retries=['retry-link'])
    write(store.root / (old_id + '.json'), record)
    restored = tmp_path / 'restored'
    shutil.copytree(root, restored)
    imported = FeedbackStore(restored / 'feedback')
    report = imported.import_campaign(restored)
    assert report['imported'] == 0
    assert report['existing'] == 1
    assert report['warnings'] == []
    assert imported.records() == [record]
    # An independent execution of the same authored task is not a duplicate.
    campaign(tmp_path / 'independent', result={'verified': False})
    assert imported.import_campaign(tmp_path / 'independent')['imported'] == 1
