import json
from pathlib import Path

import pytest

from benchmarks.ascend.phase2_feedback import FeedbackStore
from optimizer.jobs import canonical, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def campaign(root, *, result=None, task=None, attempt='attempt-001', baseline=True):
    task = task or dict(task_id='relu', submission={'source': 'reference', 'entrypoint': 'forward', 'signatures': []}, rationale='learn')
    task = json.loads(canonical(task))
    spec = task.setdefault('submission', {'source': 'reference'})
    spec.setdefault('entrypoint', 'forward')
    spec.setdefault('signatures', [])
    row = dict(attempt_id=attempt, kind='target', task=task,
               result=result or dict(verified=False, summary='agent hypothesis'))
    if baseline and not row['result'].get('job_id'):
        from optimizer.jobs import JobStore
        jobs = JobStore(root / 'execution' / 'jobs')
        spec = task.get('submission', {'source': 'reference'})
        job = jobs.create(spec['source'], spec.get('entrypoint', 'forward'),
            {'signatures': spec.get('signatures', [])}, {}, idempotency_key=str(root),
            init_args=spec.get('init_args'), init_kwargs=spec.get('init_kwargs'))
        value = baseline if isinstance(baseline, dict) else dict(suite_sha256='1' * 64,
            manifest=dict(policy=dict(name='host-policy', rtol=.01)))
        value.setdefault('cases', [dict(case='case-0')])
        ref = jobs.put_artifact(job['job_id'], 'baseline.json', canonical(value))
        row.update(execution_directory='execution')
        row['result'].update(job_id=job['job_id'], baseline=ref)
        if row['result'].get('verified'):
            source = 'actual candidate source'
            candidate = jobs.put_artifact(job['job_id'], 'candidate.json', canonical(source))
            row['result'].update(candidate=candidate, source_sha256=sha(source.encode()),
                result=dict(status='verified', candidate_sha256=sha(source.encode()),
                    suite_sha256=value['suite_sha256'], cases=[dict(case='case-0')]))
        row['result'].update(attempt_id=attempt, task_id=task['task_id'])
        result_ref = jobs.put_artifact(job['job_id'], 'result.json', canonical(row['result']))
        row['result']['result_artifact'] = result_ref
        jobs.add_event(job['job_id'], 'phase2-attempt-result', row['result'])
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
    jobs.add_event(job['job_id'], 'phase2-attempt-request', dict(attempt_id=row['attempt_id']))
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
    jobs.add_event(job['job_id'], 'phase2-attempt-request', dict(attempt_id=row['attempt_id']))
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


@pytest.mark.parametrize('verified', [False, True])
def test_completed_attempt_event_precedes_interruption_synthesis(tmp_path, verified):
    from optimizer.jobs import JobStore
    source = tmp_path / 'source'
    row = campaign(source)
    write(source / 'state.json', dict(phase='optimizer-phase2', attempts=[]))
    write(source / 'pending.json', dict(attempt_id=row['attempt_id'], kind='target', task=row['task']))
    relative = 'tasks/relu-' + sha(canonical(row['task']))[:12]
    jobs = JobStore(source / relative / 'jobs')
    job = jobs.create('def f(x): return x', 'f', {}, {}, idempotency_key='pending')
    jobs.transition(job['job_id'], 'running')
    jobs.transition(job['job_id'], 'budget-exhausted')
    result = dict(attempt_id=row['attempt_id'], job_id=job['job_id'], verified=verified,
                  speedup=1.1, source_sha256='actual-candidate')
    jobs.add_event(job['job_id'], 'phase2-attempt-result', result)
    store = FeedbackStore(tmp_path / 'feedback')
    assert store.import_campaign(source)['imported'] == 1
    outcome = store.records()[0]['outcome']
    assert outcome['completed'] is True
    assert outcome['verified'] is verified
    assert outcome['category'] == ('verified-success' if verified else 'incorrect')
    assert store.import_campaign(source)['existing'] == 1


@pytest.mark.parametrize('retry_policy', ['matching', 'different-suite', 'different-policy'])
def test_prepared_interruption_retains_frozen_baseline_for_retry_resolution(tmp_path, retry_policy):
    from optimizer.jobs import JobStore
    root = tmp_path / 'interrupted'
    row = campaign(root, baseline=False)
    write(root / 'state.json', dict(phase='optimizer-phase2', attempts=[]))
    write(root / 'pending.json', dict(attempt_id=row['attempt_id'], kind='target', task=row['task']))
    relative = 'tasks/relu-' + sha(canonical(row['task']))[:12]
    jobs = JobStore(root / relative / 'jobs')
    job = jobs.create('def f(x): return x', 'f', {}, {}, idempotency_key='interrupted')
    baseline = dict(suite_sha256='1' * 64, cases=[dict(case='case-0')],
                    manifest=dict(policy=dict(name='host-policy', rtol=.01)))
    baseline_ref = jobs.put_artifact(job['job_id'], 'baseline.json', canonical(baseline))
    jobs.add_event(job['job_id'], 'phase2-attempt-request', dict(attempt_id=row['attempt_id']))
    candidate_ref = jobs.put_artifact(job['job_id'], 'candidate.cpp', canonical('unfinished candidate'))
    jobs.add_event(job['job_id'], 'optimizer-state', dict(baseline=baseline_ref, candidate=candidate_ref))
    jobs.transition(job['job_id'], 'stopped')
    store = FeedbackStore(tmp_path / 'feedback')
    store.import_campaign(root)
    original = store.records()[0]
    assert original['outcome']['completed'] is False
    assert original['outcome']['verified'] is False
    assert original['outcome']['evidence']['baseline']['sha256'] == baseline_ref['sha256']
    assert original['outcome']['evidence']['candidate']['sha256'] == candidate_ref['sha256']
    if retry_policy == 'different-suite':
        baseline['suite_sha256'] = '2' * 64
    if retry_policy == 'different-policy':
        baseline['manifest']['policy']['rtol'] = .5
    campaign(tmp_path / 'retry', task=row['task'], result={'verified': True}, baseline=baseline)
    store.import_campaign(tmp_path / 'retry')
    retry = next(record for record in store.records() if record['outcome']['verified'])
    if retry_policy == 'matching':
        assert store.update_investigation(original['feedback_id'], status='resolved',
            retries=[retry['feedback_id']])['investigation']['status'] == 'resolved'
    else:
        with pytest.raises(ValueError, match='successful original-task retry'):
            store.update_investigation(original['feedback_id'], status='resolved', retries=[retry['feedback_id']])


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


def test_stable_reimport_does_not_scan_feedback_records(tmp_path, monkeypatch):
    root = tmp_path / 'campaign'
    campaign(root, result={'verified': False})
    store = FeedbackStore(tmp_path / 'feedback')
    assert store.import_campaign(root)['imported'] == 1
    def unexpected_scan():
        raise AssertionError('stable existing identity must use direct lookup')
    monkeypatch.setattr(store, 'records', unexpected_scan)
    assert store.import_campaign(root)['existing'] == 1


def test_recreated_campaign_same_path_job_id_is_distinct_incarnation(tmp_path):
    import shutil
    root = tmp_path / 'campaign'
    store = FeedbackStore(tmp_path / 'external-feedback')
    first = campaign(root, result={'verified': False})
    store.import_campaign(root)
    shutil.rmtree(root)
    second = campaign(root, result={'verified': True})
    assert first['result']['job_id'] == second['result']['job_id']
    assert store.import_campaign(root)['imported'] == 1
    records = store.records()
    assert len(records) == 2
    assert len({row['outcome']['job_incarnation'] for row in records}) == 2


@pytest.mark.parametrize('mismatch', ['stale-flag', 'missing-native-case'])
def test_retry_resolution_authenticates_retained_native_verdict(tmp_path, mismatch):
    from optimizer.jobs import JobStore
    campaign(tmp_path / 'failure', result={'verified': False})
    root = tmp_path / 'retry'
    row = campaign(root, result={'verified': mismatch != 'stale-flag'})
    result = row['result']
    result['verified'] = True
    if mismatch == 'missing-native-case':
        result['result']['cases'] = []
        jobs = JobStore(root / 'execution' / 'jobs')
        body = {key: value for key, value in result.items() if key != 'result_artifact'}
        result['result_artifact'] = jobs.put_artifact(result['job_id'], 'result.json', canonical(body))
        jobs.add_event(result['job_id'], 'phase2-attempt-result', result)
    write(root / 'state.json', dict(phase='optimizer-phase2', attempts=[row]))
    store = FeedbackStore(tmp_path / 'feedback')
    store.import_campaign(tmp_path / 'failure')
    store.import_campaign(root)
    records = {r['outcome']['verified']: r for r in store.records()}
    with pytest.raises(ValueError, match='successful original-task retry'):
        store.update_investigation(records[False]['feedback_id'], status='resolved',
                                   retries=[records[True]['feedback_id']])


def test_new_pending_does_not_own_previous_attempt_terminal_status(tmp_path):
    from optimizer.jobs import JobStore
    root = tmp_path / 'source'
    old = campaign(root, baseline=False)
    relative = 'tasks/relu-' + sha(canonical(old['task']))[:12]
    jobs = JobStore(root / relative / 'jobs')
    job = jobs.create('def f(x): return x', 'f', {}, {}, idempotency_key='pending')
    jobs.transition(job['job_id'], 'running')
    jobs.add_event(job['job_id'], 'phase2-attempt-request', dict(attempt_id='attempt-001'))
    jobs.transition(job['job_id'], 'budget-exhausted')
    old.update(execution_directory=relative)
    old['result'].update(job_id=job['job_id'], attempt_id='attempt-001')
    jobs.add_event(job['job_id'], 'phase2-attempt-result', old['result'])
    write(root / 'state.json', dict(phase='optimizer-phase2', attempts=[old]))
    write(root / 'pending.json', dict(attempt_id='attempt-002', kind='target', task=old['task']))
    store = FeedbackStore(tmp_path / 'feedback')
    assert store.import_campaign(root)['imported'] == 1
    assert len(store.records()) == 1
    jobs.add_event(job['job_id'], 'phase2-prepare-request', dict(attempt_id='attempt-002'))
    assert store.import_campaign(root)['imported'] == 0  # Old terminal predates this preparation boundary.
    jobs.resume(job['job_id'], additional_attempts=1, additional_seconds=60)
    jobs.add_event(job['job_id'], 'phase2-attempt-request', dict(attempt_id='attempt-002'))
    jobs.transition(job['job_id'], 'failed')
    assert store.import_campaign(root)['imported'] == 1
    interrupted = next(row for row in store.records() if row['outcome']['attempt_id'] == 'attempt-002')
    assert interrupted['outcome']['completed'] is False


@pytest.mark.parametrize('field,value', [
    ('source', 'another reference'), ('entrypoint', 'another_entry'),
    ('signatures', [{'args': [{'shape': [99], 'dtype': 'float32'}]}]),
    ('init_args', [2]), ('init_kwargs', {'mode': 'different'}),
])
def test_resolution_rejects_misattached_frozen_task_submission(tmp_path, field, value):
    root = tmp_path / 'retry'
    row = campaign(root, result={'verified': True})
    changed_task = json.loads(canonical(row['task']))
    changed_task['submission'][field] = value
    row['task'] = changed_task  # Ordinary stale/misattached campaign row, same task ID.
    write(root / 'state.json', dict(phase='optimizer-phase2', attempts=[row]))
    campaign(tmp_path / 'failure', task=changed_task, result={'verified': False})
    store = FeedbackStore(tmp_path / 'feedback')
    store.import_campaign(tmp_path / 'failure')
    store.import_campaign(root)
    records = {record['outcome']['verified']: record for record in store.records()}
    with pytest.raises(ValueError, match='successful original-task retry'):
        store.update_investigation(records[False]['feedback_id'], status='resolved',
                                   retries=[records[True]['feedback_id']])


@pytest.mark.parametrize('window', ['result-event', 'pending-evaluation'])
def test_evaluated_recovery_defers_import_until_real_attempt_result(tmp_path, monkeypatch, window):
    from benchmarks.ascend.phase2_optimizer_execution import OptimizerTaskExecutor
    from optimizer.loop import OptimizationLoop
    from test_phase2_optimizer_execution import Agent, Backend, snapshot, candidate
    from test_ascend_phase2_optimizer_tasks import proposal
    root = tmp_path / 'campaign'
    task = proposal()
    relative = 'tasks/' + task['task_id'] + '-' + sha(canonical(task))[:12]
    agent = Agent([dict(action='cases', signatures=[], summary='representative'), candidate('good')])
    backend = Backend()
    config = dict(target='bz-a3-1', physical_device=14, validation_wrapper=str(tmp_path / 'validation.sh'), max_cases=1)
    execution = OptimizerTaskExecutor(root / relative, task, snapshot(), config, agent=agent, backend=backend)
    execution.prepare()
    write(root / 'state.json', dict(phase='optimizer-phase2', attempts=[]))
    write(root / 'pending.json', dict(attempt_id='attempt-001', kind='target', task=task))
    if window == 'result-event':
        original = execution.store.add_event
        def crash(kind_key, kind, payload):
            if kind == 'phase2-attempt-result':
                raise RuntimeError('checkpoint crash')
            return original(kind_key, kind, payload)
        monkeypatch.setattr(execution.store, 'add_event', crash)
    else:
        original = OptimizationLoop._save
        def crash(loop):
            original(loop)
            pending = loop.state.get('pending', {})
            if pending.get('evaluation') and not pending.get('history_recorded'):
                raise RuntimeError('checkpoint crash')
        monkeypatch.setattr(OptimizationLoop, '_save', crash)
    with pytest.raises(RuntimeError, match='checkpoint crash'):
        execution.attempt('attempt-001')
    calls = (len(agent.contexts), len(backend.evaluations))
    store = FeedbackStore(tmp_path / 'feedback')
    report = store.import_campaign(root)
    assert report['imported'] == 0
    if execution.store.get(execution.job_id)['status'] == 'running':
        # A result-event interruption leaves an active job; only terminal jobs
        # need an explicit warning that evaluated evidence awaits reconciliation.
        assert report['warnings'] == []
    else:
        assert any('import deferred' in warning for warning in report['warnings'])
    if window == 'result-event':
        monkeypatch.setattr(execution.store, 'add_event', original)
    else:
        monkeypatch.setattr(OptimizationLoop, '_save', original)
    assert execution.attempt('attempt-001')['verified'] is True
    assert (len(agent.contexts), len(backend.evaluations)) == calls
    assert store.import_campaign(root)['imported'] == 1
    assert len(store.records()) == 1 and store.records()[0]['outcome']['completed'] is True


@pytest.mark.parametrize('new_candidate', [False, True])
def test_interrupted_retry_does_not_inherit_previous_candidate(tmp_path, new_candidate):
    from optimizer.jobs import JobStore
    root = tmp_path / 'campaign'
    old = campaign(root, baseline=False)
    relative = 'tasks/relu-' + sha(canonical(old['task']))[:12]
    jobs = JobStore(root / relative / 'jobs')
    job = jobs.create('def f(x): return x', 'f', {}, {}, idempotency_key='retry')
    previous = jobs.put_artifact(job['job_id'], 'candidate.cpp', canonical('previous completed candidate'))
    baseline = jobs.put_artifact(job['job_id'], 'baseline.json', canonical({'suite_sha256': '1' * 64}))
    old.update(execution_directory=relative)
    old['result'].update(job_id=job['job_id'], attempt_id='attempt-001', candidate=previous)
    jobs.add_event(job['job_id'], 'phase2-attempt-result', old['result'])
    write(root / 'state.json', dict(phase='optimizer-phase2', attempts=[old]))
    write(root / 'pending.json', dict(attempt_id='attempt-002', kind='target', task=old['task']))
    jobs.add_event(job['job_id'], 'phase2-attempt-request', dict(attempt_id='attempt-002'))
    source = jobs.put_artifact(job['job_id'], 'candidate.cpp', canonical('new interrupted candidate')) if new_candidate else previous
    # A newer state checkpoint alone does not establish ownership of a persistent old candidate.
    jobs.add_event(job['job_id'], 'optimizer-state', dict(baseline=baseline, candidate=source))
    jobs.add_event(job['job_id'], 'phase2-prepare-request', dict(attempt_id='attempt-002'))
    jobs.transition(job['job_id'], 'stopped')
    store = FeedbackStore(tmp_path / 'feedback')
    report = store.import_campaign(root)
    interrupted = next(row['outcome'] for row in store.records() if not row['outcome']['completed'])
    assert interrupted['evidence']['baseline']['sha256'] == baseline['sha256']
    if new_candidate:
        assert interrupted['evidence']['candidate']['sha256'] == source['sha256']
    else:
        assert 'candidate' not in interrupted['evidence']
        assert any('candidate ownership' in warning for warning in report['warnings'])


def test_multisource_cli_reports_corrupt_copies_and_imports_valid_outcomes(tmp_path, capsys):
    from benchmarks.ascend.campaign_cli import main
    first, second = tmp_path / 'first', tmp_path / 'second'
    campaign(first, result={'verified': False})
    campaign(second, result={'verified': True})
    malformed = first / 'attempts' / 'truncated.json'
    malformed.parent.mkdir()
    malformed.write_text('{"attempt_id":')
    write(first / 'attempts' / 'wrong-shape.json', {'unexpected': 'copied partial metadata'})
    (first / 'campaign-run.json').write_text('{')
    (first / 'pending.json').write_text('{')
    main(['feedback-phase2', '--feedback-root', str(tmp_path / 'feedback'),
          '--source-root', str(first), '--source-root', str(second)])
    reports = json.loads(capsys.readouterr().out)['reports']
    assert [report['imported'] for report in reports] == [1, 1]
    warnings = reports[0]['warnings']
    assert any('truncated.json' in warning for warning in warnings)
    assert any('wrong-shape.json' in warning for warning in warnings)
    assert any('campaign-run.json' in warning for warning in warnings)
    assert any('pending.json' in warning for warning in warnings)
    assert len(FeedbackStore(tmp_path / 'feedback').records()) == 2
