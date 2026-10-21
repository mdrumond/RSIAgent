import json

import pytest

from benchmarks.ascend.phase2_feedback import FeedbackStore
from benchmarks.ascend.failure_curriculum import FailureCurriculum
from optimizer.jobs import JobStore, canonical, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def phase1(root, verified=False, spec=None):
    spec = spec or dict(project_id='original', acceptance={'rtol': .01}, cases=['frozen'])
    branch = root / 'waves/wave-1/branches' / spec['project_id']
    payload = canonical(dict(host_verified=verified))
    write(branch / 'execution-evidence.bin', json.loads(payload))
    write(branch / 'outcome.json', dict(verified=verified, evidence_sha256=sha(payload),
          wave_fingerprint='frozen-wave', project_id=spec['project_id']))
    write(branch.parent.parent / 'plan.json', dict(wave_fingerprint='frozen-wave',
          wave=dict(projects=[dict(spec=spec)])))
    write(root / 'campaign-identity.json', dict(schema='ascend-native-adaptive-identity-v1', target='Ascend910B4', language='ascend-c'))
    row = dict(project_id=spec['project_id'], spec=spec, verified=verified,
               evidence=dict(classification='host-mismatch'))
    write(root / 'state.json', dict(schema='a5-phase1-adaptive-state-v2',
          completed_waves=['wave-1'], failed_outcomes=[] if verified else [row],
          prior_outcomes=[row] if verified else []))


def test_phase1_committed_import_and_retry_lifecycle(tmp_path):
    root, retry = tmp_path/'phase1', tmp_path/'retry'
    phase1(root)
    phase1(retry, verified=True)
    store = FeedbackStore(tmp_path/'feedback')
    assert store.import_phase1(root)['imported'] == 1
    original = store.records()[0]
    assert original['lifecycle'] == 'failed'
    assert original['outcome']['target'] == 'a3'
    assert store.import_phase1(root)['existing'] == 1
    record = store.update_investigation(original['feedback_id'], status='ready-to-retry', evidence=['probe'])
    assert record['lifecycle'] == 'remediated'
    store.import_phase1(retry)
    attempt = next(row for row in store.records() if row['feedback_id'] != original['feedback_id'])
    fixed = store.link_original_retry(original['feedback_id'], attempt['feedback_id'])
    assert fixed['lifecycle'] == 'fixed'
    digest = attempt['outcome']['evidence']['execution-evidence.bin']['sha256']
    assert store.read_evidence(digest)['text'] == '{"host_verified":true}'


def test_phase1_retry_requires_original_spec_and_authentic_evidence(tmp_path):
    phase1(tmp_path/'original')
    phase1(tmp_path/'retry', True, dict(project_id='original', cases=['easier']))
    store = FeedbackStore(tmp_path/'feedback')
    store.import_phase1(tmp_path/'original')
    original = store.records()[0]
    store.import_phase1(tmp_path/'retry')
    attempt = next(row for row in store.records() if row['feedback_id'] != original['feedback_id'])
    assert store.link_original_retry(original['feedback_id'], attempt['feedback_id'])['lifecycle'] == 'failed'
    assert len(store.records()) == 2
    with pytest.raises(ValueError, match='distinct'):
        store.link_original_retry(original['feedback_id'], original['feedback_id'])


def test_phase1_import_ignores_pending_proposals(tmp_path):
    root = tmp_path/'source'
    phase1(root)
    state = json.loads((root/'state.json').read_bytes())
    state['pending'] = dict(wave=dict(projects=[dict(project_id='unexecuted')]))
    write(root/'state.json', state)
    store = FeedbackStore(tmp_path/'feedback')
    assert store.import_phase1(root)['imported'] == 1
    assert [r['outcome']['attempt_id'] for r in store.records()] == ['original']


def test_standalone_optimizer_import_is_terminal_and_deduplicated(tmp_path):
    jobs = JobStore(tmp_path/'jobs')
    job = jobs.create('source', 'forward', {}, dict(source_identity=dict(
        schema='ascend-native-adaptive-identity-v1', target='Ascend910B4', language='ascend-c')))
    store = FeedbackStore(tmp_path/'feedback')
    assert store.import_optimizer(jobs.root)['imported'] == 0
    jobs.transition(job['job_id'], 'running')
    jobs.add_event(job['job_id'], 'optimizer-state', dict(history=[]))
    jobs.transition(job['job_id'], 'failed')
    report = store.import_optimizer(jobs.root)
    assert report['imported'] == 1
    record = store.records()[0]
    assert record['outcome']['source_type'] == 'optimizer'
    assert record['outcome']['target'] == 'a3'
    assert record['outcome']['language'] == 'ascend-c'
    assert 'name' not in record['outcome']['original_spec']
    assert record['lifecycle'] == 'failed'
    assert record['outcome']['original_spec']['source'] == 'source'
    assert store.import_optimizer(jobs.root, target='a3', language='ascend-c')['existing'] == 1


def test_feedback_refresh_drops_fixed_items(tmp_path):
    phase1(tmp_path/'source')
    store = FeedbackStore(tmp_path/'feedback')
    store.import_phase1(tmp_path/'source')
    root = tmp_path/'runner'
    root.mkdir()
    curriculum = FailureCurriculum(root, store.root, dict(selected=[], controls=[]))
    assert len(curriculum.refresh('a3')['selected']) == 1
    store.update_investigation(store.records()[0]['feedback_id'], status='deferred', reason='provider unavailable')
    assert curriculum.refresh('a3')['selected'] == []
