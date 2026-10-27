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


def test_legacy_ready_phase2_retains_history_and_dispatches_original_retry(tmp_path):
    from benchmarks.ascend.failure_curriculum import retry_spec, select_feedback
    from test_phase2_feedback import campaign

    source = tmp_path / 'phase2'
    task = dict(task_id='original', submission=dict(source='reference', entrypoint='forward', signatures=[]))
    baseline = dict(suite_sha256='1' * 64, manifest=dict(
        policy=dict(name='frozen-policy'), submission=task['submission']))
    campaign(source, task=task, baseline=baseline)
    store = FeedbackStore(tmp_path / 'feedback')
    store.import_campaign(source)
    record = store.records()[0]
    feedback_id = record['feedback_id']
    path = store.root / (feedback_id + '.json')
    legacy = json.loads(path.read_bytes())
    legacy['outcome'].pop('source_type')
    legacy['outcome'].pop('original_spec')
    legacy['investigation'].update(status='ready-to-retry', evidence=['committed-probe'],
                                   projects=['probe'], retries=['older-retry'])
    write(path, legacy)
    retained = path.read_bytes()

    assert store.import_campaign(source)['existing'] == 1
    assert path.read_bytes() == retained
    selected = select_feedback(store.root, 'a3', retry_id=feedback_id)
    ready = selected['selected'][0]
    assert ready['outcome']['source_type'] == 'phase2'
    assert ready['outcome']['original_spec'] == legacy['outcome']['task']
    assert ready['investigation'] == legacy['investigation']
    assert ready['lifecycle'] == 'remediated'
    # This is the optimizer branch selected by automatic retry dispatch.
    retry = retry_spec(selected, 'a3')
    assert retry['task'] == legacy['outcome']['task']
    assert retry['frozen_submission'] == task['submission']
    assert retry['suite_sha256'] == baseline['suite_sha256']
    assert path.read_bytes() == retained


def test_later_failed_original_retry_preserves_authenticated_fix(tmp_path):
    store = FeedbackStore(tmp_path / 'feedback')
    ids = {}
    for name, verified in [('original', False), ('successful', True), ('later-failed', False)]:
        root = tmp_path / name
        phase1(root, verified=verified)
        store.import_phase1(root)
        ids[name] = next(row['feedback_id'] for row in store.records()
                         if row['outcome']['source_root'] == str(root.resolve()))
    store.link_original_retry(ids['original'], ids['successful'])
    record = store.link_original_retry(ids['original'], ids['later-failed'])
    assert record['lifecycle'] == 'fixed'
    assert record['investigation']['status'] == 'resolved'
    assert record['investigation']['retries'] == [ids['successful'], ids['later-failed']]
    # Retaining fixed status still requires intact evidence, not a stale status bit.
    path = tmp_path / 'successful/waves/wave-1/branches/original/execution-evidence.bin'
    path.write_bytes(b'changed host evidence')
    assert store.link_original_retry(ids['original'], ids['later-failed'])['lifecycle'] == 'failed'


def test_failure_cases_keep_failed_action_after_verified_repair(tmp_path):
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.ascend.failure_cases import retain_phase1_failure
    from test_a5_phase1_wave_runtime import project

    spec = project('original')
    root = tmp_path / 'phase1'
    phase1(root, verified=True, spec=spec.as_dict())
    branch = root / 'waves/wave-1/branches/original'
    artifacts = ContentAddressedArtifacts(branch / 'artifacts')
    source = artifacts.put_bytes(b'broken source')
    payloads = dict(inputs={'tensors': [dict(storage=[1, 2])]}, expected={'storage': [3]},
                    observed={'output': [0]})
    retain_phase1_failure(artifacts, spec, source, 'host-mismatch', 'zero differs from three', payloads, 'float32')
    # A later run of the same candidate must not overwrite the first observation.
    retain_phase1_failure(artifacts, spec, source, 'host-mismatch', 'different case', {}, 'float32')
    store = FeedbackStore(tmp_path / 'feedback')
    store.import_phase1(root)
    record = store.records()[0]
    retained = (store.root / (record['feedback_id'] + '.json')).read_bytes()
    case = store.failure_cases(record['feedback_id'])[0]
    assert case['availability']['complete']
    assert case['description'] == 'zero differs from three'
    assert case['project_contract']['policy'] == 'phase1-classify-execution-v1'
    observed = case['artifacts']['observed']['sha256']
    assert json.loads(store.failure_artifact_bytes(record['feedback_id'], observed)) == {'output': [0]}
    assert store.read_failure_artifact(record['feedback_id'], observed, limit=2)['has_more']
    store.import_phase1(root)
    assert retained == (store.root / (record['feedback_id'] + '.json')).read_bytes()
    path = case['artifacts']['observed']['retained_path']
    from pathlib import Path
    Path(path).write_bytes(b'changed')
    assert not store.failure_cases(record['feedback_id'])[0]['availability']['complete']
    with pytest.raises(ValueError, match='digest'):
        store.failure_artifact_bytes(record['feedback_id'], observed)


def test_optimizer_failed_candidate_survives_successful_best(tmp_path):
    jobs = JobStore(tmp_path / 'jobs')
    job = jobs.create('reference', 'forward', {}, {})
    key = job['job_id']
    jobs.transition(key, 'running')
    source = jobs.put_artifact(key, 'candidate', canonical('bad source'))
    content = dict(status='incorrect', candidate_sha256=sha(b'bad source'),
        failure_case=dict(schema='failure-case-v1', classification='host-mismatch', description='actual zero, expected one',
            inputs={'x': 1}, expected={'output': 1}, observed={'output': 0}, state={}, policy={'exact': True}))
    evaluation = jobs.put_artifact(key, 'evaluation', canonical(content))
    frozen = dict(suite_sha256='1' * 64,
        manifest=dict(policy={'exact': True}, submission=jobs.submission(key)))
    baseline = jobs.put_artifact(key, 'baseline', canonical(frozen))
    jobs.add_event(key, 'optimizer-state', dict(best=dict(speedup=2, content={'status': 'verified'}), baseline=baseline,
        history=[dict(candidate_sha256=sha(b'bad source'), source=source, result=evaluation, content=content)]))
    jobs.transition(key, 'completed')
    store = FeedbackStore(tmp_path / 'feedback')
    assert store.import_optimizer(jobs.root)['imported'] == 2
    records = store.records()
    failed = next(row for row in records if not row['outcome']['verified'])
    assert any(row['outcome']['verified'] for row in records)
    assert store.failure_cases(failed['feedback_id'])[0]['availability']['complete']
    from benchmarks.ascend.failure_curriculum import retry_spec
    from benchmarks.ascend.phase2_feedback import _baseline_identity
    retry = retry_spec({'selected': [failed]}, 'a3')
    assert retry['suite_sha256'] == frozen['suite_sha256']
    assert retry['policy'] == frozen['manifest']['policy']
    assert retry['frozen_submission'] == frozen['manifest']['submission']
    assert _baseline_identity(failed['outcome']) == (frozen['suite_sha256'], frozen['manifest']['policy'])
    assert store.import_optimizer(jobs.root)['existing'] == 2
    jobs.artifact_path(key, baseline['sha256']).write_bytes(b'changed baseline')
    with pytest.raises(ValueError, match='digest'):
        retry_spec({'selected': [failed]}, 'a3')


def test_legacy_failure_explains_missing_reproducer(tmp_path):
    phase1(tmp_path / 'source')
    store = FeedbackStore(tmp_path / 'feedback')
    store.import_phase1(tmp_path / 'source')
    case = store.failure_cases(store.records()[0]['feedback_id'])[0]
    assert not case['availability']['complete']
    assert 'unavailable' in case['availability']['reason']


def test_legacy_exact_execution_reproducer_imports_without_index(tmp_path):
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from test_a5_phase1_wave_runtime import project
    spec = project('original')
    root = tmp_path / 'source'
    phase1(root, spec=spec.as_dict())
    branch = root / 'waves/wave-1/branches/original'
    artifacts = ContentAddressedArtifacts(branch / 'artifacts')
    refs = {role: artifacts.put_json(value) for role, value in dict(
        inputs={'tensors': []}, expected={'dtype': 'float32', 'storage': [1]}, observed={'output': [0]}).items()}
    write(branch / 'execution-evidence.bin', dict(classification='host-mismatch',
        detail='observed zero, expected one', source_sha256=artifacts.put_bytes(b'zero kernel'), artifacts=refs))
    store = FeedbackStore(tmp_path / 'feedback')
    assert store.import_phase1(root)['imported'] == 1
    case = store.failure_cases(store.records()[0]['feedback_id'])[0]
    assert case['availability']['complete']
    assert case['project_contract']['inputs_sha256'] == refs['inputs']


def test_phase2_failure_payload_roundtrip(tmp_path):
    from test_phase2_feedback import campaign
    root = tmp_path / 'phase2'
    row = campaign(root, result=dict(verified=False, result=dict(status='incorrect',
        failure_case=dict(schema='failure-case-v1', classification='host-mismatch', description='zero differs',
            inputs={'x': [1]}, expected={'x': [1]}, observed={'x': [0]}, state={}, policy={'exact': True}))))
    jobs = JobStore(root / 'execution/jobs')
    row['result']['candidate'] = jobs.put_artifact(row['result']['job_id'], 'source', canonical('bad candidate'))
    write(root / 'state.json', dict(phase='optimizer-phase2', attempts=[row]))
    store = FeedbackStore(tmp_path / 'feedback')
    assert store.import_campaign(root)['imported'] == 1
    feedback_id = store.records()[0]['feedback_id']
    case = store.failure_cases(feedback_id)[0]
    assert case['availability']['complete']
    assert json.loads(store.failure_artifact_bytes(feedback_id, case['artifacts']['expected']['sha256'])) == {'x': [1]}


@pytest.mark.parametrize('direct_run', [False, True])
def test_compile_failure_retains_contract_and_diagnostics_without_output(tmp_path, direct_run):
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.ascend.failure_cases import retain_phase1_failure
    from test_a5_phase1_wave_runtime import project
    spec = project('original')
    root = tmp_path / 'source'
    phase1(root, spec=spec.as_dict())
    artifacts = ContentAddressedArtifacts(root / 'waves/wave-1/branches/original/artifacts')
    payload = {'compile_passed': False, 'compile_diagnostic': 'unknown identifier', 'output': None}
    retain_phase1_failure(artifacts, spec, artifacts.put_bytes(b'invalid source'), 'compile-failure',
        'unknown identifier', {'observed' if direct_run else 'diagnostics': payload}, 'float32')
    store = FeedbackStore(tmp_path / 'feedback')
    store.import_phase1(root)
    case = store.failure_cases(store.records()[0]['feedback_id'])[0]
    assert case['availability']['complete']
    assert case['project_contract']['inputs_sha256'] == case['artifacts']['inputs']['sha256']
    assert case['project_contract']['expected_sha256'] == case['artifacts']['expected']['sha256']
    diagnostics = json.loads(store.failure_artifact_bytes(store.records()[0]['feedback_id'], case['artifacts']['diagnostics']['sha256']))
    assert diagnostics['compile_diagnostic'] == 'unknown identifier'
    assert diagnostics['output'] is None


def test_failure_artifact_paging_streams_requested_file_only(tmp_path, monkeypatch):
    from pathlib import Path
    phase1(tmp_path / 'source')
    store = FeedbackStore(tmp_path / 'feedback')
    store.import_phase1(tmp_path / 'source')
    feedback_id = store.records()[0]['feedback_id']
    data = b'0123456789' * 200000
    artifact = tmp_path / 'large-inputs.json'
    artifact.write_bytes(data)
    digest = sha(data)
    missing = tmp_path / 'missing-output.json'
    write(store.root / 'failure-cases' / (feedback_id + '.json'), [dict(
        classification='host-mismatch', artifacts=dict(
            inputs=dict(sha256=digest, retained_path=str(artifact)),
            observed=dict(sha256='1' * 64, retained_path=str(missing))))])
    original_read_bytes = Path.read_bytes
    def forbid_payload_materialization(path):
        if path in {artifact, missing}:
            pytest.fail('paged reads must not materialize payloads or read sibling artifacts')
        return original_read_bytes(path)
    monkeypatch.setattr(Path, 'read_bytes', forbid_payload_materialization)
    page = store.read_failure_artifact(feedback_id, digest, offset=123, limit=17)
    assert page['text'] == data[123:140].decode()
    assert page['size'] == len(data)
    assert page['has_more'] and page['next_offset'] == 140
    # Availability checks also authenticate incrementally and explain missing siblings.
    assert not store.failure_cases(feedback_id)[0]['availability']['complete']
    artifact.write_bytes(b'changed' + data[7:])
    with pytest.raises(ValueError, match='digest'):
        store.read_failure_artifact(feedback_id, digest, offset=123, limit=17)
    with pytest.raises(FileNotFoundError, match='not linked'):
        store.read_failure_artifact(feedback_id, '2' * 64)


@pytest.mark.parametrize('source_type', ['phase1', 'phase2', 'optimizer'])
@pytest.mark.parametrize('configured', [False, True])
def test_feedback_cli_imports_retained_sources_without_dispatch(tmp_path, capsys, source_type, configured):
    from benchmarks.ascend.campaign_cli import main
    root = tmp_path / 'source'
    if source_type == 'phase1':
        phase1(root)
    elif source_type == 'phase2':
        from test_phase2_feedback import campaign
        campaign(root)
    else:
        jobs = JobStore(root)
        job = jobs.create('source', 'forward', {}, {})
        jobs.transition(job['job_id'], 'running')
        jobs.transition(job['job_id'], 'failed')
    feedback = tmp_path / 'feedback'
    arguments = ['feedback', '--feedback-root', str(feedback), '--source-type', source_type]
    if configured:
        configuration = tmp_path / 'dashboard.json'
        write(configuration, dict(campaigns=[dict(root=str(root), phase=1 if source_type == 'phase1' else 2)],
                                  optimizer=dict(root=str(root))))
        arguments += ['--dashboard-config', str(configuration)]
    else:
        arguments += ['--source-root', str(root)]
    main(arguments, screen=lambda *args: pytest.fail('imports must not dispatch agents'))
    report = json.loads(capsys.readouterr().out)
    assert len(report['reports']) == 1 and report['reports'][0]['imported'] == 1
    records = FeedbackStore(feedback).records()
    assert len(records) == 1 and records[0]['outcome']['source_type'] == source_type
    # The historical command remains a Phase2 import by default.
    if source_type == 'phase2':
        main(['feedback-phase2', '--feedback-root', str(feedback), '--source-root', str(root)],
             screen=lambda *args: pytest.fail('imports must not dispatch agents'))
        assert json.loads(capsys.readouterr().out)['reports'][0]['existing'] == 1


def test_curriculum_discovers_and_pages_only_selected_failure_cases(tmp_path, monkeypatch):
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.ascend.failure_cases import retain_phase1_failure
    from test_a5_phase1_wave_runtime import project

    store = FeedbackStore(tmp_path / 'feedback')
    ids = {}
    for name, observed in [('selected', [0]), ('unselected', [99])]:
        root = tmp_path / name
        spec = project(name)
        phase1(root, spec=spec.as_dict())
        artifacts = ContentAddressedArtifacts(root / 'waves/wave-1/branches' / name / 'artifacts')
        source = artifacts.put_bytes(b'broken source')
        retain_phase1_failure(artifacts, spec, source, 'host-mismatch', 'observed mismatch',
            dict(inputs={'tensors': []}, expected={'storage': [3]}, observed={'output': observed}), 'float32')
        store.import_phase1(root)
        ids[name] = next(row['feedback_id'] for row in store.records()
                         if row['outcome']['source_root'] == str(root.resolve()))
    selected = next(row for row in store.records() if row['feedback_id'] == ids['selected'])
    curriculum = FailureCurriculum(tmp_path / 'curriculum', store.root,
                                   dict(selected=[selected], controls=[]))
    context = curriculum.context({})
    index = next(row for row in context if row.get('role') == 'selection-index')
    manifest = index['selected'][0]['failure_cases_file']
    offset, chunks = 0, []
    while offset is not None:
        page = curriculum.read(manifest['sha256'], offset, 73)
        assert len(page['text'].encode()) <= 73
        chunks.append(page['text'])
        offset = page['next_offset']
    cases = json.loads(''.join(chunks))
    assert sha(''.join(chunks).encode()) == manifest['sha256']
    assert cases[0]['availability']['complete']
    observed = cases[0]['artifacts']['observed']['sha256']
    monkeypatch.setattr(FeedbackStore, 'failure_artifact_bytes',
                        lambda *args: pytest.fail('curriculum reads must use bounded pages'))
    assert curriculum.read(observed, limit=2)['next_offset'] == 2
    assert json.loads(curriculum.read(observed)['text']) == {'output': [0]}
    other = store.failure_cases(ids['unselected'])[0]['artifacts']['observed']['sha256']
    with pytest.raises(ValueError):
        curriculum.read(other)
    from pathlib import Path
    Path(cases[0]['artifacts']['observed']['retained_path']).write_bytes(b'changed')
    with pytest.raises(ValueError):
        curriculum.read(observed)



@pytest.mark.parametrize('historical_sources', [False, True])
def test_optimizer_import_keeps_failed_candidates_after_interleaved_real_tools(tmp_path, historical_sources):
    from optimizer.loop import OptimizationLoop
    from test_optimizer_loop import Agent, Backend, candidate
    body = dict(files={}, memory_entries=[{'lesson': 'Retained context'}],
                source_identity=dict(target='a3', language='ascend-c'))
    jobs = JobStore(tmp_path / 'jobs')
    job = jobs.create('def forward(x): return x', 'forward',
        {'args': [{'shape': [5], 'dtype': 'float32'}]},
        dict(body, snapshot_id=sha(canonical(body))))
    key = job['job_id']
    readable = jobs.put_artifact(key, 'retained-context', b'Actual context bytes')

    class FailingBackend(Backend):
        def compile(self, source, directory, *, block_count):
            return {'status': 'compile-error', 'stderr': 'Observed compile diagnostic'}
        def evaluate(self, source, directory, *, block_count):
            self.evaluations.append(source)
            if historical_sources:
                return dict(status='compile-error', suite_sha256='suite',
                    candidate_sha256=sha(source.encode()), cases=[dict(case='case-000',
                        error='compiler rejected candidate', compiler_diagnostics='Actual retained diagnostic')])
            return dict(status='incorrect', suite_sha256='suite', cases=[],
                candidate_sha256=sha(source.encode()), failure_case=dict(
                    schema='failure-case-v1', classification='host-mismatch',
                    description='Observed zero differs from expected one',
                    inputs={'x': [1]}, expected={'output': [1]}, observed={'output': [0]}))

    backend = FailingBackend()
    actions = [dict(action='cases', signatures=[], summary='Retain frozen input'),
        dict(action='read-memory', index=0),
        dict(action='read-artifact', sha256=readable['sha256'], offset=0),
        dict(action='query-knowledge', query='First retrieval'),
        dict(action='compile', source='compile probe', block_count=1, summary='Read compiler error'),
        candidate('bad-one'), dict(action='query-knowledge', query='Second retrieval'),
        candidate('bad-two'), dict(action='read-memory', index=0),
        dict(action='finish', summary='Retained failed candidates')]
    loop = OptimizationLoop(jobs, key, Agent(actions), backend)
    if historical_sources:
        # The old producer retained canonical source artifacts and current
        # candidate state, but not the additive history.source reference.
        def save_legacy_state():
            state = json.loads(canonical(loop.state))
            for row in state['history']:
                row.pop('source', None)
            jobs.add_event(key, 'optimizer-state', state)
        loop._save = save_legacy_state
    result = loop.run()
    assert result['status'] == 'completed' and backend.evaluations == ['bad-one', 'bad-two']
    history = OptimizationLoop(jobs, key, Agent([]), backend).state['history']
    assert [row.get('tool', {}).get('action') for row in history if 'tool' in row] == [
        'read-memory', 'read-artifact', 'query-knowledge', 'compile', 'query-knowledge', 'read-memory']
    store = FeedbackStore(tmp_path / 'feedback')
    report = store.import_optimizer(jobs.root)
    assert report['imported'] == 3 and report['warnings'] == []
    records = store.records()
    assert sum(row['outcome']['attempt_id'] == key for row in records) == 1
    candidates = [row for row in records if ':candidate:' in row['outcome']['attempt_id']]
    assert {row['outcome']['result']['source_sha256'] for row in candidates} == {
        sha(b'bad-one'), sha(b'bad-two')}
    for row in candidates:
        source = store.failure_cases(row['feedback_id'])[0]['artifacts']['source']['sha256']
        assert sha(json.loads(store.failure_artifact_bytes(row['feedback_id'], source)).encode()) == row['outcome']['result']['source_sha256']
    assert all(store.failure_cases(row['feedback_id'])[0]['availability']['complete'] for row in candidates)
    repeated = store.import_optimizer(jobs.root)
    assert repeated['existing'] == 3 and repeated['warnings'] == []


def test_a5_production_identity_imports_selects_and_authenticates_original_retry(tmp_path, capsys):
    from benchmarks.a5kernels.phase1_adaptive_production import _bind_campaign_identity
    from benchmarks.ascend.campaign_cli import main
    from benchmarks.ascend.failure_curriculum import select_feedback
    from test_a5_phase1_adaptive_production import _cell
    from test_a5_phase1_wave_runtime import project

    identity_root = tmp_path / 'producer'
    _bind_campaign_identity(identity_root, _cell())
    identity = (identity_root / 'campaign-identity.json').read_bytes()
    original_spec = project('original').as_dict()
    roots = dict(original=tmp_path / 'original', changed=tmp_path / 'changed', retry=tmp_path / 'retry')
    for name, root in roots.items():
        phase1(root, verified=name != 'original',
               spec=project('different').as_dict() if name == 'changed' else original_spec)
        (root / 'campaign-identity.json').write_bytes(identity)
    feedback_root = tmp_path / 'feedback'
    main(['feedback', '--source-type', 'phase1', '--feedback-root', str(feedback_root),
          '--source-root', str(roots['original'])], screen=lambda *args: pytest.fail('import dispatched work'))
    report = json.loads(capsys.readouterr().out)['reports'][0]
    assert report['imported'] == 1
    selection = select_feedback(feedback_root, 'a5')
    original = selection['selected'][0]
    assert (original['outcome']['target'], original['outcome']['language']) == ('a5', 'catlass-dsl')
    assert original['outcome']['original_spec'] == original_spec
    store = FeedbackStore(feedback_root)
    ids = {}
    for name in ('changed', 'retry'):
        assert store.import_phase1(roots[name])['imported'] == 1
        ids[name] = next(row['feedback_id'] for row in store.records()
                         if row['outcome']['source_root'] == str(roots[name].resolve()))
    assert store.link_original_retry(original['feedback_id'], ids['changed'])['lifecycle'] == 'failed'
    assert store.link_original_retry(original['feedback_id'], ids['retry'])['lifecycle'] == 'fixed'
    assert all((root / 'campaign-identity.json').read_bytes() == identity for root in roots.values())


@pytest.mark.parametrize('identity,target,language', [
    ({'schema': 'future-production-v2', 'cell': {'target': 'a5', 'language': 'catlass-dsl'}}, 'unknown', 'unknown'),
    ({'schema': 'a5-phase1-production-identity-v1', 'cell': {'target': 'a5', 'language': 'other'}}, 'unknown', 'unknown'),
    ({'target': 'native-a5', 'language': 'catlass-dsl'}, 'a5', 'catlass-dsl'),
    ({'target': 'native-a3', 'language': 'ascend-c'}, 'a3', 'ascend-c'),
])
def test_identity_normalization_keeps_unknown_schemas_and_existing_aliases(tmp_path, identity, target, language):
    assert FeedbackStore._identity(tmp_path, None, None, identity) == (target, language)


@pytest.mark.parametrize('change_contract', [False, True])
def test_retained_campaign_retry_links_standalone_optimizer_without_new_launch(
        tmp_path, change_contract):
    from benchmarks.ascend.campaign_cli import main, campaign_worker
    from benchmarks.ascend.failure_curriculum import link_retry, retry_spec
    from benchmarks.ascend.phase2_optimizer_campaign import OptimizerPhase2Coordinator
    from test_ascend_phase2_optimizer_tasks import seed

    submission = dict(source='def forward(x): return x', entrypoint='forward',
        signatures=[dict(args=[dict(shape=[5], dtype='float32')])], init_args=[], init_kwargs={})
    memory = seed()
    memory['source_identity'] = dict(target='a3', language='ascend-c')
    memory['snapshot_id'] = sha(canonical({k: v for k, v in memory.items() if k != 'snapshot_id'}))
    jobs = JobStore(tmp_path / 'original-jobs')
    original_job = jobs.create(submission['source'], submission['entrypoint'],
        {'signatures': submission['signatures']}, memory, name='original-task')
    key = original_job['job_id']
    frozen = dict(suite_sha256='1' * 64,
        manifest=dict(policy={'rtol': 1e-5}, submission=submission),
        cases=[dict(case='case-000', timing={'torch': {'latency_us': 10}})])
    baseline = jobs.put_artifact(key, 'baseline', canonical(frozen))
    jobs.transition(key, 'running')
    jobs.add_event(key, 'optimizer-state', dict(baseline=baseline, history=[]))
    jobs.transition(key, 'failed')
    store = FeedbackStore(tmp_path / 'feedback')
    assert store.import_optimizer(jobs.root)['imported'] == 1
    original = store.records()[0]
    pinned = retry_spec(dict(selected=[original]), 'a3')
    assert pinned['task']['submission'] == submission
    task = json.loads(canonical(pinned['task']))
    if change_contract:
        task['submission']['source'] = 'def forward(x): return x + 1'
    repeated = json.loads(canonical(frozen))
    repeated['cases'][0]['timing']['torch']['latency_us'] = 12

    class Retried:
        def __init__(self, directory, selected, memory):
            self.task = selected
            spec = selected['submission']
            self.jobs = JobStore(directory / 'jobs')
            self.key = self.jobs.create(spec['source'], spec['entrypoint'],
                {'signatures': spec['signatures']}, memory,
                init_args=spec.get('init_args'), init_kwargs=spec.get('init_kwargs'))['job_id']
            self.baseline = self.jobs.put_artifact(self.key, 'baseline', canonical(repeated))
        def prepare(self):
            return repeated
        def attempt(self, attempt_id, **kwargs):
            source = 'candidate for the retained task'
            result = dict(attempt_id=attempt_id, task_id=self.task['task_id'], job_id=self.key,
                verified=True, speedup=2, baseline=self.baseline, source_sha256=sha(source.encode()),
                candidate=self.jobs.put_artifact(self.key, 'candidate', canonical(source)),
                result=dict(status='verified', candidate_sha256=sha(source.encode()),
                    suite_sha256=frozen['suite_sha256'], cases=[dict(case='case-000',
                        timing={'ascendc': {'latency_us': 5}})]))
            result['result_artifact'] = self.jobs.put_artifact(self.key, 'result', canonical(result))
            self.jobs.add_event(self.key, 'phase2-attempt-result', result)
            return result

    root = tmp_path / 'retained-retry'
    campaign = OptimizerPhase2Coordinator(root, 'Retained original retry', memory,
        lambda _: json.dumps(dict(action='ready', reason='Retained host result')), Retried,
        feedback_root=store.root, fixed_task=task)
    assert campaign.resume()['status'] == 'READY'
    # This is retained-result finalization through the existing worker API, not
    # permission to launch a retired Phase2 campaign from the CLI.
    control = tmp_path / 'control'
    control.mkdir()
    job = dict(root=str(root), control=str(control), feedback_root=str(store.root),
               retry_feedback_id=original['feedback_id'])
    assert campaign_worker(job, builder=lambda _: campaign)['status'] == 'READY'
    updated = next(row for row in store.records() if row['feedback_id'] == original['feedback_id'])
    assert (updated['investigation']['status'] == 'resolved') is not change_contract
    assert len(updated['investigation']['retries']) == 1
    assert json.loads((control / 'worker.json').read_bytes())['status'] == 'complete'
    link_retry(job)
    assert next(row for row in store.records() if row['feedback_id'] == original['feedback_id']) == updated
    before = (store.root / (original['feedback_id'] + '.json')).read_bytes()
    with pytest.raises(SystemExit) as error:
        main(['start', '--phase', '2', '--target', 'a3', '--control', str(tmp_path / 'new-control'),
            '--root', str(tmp_path / 'new-campaign'), '--objective', 'Original retry',
            '--config', 'absent', '--initial-memory', 'absent', '--project-target', '1',
            '--retry-feedback-id', original['feedback_id'], '--feedback-root', str(store.root)],
            screen=lambda *args: pytest.fail('retired retry dispatched an agent'))
    assert error.value.code == 2
    assert not (tmp_path / 'new-control').exists() and not (tmp_path / 'new-campaign').exists()
    assert (store.root / (original['feedback_id'] + '.json')).read_bytes() == before
