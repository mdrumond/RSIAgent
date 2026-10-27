import hashlib
import io
import json
from pathlib import Path
from contextlib import redirect_stdout
import subprocess
import time

import pytest

from dashboard.app import create_app
from benchmarks.ascend.status_descriptions import describe_statuses


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def feedback(tmp_path):
    root = tmp_path / 'feedback'
    row = dict(feedback_id='a' * 64, outcome=dict(
        source_root=str(tmp_path / 'old-phase2'), attempt_id='attempt-001',
        target='a3', language='ascend-c', runtime={}, task={'task_id': 'relu'},
        verified=True, completed=True, category='verified-slow',
        result={'speedup': .4}, evidence={}), investigation=dict(
            status='open', classification=None, reason=None, projects=[], retries=[], evidence=[]))
    write(root / (row['feedback_id'] + '.json'), row)
    presets = [dict(id=f'phase{phase}', phase=phase, target='a3',
                    root=str(tmp_path / f'new{phase}'), control=str(tmp_path / f'control{phase}'),
                    config='/runtime.json', initial_memory='/memory.bin', objective='Investigate',
                    project_target=4, phase1_root='/reviewed-phase1', review='/review.json')
               for phase in (1, 2)]
    config = tmp_path / 'dashboard.json'
    write(config, dict(feedback_root=str(root), campaigns=[], controls=dict(
        enabled=True, runner='/repo/run_ascend_campaign.py', presets=presets)))
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, '{}', '')

    app = create_app(config, tmp_path / 'cache.sqlite', command_runner=runner)
    client = app.test_client()
    yield client, app, config, row, calls
    app.extensions['dashboard_controls'].close()


def presented(row):
    return describe_statuses({**row, 'lifecycle': 'failed', 'outcome': {
        **row['outcome'], 'source_type': 'phase2', 'original_spec': row['outcome']['task']}})


def submit(client, action, body):
    csrf = client.get('/api/csrf').get_json()['token']
    body['confirmation'] = f"{action}:{body.get('preset_id', 'phase2')}"
    return client.post('/api/control/' + action, json=body,
                       headers={'X-CSRF-Token': csrf, 'Idempotency-Key': json.dumps(body)})


def completed(client, response):
    assert response.status_code == 202, response.get_json()
    for _ in range(200):
        value = client.get('/api/operations/' + response.get_json()['operation_id']).get_json()
        if value['status'] not in {'queued', 'running'}:
            assert value['status'] == 'succeeded', value
            return
        time.sleep(.005)
    pytest.fail('operation did not finish')


def test_feedback_readonly_separates_correctness_performance_and_investigation(feedback):
    client, app, config, row, calls = feedback
    retained = config.parent / 'feedback' / (row['feedback_id'] + '.json')
    original_bytes = retained.read_bytes()
    body = client.get('/feedback').get_data(as_text=True)
    assert 'verified-slow' in body and 'shortfall' in body and 'Host correctness:' in body
    assert 'original attempt attempt-001' in body and 'Investigation:</b> open' in body
    assert client.get('/api/feedback').get_json() == [presented(row)]
    assert not calls
    assert retained.read_bytes() == original_bytes
    settings = json.loads(config.read_text())
    settings['controls']['enabled'] = False
    write(config, settings)
    readonly = create_app(config, config.parent / 'readonly.sqlite')
    try:
        view = readonly.test_client()
        assert 'data-feedback-action' not in view.get('/feedback').get_data(as_text=True)
        assert view.get('/api/feedback').get_json() == [presented(row)]
        assert view.post('/api/control/investigate', json={}).status_code == 403
    finally:
        readonly.extensions['dashboard_controls'].close()


def test_import_and_bounded_phase1_use_existing_async_runner(feedback):
    client, app, config, row, calls = feedback
    completed(client, submit(client, 'import-feedback', {}))
    assert calls[0][2:] == ['feedback', '--source-type', 'all', '--feedback-root',
                           str(config.parent / 'feedback'), '--dashboard-config', str(config)]
    completed(client, submit(client, 'investigate', dict(
        preset_id='phase1', feedback_ids=[row['feedback_id']], project_target=3)))
    argv = calls[-1]
    assert argv[2] == 'start' and '--approval-mode' in argv
    assert argv[argv.index('--project-target') + 1] == '3'
    assert argv[-4:] == ['--feedback-root', str(config.parent / 'feedback'),
                        '--failure-id', row['feedback_id']]


def test_retired_retry_is_refused_before_runner_and_preserves_history(feedback):
    client, app, config, row, calls = feedback
    response = submit(client, 'retry-feedback', dict(
        preset_id='phase2', feedback_ids=[row['feedback_id']]))
    assert response.status_code == 409
    assert calls == []
    assert not (config.parent / 'new2').exists()
    assert not (config.parent / 'control2').exists()
    assert client.get('/api/feedback').get_json() == [presented(row)]



@pytest.mark.parametrize('status', ['resolved', 'deferred'])
def test_deferred_and_resolved_visible_but_not_selected(feedback, status):
    client, app, config, row, calls = feedback
    row['investigation'].update(status=status, reason='unsupported dtype')
    write(config.parent / 'feedback' / (row['feedback_id'] + '.json'), row)
    assert 'unsupported dtype' in client.get('/feedback').get_data(as_text=True)
    assert submit(client, 'retry-feedback', dict(
        preset_id='phase2', feedback_ids=[row['feedback_id']])).status_code == 409
    assert not calls


@pytest.mark.parametrize('classification', ['infrastructure', 'provider'])
def test_classified_infrastructure_is_readable_but_no_retired_retry_or_investigation(feedback, classification):
    client, app, config, row, calls = feedback
    row['outcome'].update(category='incorrect', verified=False)
    row['investigation'].update(classification=classification, reason='not a kernel defect')
    write(config.parent / 'feedback' / (row['feedback_id'] + '.json'), row)
    page = client.get('/feedback').get_data(as_text=True)
    assert 'not a kernel defect' in page and 'name="feedback_id"' not in page
    assert submit(client, 'investigate', dict(preset_id='phase1',
        feedback_ids=[row['feedback_id']], project_target=2)).status_code == 409
    assert not calls
    assert submit(client, 'retry-feedback', dict(
        preset_id='phase2', feedback_ids=[row['feedback_id']])).status_code == 409
    assert calls == []


def test_unclassified_budget_failure_remains_available_for_curriculum_triage(feedback):
    client, app, config, row, calls = feedback
    row['outcome'].update(category='budget-exhausted', verified=False)
    write(config.parent / 'feedback' / (row['feedback_id'] + '.json'), row)
    assert 'name="feedback_id"' in client.get('/feedback').get_data(as_text=True)
    completed(client, submit(client, 'investigate', dict(preset_id='phase1',
        feedback_ids=[row['feedback_id']], project_target=2)))


def test_lineage_projects_and_retries_are_linked_without_changing_results(feedback):
    client, app, config, row, calls = feedback
    root = config.parent / 'investigation'
    app.extensions['dashboard_controls'].campaigns['probe'] = dict(phase=1, root=str(root))
    row['investigation'].update(projects=['historical-probe', dict(campaign_root=str(root), project_id='probe-control')],
                                retries=['c' * 64])
    write(config.parent / 'feedback' / (row['feedback_id'] + '.json'), row)
    page = client.get('/feedback').get_data(as_text=True)
    assert 'href="/phase/1/project/probe-control"' in page
    assert 'href="#' + 'c' * 64 + '"' in page


@pytest.mark.parametrize('change', [dict(target='a5'), dict(language='catlass-dsl'),
                                    dict(category='interrupted'), dict(category='verified-success')])
def test_incompatible_or_nonlearnable_selection_rejected(feedback, change):
    client, app, config, row, calls = feedback
    row['outcome'].update(change)
    write(config.parent / 'feedback' / (row['feedback_id'] + '.json'), row)
    response = submit(client, 'investigate', dict(
        preset_id='phase1', feedback_ids=[row['feedback_id']], project_target=1))
    assert response.status_code == 409 and not calls


@pytest.mark.parametrize('payload', [dict(feedback_ids=[]), dict(project_target=0),
                                   dict(project_target=True), dict(feedback_ids=['missing'])])
def test_investigation_requires_selection_and_explicit_budget(feedback, payload):
    client, app, config, row, calls = feedback
    body = dict(preset_id='phase1', feedback_ids=[row['feedback_id']], project_target=2)
    body.update(payload)
    assert submit(client, 'investigate', body).status_code == 409
    assert not calls


def test_evidence_preview_bounded_authenticated_and_missing_reported(feedback):
    client, app, config, row, calls = feedback
    data = b'x' * 20000
    digest = hashlib.sha256(data).hexdigest()
    jobs = config.parent / 'jobs'
    artifact = jobs / ('b' * 32) / 'artifacts' / digest
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(data)
    row['outcome']['evidence']['profiling'] = dict(
        name='profile.json', sha256=digest, size=len(data), job_root=str(jobs), job_id='b' * 32)
    write(config.parent / 'feedback' / (row['feedback_id'] + '.json'), row)
    url = '/api/feedback/' + row['feedback_id'] + '/evidence/profiling'
    preview = client.get(url).get_json()
    assert preview['has_more'] and len(preview['text']) == 8192
    artifact.write_bytes(b'wrong')
    assert client.get(url).status_code == 409
    artifact.unlink()
    assert client.get(url).status_code == 404


def test_missing_feedback_root_not_created(feedback):
    client, app, config, row, calls = feedback
    absent = config.parent / 'absent'
    app.extensions['dashboard_controls'].feedback.root = absent
    assert client.get('/api/feedback').get_json() == []
    assert not absent.exists()


@pytest.mark.parametrize('action,phase,classification', [
    ('investigate', 1, None), ('retry-feedback', 2, None), ('import-feedback', None, None),
    ('retry-feedback', 2, 'infrastructure'), ('retry-feedback', 2, 'provider')])
def test_feedback_controls_reach_real_cli_without_providers_or_screens(feedback, action, phase, classification):
    from benchmarks.ascend import campaign_cli
    from optimizer.jobs import JobStore
    client, app, config, row, calls = feedback
    root = config.parent
    manifest = root / 'sources.json'
    write(manifest, dict(schema='ascend-sft-sources-v2', sources=[dict(name='catlass',
        repository='https://example.invalid/catlass', revision='a' * 40,
        license='Apache-2.0', include=['**/*.py'])]))
    settings = root / 'runtime.json'
    write(settings, dict(model='openai/gpt-5.6-sol', phase1_arguments=[], source_manifest=str(manifest)))
    seed = root / 'seed.bin'
    seed.write_bytes(b'{}')
    manager = app.extensions['dashboard_controls']
    for preset in manager.presets.values():
        preset.update(config=str(settings), initial_memory=str(seed))
    # Retry pins exactly this locally retained original suite; it does not generate one.
    row['outcome']['task']['submission'] = {'source': 'original reference'}
    row['investigation']['classification'] = classification
    frozen = json.dumps(dict(suite_sha256='d' * 64, manifest=dict(
        policy={'rtol': 1e-5}, submission={'signatures': [{'shape': [8]}]}))).encode()
    jobs = JobStore(root / 'jobs')
    job_id = jobs.create('def relu(x): return x', 'relu', {'signatures': []}, {})['job_id']
    baseline = jobs.put_artifact(job_id, 'baseline.json', frozen)
    row['outcome']['evidence']['baseline'] = dict(baseline, job_root=str(root / 'jobs'), job_id=job_id)
    write(root / 'feedback' / (row['feedback_id'] + '.json'), row)
    screens, invocations = [], []

    def real_cli(argv, **kwargs):
        invocations.append(argv)
        output = io.StringIO()
        with redirect_stdout(output):
            campaign_cli.main(argv[2:], screen=lambda job, role: screens.append(role) or 'not-started')
        return subprocess.CompletedProcess(argv, 0, output.getvalue(), '')

    manager.runner = real_cli
    payload = {} if phase is None else dict(preset_id=f'phase{phase}',
        feedback_ids=[row['feedback_id']], project_target=2)
    response = submit(client, action, payload)
    if phase == 2:
        assert response.status_code == 409
        assert invocations == [] and screens == []
        assert not (root / 'control2').exists() and not (root / 'new2').exists()
        assert client.get('/api/feedback').get_json() == [row]
        return
    completed(client, response)
    if phase:
        job = json.loads((root / f'control{phase}' / 'job.json').read_text())
        assert job['failure_ids'] == [row['feedback_id']]
        assert job['project_target'] == (2 if phase == 1 else 1)
        assert job.get('approval_mode', False) is (phase == 1)
        assert bool(job.get('retry_feedback_id')) is (phase == 2)
        assert screens == ['collector', 'campaign']  # callbacks only; no processes launched
    else:
        assert not screens


def test_phase1_original_missing_native_acceptance_rejected(feedback):
    client, app, config, row, calls = feedback
    row['outcome'].update(source_type='phase1', original_spec=None)
    write(config.parent / 'feedback' / (row['feedback_id'] + '.json'), row)
    assert submit(client, 'retry-feedback', dict(preset_id='phase2', feedback_ids=[row['feedback_id']])).status_code == 409
    assert not calls


def test_failure_case_paging_download_and_legacy_unavailability(feedback):
    client, app, config, row, calls = feedback
    identifier = row['feedback_id']
    base = f'/api/feedback/{identifier}'
    legacy = client.get(base + '/failure-cases').get_json()[0]
    assert not legacy['availability']['complete']
    assert 'historical observation' in legacy['description']
    root = config.parent / 'feedback'
    source = b'kernel source\n'
    inputs = b'{"values":[' + b'1,' * 5000 + b'1]}'
    refs = {}
    for role, data in dict(source=source, inputs=inputs, expected=b'[2]', observed=b'[0]',
                           diagnostics=b'index 0: expected 2; observed 0').items():
        digest = hashlib.sha256(data).hexdigest()
        path = root / 'retained' / digest
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        refs[role] = dict(sha256=digest, retained_path=str(path))
    write(root / 'failure-cases' / (identifier + '.json'), [dict(schema='failure-case-v1',
        classification='host-mismatch', description='Addition returned zeros.',
        policy=dict(atol=1e-5), artifacts=refs)])
    before = (root / (identifier + '.json')).read_bytes()
    cases = client.get(base + '/failure-cases').get_json()
    assert cases[0]['availability']['complete']
    assert cases[0]['policy'] == dict(atol=1e-5)
    assert 'differs from the host reference' in cases[0]['status_explanations'][0]['description']
    url = base + '/failure-artifacts/' + refs['inputs']['sha256']
    first = client.get(url).get_json()
    assert first['size'] == len(inputs) and first['has_more']
    second = client.get(url + '?offset=' + str(first['next_offset'])).get_json()
    assert (first['text'] + second['text']).encode() == inputs and not second['has_more']
    download = client.get(url + '/download')
    assert download.data == inputs
    assert download.headers['Content-Disposition'].startswith('attachment')
    assert client.get(url + '?offset=-1').status_code == 409
    assert client.get(base + '/failure-artifacts/' + 'f' * 64).status_code == 404
    assert client.get('/feedback').status_code == 200
    assert 'Failing examples: inputs, expected and actual results' in client.get('/feedback').text
    Path(refs['inputs']['retained_path']).write_bytes(b'corrupt')
    unavailable = client.get(base + '/failure-cases').get_json()[0]
    assert not unavailable['availability']['complete']
    assert 'digest mismatch' in unavailable['availability']['reason']
    assert client.get(url + '/download').status_code == 409
    assert (root / (identifier + '.json')).read_bytes() == before and not calls


def test_feedback_destinations_and_actions_exclude_retired_phase2(feedback):
    from html.parser import HTMLParser
    class Controls(HTMLParser):
        def __init__(self, html):
            super().__init__()
            self.destinations, self.actions = [], []
            self.feed(html)
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'option':
                self.destinations.append(attrs.get('value'))
            if tag == 'button' and 'data-feedback-action' in attrs:
                self.actions.append(attrs['data-feedback-action'])
    client, _, _, _, _ = feedback
    controls = Controls(client.get('/feedback').get_data(as_text=True))
    assert controls.destinations == ['phase1']
    assert controls.actions == ['investigate']
