from contextlib import contextmanager
import hashlib
import json
import time
from types import SimpleNamespace

import pytest

from dashboard.app import create_app
from dashboard.subject_services import bind_subject
from benchmarks.ascend.subjects import Subject, SubjectScheduler
from benchmarks.ascend.subject_memory import SubjectMemory
from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
from benchmarks.a5kernels.phase1_provider import direct_profile
from optimizer.jobs import canonical


def item(identifier):
    return dict(project_id=identifier, repairs=[], spec=dict(project_id=identifier,
        capabilities=['gm-ub-movement'], shape=[16], dtype='float32', block_count=1,
        padding_multiple=1, failure_mode='none', timing_lengths=[], profile_metric=None,
        hypothesis='Can a copy preserve all values?', graph=dict(inputs=[dict(name='x', shape=[16])],
        nodes=[dict(name='y', op='copy', operands=['x'], axis=None)], output='y')))


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / 'campaign'
    scheduler = SubjectScheduler(root, [Subject('current', 'Current'), Subject('future', 'Future')])
    SubjectMemory(root / 'memory', b'initial')
    (root / 'subject-campaign.json').write_bytes(canonical(dict(projects={
        'original': dict(plan=item('original'), objective='Current', wave_index=1)}, waves=[], pending=None)))
    config = tmp_path / 'dashboard.json'
    config.write_bytes(canonical(dict(controls=dict(enabled=True), subject_campaigns=[dict(
        id='research', target='a3', root=str(root), config=str(tmp_path / 'runtime.json'),
        initial_memory=str(tmp_path / 'seed'))])))
    return config, root, scheduler


def application(setup, responses, restart=None):
    config, root, _ = setup
    calls = []
    values = iter(responses)
    @contextmanager
    def binder(config_path, campaign_id, **options):
        assert config_path == config and campaign_id == 'research'
        def complete(role, messages, **limits):
            calls.append((role, messages, limits))
            return json.dumps(next(values))
        yield SimpleNamespace(root=root, entry=dict(target='a3'), complete=complete,
                              policy=SimpleNamespace(admit=lambda _: None, grammar=lambda grammar: grammar))
    return create_app(config, root.parent / 'cache.sqlite', curriculum_binder=binder,
                      curriculum_restart=restart), calls


def submit(client, action, payload, key):
    subject = payload.get('proposal_id', payload.get('campaign_id'))
    return client.post('/api/control/' + action, json=dict(payload, confirmation=action + ':' + subject),
        headers={'X-CSRF-Token': client.get('/api/csrf').get_json()['token'], 'Idempotency-Key': key})


def settled(app, response):
    assert response.status_code == 202, response.data
    manager = app.extensions['dashboard_controls']
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        operation = manager.operation(response.get_json()['operation_id'])
        if operation['status'] in {'succeeded', 'failed'}:
            return operation
        time.sleep(.01)
    pytest.fail('operation did not settle')


def test_request_project_preview_then_approval_is_idempotent_and_durable(setup):
    rows = [dict(subject_id='current', item=item('new'))]
    app, calls = application(setup, [rows])
    client = app.test_client()
    try:
        body = dict(campaign_id='research', kind='request-project', instructions='Try a copy', format_seconds=10)
        response = submit(client, 'curriculum-preview', body, 'preview')
        proposal = settled(app, response)['result']
        assert setup[2].state().get('project_queue', []) == []
        assert proposal['after'] == rows and calls[0][0] == 'curriculum'
        assert 0 < calls[0][2]['remaining_seconds'] <= 10
        assert submit(client, 'curriculum-preview', body, 'preview').get_json() == response.get_json()
        approved = dict(campaign_id='research', proposal_id=proposal['id'])
        result = settled(app, submit(client, 'curriculum-approve', approved, 'approve'))
        assert result['status'] == 'succeeded'
        assert setup[2].state()['project_queue'] == rows
        assert settled(app, submit(client, 'curriculum-approve', approved, 'another-key'))['status'] == 'succeeded'
        assert setup[2].state()['revision'] == 1
    finally:
        app.extensions['dashboard_controls'].close()


def test_subject_edit_stale_approval_fails_without_overwriting_progress(setup):
    rows = [Subject('current', 'Current').as_dict(), Subject('new', 'Reviewed future').as_dict()]
    app, _ = application(setup, [rows])
    client = app.test_client()
    try:
        proposal = settled(app, submit(client, 'curriculum-preview', dict(campaign_id='research',
            kind='subjects', instructions='Replace future', format_seconds=10), 'p'))['result']
        setup[2].start_wave('started', 'a' * 64)
        result = settled(app, submit(client, 'curriculum-approve', dict(campaign_id='research',
            proposal_id=proposal['id']), 'a'))
        assert result['status'] == 'failed' and 'stale' in result['error']
        assert setup[2].context()['subjects'][1]['subject_id'] == 'future'
    finally:
        app.extensions['dashboard_controls'].close()


def test_restart_uses_reviewed_prompt_and_explicit_allocation(setup):
    reviewed = dict(item=item('retry'), instruction='Inspect physical padding before submit.')
    restarts = []
    def restart(*args):
        restarts.append(args)
        return dict(id='request', status='queued', project_id='retry')
    app, _ = application(setup, [reviewed], restart)
    client = app.test_client()
    try:
        proposal = settled(app, submit(client, 'curriculum-preview', dict(campaign_id='research',
            kind='restart', project_id='original', instructions='Fix prompt', format_seconds=10), 'p'))['result']
        body = dict(campaign_id='research', proposal_id=proposal['id'])
        assert submit(client, 'curriculum-approve', body, 'missing-time').status_code == 409
        result = settled(app, submit(client, 'curriculum-approve', dict(body, additional_seconds=60), 'a'))
        assert result['result']['restart']['project_id'] == 'retry'
        assert restarts[0][2]['after'] == reviewed and restarts[0][3] == 60
        assert restarts[0][2]['status'] == 'approved'
    finally:
        app.extensions['dashboard_controls'].close()


def test_controls_require_csrf_allowance_and_enabled_mode(setup):
    app, _ = application(setup, [])
    client = app.test_client()
    try:
        assert client.post('/api/control/curriculum-preview', json={}).status_code == 403
        body = dict(campaign_id='research', kind='projects', instructions='Add projects')
        assert submit(client, 'curriculum-preview', body, 'missing').status_code == 409
        assert client.get('/campaign/research').status_code == 200
        app.extensions['dashboard_controls'].enabled = False
        assert submit(client, 'curriculum-preview', dict(body, format_seconds=5), 'disabled').status_code == 403
        assert client.get('/api/curriculum/research/proposals').status_code == 403
    finally:
        app.extensions['dashboard_controls'].close()


def test_role_binding_uses_frozen_runtime_without_campaign_or_admission_builder(setup, monkeypatch):
    config, root, _ = setup
    settings = dict(phase1_arguments=[])
    runtime = root.parent / 'runtime.json'
    runtime.write_bytes(canonical(settings))
    (root.parent / 'seed').write_bytes(b'initial')
    (root / 'subject-runtime.json').write_bytes(canonical(dict(target='a3', max_seconds=90,
        settings_sha256=hashlib.sha256(canonical(settings)).hexdigest(), referenced_json_sha256={})))
    calls = []
    def compose(received, **options):
        assert received == settings and options['root'] != root
        calls.append(options['root'])
        return dict(provider=object(), profile=direct_profile(MODEL_IDENTITIES[0]),
            coordinator=SimpleNamespace(curriculum=SimpleNamespace(target_policy=None, query_knowledge=None)))
    with bind_subject(config, 'research', max_seconds=5, compose=compose) as service:
        assert service.root == root and service.knowledge.memory_getter().records() == []
        assert calls[0].is_dir()
        assert not (root / 'memory-reviews/.admission.lock').exists()
    assert not calls[0].exists()
    runtime.write_bytes(canonical(dict(settings, curriculum_model='changed')))
    with pytest.raises(ValueError, match='settings changed'):
        with bind_subject(config, 'research', max_seconds=5, compose=compose):
            pass
    assert len(calls) == 1


def test_approved_restart_reconciles_failed_launch_and_keeps_durable_request_status(setup):
    config, root, _ = setup
    reviewed = dict(item=item('retry'), instruction='Inspect physical padding.')
    calls = []
    def restart(config_path, campaign_id, proposal, seconds, key):
        calls.append((proposal['status'], seconds, key))
        identifier = hashlib.sha256(key.encode()).hexdigest()[:32]
        path = root / 'dashboard-requests' / (identifier + '.json')
        path.parent.mkdir(exist_ok=True)
        if len(calls) == 1:
            path.write_bytes(canonical(dict(id=identifier, status='failed', additional_seconds=seconds,
                project_id='original', error='launch failed', proposal=proposal)))
            raise RuntimeError('launch failed')
        request = dict(id=identifier, status='queued', additional_seconds=seconds,
                       project_id='original', proposal=proposal)
        path.write_bytes(canonical(request))
        return request
    app, formatted = application(setup, [reviewed], restart)
    client = app.test_client()
    try:
        proposal = settled(app, submit(client, 'curriculum-preview', dict(campaign_id='research',
            kind='restart', project_id='original', instructions='Fix prompt', format_seconds=10), 'p'))['result']
        body = dict(campaign_id='research', proposal_id=proposal['id'], additional_seconds=60)
        assert settled(app, submit(client, 'curriculum-approve', body, 'a'))['status'] == 'failed'
        service = app.extensions['dashboard_curriculum_controls']
        assert service.proposal(root, proposal['id'])['status'] == 'approved'
        url = '/api/curriculum/research/proposals/' + proposal['id'] + '/restart'
        retained = client.get(url).get_json()['request']
        assert retained['status'] == 'failed' and retained['additional_seconds'] == 60
        assert 'proposal' not in retained
        assert client.get('/campaign/research/project/original').status_code == 200
        result = settled(app, submit(client, 'curriculum-approve', body, 'retry'))
        assert result['status'] == 'succeeded'
        assert client.get(url).get_json()['request']['status'] == 'queued'
        assert calls[0] == calls[1] and len(formatted) == 1
        app.extensions['dashboard_controls'].enabled = False
        assert client.get(url).status_code == 403
    finally:
        app.extensions['dashboard_controls'].close()


def test_restart_reconcile_affordance_follows_retained_execution_state(setup):
    from html.parser import HTMLParser
    class Actions(HTMLParser):
        def __init__(self, html):
            super().__init__()
            self.approvals = []
            self.feed(html)
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'button' and 'data-curriculum-approve' in attrs:
                self.approvals.append(attrs['data-curriculum-approve'])
    config, root, _ = setup
    reviewed = dict(item=item('retry'), instruction='Inspect physical padding.')
    app, _ = application(setup, [reviewed], lambda *args: dict(status='queued'))
    client = app.test_client()
    try:
        proposal = settled(app, submit(client, 'curriculum-preview', dict(campaign_id='research',
            kind='restart', project_id='original', instructions='Fix prompt', format_seconds=10), 'p'))['result']
        settled(app, submit(client, 'curriculum-approve', dict(campaign_id='research',
            proposal_id=proposal['id'], additional_seconds=60), 'a'))
        page = '/campaign/research/project/original'
        assert Actions(client.get(page).get_data(as_text=True)).approvals == [proposal['id']]
        identifier = hashlib.sha256(('curriculum-restart:' + proposal['id']).encode()).hexdigest()[:32]
        path = root / 'dashboard-requests' / (identifier + '.json')
        path.parent.mkdir(exist_ok=True)
        for status in ('succeeded', 'observation-required'):
            path.write_bytes(canonical(dict(id=identifier, status=status, additional_seconds=60)))
            assert Actions(client.get(page).get_data(as_text=True)).approvals == []
    finally:
        app.extensions['dashboard_controls'].close()
