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


def application(setup, responses, restart=None, *, settings=None, target='a3'):
    config, root, _ = setup
    # Unit fixtures explicitly supply both interaction callbacks. Production
    # availability without callbacks is exercised separately below.
    if restart is None:
        restart = lambda *args: dict(status='queued')
    calls = []
    values = iter(responses)
    @contextmanager
    def binder(config_path, campaign_id, **options):
        assert config_path == config and campaign_id == 'research'
        def complete(role, messages, **limits):
            calls.append((role, messages, limits))
            return json.dumps(next(values))
        yield SimpleNamespace(root=root, entry=dict(target=target), complete=complete, bound={},
                              settings=settings or {},
                              policy=SimpleNamespace(admit=lambda _: None, grammar=lambda grammar: grammar))
    def status(config_path, campaign_id, identifier):
        path = root / 'dashboard-requests' / (identifier + '.json')
        return json.loads(path.read_bytes()) if path.exists() else None
    return create_app(config, root.parent / 'cache.sqlite', curriculum_binder=binder,
                      curriculum_restart=restart, curriculum_status=status), calls


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


def test_reserved_unstarted_project_reaches_curriculum_restart_preview(setup):
    config, root, scheduler = setup
    pending = item('reserved')
    state = dict(projects={}, waves=[], pending=dict(wave_id='wave-reserved',
        subject=Subject('current', 'Current').as_dict(), projects=['reserved'], items=[pending],
        memory_sha256='a' * 64))
    (root / 'subject-campaign.json').write_bytes(canonical(state))
    scheduler.start_wave('wave-reserved', 'a' * 64)
    reviewed = dict(item=item('fresh'), instruction='Inspect the physical tail.')
    app, calls = application(setup, [reviewed])
    client = app.test_client()
    try:
        assert client.get('/campaign/research/project/reserved').status_code == 200
        response = submit(client, 'curriculum-preview', dict(campaign_id='research',
            kind='restart', project_id='reserved', instructions='Review the prompt', format_seconds=10), 'pending')
        result = settled(app, response)
        assert result['status'] == 'succeeded', result
        assert result['result']['project'] == 'reserved'
        assert result['result']['before']['plan'] == pending
        assert len(calls) == 1
        assert json.loads((root / 'subject-campaign.json').read_bytes()) == state
        active = json.loads((root / 'memory/active.json').read_bytes())
        active['admissions']['reserved'] = dict(version='b' * 64)
        (root / 'memory/active.json').write_bytes(canonical(active))
        assert submit(client, 'curriculum-preview', dict(campaign_id='research', kind='restart',
            project_id='reserved', instructions='Do not rerun merged work', format_seconds=10), 'merged').status_code == 409
        assert len(calls) == 1
    finally:
        app.extensions['dashboard_controls'].close()


def test_restart_status_uses_authoritative_accessor_without_dispatch(setup):
    config, root, _ = setup
    app, _ = application(setup, [dict(item=item('retry'), instruction='Inspect padding.')])
    client = app.test_client()
    try:
        proposal = settled(app, submit(client, 'curriculum-preview', dict(campaign_id='research',
            kind='restart', project_id='original', instructions='Fix prompt', format_seconds=10), 'p'))['result']
        from dashboard.subject_services import subject_editor
        subject_editor(root).approve(proposal['id'])
        identifier = hashlib.sha256(('curriculum-restart:' + proposal['id']).encode()).hexdigest()[:32]
        path = root / 'dashboard-requests' / (identifier + '.json')
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(canonical(dict(id=identifier, status='running', additional_seconds=60)))
        calls = []
        def authoritative(config_path, campaign_id, request_id):
            calls.append((config_path, campaign_id, request_id))
            return dict(id=request_id, status='observation-required', additional_seconds=60)
        service = app.extensions['dashboard_curriculum_controls']
        service.status = authoritative
        service.restart = lambda *args: pytest.fail('status read must never dispatch a restart')
        result = client.get('/api/curriculum/research/proposals/' + proposal['id'] + '/restart')
        assert result.get_json()['request']['status'] == 'observation-required'
        assert calls == [(config, 'research', identifier)]
        assert app.extensions['dashboard_controls'].recent()[0]['action'] == 'curriculum-preview'
    finally:
        app.extensions['dashboard_controls'].close()


def test_target_preview_checks_observed_runtime_before_accepting_qualification(setup, monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / 'tests'))
    from test_ascend_phase2_targets import qualified, RUNTIME
    from benchmarks.ascend.targets import a3_curriculum_policy
    from copy import deepcopy
    config, root, _ = setup
    target, artifacts, _ = qualified(root.parent / 'qualification')
    plan = dict(project_id=target.spec.project_id, spec=target.spec.as_dict(), repairs=[],
        task=dict(kind='target', contract=target.as_dict(), artifact_root=str(artifacts.root)))
    changed = deepcopy(plan)
    changed['task']['contract']['runtime']['device'] = 'wrong-device'
    responses, calls = iter([changed, plan]), []
    @contextmanager
    def binder(*args, **kwargs):
        def complete(role, messages, **options):
            calls.append(messages)
            return json.dumps([dict(subject_id='current', item=next(responses))])
        yield SimpleNamespace(root=root, entry=dict(target='a3'), complete=complete, settings={},
            bound=dict(actor=SimpleNamespace(executor=SimpleNamespace(runtime_identity=dict(RUNTIME)))),
            policy=a3_curriculum_policy())
    app = create_app(config, root.parent / 'target-cache.sqlite', curriculum_binder=binder)
    client = app.test_client()
    try:
        result = settled(app, submit(client, 'curriculum-preview', dict(campaign_id='research',
            kind='projects', instructions='Use the qualified reference', format_seconds=10), 'target'))
        assert result['status'] == 'succeeded', result
        assert len(calls) == 2
        assert result['result']['after'][0]['item'] == plan
        assert app.extensions['dashboard_subjects'].campaign('research')['project_queue'] == []
    finally:
        app.extensions['dashboard_controls'].close()


def optimizer_item():
    return dict(project_id='optimizer-copy', repairs=[], task=dict(kind='optimizer', task=dict(
        task_id='torch-copy', submission=dict(source='def copy(x): return x', entrypoint='copy',
        signatures=[dict(args=[dict(shape=[8], dtype='float32')])]))))


@pytest.mark.parametrize('settings,target', [({}, 'a3'), ({'optimizer_worker_config': {}}, 'a5')])
def test_curriculum_repairs_optimizer_request_unavailable_in_campaign(setup, settings, target):
    graph = dict(subject_id='current', item=item('graph-copy'))
    app, calls = application(setup, [[dict(subject_id='current', item=optimizer_item())], [graph]],
                             settings=settings, target=target)
    client = app.test_client()
    try:
        result = settled(app, submit(client, 'curriculum-preview', dict(campaign_id='research',
            kind='request-project', instructions='Request a useful copy experiment', format_seconds=10), 'optimizer'))
        assert result['status'] == 'succeeded', result
        assert len(calls) == 2 and result['result']['after'] == [graph]
        assert setup[2].state().get('project_queue', []) == []
        assert not list(setup[1].glob('**/jobs'))
    finally:
        app.extensions['dashboard_controls'].close()


def test_configured_a3_optimizer_can_be_previewed_without_execution(setup):
    queued = [dict(subject_id='current', item=optimizer_item())]
    app, calls = application(setup, [queued], settings={'optimizer_worker_config': {}}, target='a3')
    client = app.test_client()
    try:
        result = settled(app, submit(client, 'curriculum-preview', dict(campaign_id='research',
            kind='projects', instructions='Use the configured optimizer', format_seconds=10), 'supported'))
        assert result['status'] == 'succeeded', result
        assert len(calls) == 1 and result['result']['after'] == queued
        assert setup[2].state().get('project_queue', []) == []
        assert not list(setup[1].glob('**/jobs'))
    finally:
        app.extensions['dashboard_controls'].close()


def test_old_restart_recovery_remains_visible_after_twenty_other_previews(setup):
    import os
    from dashboard.subject_services import subject_editor
    config, root, _ = setup
    app, _ = application(setup, [dict(item=item('retry'), instruction='Inspect padding.')])
    client = app.test_client()
    try:
        proposal = settled(app, submit(client, 'curriculum-preview', dict(campaign_id='research',
            kind='restart', project_id='original', instructions='Fix prompt', format_seconds=10), 'old'))['result']
        subject_editor(root).approve(proposal['id'])
        directory = root / 'curriculum-edits'
        os.utime(directory / (proposal['id'] + '.json'), (100, 100))
        for index in range(20):
            other = dict(proposal, id=f'{index:032x}', project='another', status='proposed')
            path = directory / (other['id'] + '.json')
            path.write_bytes(canonical(other))
            os.utime(path, (101 + index, 101 + index))
        proposals = client.get('/api/curriculum/research/proposals').get_json()['proposals']
        assert proposal['id'] in [row['id'] for row in proposals]
        from html.parser import HTMLParser
        class Actions(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids = []
            def handle_starttag(self, tag, attrs):
                values = dict(attrs)
                if tag == 'button' and 'data-curriculum-approve' in values:
                    self.ids.append(values['data-curriculum-approve'])
        actions = Actions()
        actions.feed(client.get('/campaign/research/project/original').get_data(as_text=True))
        assert actions.ids == [proposal['id']]
    finally:
        app.extensions['dashboard_controls'].close()


def test_missing_restart_runtime_refuses_without_approval_or_provider_effects(setup, monkeypatch):
    from dashboard.subject_services import subject_editor
    app, calls = application(setup, [dict(item=item('retry'), instruction='Inspect padding.')])
    client = app.test_client()
    try:
        body = dict(campaign_id='research', kind='restart', project_id='original',
                    instructions='Fix prompt', format_seconds=10)
        proposal = settled(app, submit(client, 'curriculum-preview', body, 'retained'))['result']
        service = app.extensions['dashboard_curriculum_controls']
        service.restart = service.status = None
        monkeypatch.setattr('dashboard.curriculum_controls.find_spec', lambda _: None)
        assert not service.restart_available
        count = len(calls)
        before = service.proposal(setup[1], proposal['id'])
        assert submit(client, 'curriculum-preview', body, 'missing-runtime').status_code == 409
        approval = dict(campaign_id='research', proposal_id=proposal['id'], additional_seconds=60)
        assert submit(client, 'curriculum-approve', approval, 'approval').status_code == 409
        with pytest.raises(ValueError):
            service.approve(approval)
        assert len(calls) == count and service.proposal(setup[1], proposal['id']) == before
        # A retained approval/request from a newer installation stays readable.
        subject_editor(setup[1]).approve(proposal['id'])
        identifier = hashlib.sha256(('curriculum-restart:' + proposal['id']).encode()).hexdigest()[:32]
        directory = setup[1] / 'dashboard-requests'
        directory.mkdir()
        (directory / (identifier + '.json')).write_bytes(canonical(dict(id=identifier, status='queued')))
        assert service.restart_request('research', proposal['id']) is None
        from html.parser import HTMLParser
        class Actions(HTMLParser):
            def __init__(self):
                super().__init__()
                self.mutations = []
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if 'data-curriculum-preview' in attrs or 'data-curriculum-approve' in attrs:
                    self.mutations.append(tag)
        response = client.get('/campaign/research/project/original')
        assert response.status_code == 200
        actions = Actions(); actions.feed(response.get_data(as_text=True))
        assert actions.mutations == []
        # Installing the real runtime restores production availability without
        # injecting a restart callback or changing retained proposals.
        monkeypatch.setattr('dashboard.curriculum_controls.find_spec', lambda _: object())
        assert service.restart_available
    finally:
        app.extensions['dashboard_controls'].close()


def test_installed_restart_runtime_dispatches_through_default_production_callback(setup, monkeypatch):
    interaction = pytest.importorskip('benchmarks.ascend.subject_interaction')
    config, root, _ = setup
    state = json.loads((root / 'subject-campaign.json').read_bytes())
    state['projects']['original']['directory'] = 'original'
    (root / 'subject-campaign.json').write_bytes(canonical(state))
    (root.parent / 'runtime.json').write_text('{}')
    (root.parent / 'seed').write_bytes(b'initial')
    launched = []
    monkeypatch.setattr(interaction, '_launch', lambda *args: launched.append(args))
    app, _ = application(setup, [dict(item=item('retry'), instruction='Inspect padding.')])
    service = app.extensions['dashboard_curriculum_controls']
    service.restart = service.status = None
    assert service.restart_available
    client = app.test_client()
    try:
        proposal = settled(app, submit(client, 'curriculum-preview', dict(campaign_id='research',
            kind='restart', project_id='original', instructions='Fix prompt', format_seconds=10), 'p'))['result']
        result = settled(app, submit(client, 'curriculum-approve', dict(campaign_id='research',
            proposal_id=proposal['id'], additional_seconds=60), 'a'))
        assert result['status'] == 'succeeded', result
        assert len(launched) == 1 and launched[0][2]['proposal']['after'] == proposal['after']
        assert launched[0][2]['additional_seconds'] == 60
        url = '/api/curriculum/research/proposals/' + proposal['id'] + '/restart'
        assert client.get(url).get_json()['request']['status'] == 'queued'
    finally:
        app.extensions['dashboard_controls'].close()
