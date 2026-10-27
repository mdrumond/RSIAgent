import io
import json

import pytest

from dashboard.app import create_app


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = tmp_path / 'config.json'
    worker = tmp_path / 'worker.json'
    worker.write_text('{}')
    config.write_text(json.dumps({'campaigns': [], 'optimizer': {
        'enabled': True, 'root': str(tmp_path / 'jobs'), 'memory_roots': ['/memory'],
        'worker_config': str(worker)}}))
    import dashboard.optimizer as module
    monkeypatch.setattr(module, 'discover_sources', lambda *a, **k: [
        {'root': '/memory', 'available': True, 'snapshot_id': 'snapshot', 'memory_count': 4}])
    monkeypatch.setattr(module, 'freeze_source', lambda *a, **k: {'snapshot_id': 'snapshot'})
    launched = []
    app = create_app(config, tmp_path / 'cache', optimizer_launcher=lambda *a, **k: launched.append((a, k)))
    return app, config, launched


def headers(client, key='create'):
    return {'X-CSRF-Token': client.get('/api/csrf').json['token'], 'Idempotency-Key': key}


def test_upload_lifecycle_and_restart(setup, tmp_path):
    app, config, launched = setup
    client = app.test_client()
    assert b'Kernel Optimizer' in client.get('/').data
    assert b'Python source' in client.get('/optimizer').data
    assert client.get('/api/optimizer/sources').json[0]['available']
    data = {'name': 'demo', 'entrypoint': 'forward', 'signature': '[{"shape":[3]}]',
            'memory_root': '/memory', 'source_file': (io.BytesIO(b'def forward(x): return x'), 'kernel.py')}
    response = client.post('/api/optimizer/jobs', data=data, headers=headers(client))
    assert response.status_code == 202
    job_id = response.json['job_id']
    assert len(launched) == 1
    assert launched[0][1]['start_new_session'] is True
    assert 'optimizer.worker' in launched[0][0][0]
    store = app.extensions['optimizer'].store
    assert store.submission(job_id)['memory_snapshot']['snapshot_id'] == 'snapshot'
    assert store.submission(job_id)['signature'] == {'signatures': [{'shape': [3]}]}
    store.transition(job_id, 'running', stage='checking-correctness')
    store.add_event(job_id, 'provider', {'prompt': 'large hidden prompt'})
    status = client.get(f'/api/optimizer/jobs/{job_id}').json
    assert status['stage'] == 'checking-correctness'
    events = client.get(f'/api/optimizer/jobs/{job_id}/events').json
    assert 'large hidden prompt' not in json.dumps(events)
    assert client.get(f'/api/optimizer/jobs/{job_id}/events/2').json['payload']['prompt']
    assert client.post(f'/api/optimizer/jobs/{job_id}/stop', json={}, headers=headers(client, 'stop')).status_code == 202
    store.transition(job_id, 'stopped')
    resumed = client.post(f'/api/optimizer/jobs/{job_id}/resume', json={
        'additional_attempts': 2, 'additional_seconds': 60}, headers=headers(client, 'resume'))
    assert resumed.status_code == 202
    restarted = create_app(config, tmp_path / 'cache2', optimizer_launcher=lambda *a, **k: launched.append(a))
    other = restarted.test_client()
    repeated = other.post(f'/api/optimizer/jobs/{job_id}/resume', json={
        'additional_attempts': 2, 'additional_seconds': 60}, headers=headers(other, 'resume'))
    assert repeated.status_code == 202 and len(launched) == 2
    assert other.get(f'/api/optimizer/jobs/{job_id}').json['max_attempts'] == 27


def test_disabled_and_action_validation(setup, tmp_path):
    app, _, launched = setup
    client = app.test_client()
    assert client.post('/api/optimizer/jobs', json={}).status_code == 403
    assert client.post('/api/optimizer/jobs', json={'source': 'x', 'entrypoint': 'f',
        'signature': [], 'memory_root': '/wrong'}, headers=headers(client)).status_code == 409
    config = tmp_path / 'disabled.json'
    config.write_text('{"campaigns": []}')
    disabled = create_app(config, tmp_path / 'cache3').test_client()
    assert b'Kernel Optimizer' in disabled.get('/').data
    assert b'Configure' in disabled.get('/optimizer').data
    assert disabled.post('/api/optimizer/jobs', json={}, headers=headers(disabled)).status_code == 403
    assert not launched


def test_launch_failure_and_live_http(setup):
    from threading import Thread
    from urllib.request import urlopen, Request
    from werkzeug.serving import make_server

    app, _, _ = setup
    def fail(*args, **kwargs):
        raise OSError('worker could not start')
    app.extensions['optimizer'].launcher = fail
    server = make_server('127.0.0.1', 0, app)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        token = json.load(urlopen(base + '/api/csrf'))['token']
        payload = {'source': 'def f(x): return x', 'entrypoint': 'f',
                   'signature': [{'shape': [3]}], 'memory_root': '/memory'}
        request = Request(base + '/api/optimizer/jobs', data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json', 'X-CSRF-Token': token,
                     'Idempotency-Key': 'runtime-proof'})
        response = json.load(urlopen(request))
        assert response['status'] == 'failed'
        job_id = response['job_id']
        assert b'Execution trace' in urlopen(base + '/optimizer/' + job_id).read()
        events = json.load(urlopen(base + '/api/optimizer/jobs/' + job_id + '/events'))
        assert any(row['kind'] == 'worker-launch-failed' for row in events)
        pinned = json.load(urlopen(base + '/api/optimizer/jobs/' + job_id + '/submission'))
        assert pinned['source'] == payload['source']
        assert pinned['memory_snapshot']['snapshot_id'] == 'snapshot'
    finally:
        server.shutdown()
        thread.join()


def test_optimizer_status_and_event_explanations_preserve_raw_observation(setup):
    app, _, _ = setup
    client = app.test_client()
    response = client.post('/api/optimizer/jobs', json=dict(name='repair', source='def forward(x): return x',
        entrypoint='forward', signature=[{'shape': [3]}], memory_root='/memory'), headers=headers(client))
    assert response.status_code == 202
    identifier = response.json['job_id']
    store = app.extensions['optimizer'].store
    store.transition(identifier, 'running', stage='optimize')
    raw = dict(classification='host-mismatch', missing=['observed-host-mismatch'])
    store.add_event(identifier, 'verification', raw)
    state = client.get(f'/api/optimizer/jobs/{identifier}').json
    assert {row['code'] for row in state['status_explanations']} == {'running', 'optimize'}
    page = client.get('/optimizer/' + identifier).text
    assert 'frozen reference and correctness suite' in page
    assert 'Next:' in page
    events = store.events(identifier)
    sequence = events[-1]['sequence']
    event = client.get(f'/api/optimizer/jobs/{identifier}/events/{sequence}').json
    assert event['payload'] == raw
    assert any(row['code'] == 'observed-host-mismatch' for row in event['presentation']['status_explanations'])
    assert store.events(identifier)[-1]['payload'] == raw
