"""Portable exports retain real read views without their execution dependencies."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tarfile

import pytest

from dashboard.demo import export_demo
from dashboard.store import DashboardStore
from dashboard.tests.test_dashboard import campaign  # reusable on-disk evidence fixture
from optimizer.jobs import JobStore, canonical


def route(output, path):
    manifest = json.loads((output / 'snapshot.json').read_text())
    entry = manifest['routes'][path]
    return entry, (output / entry['file'].lstrip('/')).read_bytes()


def test_export_campaign_refresh_and_lazy_trace(campaign, tmp_path):
    cache = tmp_path / 'cache'
    store = DashboardStore(campaign, cache / 'dashboard.sqlite3')
    store.refresh()
    before = (campaign.read_bytes(), store.cached())
    output = export_demo(campaign, cache, tmp_path / 'demo')
    _, page = route(output, '/')
    assert b'Demo snapshot' in page and b'Campaign controls' not in page
    assert b'http' not in (output / 'requirements.txt').read_bytes()
    _, status = route(output, '/api/status')
    view = json.loads(status)
    assert view['campaigns'][0]['stats']['verified'] == 1
    assert view['campaigns'][1]['stats']['attempted'] == 0
    project = view['campaigns'][0]['projects'][0]['id']
    base = f'/api/phase/1/project/{project}'
    _, trace = route(output, base + '/trace')
    available = [row for row in json.loads(trace)['events'] if row['available']]
    assert available
    for event in available:
        entry, payload = route(output, event['download_url'])
        assert entry['status'] in (200, 409)  # fixture includes deliberately invalid hashes
        if entry['status'] == 200:
            assert hashlib.sha256(payload).hexdigest() == event['sha256']
    _, performance = route(output, base + '/performance')
    assert json.loads(performance)
    assert campaign.read_bytes() == before[0] and store.cached() == before[1]
    with pytest.raises(ValueError, match='already exists'):
        export_demo(campaign, cache, output)
    # Later local results are imported by exporting into a fresh directory.
    config = json.loads(campaign.read_text())
    config['campaigns'][1]['root'] = config['campaigns'][0]['root']
    campaign.write_text(json.dumps(config))
    updated = export_demo(campaign, cache, tmp_path / 'updated')
    assert json.loads(route(updated, '/api/status')[1])['campaigns'][1]['stats']['verified'] == 1
    assert json.loads(route(output, '/api/status')[1])['campaigns'][1]['stats']['attempted'] == 0


def test_optimizer_export_without_worker_and_portable_archive(tmp_path):
    jobs = JobStore(tmp_path / 'jobs')
    job = jobs.create('def forward(x): return x + 1', 'forward', {'signatures': []},
                      {'snapshot_id': 'retained-memory', 'memory_entries': []}, name='Real fixture')
    key = job['job_id']
    source = jobs.put_artifact(key, 'candidate.cpp', canonical('void kernel() {}\n'))
    excluded = jobs.put_artifact(key, 'timeline.json', canonical({'large': 'excluded'}))
    baseline = jobs.put_artifact(key, 'baseline', canonical({'cases': [
        {'case': 'small', 'timing': {'torch': {'latency_us': 10}}}]}))
    best = {'candidate_sha256': 'a' * 64, 'speedup': 2, 'source': source,
            'content': {'cases': [{'case': 'small', 'timing': {'ascendc': {'latency_us': 5}}}]}}
    jobs.add_event(key, 'optimizer-state', {'baseline': baseline, 'best': best, 'history': [best]})
    jobs.transition(key, 'running', stage='optimizing')
    jobs.transition(key, 'completed', stage='done')
    (jobs._directory(key) / 'worker.json').write_text('PRIVATE_CONFIG_MUST_NOT_EXPORT')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'campaigns': [], 'optimizer': {
        'enabled': True, 'root': str(jobs.root), 'worker_config': '/deleted/worker.json'}}))
    output = export_demo(config, tmp_path / 'cache', tmp_path / 'demo')
    assert b'Real fixture' in route(output, '/optimizer')[1]
    assert b'optimizer-create' not in route(output, '/optimizer')[1]
    page = route(output, '/optimizer/' + key)[1]
    assert b'optimizer-stop' not in page and b'optimizer-resume' not in page
    base = '/api/optimizer/jobs/' + key
    assert json.loads(route(output, base + '/results')[1])['cases'][0]['speedup'] == 2
    assert json.loads(route(output, base + '/sources/0')[1])['source'] == 'void kernel() {}\n'
    assert route(output, base + '/sources/0/download')[1] == b'void kernel() {}\n'
    manifest = json.loads((output / 'snapshot.json').read_text())
    assert base + '/artifacts/' + excluded['sha256'] not in manifest['routes']
    assert not any(b'PRIVATE_CONFIG_MUST_NOT_EXPORT' in p.read_bytes() for p in output.rglob('*') if p.is_file())
    archive = tmp_path / 'demo.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(output, arcname='demo')
    extracted = tmp_path / 'extracted'
    with tarfile.open(archive) as tar:
        tar.extractall(extracted, filter='data')
    shutil.rmtree(jobs.root)
    assert b'void kernel' in route(extracted / 'demo', base + '/sources/0/download')[1]
    result = subprocess.run([sys.executable, str(extracted / 'demo/serve.py'), '--help'],
                            capture_output=True, text=True, check=True)
    assert '--port' in result.stdout


def test_cli_help_and_playback_syntax():
    result = subprocess.run([sys.executable, '-m', 'dashboard', 'export-demo', '--help'],
                            capture_output=True, text=True, check=True)
    assert '--output' in result.stdout
    node = shutil.which('node')
    if node:
        from dashboard.demo import _PLAYBACK
        subprocess.run([node, '--check'], input=_PLAYBACK.replace('__MANIFEST__', '{}'),
                       text=True, check=True)


def test_export_ledger_authentication_shared_and_invalidated(campaign, tmp_path, monkeypatch):
    from collections import Counter
    from dashboard.app import create_app
    from dashboard.demo import _ExportStore
    from dashboard.tests.test_dashboard import evidence_ledger_rows
    original = DashboardStore._authenticated_ledger
    calls = Counter()
    def authenticate(path):
        calls[path] += 1
        return original(path)
    monkeypatch.setattr(DashboardStore, '_authenticated_ledger', staticmethod(authenticate))
    store = _ExportStore(campaign, tmp_path / 'private.sqlite3')
    snapshot = store.refresh()
    project = snapshot['campaigns'][0]['projects'][0]['id']
    trace = store.trace_index(1, project)
    app = create_app(campaign, store.cache_path, refresh_seconds=float('inf'), dashboard_store=store)
    client = app.test_client()
    client.get(f'/phase/1/project/{project}')
    for event in trace['events']:
        if event['available']:
            client.get(event['payload_url'])
            client.get(event['download_url'])
    assert calls and all(count == 1 for count in calls.values())
    ledger = tmp_path / 'changing.jsonl'
    ledger.write_text('\n'.join(json.dumps(row) for row in evidence_ledger_rows([{'message': 'first'}])) + '\n')
    first = store._authenticated_ledger(ledger)
    assert first[1] is None
    assert store._authenticated_ledger(ledger) is first
    assert calls[ledger] == 1
    previous_stat = ledger.stat()
    ledger.write_text(ledger.read_text().replace('first', 'wrong'))  # same length, bad digest
    # Some local filesystems round sub-millisecond writes to the same timestamp.
    os.utime(ledger, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns + 1_000_000_000))
    assert store._authenticated_ledger(ledger)[1] == 'evidence ledger integrity verification failed'
    assert calls[ledger] == 2
    ledger.unlink()
    assert store._authenticated_ledger(ledger)[1] == 'retained evidence ledger unavailable'
    assert calls[ledger] == 3
    app.extensions['dashboard_controls'].close()


def test_multiple_optimizer_phase2_campaigns_keep_attempts_and_traces_distinct(tmp_path):
    campaigns = []
    expected = []
    for index, name in enumerate(('Production goal', 'Completed goal', 'Validation smoke')):
        root = tmp_path / f'campaign-{index}'
        jobs = JobStore(root / 'tasks/shared-task/jobs')
        key = jobs.create('torch reference', 'forward', {}, {})['job_id']
        source = f'// distinct native kernel {index}'
        artifact = jobs.put_artifact(key, 'candidate.cpp', canonical(source))
        baseline = jobs.put_artifact(key, 'baseline', canonical({'cases': [
            {'case': 'small', 'timing': {'torch': {'latency_us': 12 + index}}}]}))
        task = {'task_id': 'shared-task', 'submission': {'source': 'torch reference'},
                'rationale': name}
        result = {'job_id': key, 'classification': 'host-verified',
                  'summary': name, 'candidate': artifact, 'baseline': baseline,
                  'lessons': [{'category': 'movement', 'claim': name, 'confidence': 'verified'}],
                  'result': {'cases': [{'case': 'small', 'timing': {
                      'ascendc': {'latency_us': 6 + index}}}]}}
        jobs.add_event(key, 'phase2-attempt-result', result)
        attempt = {'attempt_id': 'attempt-001', 'task_id': 'shared-task',
                   'kind': 'target', 'task': task, 'verified': True,
                   'execution_directory': 'tasks/shared-task', 'result': result}
        (root / 'attempts').mkdir()
        (root / 'attempts/attempt-001.json').write_bytes(canonical(attempt))
        (root / 'state.json').write_bytes(canonical({
            'phase': 'optimizer-phase2', 'attempts': [attempt], 'target': task,
            'terminal': {'status': 'READY'}, 'memory_entries': [{'lesson': name}]}))
        campaigns.append({'id': f'goal-{index}', 'name': name, 'phase': 2, 'root': str(root)})
        expected.append((source, artifact['sha256']))
    config = tmp_path / 'config.json'
    config.write_bytes(canonical({'campaigns': campaigns}))
    output = export_demo(config, tmp_path / 'cache', tmp_path / 'demo')
    manifest = json.loads((output / 'snapshot.json').read_text())['routes']
    index_page = route(output, '/')[1]
    for index, (source, digest) in enumerate(expected):
        scope = f'/campaign/{index}'
        phase = scope + '/phase/2'
        assert manifest[phase]['file'].encode() in index_page
        project = phase + '/project/attempt-001'
        assert manifest[project]['file'].encode() in route(output, phase)[1]
        assert manifest[project]['file'].encode() in route(output, '/memory')[1]
        page = route(output, project)[1]
        assert f'distinct native kernel {index}'.encode() in page
        assert f'distinct native kernel {(index + 1) % 3}'.encode() not in page
        assert (scope + '/api/phase/2/project/attempt-001/performance').encode() in page
        assert b'Optimizer measurements' in page
        trace_url = scope + '/api/phase/2/project/attempt-001/trace'
        trace = json.loads(route(output, trace_url)[1])
        event = next(row for row in trace['events'] if row['label'] == 'candidate.cpp')
        assert event['download_url'].startswith(scope + '/')
        raw = route(output, event['download_url'])[1]
        assert hashlib.sha256(raw).hexdigest() == digest
        assert json.loads(raw) == source
    assert '/phase/2/project/attempt-001' not in manifest
