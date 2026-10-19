import json

from dashboard.tests.test_optimizer_dashboard import setup, headers
from optimizer.jobs import canonical


def test_results_sources_and_profiles(setup):
    app, _, _ = setup
    client = app.test_client()
    response = client.post('/api/optimizer/jobs', json={'source': 'def f(x): return x',
        'entrypoint': 'f', 'signature': [{'args': [{'shape': [3]}]}], 'memory_root': '/memory'},
        headers=headers(client))
    key = response.json['job_id']
    store = app.extensions['optimizer'].store
    source = store.put_artifact(key, 'candidate.cpp', canonical('void kernel() {}\n'))
    baseline = store.put_artifact(key, 'baseline', canonical({'cases': [
        {'case': 'case-0', 'timing': {'torch': {'latency_us': 10}}}]}))
    profile = {'groups': [{'group': 'Memory', 'available': True, 'reports': [{'metrics': [
        {'column': 'mte_bw(GB/s)', 'name': 'mte_bw', 'unit': 'GB/s', 'row': 0,
         'identity': {'Name': 'kernel'}, 'value': 0, 'available': True}]}]},
        {'group': 'ResourceConflictRatio', 'available': False, 'reason': 'unsupported'}]}
    profiling = store.put_artifact(key, 'profiling', canonical({'cases': [
        {'case': 'case-0', 'profiling': {'ascendc': profile}}]}))
    row = {'candidate_sha256': 'a' * 64, 'speedup': 2, 'summary': 'vectorized',
           'content': {'status': 'verified', 'cases': [
               {'case': 'case-0', 'timing': {'ascendc': {'latency_us': 5}}}]}}
    store.add_event(key, 'optimizer-state', {'baseline': baseline, 'history': [row],
        'best': {**row, 'source': source, 'profiling': profiling}})
    root = f'/api/optimizer/jobs/{key}'
    result = client.get(root + '/results').json
    assert result['cases'] == [{'case': 'case-0', 'torch_us': 10, 'ascendc_us': 5, 'speedup': 2}]
    assert result['attempts'][0]['best_speedup'] == 2
    assert result['metrics'][0]['value'] == 0
    assert result['metrics'][0]['unit'] == 'GB/s'
    assert result['unavailable'][0]['reason'] == 'unsupported'
    assert 'void kernel' not in json.dumps(result)
    assert client.get(root + '/sources/0').json['source'] == 'void kernel() {}\n'
    assert client.get(root + '/sources/0/download').data == b'void kernel() {}\n'
    assert client.get(root + '/sources/99').status_code == 404
    assert client.get(root + '/artifacts/' + source['sha256']).data == canonical('void kernel() {}\n')
    assert client.get(root + '/artifacts/' + '0' * 64).status_code == 404
    second = store.put_artifact(key, 'candidate.cpp', canonical('void next() {} // <script>\n'))
    changed = client.get(root + '/sources/1').json
    assert '-void kernel() {}' in changed['diff'] and '+void next()' in changed['diff']
    assert '<script>' not in changed['html']
    assert client.get(root + '/sources/1/download').data == b'void next() {} // <script>\n'
    html = client.get('/optimizer/' + key).data
    assert html.index(b'optimizer-results') < html.index(b'optimizer-submission') < html.index(b'optimizer-trace')


def test_empty_results_are_not_zero_measurements(setup):
    app, _, _ = setup
    store = app.extensions['optimizer'].store
    key = store.create('def f(x): return x', 'f', {}, {'snapshot_id': 's'})['job_id']
    result = app.test_client().get(f'/api/optimizer/jobs/{key}/results').json
    assert result['cases'] == [] and result['attempts'] == [] and result['best'] is None


def test_attempts_and_missing_timings_are_not_zero(setup):
    app, _, _ = setup
    store = app.extensions['optimizer'].store
    key = store.create('def f(x): return x', 'f', {}, {'snapshot_id': 's'})['job_id']
    base = store.put_artifact(key, 'baseline', canonical({'cases': [
        {'case': 'a', 'timing': {'torch': {'latency_us': 3}}}, {'case': 'b'}],
        'profiling': {'status': 'unavailable', 'error': 'not supported'}}))
    store.add_event(key, 'optimizer-state', {'baseline': base, 'history': [
        {'candidate_sha256': 'a', 'speedup': 2, 'content': {'status': 'verified'}},
        {'candidate_sha256': 'b', 'speedup': None, 'content': {'status': 'incorrect'}},
        {'candidate_sha256': 'b', 'status': 'duplicate-skipped'},
        {'tool': {}, 'result': {}},
        {'candidate_sha256': 'c', 'speedup': 1, 'content': {'status': 'verified'}}]})
    result = app.test_client().get(f'/api/optimizer/jobs/{key}/results').json
    assert [row['best_speedup'] for row in result['attempts']] == [2, 2, 2]
    assert result['cases'][0]['ascendc_us'] is None
    assert result['cases'][1]['torch_us'] is None
    assert result['unavailable'][0]['reason'] == 'not supported'
