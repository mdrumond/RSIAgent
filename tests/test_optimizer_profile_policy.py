"""Diagnostic coverage is bounded without reducing verified/timed coverage."""
import json

import pytest

torch = pytest.importorskip('torch')
from optimizer import native, profiles
from optimizer.profile_policy import policy, select
from optimizer.reference import prepare_reference


def suite():
    return prepare_reference({'source': 'def forward(x): return x + 1', 'entrypoint': 'forward',
        'signatures': [{'args': [{'shape': [i + 1], 'dtype': 'float32'}]} for i in range(24)]}, max_cases=24)


def instrumentation(monkeypatch):
    calls = {'compile': [], 'profile': [], 'timing': []}
    def compile_case(source, contract, directory, **kwargs):
        calls['compile'].append(contract)
        return lambda inputs, states: [inputs[0] + 1], {}
    def capture(fn, directory, **kwargs):
        fn()
        calls['profile'].append(kwargs)
        return {'groups': [{'group': g, 'available': False, 'reason': 'unsupported'} for g in kwargs['groups']]}
    def timing(fn, directory, **kwargs):
        calls['timing'].append(kwargs)
        fn()
        return {'latency_us': 1}
    monkeypatch.setattr(native, '_compile_case', compile_case)
    monkeypatch.setattr(profiles, 'profile', capture)
    monkeypatch.setattr(native, '_benchmark', timing)
    return calls


def test_full_suite_verification_and_timing_with_one_case_diagnostic(tmp_path, monkeypatch):
    prepared = suite()
    calls = instrumentation(monkeypatch)
    result = native.evaluate(prepared, 'source', tmp_path, device='cpu', arch='test', benchmark=True, profile=True)
    assert result['status'] == 'verified'
    assert len(result['cases']) == len(calls['compile']) == 24
    assert len(calls['timing']) == 48
    assert len(calls['profile']) == 2  # one Torch and one native replay
    assert all(x['groups'] == ['PipeUtilization'] for x in calls['profile'])
    assert all(row['correctness'] for row in result['cases'])
    selected = result['profiling_coverage']['selected_cases']
    assert selected == [prepared.cases[-1]['case']['id']]
    assert sum('profiling' in row for row in result['cases']) == 1


def test_separate_profile_rechecks_only_selected_cases_and_is_not_full_verification(tmp_path, monkeypatch):
    calls = instrumentation(monkeypatch)
    result = native.evaluate(suite(), 'source', tmp_path, device='cpu', arch='test',
                             profile=True, diagnostic_only=True)
    assert result['status'] == 'diagnostic'
    assert len(result['cases']) == len(calls['compile']) == 1
    assert len(calls['profile']) == 1
    assert result['profiling_coverage']['implementations'] == ['ascendc']
    assert calls['timing'] == []
    assert result['profiling_coverage']['suite_case_count'] == 24
    assert result['profiling_coverage']['correctness_scope'].startswith('selected cases only')


def test_full_profile_is_explicit_opt_in(tmp_path, monkeypatch):
    calls = instrumentation(monkeypatch)
    result = native.evaluate(suite(), 'source', tmp_path, device='cpu', arch='test',
        profile=True, diagnostic_only=True, profile_policy={'max_cases': 24, 'groups': list(profiles.GROUPS)})
    assert len(result['cases']) == 24
    assert len(calls['profile']) == 24
    assert all(x['groups'] == list(profiles.GROUPS) for x in calls['profile'])


def test_profile_failure_does_not_erase_full_suite_verification(tmp_path, monkeypatch):
    calls = instrumentation(monkeypatch)
    monkeypatch.setattr(profiles, 'profile', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('missing metrics')))
    result = native.evaluate(suite(), 'source', tmp_path, device='cpu', arch='test', benchmark=True, profile=True)
    assert result['status'] == 'verified'
    assert len(calls['timing']) == 48
    diagnostic = next(row['profiling'] for row in result['cases'] if 'profiling' in row)
    assert diagnostic['ascendc']['status'] == 'unavailable'
    assert json.loads((tmp_path / 'result.json').read_text()) == result


@pytest.mark.parametrize('value', [{'max_cases': 0}, {'max_cases': True}, {'groups': []}, {'groups': ['unknown']}])
def test_invalid_policy_rejected(value):
    with pytest.raises(ValueError):
        policy(value)


@pytest.mark.parametrize('backend_name', ['RemoteBackend', 'LocalBackend'])
def test_remote_backend_forwards_effective_policy(tmp_path, monkeypatch, backend_name):
    from optimizer import worker
    backend = object.__new__(getattr(worker, backend_name))
    backend.config = {}
    monkeypatch.setattr(backend, '_operation', lambda operation, directory, **payload: payload)
    assert backend.profile('source', tmp_path)['profiling'] == {'max_cases': 1, 'groups': ['PipeUtilization']}
    backend.config = {'profiling': {'max_cases': 2, 'groups': ['Memory', 'PipeUtilization']}}
    assert backend.profile(None, tmp_path)['profiling']['max_cases'] == 2


def test_local_worker_validates_profiling_before_early_return(tmp_path):
    from optimizer.worker import load_config
    settings = {'execution': 'local', 'guide': str(tmp_path / 'guide'),
                'workspace': str(tmp_path / 'workspace'),
                'physical_device': 0, 'arch': 'test-arch',
                'profiling': {'max_cases': 0}}
    path = tmp_path / 'worker.json'
    path.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match='max_cases'):
        load_config(path)
    settings['profiling'] = {'max_cases': 2, 'groups': ['Memory']}
    path.write_text(json.dumps(settings))
    assert load_config(path)['profiling'] == settings['profiling']


def test_remote_reference_profile_selects_one_and_binds_suite_identity(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from optimizer.remote_worker import execute
    prepared = suite()
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(set_device=lambda device: None), raising=False)
    monkeypatch.setattr(native, 'load_reference', lambda *a: prepared)
    calls = []
    monkeypatch.setattr(profiles, 'profile', lambda fn, directory, **kwargs: calls.append(kwargs) or {'groups': []})
    result = execute({'operation': 'profile', 'directory': str(tmp_path), 'baseline': '/frozen'})
    assert len(calls) == len(result['cases']) == 1
    assert calls[0]['groups'] == ['PipeUtilization']
    assert calls[0]['identity']['suite_sha256'] == prepared.manifest['suite_sha256']
    assert calls[0]['identity']['case'] == prepared.cases[-1]['case']['id']


def test_remote_canonical_evaluation_does_not_select_diagnostic_subset(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from optimizer.remote_worker import execute
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(set_device=lambda device: None), raising=False)
    calls = []
    monkeypatch.setattr(native, 'evaluate', lambda *a, **kwargs: calls.append(kwargs) or {})
    request = {'operation': 'evaluate', 'directory': str(tmp_path), 'baseline': '/frozen',
               'source': 'kernel', 'arch': 'test', 'block_count': 1}
    execute(request)
    assert calls[0]['benchmark'] is True
    assert calls[0]['diagnostic_only'] is False
    execute(request | {'operation': 'profile'})
    assert calls[1]['diagnostic_only'] is True


def test_remote_digest_keeps_only_requested_counter_and_exact_kernel_rows(tmp_path, monkeypatch):
    import csv
    def capture(fn, root, *args, **kwargs):
        root.mkdir(parents=True)
        path = root / 'op_summary.csv'
        with path.open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['Op Name', 'Task Duration(us)', 'aiv_vec_ratio'])
            writer.writerow(['other_kernel', '1', '90'])
            writer.writerow(['optimizer_abc_kernel', '2', '0'])
        return root
    monkeypatch.setattr(profiles, 'capture', capture)
    monkeypatch.setattr(profiles, 'discover', lambda: {'groups': ['PipeUtilization']})
    result = profiles.profile(lambda: None, tmp_path / 'case-ascendc-profile',
        groups=['PipeUtilization'], identity={'kernel_symbol': 'optimizer_abc_kernel'})
    group = result['groups'][0]
    assert group['available'] is True  # genuinely measured zero is not missing
    metrics = group['reports'][0]['metrics']
    assert len(metrics) == 1
    assert metrics[0]['name'] == 'aiv_vec_ratio'
    assert metrics[0]['value'] == 0
    assert metrics[0]['identity']['Op Name'] == 'optimizer_abc_kernel'
    report = group['reports'][0]
    assert all(report[key]['sha256'] for key in ('source', 'json', 'csv'))
    assert len(json.loads(open(report['json']['path']).read())) > 1  # complete table retained remotely


def test_publish_transfers_digests_not_raw_diagnostic_tables(tmp_path):
    import tarfile
    from optimizer.remote_worker import publish
    raw = tmp_path / 'case-ascendc-profile/PipeUtilization/raw/op_summary.csv'
    raw.parent.mkdir(parents=True)
    raw.write_text('raw diagnostic')
    normalized = tmp_path / 'case-ascendc-profile/PipeUtilization/compact/metrics-0.json'
    normalized.parent.mkdir(parents=True)
    normalized.write_text('normalized diagnostic')
    digest = tmp_path / 'case-ascendc-profile/profile.json'
    digest.write_text('{"diagnostic":true}')
    timing = tmp_path / 'case-ascendc-timing/raw/op_summary.csv'
    timing.parent.mkdir(parents=True)
    timing.write_text('canonical timing samples')
    publish(tmp_path, {'status': 'diagnostic'})
    with tarfile.open(tmp_path / 'compact.tar.gz') as archive:
        names = archive.getnames()
    assert str(raw.relative_to(tmp_path)) not in names
    assert str(normalized.relative_to(tmp_path)) not in names
    assert str(digest.relative_to(tmp_path)) in names
    assert str(timing.relative_to(tmp_path)) in names
    assert raw.exists() and normalized.exists()
