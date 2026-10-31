"""Behavioral admission tests; synthetic measurements are not device evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi.cpl_execution import CplConfig, CplExecutor
from rsi.device_admission import guard
from rsi.python_execution import atomic_json


def probe(product='a3', *, health='OK', error='NA', memory=5, busy=False):
    values = {
        'board': 'NPU ID : 2\nChip Name : ' + ('Ascend950PR' if product == 'a5' else '910B4-1') + '\n',
        'health': f'Health Status : {health}\nError Code : {error}\n',
        'usages': (f'HBM Capacity(MB) : 65536\nHBM Usage Rate(%) : {memory}\n'
                   f'Aicore Usage Rate(%) : {30 if busy else 0}\nAivector Usage Rate(%) : 0\n'),
        'proc-mem': 'Process id:123 Process name:python Process memory(MB):50\n' if busy
                    else 'No process in device.\n',
    }
    if product == 'a5':
        values['usages'] += 'NPU Utilization(%) : 0\nAicube Usage Rate(%) : 0\n'
        values['current-fault-event'] = 'Fault Event Count : 0\nFault Event Information : NA\n'
    return dict(physical_device=2, queries={k: dict(exit=0, stdout=v, stderr='') for k, v in values.items()})


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = CplConfig('test-target', 'test-runtime', '/workspace', 2,
                       tmp_path / 'skill/scripts/cpl-remote', 'frozen', '/workspace/commands')
    executor = CplExecutor('frozen', config)
    metadata = dict(executor.metadata, controller_placement='here', outer_handle=None)
    configuration = dict(executor=executor, evidence_root=tmp_path, probe_id='host-request',
                         product='a3', chip=0)
    state = dict(probe=probe(), calls=[], terminal='completed', fail_submit=False)
    handle = 'remote:test-target:job:original'
    def cli(args, remaining, response_file=None, *, result_path=None):
        response_file = result_path if result_path is not None else response_file
        assert remaining() > 0
        state['calls'].append(args)
        action = args[0]
        value = dict(target='test-target', handle=handle, state='completed')
        if action == 'run':
            value.update(state='queued', runtime='test-runtime', npu=dict(devices=[2], checked_at='now', pool='pool'))
            receipt = args[args.index('--receipt-file') + 1]
            atomic_json(receipt, value)
            if state['fail_submit']:
                raise OSError('disconnected after receipt')
        elif action == 'observe':
            assert args[1] == handle
            value.update(state=state['terminal'], exit=0)
        elif action == 'logs':
            assert args[1] == handle
            value['content'] = json.dumps(state['probe'])
        else:
            raise AssertionError(args)
        if response_file:
            atomic_json(response_file, value)
        return value
    monkeypatch.setattr(executor, '_cli', cli)
    def run(**overrides):
        return guard(SimpleNamespace(), metadata=metadata, configuration=configuration,
                     remaining=lambda: 60, **overrides)
    return run, configuration, metadata, state


def test_remote_admission_retains_actual_receipts_and_decision(setup):
    run, config, _, state = setup
    result = run()
    assert result['admitted'] and result['physical_device'] == 2
    assert result['measurements']['idle']
    assert json.loads(Path(result['evidence_path']).read_text()) == result
    assert result['raw_measurements']['submission']['handle'].endswith(':original')
    args = state['calls'][0]
    assert args[args.index('--npu') + 1] == 2
    assert args[args.index('--runtime') + 1] == 'test-runtime'


@pytest.mark.parametrize('change,failure', [
    (dict(health='Alarm'), 'unhealthy'), (dict(error='81B18603'), 'unhealthy'),
    (dict(busy=True), 'device_busy'), (dict(memory=50, busy=True), 'device_busy'),
])
def test_excludes_unhealthy_and_unconfigured_busy(setup, change, failure):
    run, _, _, state = setup
    state['probe'] = probe(**change)
    result = run()
    assert not result['admitted'] and result['failure']['type'] == failure
    assert result['raw_measurements']['probe'] == state['probe']


def test_ordinary_busy_requires_explicit_configuration_and_less_than_half_memory(setup):
    run, config, _, state = setup
    config['allow_busy'] = True
    state['probe'] = probe(busy=True, memory=49)
    assert run()['admitted']
    config['require_idle'] = True
    assert not run()['admitted']


@pytest.mark.parametrize('query', ['health', 'usages', 'proc-mem'])
def test_missing_metric_is_unavailable_never_zero(setup, query):
    run, _, _, state = setup
    del state['probe']['queries'][query]
    result = run()
    assert not result['admitted'] and result['failure']['type'] == 'metric_unavailable'
    assert 'measurements' not in result


def test_disconnect_observes_original_handle_without_duplicate_probe(setup):
    run, _, _, state = setup
    state['fail_submit'] = True
    assert not run()['admitted']
    state['fail_submit'] = False
    assert run()['admitted']
    assert sum(call[0] == 'run' for call in state['calls']) == 1


def test_pending_handle_is_retained_and_reconciled(setup):
    run, _, _, state = setup
    state['terminal'] = 'observation-unavailable'
    result = run()
    assert result['failure']['type'] == 'probe_pending'
    state['terminal'] = 'completed'
    assert run()['admitted']
    assert sum(call[0] == 'run' for call in state['calls']) == 1


def test_expired_budget_does_not_dispatch(setup):
    _, config, metadata, state = setup
    result = guard({}, metadata=metadata, configuration=config, remaining=lambda: 0)
    assert result['failure']['type'] == 'budget_exhausted'
    assert not state['calls']


def test_a5_local_validates_outer_admission_and_has_no_loopback(setup, tmp_path, monkeypatch):
    run, config, metadata, state = setup
    metadata.update(controller_placement='on', executor_identity=None,
                    outer_handle='remote:test-target:job:outer')
    receipt = tmp_path / 'outer.json'
    atomic_json(receipt, dict(target='test-target', handle=metadata['outer_handle'],
                             runtime='test-runtime', npu=dict(devices=[2], pool='pool', checked_at='now')))
    config.update(product='a5', chip=None, outer_receipt=receipt, outer_runtime='test-runtime')
    monkeypatch.setenv('RSI_OUTER_CPL_HANDLE', metadata['outer_handle'])
    monkeypatch.setenv('RSI_CPL_RUNTIME', metadata['runtime'])
    captured = []
    def local(args, **kwargs):
        captured.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(probe('a5')).encode(), stderr=b'')
    monkeypatch.setattr('rsi.device_admission.subprocess.run', local)
    result = run()
    assert result['admitted'] and not state['calls']
    spec = json.loads(captured[0][0][-1])
    assert spec['chip'] is None and spec['device'] == 2
    assert 'current-fault-event' in spec['queries']
    assert captured[0][1]['timeout'] <= 60
    config['probe_id'] = 'second'
    atomic_json(receipt, dict(target='test-target', handle=metadata['outer_handle'], npu=dict(devices=[])))
    assert run()['failure']['type'] == 'outer_admission_invalid'
    assert len(captured) == 1


def test_a5_fault_event_denies_even_with_ok_health(setup):
    run, config, _, state = setup
    config.update(product='a5', chip=None)
    state['probe'] = probe('a5')
    state['probe']['queries']['current-fault-event']['stdout'] = 'Fault Event Count : 1\n'
    assert run()['failure']['type'] == 'unhealthy'


def test_command_profile_requires_idle_even_when_sharing_allowed(setup):
    _, config, metadata, state = setup
    config['allow_busy'] = True
    state['probe'] = probe(busy=True)
    request = SimpleNamespace(operation_id='profile-operation', category='profiling', profile_path=None)
    result = guard(request, phase='command', metadata=metadata, configuration=config, remaining=lambda: 60)
    assert result['failure']['type'] == 'device_busy'


def test_unsupported_measurement_and_query_timeout_are_not_idle(setup):
    run, _, _, state = setup
    state['probe']['queries']['usages']['stdout'] = 'HBM Capacity(MB) : NA\n'
    assert run()['failure']['type'] == 'metric_unavailable'
    state['probe']['queries']['usages'] = dict(failure='query_timeout', stdout='partial', stderr='')
    result = run()
    assert result['failure']['type'] == 'query_timeout'
    assert result['raw_measurements']['probe']['queries']['usages']['stdout'] == 'partial'


def test_wrong_board_identity_excluded(setup):
    run, _, _, state = setup
    state['probe']['queries']['board']['stdout'] = 'NPU ID : 7\nChip Name : 910B4-1\n'
    assert run()['failure']['type'] == 'identity_mismatch'


@pytest.mark.parametrize('chip', [0, None])
def test_query_engine_executes_selected_device_syntax(tmp_path, monkeypatch, chip):
    import os
    import subprocess
    import sys
    from rsi.device_admission import _PROBE
    executable = tmp_path / 'npu-smi'
    executable.write_text('#!/bin/sh\nprintf "measured\\n"\n')
    executable.chmod(0o700)
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ['PATH'])
    spec = dict(device=7, chip=chip, queries=['health'], query_timeout=1, remaining=2)
    output = subprocess.run([sys.executable, '-I', '-c', _PROBE, json.dumps(spec)],
                            capture_output=True, timeout=3, check=True)
    measured = json.loads(output.stdout)['queries']['health']
    expected = ['npu-smi', 'info', '-t', 'health', '-i', '7']
    assert measured['argv'] == expected + (['-c', '0'] if chip == 0 else [])
    assert measured['exit'] == 0 and measured['stdout'] == 'measured\n'


def test_frozen_stdlib_export_queries_once_and_counts_existing_processes(monkeypatch):
    import hashlib
    from rsi.device_health_probe import standalone_hash, standalone_source
    source = standalone_source()
    assert standalone_hash() == hashlib.sha256(source.encode('utf-8')).hexdigest()
    namespace = {}
    exec(compile(source, '<frozen-device-health>', 'exec'), namespace)
    calls = []
    measured = probe('a5', busy=True)
    def query(argv, **kwargs):
        calls.append((argv, kwargs['timeout']))
        name = argv[argv.index('-t') + 1]
        return SimpleNamespace(returncode=0, stdout=measured['queries'][name]['stdout'].encode(), stderr=b'')
    monkeypatch.setattr(namespace['subprocess'], 'run', query)
    result = namespace['local_probe'](physical_device=2,
        configuration=dict(product='a5', chip=None, require_idle=True, allow_busy=True),
        remaining=lambda: 4)
    assert not result['admitted'] and result['failure']['type'] == 'device_busy'
    assert len(calls) == 5 and all(timeout <= 4 for _, timeout in calls)
    assert result['raw_measurements']['queries']['proc-mem']['stdout'] == measured['queries']['proc-mem']['stdout']


def test_standalone_probe_preserves_unavailable_output(monkeypatch):
    import subprocess
    from rsi.device_health_probe import local_probe
    def query(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs['timeout'], output=b'partial measurement', stderr=b'busy driver')
    monkeypatch.setattr('rsi.device_health_probe.subprocess.run', query)
    result = local_probe(physical_device=0, configuration=dict(product='a3', chip=0), remaining=lambda: 2)
    assert result['failure']['type'] == 'query_timeout'
    item = result['raw_measurements']['queries']['board']
    assert item['stdout'] == 'partial measurement' and item['stderr'] == 'busy driver'


def test_explicit_alarm_authority_is_ordinary_only_and_retains_faults(setup):
    run, config, metadata, state = setup
    config.update(product='a5', chip=None)
    state['probe'] = probe('a5', health='Alarm', error='80F78009')
    state['probe']['queries']['current-fault-event']['stdout'] = 'Fault Event Count : 1\nFault Event Information : event id=0x80F78009\n'
    assert run()['failure']['type'] == 'unhealthy'
    config['allow_alarm'] = True
    admitted = run()
    assert admitted['admitted'] and admitted['degraded']
    assert admitted['measurements']['health'] == 'Alarm'
    assert admitted['measurements']['hardware_error_code'] == '80F78009'
    assert admitted['measurements']['fault_event_count'] == 1
    assert admitted['raw_measurements']['probe'] == state['probe']
    request = SimpleNamespace(operation_id='alarm-timing', command='timing', category='execution', profile_path=None)
    timing = guard(request, phase='command', metadata=metadata, configuration=config, remaining=lambda: 60)
    assert timing['failure']['type'] == 'unhealthy'
    state['probe'] = probe('a5', health='Alarm', error='80F78009', busy=True)
    assert run()['failure']['type'] == 'device_busy'
    state['probe'] = probe('a5', health='Failed', error='80F78009')
    assert run()['failure']['type'] == 'unhealthy'


def test_invalid_cli_diagnostics_retained_unknown_submission_never_replayed(setup, monkeypatch):
    run, config, _, _ = setup
    executor = config['executor']
    skill = executor.config.installed_client.parent.parent
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('test-installed-policy')
    monkeypatch.setattr(executor, '_cli', CplExecutor._cli.__get__(executor))
    calls = []
    def invalid(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=2, stdout=b'invalid json\n', stderr=b'native CLI rejected request\n')
    monkeypatch.setattr('rsi.cpl_execution.subprocess.run', invalid)
    first = run()
    assert first['failure']['type'] == 'admission_unavailable'
    evidence = first['raw_measurements']['client_results'][0]
    assert evidence['stdout'] == 'invalid json\n'
    assert evidence['stderr'] == 'native CLI rejected request\n' and evidence['returncode'] == 2
    directory = Path(first['evidence_path']).parent
    intent = (directory / 'intent.json').read_bytes()
    assert not (directory / 'submission.json').exists()
    second = run()
    assert second['failure']['type'] == 'submission_unknown'
    assert len(calls) == 1 and (directory / 'intent.json').read_bytes() == intent
    assert 'submission' not in second['raw_measurements']


def local_setup(setup, tmp_path, monkeypatch, *, health='OK', returncode=0):
    run, config, metadata, state = setup
    metadata.update(controller_placement='on', executor_identity=None,
                    outer_handle='remote:test-target:job:outer')
    receipt = tmp_path / 'outer.json'
    atomic_json(receipt, dict(target='test-target', runtime='test-runtime',
        handle=metadata['outer_handle'], npu=dict(devices=[2], pool='pool', checked_at='now')))
    config.update(product='a5', chip=None, outer_receipt=receipt, outer_runtime='test-runtime')
    monkeypatch.setenv('RSI_OUTER_CPL_HANDLE', metadata['outer_handle'])
    monkeypatch.setenv('RSI_CPL_RUNTIME', metadata['runtime'])
    calls = []
    def query(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=returncode, stdout=json.dumps(probe('a5', health=health)).encode(), stderr=b'')
    monkeypatch.setattr('rsi.device_admission.subprocess.run', query)
    return run, calls


@pytest.mark.parametrize('health,returncode', [('OK', 0), ('Alarm', 0), ('OK', 7)])
def test_completed_local_probe_reconciles_exact_evidence_without_requery_or_double_charge(
        setup, tmp_path, monkeypatch, health, returncode):
    run, calls = local_setup(setup, tmp_path, monkeypatch, health=health, returncode=returncode)
    ticks = iter([100., 107., 200., 200.125])
    monkeypatch.setattr('rsi.device_admission.time', SimpleNamespace(monotonic=lambda: next(ticks)))
    first = run()
    directory = Path(first['evidence_path']).parent
    before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    second = run()
    assert second['admitted'] == first['admitted'] == (health == 'OK' and returncode == 0)
    assert second.get('failure') == first.get('failure')
    assert second['raw_measurements'] == first['raw_measurements']
    assert second['evidence_path'] == first['evidence_path'] and second['reconciled']
    assert second['elapsed_seconds'] == 0.125 and second['retained_probe_elapsed_seconds'] == 7
    assert len(calls) == 1
    assert {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()} == before


@pytest.mark.parametrize('keep_measurements', [True, False])
def test_local_marker_reconciles_only_durable_completion(setup, tmp_path, monkeypatch, keep_measurements):
    run, calls = local_setup(setup, tmp_path, monkeypatch)
    first = run()
    directory = Path(first['evidence_path']).parent
    (directory / 'decision.json').unlink()  # Interruption after durable measurements, before final decision.
    if not keep_measurements:
        (directory / 'measurements.json').unlink()
    before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    second = run()
    assert second['admitted'] is keep_measurements
    if keep_measurements:
        assert second['raw_measurements'] == json.loads(before['measurements.json'])
    else:
        assert second['failure']['type'] == 'probe_unknown'
    assert len(calls) == 1 and not (directory / 'decision.json').exists()
    assert {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()} == before
