"""Host-profile context and guard behavior; scripted guards are not NPU evidence."""
import json
import os
import sys

import pytest

from rsi.configuration import set_configuration, show_configuration
from rsi.executor_profiles import ExecutorProfiles
from rsi.python_execution import PythonExecution
from test_rsi_service_wave import wave_instance
from test_rsi_services import instance


def scripted_guard(request, *, phase, metadata, configuration, remaining):
    assert remaining() is None or remaining() > 0
    if phase == 'admission':
        folder = request.root.parent
    else:
        folder = request.project.parent.parent
    path = folder / 'scripted-guard.jsonl'
    with path.open('a') as stream:
        stream.write(json.dumps(dict(phase=phase, mask=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
            cache=os.environ.get('CATLASS_DSL_CACHE_DIR'), expected_source=os.environ.get('RSI_EXPECTED_SOURCE_HASH'), outer=os.environ.get('RSI_OUTER_CPL_HANDLE'))) + '\n')
    return dict(admitted=configuration.get('admitted', True), physical_device=metadata['physical_device'],
                observation='explicit test-only admission')


def cpu_worker_environment(*, inherited, outer_cpl_handle, physical_device, cache_directory):
    """Explicit test callback; no target compiler, registry or CANN module imports."""
    result = dict(inherited)
    result.pop('RSI_EXPECTED_SOURCE_HASH', None)
    result.update(ASCEND_RT_VISIBLE_DEVICES=str(physical_device),
        CATLASS_DSL_CACHE_DIR=str(cache_directory), RSI_OUTER_CPL_HANDLE=outer_cpl_handle,
        RSI_CPL_RUNTIME='cpu-test-runtime')
    return result


def profiles(path):
    outer = 'remote:cpu-test:session:test-only'
    entries = {}
    for device in (0, 1):
        entries[f'worker-{device}'] = dict(metadata=dict(target='test-only', runtime='cpu-test-runtime',
            executor_identity=None, physical_device=device, controller_placement='local', outer_handle=outer),
            worker_environment='test_rsi_worker_admission:cpu_worker_environment',
            worker_configuration=dict(outer_cpl_handle=outer, physical_device=device),
            guard='test_rsi_worker_admission:scripted_guard')
    path.write_text(json.dumps({'profiles': entries}))
    return ExecutorProfiles(str(path))


def test_two_real_service_children_get_owned_context_and_command_guards(wave_instance, monkeypatch):
    resolver = profiles(wave_instance.root / 'ignored-test-profiles.json')
    settings = show_configuration(wave_instance.root)['payload']
    settings['project_wave']['devices'] = [dict(device=str(i), worker_profile=f'worker-{i}') for i in (0, 1)]
    set_configuration(wave_instance.root, settings)
    wave_instance.executor_factory = resolver
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES', 'parent-mask')
    monkeypatch.setenv('CATLASS_DSL_CACHE_DIR', 'parent-cache')
    monkeypatch.setenv('RSI_EXPECTED_SOURCE_HASH', 'stale-parent-hash')
    cycle = wave_instance.create_cycle('context', 'phase1')
    try:
        assert cycle.step().outcome == 'accepted'
        wave = cycle.scheduler.state['active_wave']
        rows = cycle.waves._pool(wave).status()['projects']
        caches = set()
        for operation, row in rows.items():
            branch = cycle.root / 'waves' / wave['id'] / 'projects' / operation
            records = [json.loads(line) for line in (branch / 'scripted-guard.jsonl').read_text().splitlines()]
            assert [r['phase'] for r in records] == ['admission', 'command', 'command', 'command']
            assert {r['mask'] for r in records} == {row['lease_device']}
            assert {r['outer'] for r in records} == {'remote:cpu-test:session:test-only'}
            assert all(r['expected_source'] is None for r in records)
            caches.add(records[0]['cache'])
            assert records[0]['cache'] == str(branch / 'target-cache')
        assert len(caches) == 2
        assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == 'parent-mask'
        assert os.environ['CATLASS_DSL_CACHE_DIR'] == 'parent-cache'
        assert os.environ['RSI_EXPECTED_SOURCE_HASH'] == 'stale-parent-hash'
    finally:
        cycle.close()


def test_profile_without_guard_fails_closed_before_target_construction(tmp_path):
    path = tmp_path / 'profiles.json'
    resolver = profiles(path)
    value = json.loads(path.read_text())
    del value['profiles']['worker-0']['guard']
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        resolver.metadata('worker-0')
    with pytest.raises(ValueError):
        resolver('worker-0')


def test_denied_local_command_guard_prevents_process_and_retains_failure(tmp_path, monkeypatch):
    from rsi.contracts import BudgetLimits
    from rsi.runtime import Workflow
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'solution.py').write_text('pass\n')
    verifier = tmp_path / 'verify.py'
    verifier.write_text("from pathlib import Path; Path('must-not-run').touch()\n")
    backend = PythonExecution.snapshot(tmp_path / 'runner', source, verifier,
        commands={'test': [sys.executable, '{verifier}']})
    now = [100.]
    monkeypatch.setattr('rsi.python_execution.time.monotonic', lambda: now[0])
    workflow = Workflow(tmp_path / 'state.json', lambda *args: True,
        source_hash=backend.project_hash(), limits=BudgetLimits(actor_seconds=20), clock=lambda: now[0])
    backend.workflow = workflow
    calls = []
    def denied(request, *, remaining):
        calls.append(request.operation_id)
        now[0] += 1.
        raise ValueError('scripted policy denial')
    backend.command_guard = denied
    with pytest.raises(ValueError):
        backend.run('denied', 'test')
    assert calls == ['denied']
    assert not (backend.project / 'must-not-run').exists()
    assert backend.receipt('denied')['classification'] == 'interrupted'
    assert workflow.accounting.seconds() == 1.
