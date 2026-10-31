"""Common review regressions: negative admission, parent trials and parent stop."""
from dataclasses import replace
import json
import os
from pathlib import Path
import threading
import time

import pytest

from rsi.configuration import set_configuration, show_configuration
from test_rsi_services import instance, Roles
from test_rsi_service_wave import wave_instance
from test_rsi_worker_admission import profiles


def transient_guard(request, *, phase, metadata, configuration, remaining):
    root = Path(configuration['coordination'])
    root.mkdir(exist_ok=True)
    device = str(metadata['physical_device'])
    if phase == 'admission':
        marker = root / ('first-' + device)
        if not marker.exists():
            marker.write_text(request.dispatch_id)
        with (root / 'calls.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(dispatch=request.dispatch_id, device=device)) + '\n')
        time.sleep(.01)  # Real retained elapsed, not a fabricated duration.
        if marker.read_text() == request.dispatch_id and not (root / 'release').exists():
            return dict(admitted=False, physical_device=metadata['physical_device'],
                observation='explicit CPU admission failure',
                failure=dict(type=configuration['failure']), handle='retained-fixture-probe-' + device)
    return dict(admitted=True, physical_device=metadata['physical_device'], observation='CPU admitted')


@pytest.mark.parametrize('failure', ['busy', 'probe_pending'])
def test_pre_runner_admission_can_settle_original_probe_then_resume(wave_instance, tmp_path, failure):
    path = tmp_path / 'profiles.json'
    resolver = profiles(path)
    config = json.loads(path.read_text())
    coordination = tmp_path / 'admission-state'
    for profile in config['profiles'].values():
        profile.update(guard='test_rsi_wave_review_recovery:transient_guard',
            guard_configuration=dict(coordination=str(coordination), failure=failure))
    path.write_text(json.dumps(config))
    settings = show_configuration(wave_instance.root)['payload']
    settings['project_wave']['devices'] = [dict(device=str(i), worker_profile=f'worker-{i}') for i in (0, 1)]
    set_configuration(wave_instance.root, settings)
    wave_instance.executor_factory = resolver
    cycle = wave_instance.create_cycle('admission-retry', 'phase1')
    try:
        with pytest.raises(RuntimeError):
            cycle.step()
        wave = cycle.scheduler.state['active_wave']
        pool = cycle.waves._pool(wave)
        before = pool.status()['projects']
        original_dispatches = {r['dispatch_id'] for r in before.values()}
        assert not any((pool._branch(op) / 'runner').exists() for op in before)
        if failure == 'probe_pending':
            assert {r['status'] for r in before.values()} == {'pending'}
            assert all(r['device'] is not None for r in before.values())
        (coordination / 'release').touch()
        for op, row in before.items():
            if row['status'] == 'pending':
                cycle.waves.apply_observation(op, {})
        closed = pool.status()['projects']
        assert all(r['status'] == 'ready' and not r['reopen'] and r['actor_seconds'] > 0
                   and r['device'] is None for r in closed.values())
        calls = [json.loads(line) for line in (coordination / 'calls.jsonl').read_text().splitlines()]
        assert {r['dispatch'] for r in calls} == original_dispatches
        cycle.reconcile(dispatch_ready=True)
        settled = pool.status()['projects']
        assert all(r['status'] == 'completed' and r['receipt']['result']['status'] == 'accepted'
                   for r in settled.values())
        for op, row in settled.items():
            assert row['actor_seconds'] >= closed[op]['actor_seconds']
            runner = cycle.open_project(op)
            assert runner.workflow.accounting.seconds() >= closed[op]['actor_seconds']
        assert not cycle.memory.records()
    finally:
        cycle.close()


def test_paired_local_trials_keep_frozen_device_guard_and_owned_cache(wave_instance, tmp_path, monkeypatch):
    path = tmp_path / 'profiles.json'
    resolver = profiles(path)
    values = json.loads(path.read_text())
    coordination = tmp_path / 'binding-order'
    coordination.mkdir()
    (coordination / 'third.bound').touch()  # No forced third sibling in this two-project case.
    for profile in values['profiles'].values():
        profile.update(project_binder='test_rsi_lease_binding:cpu_bind_project',
            project_configuration=dict(coordination=str(coordination)))
    path.write_text(json.dumps(values))
    settings = show_configuration(wave_instance.root)['payload']
    settings['project_wave']['devices'] = [dict(device=str(i), worker_profile=f'worker-{i}') for i in (0, 1)]
    set_configuration(wave_instance.root, settings)
    wave_instance.executor_factory = resolver
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES', 'parent-unleased')
    parent = dict(os.environ)
    cycle = wave_instance.create_cycle('local-trials', 'phase2')
    try:
        assert cycle.step().outcome == 'accepted'
        original = cycle.scheduler.state['originals'][0]
        checkpoint = cycle.open_project(original['operation_id'])
        assert checkpoint.backend.command_guard is not None
        baseline = cycle.scheduler.state['skills']
        pair = cycle.pair(dict(original=original, memory=cycle.scheduler.state['memory'],
            incumbent=baseline, candidate=baseline), 'test-pair', cycle.scheduler.remaining())
        caches = []
        for arm in ('incumbent', 'candidate'):
            request = pair.request(arm)
            pair.step()
            runner = cycle.trial(request, reopen=True)
            assert runner.workflow.outcome == 'accepted'
            assert runner.context()['execution_context']['physical_device'] == '0'
            assert runner.backend.command_guard is not None
            receipts = [json.loads(p.read_text()) for p in (runner.root / 'receipts').glob('*.json')]
            complete = [runner.store.read_json(r['artifact']) for r in receipts if r.get('status') == 'complete']
            assert complete and all(r['guard_receipt']['admitted'] for r in complete)
            env = runner.backend.command_guard.environment(type('Command', (),
                dict(project=runner.backend.project))(), inherited={})
            caches.append(env['CATLASS_DSL_CACHE_DIR'])
            assert env['ASCEND_RT_VISIBLE_DEVICES'] == '0'
            assert dict(os.environ) == parent
        assert len(set(caches)) == 2
        # A negative command guard on a fresh local trial must prevent Popen,
        # even though its frozen backend identity uses default-local execution.
        denied_request = replace(pair.request('incumbent'), operation_id='denied-local-trial',
                                 root=cycle.root / 'denied-local-trial')
        denied = cycle.trial(denied_request, reopen=False)
        for profile in values['profiles'].values():
            profile['guard_configuration'] = {'admitted': False}
        path.write_text(json.dumps(values))
        def unexpected(*args, **kwargs):
            pytest.fail('denied paired command must not launch a subprocess')
        monkeypatch.setattr('rsi.python_execution.subprocess.Popen', unexpected)
        from rsi.executor_profiles import GuardDenied
        with pytest.raises(GuardDenied):
            denied.backend.run('denied-command', 'compile')
        assert denied.backend.receipt('denied-command')['guard_receipt']['admitted'] is False
        assert dict(os.environ) == parent
    finally:
        cycle.close()


class BlockingWriter(Roles):
    stop = None
    interrupted = None

    def skillwriter(self, *args, **kwargs):
        self.stop.set()
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            self.interrupted.append(True)
            raise
        return super().skillwriter(*args, **kwargs)


def test_wave_enabled_parent_writer_is_interrupted_outside_worker_drain(wave_instance, monkeypatch):
    import rsi.service
    stop = BlockingWriter.stop = threading.Event()
    interrupted = BlockingWriter.interrupted = []
    wave_instance.test = replace(wave_instance.test, roles_factory=BlockingWriter)
    cycle = wave_instance.create_cycle('parent-stop', 'phase2')
    try:
        assert cycle.step().outcome == 'accepted'
    finally:
        cycle.close()
    monkeypatch.setattr(rsi.service, 'InstanceServices', lambda _: wave_instance)
    result = rsi.service.run_job(wave_instance.root,
        dict(action='resume', cycle_id='parent-stop'), show_configuration(wave_instance.root),
        should_stop=stop.is_set, emit=lambda event: None)
    assert result['status'] == 'stopped'
    assert interrupted == [True]
