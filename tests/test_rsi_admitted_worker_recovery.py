"""Public reconciliation after real admitted worker process loss before Runner."""
from dataclasses import dataclass
import json
import os
import signal

import pytest

from rsi.configuration import set_configuration, show_configuration
from rsi.service_wave import ProjectWorkerFactory
from test_rsi_service_wave import wave_instance
from test_rsi_services import instance
from test_rsi_worker_admission import profiles


@dataclass
class KilledSetup:
    factory: ProjectWorkerFactory
    legacy: bool = False

    def prepare_worker(self, request):
        self.factory.prepare_worker(request)

    def __call__(self, request):
        if self.legacy:
            os.kill(os.getpid(), signal.SIGKILL)
        from rsi.project_composition import ProjectComposition
        def interrupted(composition, memory_ref):
            os.kill(os.getpid(), signal.SIGKILL)
        ProjectComposition.query = interrupted
        return self.factory(request)


def interrupted_wave(services, monkeypatch, *, legacy=False, unknown=False):
    services.executor_factory = profiles(services.root / 'test-profiles.json')
    if legacy:
        path = services.root / 'test-profiles.json'
        config = json.loads(path.read_text())
        for profile in config['profiles'].values():
            profile.update(project_binder=('test_rsi_lease_binding:cpu_bind_project' if unknown
                           else 'rsi.triton_worker_binding:bind_project'),
                           project_configuration={}, guard='test_rsi_admitted_worker_recovery:closed_guard')
        path.write_text(json.dumps(config))
    settings = show_configuration(services.root)['payload']
    settings['project_wave']['devices'] = [dict(device=str(i), worker_profile=f'worker-{i}') for i in (0, 1)]
    set_configuration(services.root, settings)
    cycle = services.create_cycle('interrupted', 'phase1')
    factory = cycle.waves._factory
    with monkeypatch.context() as paused:
        paused.setattr(cycle.waves, '_factory', lambda wave: KilledSetup(factory(wave), legacy))
        paused.setattr(cycle.waves, 'observe', lambda: cycle.waves._pool(cycle.scheduler.state['active_wave']).status())
        with pytest.raises(RuntimeError):
            cycle.step()
    wave = cycle.scheduler.state['active_wave']
    pool = cycle.waves._pool(wave)
    return cycle, wave, pool


def test_killed_admitted_setup_public_reconcile_preserves_original_and_runs_once(wave_instance, monkeypatch):
    cycle, wave, pool = interrupted_wave(wave_instance, monkeypatch)
    before = pool.status()['projects']
    plan = pool.plan
    specs = [op['spec'] for op in wave['operations']]
    try:
        assert all(row['status'] == 'pending' for row in before.values())
        assert all(not (pool._branch(op) / 'runner').exists() for op in before)
        with pytest.raises(ValueError, match='wave requires observation'):
            cycle.reconcile()  # Reconciles leases but never dispatches ready work.
        after = pool.status()['projects']
        assert all(row['status'] == 'ready' for row in after.values())
        for op, row in after.items():
            assert row['dispatch_id'] == before[op]['dispatch_id']
            assert row['allowance'] == before[op]['allowance']
            assert row['actor_seconds'] > 0
            assert row['reopen'] is False and row['recovery'] is None
            retained = cycle.store.read_json(row['admission_ref'])
            assert retained['setup_closure']['setup_elapsed'] is None
            assert retained['setup_closure']['setup_actor_billable'] is False
            assert not (pool._branch(op) / 'wave-result.json').exists()
        elapsed = {op: row['actor_seconds'] for op, row in after.items()}
        with pytest.raises(ValueError, match='wave requires observation'):
            cycle.reconcile()
        assert {op: row['actor_seconds'] for op, row in pool.status()['projects'].items()} == elapsed
        assert pool.plan == plan and [op['spec'] for op in wave['operations']] == specs
        cycle.reconcile(dispatch_ready=True)
        completed = pool.status()['projects']
        assert all(row['status'] == 'completed' for row in completed.values())
        for op, row in completed.items():
            assert row['actor_seconds'] >= elapsed[op]
            assert row['receipt']['result']['status'] == 'accepted'
            assert row['dispatch_id'] != before[op]['dispatch_id']
            runner = cycle.open_project(op)
            assert runner.spec.to_dict() == next(item['spec'] for item in wave['operations'] if item['id'] == op)
    finally:
        cycle.close()


@pytest.mark.parametrize('unresolved', ['live', 'unknown-binding', 'partial-runner', 'missing-identity'])
def test_unresolved_effects_keep_original_reservation(wave_instance, unresolved, monkeypatch):
    from rsi.jobs import _identity
    cycle, wave, pool = interrupted_wave(wave_instance, monkeypatch)
    try:
        op, before = next(iter(pool.status()['projects'].items()))
        branch = pool._branch(op)
        if unresolved == 'live':
            worker = json.loads((branch / 'worker.json').read_text())
            worker.update(pid=os.getpid(), process=_identity(os.getpid()))
            (branch / 'worker.json').write_text(json.dumps(worker))
        elif unresolved == 'unknown-binding':
            (branch / ('worker-setup-' + before['dispatch_id'] + '.json')).unlink()
        elif unresolved == 'partial-runner':
            (branch / 'runner').mkdir()
        else:
            (branch / 'worker.json').unlink()
        cycle.waves.observe()
        after = pool.status()['projects'][op]
        assert after['status'] == 'pending'
        assert after['dispatch_id'] == before['dispatch_id']
        assert after['device'] == before['device']
        assert after['actor_seconds'] == before['actor_seconds']
        assert 'closure' not in after and not (branch / 'wave-result.json').exists()
    finally:
        cycle.close()


def closed_guard(request, *, phase, metadata, configuration, remaining):
    from test_rsi_worker_admission import scripted_guard
    receipt = scripted_guard(request, phase=phase, metadata=metadata, configuration=configuration, remaining=remaining)
    handle = 'remote:cpu-test:job:' + request.dispatch_id
    receipt['raw_measurements'] = dict(submission=dict(handle=handle),
        terminal=dict(handle=handle, state='completed', exit=0))
    return receipt


@pytest.mark.parametrize('unknown', [False, True])
def test_legacy_no_checkpoint_derives_only_audited_binder_closure(wave_instance, monkeypatch, unknown):
    cycle, wave, pool = interrupted_wave(wave_instance, monkeypatch, legacy=True, unknown=unknown)
    try:
        before = pool.status()['projects']
        assert not list(pool.root.glob('projects/*/worker-setup-*.json'))
        for op in before:
            if unknown:
                with pytest.raises(ValueError, match='admission remains unresolved'):
                    cycle.waves.apply_observation(op, {})
            else:
                cycle.waves.apply_observation(op, {})  # Existing public observation interface.
        rows = pool.status()['projects']
        assert {row['status'] for row in rows.values()} == ({'pending'} if unknown else {'ready'})
        for op, row in rows.items():
            assert row['dispatch_id'] == before[op]['dispatch_id']
            assert row['allowance'] == before[op]['allowance']
            assert row['recovery'] is None
            if not unknown:
                closure = cycle.store.read_json(row['admission_ref'])['setup_closure']
                assert closure['boundary'] is None and closure['setup_elapsed'] is None
                assert closure['setup_only_binding'] is True
                assert row['actor_seconds'] > 0
            assert not (pool._branch(op) / 'runner').exists()
            assert not (pool._branch(op) / 'wave-result.json').exists()
    finally:
        cycle.close()
