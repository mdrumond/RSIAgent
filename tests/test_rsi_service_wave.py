"""Opt-in product waves with real CPU children and the canonical service stack."""
from dataclasses import replace
import json

import pytest

from rsi.configuration import set_configuration, show_configuration
from rsi.services import TestBindings as Bindings
from test_rsi_services import Decisions, Roles, instance


def wave_query(payload):
    return dict(results=[], negative_records=[], provenance={'test': True})


class FailedDecisions(Decisions):
    def next_action(self, *args, **kwargs):
        from coding.pi.rpc import PiProviderError
        raise PiProviderError('scripted closed test actor failure')


@pytest.fixture
def wave_instance(instance):
    instance.test = Bindings(wave_query, Decisions, Roles, 'explicit-test-binding')
    settings = show_configuration(instance.root)['payload']
    second = dict(settings['phase1_projects'][0], project_id='other')
    settings['phase1_projects'].append(second)
    settings['phase2_projects'].append(dict(settings['phase2_projects'][0], project_id='other'))
    settings['project_configs']['other'] = dict(settings['project_configs']['answer'])
    settings['project_wave'] = dict(max_parallel=2, devices=[{'device': 'cpu0'}, {'device': 'cpu1'}])
    settings['cycle_limits'] = dict(wave_size=2, actor_seconds=30000., wall_seconds=120.)
    settings['tuning']['at_wave_end'] = False
    set_configuration(instance.root, settings)
    resource = instance.root / 'source/reference-skills/basic/scripts'
    resource.mkdir(parents=True)
    (resource / 'example.py').write_text('reference_value = 7\n')
    return instance


def test_service_wave_real_children_parent_memory_barrier_and_restart(wave_instance):
    cycle = wave_instance.create_cycle('parallel', 'phase1')
    try:
        memory, skills = cycle.scheduler.state['memory'], cycle.scheduler.state['skills']
        first = cycle.step()
        assert first.outcome == 'accepted'
        wave = cycle.scheduler.state['active_wave']
        pool = cycle.waves._pool(wave)
        children = pool.status()['projects']
        assert len({r['worker']['pid'] for r in children.values()}) == 2
        assert all(row['status'] == 'completed' for row in children.values())
        assert not cycle.memory.records()
        assert cycle.scheduler.state['memory'] == memory
        pids = {op: row['worker']['pid'] for op, row in children.items()}
        for op in wave['operations']:
            runner = cycle.open_project(op['id'])
            binding = json.loads((runner.root / 'service-binding.json').read_text())
            assert binding['memory'] == memory and binding['skills'] == skills
            assert runner.backend.read('reference-skills/basic/scripts/example.py')['content'] == 'reference_value = 7\n'
            assert runner.config.editable_files == ('solution.py',)
            with pytest.raises(ValueError):
                runner.backend.edit('reference-skills/basic/scripts/example.py', 'changed',
                    expected_file_hash='unused', expected_project_hash='unused')
        assert cycle.step().outcome == 'accepted'
        assert not cycle.memory.records()
        cycle.step()  # Existing Phase1 memory callback, only after all siblings.
        assert len(cycle.memory.records()) == 2
        assert {r['proposal']['kind'] for r in cycle.memory.records()} == {'correctness'}
        while cycle.scheduler.state['phase'] != 'complete':
            cycle.step()
        assert cycle.scheduler.state['active_wave'] is None
        assert cycle.scheduler.state['spent'] >= sum(r['actor_seconds'] for r in children.values())
    finally:
        cycle.close()
    reopened = wave_instance.open_cycle('parallel')
    try:
        reopened.step()
        retained = reopened.waves._pool(reopened.scheduler.state['wave_history'][0]).status()['projects']
        assert {op: row['worker']['pid'] for op, row in retained.items()} == pids
        assert len(reopened.scheduler.state['originals']) == 2
    finally:
        reopened.close()


def test_closed_failed_attempts_remain_failures_in_memory(wave_instance):
    wave_instance.test = replace(wave_instance.test, session_factory=FailedDecisions)
    cycle = wave_instance.create_cycle('failed', 'phase1')
    try:
        assert cycle.step().outcome == 'failed'
        assert cycle.step().outcome == 'failed'
        assert not cycle.memory.records()
        cycle.step()
        assert {r['proposal']['kind'] for r in cycle.memory.records()} == {'failure'}
        assert [r['outcome'] for r in cycle.scheduler.state['originals']] == ['failed', 'failed']
        inspection = wave_instance.status('failed')['waves'][0]
        assert {p['outcome'] for p in inspection['projects']} == {'failed'}
    finally:
        cycle.close()


def test_stop_drains_first_child_then_explicit_resume_runs_only_ready_sibling(wave_instance):
    settings = show_configuration(wave_instance.root)['payload']
    settings['project_wave']['max_parallel'] = 1
    settings['project_wave']['wave_size'] = 2
    set_configuration(wave_instance.root, settings)
    cycle = wave_instance.create_cycle('drain', 'phase1')
    def progress(state):
        if any(row['status'] == 'completed' for row in state['projects'].values()):
            cycle.waves.stop()
    cycle.wave_progress = progress
    try:
        with pytest.raises(RuntimeError):
            cycle.step()
        wave = cycle.scheduler.state['active_wave']
        children = cycle.waves._pool(wave).status()['projects']
        completed = next(op for op, row in children.items() if row['status'] == 'completed')
        original_pid = children[completed]['worker']['pid']
        assert sum(row['status'] == 'ready' for row in children.values()) == 1
        assert not cycle.memory.records()
    finally:
        cycle.close()
    reopened = wave_instance.open_cycle('drain')
    try:
        reopened.reconcile(dispatch_ready=True)
        reopened.scheduler.resume()
        assert reopened.step().outcome == 'accepted'
        children = reopened.waves._pool(reopened.scheduler.state['active_wave']).status()['projects']
        assert children[completed]['worker']['pid'] == original_pid
    finally:
        reopened.close()


def test_actual_cli_parser_reaches_two_child_service_and_memory_publication(wave_instance, monkeypatch, capsys):
    import rsi.cli
    import rsi.job_cli
    import rsi.pinned_cli
    import rsi.service
    monkeypatch.setattr(rsi.service, 'InstanceServices', lambda _: wave_instance)
    # The test instance intentionally has no installed provider/runtime. Replace
    # only pinned dispatch and the outer owned-job launcher with TestBindings;
    # actual CLI parsing, handler, service, scheduler and spawned actors execute.
    monkeypatch.setattr(rsi.pinned_cli, 'dispatch', lambda *args: None)
    events = []
    def launch(root, request):
        return rsi.service.run_job(root, request, show_configuration(root),
                                   should_stop=lambda: False, emit=events.append)
    monkeypatch.setattr(rsi.job_cli.jobs, 'start', launch)
    assert rsi.cli.main(['--instance', str(wave_instance.root), 'phase1',
                         'start', '--cycle', 'cli-wave']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'complete'
    assert any(event['event'] == 'wave_observed' for event in events)
    status = wave_instance.status('cli-wave')
    assert len(status['cycle.json']['originals']) == 2
    assert status['cycle.json']['memory'] != status['service.json']['spec']['baseline_kdb']


def test_partial_memory_admission_reconciles_same_operation_without_duplicate_rows(wave_instance, monkeypatch):
    cycle = wave_instance.create_cycle('memory-crash', 'phase1')
    try:
        cycle.step()
        cycle.step()
        accept = cycle.memory.accept
        calls = []
        def interrupted(identity, *args, **kwargs):
            value = accept(identity, *args, **kwargs)
            calls.append(identity)
            if len(calls) == 1:
                raise OSError('crash after durable local memory admission')
            return value
        monkeypatch.setattr(cycle.memory, 'accept', interrupted)
        with pytest.raises(OSError):
            cycle.step()
        operation = cycle.scheduler.state['pending']['id']
        assert len(cycle.memory.records()) == 1
    finally:
        cycle.close()
    reopened = wave_instance.open_cycle('memory-crash')
    try:
        reopened.reconcile()
        assert len(reopened.memory.records()) == 2
        assert operation in reopened.scheduler.state['completed_operations']
        assert len(reopened.learning.state['operations']) == 1
        assert len(reopened.scheduler.state['originals']) == 2
    finally:
        reopened.close()


def test_phase2_keeps_parent_learning_callbacks_after_siblings_settle(wave_instance):
    cycle = wave_instance.create_cycle('phase2-wave', 'phase2')
    try:
        baseline = cycle.scheduler.state['skills']
        assert cycle.step().outcome == 'accepted'
        wave = cycle.scheduler.state['active_wave']
        assert all(row['status'] == 'completed' for row in cycle.waves._pool(wave).status()['projects'].values())
        assert not Roles.prompts and not cycle.memory.records()
        for _ in range(20):
            cycle.step()
            if cycle.scheduler.state['phase'] == 'complete' and cycle.scheduler.state['active_wave'] is None:
                break
        assert cycle.scheduler.state['phase'] == 'complete'
        assert len(cycle.scheduler.state['originals']) == 2
        assert len(Roles.prompts) == 2
        assert len(cycle.memory.records()) == 2
        assert cycle.learning.active == cycle.scheduler.state['skills'] == baseline
    finally:
        cycle.close()
