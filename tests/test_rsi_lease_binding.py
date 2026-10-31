"""Three real CPU projects use actual leases, not authored ordinal/modulo slots."""
import hashlib
import json
from pathlib import Path
import shutil
import sys


from rsi.configuration import set_configuration, show_configuration
from rsi.project_binding import admission_hash, source_identity
from test_rsi_services import instance
from test_rsi_service_wave import wave_instance
from test_rsi_worker_admission import profiles


def cpu_bind_project(request, logical_admission, *, metadata, configuration, remaining):
    assert remaining() > 0 and not request.root.exists()
    source = request.root.parent / 'lease-inputs'
    original = logical_admission['config']
    shutil.copytree(original['source'], source)
    coordination = Path(configuration['coordination'])
    coordination.mkdir(exist_ok=True)
    project = logical_admission['spec']['project_id']
    calls = coordination / (project + '.bindings')
    with calls.open('a') as stream:
        stream.write(request.device + '\n')
    (coordination / (project + '.bound')).write_text(request.device)
    verifier = request.root.parent / 'lease-verifier.py'
    # The first project's verifier cannot finish until the third is bound. The
    # second must release device1 first, forcing third onto1 (not ordinal2%2=0).
    verifier.write_text('import json, os, runpy, sys, time\n'
        + f'expected = {request.device!r}\n'
        + "assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == expected\n"
        + "assert sys.argv[1:] == ['--leased-device', expected]\n"
        + "assert runpy.run_path('solution.py')['answer'] == 42\n"
        + (f"deadline=time.monotonic()+10\nwhile not __import__('pathlib').Path({str(coordination / 'third.bound')!r}).exists():\n"
           "    assert time.monotonic() < deadline, 'third project was not dispatched'\n    time.sleep(.01)\n"
           if project == 'answer' else ''))
    command = [sys.executable, '{verifier}', '--leased-device', request.device]
    config = dict(original, source=str(source), verifier=str(verifier),
                  commands=dict(original['commands'], test=command, validation=command),
                  executor_identity=metadata['executor_identity'])
    spec = dict(logical_admission['spec'], command=command, starting_folder=str(source),
                correctness_ref=str(verifier))
    cases = hashlib.sha256(Path(original['verifier']).read_bytes()).hexdigest()
    return dict(spec=spec, config=config, evidence=dict(
        logical_admission_hash=admission_hash(logical_admission), logical_project_id=project,
        physical_device=request.device, logical_case_identity=cases, resolved_case_identity=cases,
        logical_source_identity=source_identity(original['source'], verifier=original['verifier'],
                                               generated_paths=original.get('generated_paths', ())),
        resolved_source_identity=source_identity(source, verifier=verifier,
                                                generated_paths=config.get('generated_paths', ())),
        public_runtime=dict(soc='CPU-test', runtime=metadata['runtime'],
                            private_configuration={'not_public': True})))


def test_three_projects_bind_actual_non_modulo_lease_and_reopen_exact_binding(wave_instance, tmp_path):
    path = tmp_path / 'profiles.json'
    resolver = profiles(path)
    values = json.loads(path.read_text())
    coordination = tmp_path / 'binding-order'
    for profile in values['profiles'].values():
        profile['project_binder'] = 'test_rsi_lease_binding:cpu_bind_project'
        profile['project_configuration'] = {'coordination': str(coordination)}
    path.write_text(json.dumps(values))
    settings = show_configuration(wave_instance.root)['payload']
    settings['phase1_projects'].append(dict(settings['phase1_projects'][0], project_id='third'))
    for spec in settings['phase1_projects']:
        spec['budgets']['wall_seconds'] = 20.
    settings['project_configs']['third'] = dict(settings['project_configs']['answer'])
    settings['project_wave'].update(wave_size=3, devices=[
        dict(device=str(i), worker_profile=f'worker-{i}') for i in (0, 1)])
    settings['cycle_limits'].update(wave_size=3)
    set_configuration(wave_instance.root, settings)
    wave_instance.executor_factory = resolver
    cycle = wave_instance.create_cycle('lease-bound', 'phase1')
    try:
        assert cycle.step().outcome == 'accepted'
        wave = cycle.scheduler.state['active_wave']
        operations = {op['original_id']: op['id'] for op in wave['operations']}
        assert (coordination / 'third.bound').read_text() == '1'
        rows = cycle.waves._pool(wave).status()['projects']
        assert rows[operations['third']]['lease_device'] == '1'
        assert {row['receipt']['result']['status'] for row in rows.values()} == {'accepted'}
        before = {}
        for op in wave['operations']:
            runner = cycle.open_project(op['id'])
            binding = json.loads((runner.root / 'service-binding.json').read_text())
            resolved = runner.store.read_json(binding['resolved_binding_ref'])
            assert runner.spec.command == tuple(runner.config.commands['test'])
            assert runner.spec.command != tuple(op['spec']['command'])
            assert resolved['evidence']['logical_case_identity'] == resolved['evidence']['resolved_case_identity']
            context = runner.context()['execution_context']
            assert str(context['physical_device']) == rows[op['id']]['lease_device']
            assert context['soc'] == 'CPU-test' and 'private_configuration' not in context
            before[op['id']] = (binding['resolved_binding_ref'], runner.backend.project_hash())
        assert cycle.step().outcome == 'accepted'
        assert cycle.step().outcome == 'accepted'
        cycle.step()  # Parent memory validates logical/case proof, not physical argv equality.
        assert len(cycle.memory.records()) == 3
        while cycle.scheduler.state['phase'] != 'complete':
            cycle.step()
    finally:
        cycle.close()
    reopened = wave_instance.open_cycle('lease-bound')
    try:
        reopened.step()
        for operation, (reference, source_hash) in before.items():
            runner = reopened.open_project(operation)
            assert json.loads((runner.root / 'service-binding.json').read_text())['resolved_binding_ref'] == reference
            assert runner.backend.project_hash() == source_hash
        assert all(len((coordination / (p + '.bindings')).read_text().splitlines()) == 1
                   for p in ('answer', 'other', 'third'))
    finally:
        reopened.close()
