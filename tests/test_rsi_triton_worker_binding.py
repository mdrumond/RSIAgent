"""CPU-only real-child lease binding; shared wave scheduling has separate tests."""
import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi.contracts import ProjectSpec
from rsi.project_binding import admission_hash, source_identity, validate_resolution
from rsi.python_execution import atomic_json
from rsi.triton_project_config import budgets
from rsi.triton_worker_binding import bind_project, prepare_worker_bindings


def _child(root, admission, metadata, configuration, queue, release, hold=False, unlock=False):
    root = Path(root)
    request = SimpleNamespace(root=root, project_id=admission['spec']['project_id'],
                              device=str(metadata['physical_device']), dispatch_id=root.parent.name,
                              reopen=False)
    atomic_json(root.parent/'worker.json', {'pid': os.getpid(), 'dispatch_id': request.dispatch_id})
    try:
        result = bind_project(request, admission, metadata=metadata, configuration=configuration,
                              remaining=lambda: 30.0)
        if hold and not release.wait(15):
            raise RuntimeError('Third project never acquired the actually freed lease')
        queue.put({'device': request.device, 'root': str(root), 'resolution': result})
    finally:
        if unlock:
            release.set()


def _inputs(tmp_path):
    projects, devices = [], {}
    for device in (0, 1):
        runtime = tmp_path/f'runtime-{device}.json'
        atomic_json(runtime, {'physical_device': device, 'device_health': {'product': 'a3', 'chip': 0},
                              'soc': '910B4-1', 'private_profile': {'not_public': True}})
        metadata = {'target': 'registered-target', 'runtime': 'named-runtime',
                    'physical_device': device, 'executor_identity': f'profile-{device}'}
        devices[str(device)] = {'runtime_manifest': str(runtime), 'executor_identity': f'profile-{device}',
                                 'python': 'python3', 'metadata': metadata}
    for index in range(3):
        source = tmp_path/f'source-{index}'
        source.mkdir()
        (source/'candidate.py').write_text('def candidate():\n    return 1\n')
        (source/'reference-skills').mkdir()
        (source/'reference-skills/reference.txt').write_text('frozen read-only resource')
        cases = tmp_path/f'cases-{index}.json'
        atomic_json(cases, {'project_id': f'project-{index}', 'operation': 'add',
                            'cases': [{'case_id': 'tail', 'shape': [17], 'seed': index}]})
        spec = ProjectSpec(f'project-{index}', 'Preserve the logical workload', ('logical-template',),
                           budgets=budgets(), starting_folder=str(source), correctness_ref=str(cases))
        projects.append({'spec': spec, 'source': str(source), 'cases': str(cases)})
    configuration = prepare_worker_bindings(projects, devices, tmp_path/'bindings.json')
    manifest = json.loads(Path(configuration['manifest']).read_text())
    admissions = []
    for row in projects:
        variant = manifest['projects'][row['spec'].project_id]['variants']['0']
        admissions.append({'operation_id': row['spec'].project_id, 'spec': variant['spec'], 'config': variant['config'],
                           'memory_ref': 'frozen-memory', 'skills_ref': 'frozen-skills',
                           'configuration_ref': 'frozen-configuration'})
    return configuration, admissions, devices


def test_three_projects_two_leases_use_completion_device_not_modulo(tmp_path):
    configuration, admissions, devices = _inputs(tmp_path)
    context = multiprocessing.get_context('spawn')
    queue, release = context.Queue(), context.Event()
    processes = []

    def start(index, device, **options):
        branch = tmp_path/f'worker-{index}'
        branch.mkdir()
        process = context.Process(target=_child, args=(str(branch/'runner'), admissions[index],
            devices[device]['metadata'], configuration, queue, release), kwargs=options)
        process.start()
        processes.append(process)

    try:
        start(0, '0', hold=True)
        start(1, '1')
        first = queue.get(timeout=20)
        assert first['resolution']['spec']['project_id'] == 'project-1'
        # Device 0 is still held; project 2 must use freed device 1, not 2 % 2.
        start(2, first['device'], unlock=True)
        results = [first, queue.get(timeout=20), queue.get(timeout=20)]
        for result in results:
            resolution = result['resolution']
            index = int(resolution['spec']['project_id'].rsplit('-', 1)[1])
            metadata = devices[result['device']]['metadata']
            validate_resolution(admissions[index], resolution, device=result['device'],
                                executor_identity=metadata['executor_identity'])
            spec, config, evidence = resolution['spec'], resolution['config'], resolution['evidence']
            assert spec['command'] == config['commands']['test']
            assert config['commands']['test'][config['commands']['test'].index('--physical-device')+1] == result['device']
            runtime = Path(config['source'])/config['commands']['test'][config['commands']['test'].index('--runtime-manifest')+1]
            runtime_data = json.loads(runtime.read_text())
            assert str(runtime_data['physical_device']) == result['device']
            assert 'private_profile' not in runtime_data
            assert evidence['logical_admission_hash'] == admission_hash(admissions[index])
            assert evidence['logical_case_identity'] == evidence['resolved_case_identity']
            assert evidence['logical_source_identity'] != evidence['resolved_source_identity']
            assert evidence['resolved_source_identity'] == source_identity(config['source'],
                verifier=config['verifier'], generated_paths=config['generated_paths'])
            assert 'private_profile' not in evidence['public_runtime']
            assert not Path(result['root']).exists()
            assert config['editable_files'] == ['candidate.py']
        third = next(row for row in results if row['resolution']['spec']['project_id'] == 'project-2')
        assert third['device'] == '1'
    finally:
        release.set()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
        queue.close()
    assert all(process.exitcode == 0 for process in processes)


def test_resume_cannot_rebuild_lease_inputs(tmp_path):
    atomic_json(tmp_path/'worker.json', {'pid': os.getpid(), 'dispatch_id': 'dispatch'})
    request = SimpleNamespace(root=tmp_path/'runner', dispatch_id='dispatch', reopen=True)
    with pytest.raises(ValueError):
        bind_project(request, {}, metadata={}, configuration={}, remaining=lambda: 20.0)
    assert not (tmp_path/'lease-inputs').exists()
