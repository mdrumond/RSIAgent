"""Mixed admitted project families use their own trusted binder, not the first sibling's."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from rsi.configuration import _snapshot, set_configuration, show_configuration
from rsi.contracts import ProjectSpec
from rsi.executor_profiles import ExecutorProfiles
from rsi.python_execution import atomic_json
from test_rsi_lease_binding import cpu_bind_project
from test_rsi_service_wave import wave_instance
from test_rsi_services import instance
from test_rsi_worker_admission import profiles


def exclusive_bind_project(request, logical_admission, *, metadata, configuration, remaining):
    manifest = json.loads(Path(configuration['manifest']).read_text())
    project = logical_admission['spec']['project_id']
    admitted = manifest['projects'][project]  # Disjoint manifests must never serve a sibling.
    assert admitted['source'] == logical_admission['config']['source']
    assert admitted['command'] == logical_admission['spec']['command']
    return cpu_bind_project(request, logical_admission, metadata=metadata,
        configuration={'coordination': configuration['coordination']}, remaining=remaining)


@pytest.mark.parametrize('order', [
    ('original', 'child-a'), ('child-a', 'original'), ('child-a', 'child-b'),
])
def test_mixed_families_bind_and_reopen_their_own_registered_inputs(wave_instance, tmp_path, order):
    path = tmp_path/'mixed-profiles.json'
    resolver = profiles(path)
    settings = show_configuration(wave_instance.root)['payload']
    template = settings['phase1_projects'][0]
    base_config = settings['project_configs']['answer']
    specs, bindings = {}, {}
    for name in ('original', 'child-a', 'child-b'):
        source = tmp_path/name
        source.mkdir()
        (source/'solution.py').write_text('answer = 42\n')
        spec = dict(template, project_id=name, starting_folder=str(source))
        config = dict(base_config, source=str(source), verifier=str(wave_instance.root/'verify.py'))
        manifest = tmp_path/(name+'-manifest.json')
        atomic_json(manifest, {'projects': {name: dict(source=str(source), command=spec['command'])}})
        bindings[name] = dict(manifest=str(manifest), coordination=str(tmp_path/'calls'))
        specs[name] = spec
        settings['project_configs'][name] = config
    settings['phase1_projects'] = [specs[name] for name in order]
    settings['curriculum'] = dict(enabled=True)
    settings['project_wave'].update(wave_size=2, devices=[
        dict(device=str(i), worker_profile=f'worker-{i}') for i in (0, 1)])
    settings['cycle_limits'].update(wave_size=2, recovery_seconds=100.)
    records = json.loads(path.read_text())
    for record in records['profiles'].values():
        record.update(project_binder='test_rsi_mixed_wave_binding:exclusive_bind_project',
                      project_configuration=bindings['original'])
    atomic_json(path, records)
    set_configuration(wave_instance.root, settings)
    wave_instance.executor_factory = resolver
    cycle = wave_instance.create_cycle('mixed-binding', 'phase1')
    expected_configuration = {}
    try:
        for name in order:
            if name == 'original':
                expected_configuration[name] = cycle.spec.configuration_ref
                continue
            config = settings['project_configs'][name]
            configuration_ref = cycle.store.put_json(_snapshot(dict(cycle.config,
                project_configs={**cycle.config['project_configs'], name: config})))
            cycle.scheduler.state.setdefault('derived_projects', {})[name] = dict(
                config_ref=cycle.store.put_json(config), configuration_ref=configuration_ref,
                provenance_ref=cycle.store.put_json(dict(binding=bindings[name])))
            expected_configuration[name] = configuration_ref
        cycle.scheduler._save()
        assert cycle.step().outcome == 'accepted'
        wave = cycle.scheduler.state['active_wave']
        admission = cycle.store.read_json(wave['admission_ref'])
        rows = cycle.waves._pool(wave).status()['projects']
        retained = {}
        for op in wave['operations']:
            name = op['spec']['project_id']
            assert rows[op['id']]['receipt']['result']['status'] == 'accepted'
            admitted = admission['operations'][op['id']]
            assert admitted['binding']['configuration'] == expected_configuration[name]
            factory = cycle.project_executor_factory(ProjectSpec.from_dict(op['spec']))
            assert admitted['binding_contracts'] == {
                identity: factory.binding_contract(identity) for identity in admission['identities']}
            runner = cycle.open_project(op['id'])
            binding = json.loads((runner.root/'service-binding.json').read_text())
            proof = runner.store.read_json(binding['resolved_binding_ref'])
            assert proof['evidence']['physical_device'] == rows[op['id']]['lease_device']
            assert binding['configuration'] == expected_configuration[name]
            assert (tmp_path/'calls'/(name+'.bindings')).read_text().splitlines() == [rows[op['id']]['lease_device']]
            retained[op['id']] = binding['resolved_binding_ref']
    finally:
        cycle.close()
    # Existing runners retain their exact resolution; opening never invokes a new binder.
    reopened = wave_instance.open_cycle('mixed-binding')
    try:
        for operation, reference in retained.items():
            runner = reopened.open_project(operation)
            assert json.loads((runner.root/'service-binding.json').read_text())['resolved_binding_ref'] == reference
    finally:
        reopened.close()


@pytest.mark.parametrize('next_device', ['0', '1'])
def test_original_retry_rebuilds_actual_triton_manifest_from_latest_root(tmp_path, next_device):
    import os
    from types import SimpleNamespace
    from uuid import uuid4
    from core.artifacts import ContentAddressedArtifacts
    from rsi.phase1_recovery import Phase1Recovery
    from rsi.project_binding import validate_resolution, source_identity
    from rsi.project_runner import ProjectConfig
    from rsi.setup_telemetry import SetupTelemetry
    from rsi.triton_worker_binding import bind_project, case_identity
    from test_rsi_triton_worker_binding import _inputs

    configuration, admissions, devices = _inputs(tmp_path)
    original = ProjectSpec.from_dict(admissions[0]['spec'])  # Logical template uses physical0.
    def request(name, device, operation):
        root = tmp_path/name
        root.mkdir()
        atomic_json(root/'worker.json', dict(pid=os.getpid(), dispatch_id=name))
        return SimpleNamespace(root=root/'runner', project_id=operation,
            dispatch_id=name, device=device, reopen=False)
    initial = bind_project(request('first', '1', admissions[0]['operation_id']), admissions[0],
        metadata=devices['1']['metadata'], configuration=configuration, remaining=lambda: 30.)
    assert initial['spec']['command'] != original.to_dict()['command']
    previous = ProjectConfig.from_dict(initial['config'])
    latest = Path(previous.source)/'candidate.py'
    latest.write_text('def candidate():\n    return 2  # actor-edited ORIGINAL, not a diagnostic child\n')
    original_cases = case_identity(previous)
    profiles_path = tmp_path/'triton-profiles.json'
    atomic_json(profiles_path, dict(profiles={options['executor_identity']: dict(
        metadata=options['metadata'], guard='test_rsi_worker_admission:scripted_guard',
        project_binder='rsi.triton_worker_binding:bind_project', project_configuration=configuration)
        for options in devices.values()}))
    store = ContentAddressedArtifacts(tmp_path/'cycle-artifacts')
    services = SimpleNamespace(executor_factory=ExecutorProfiles(str(profiles_path)),
        curriculum_researcher=None, curriculum_materializer=None, setup=SetupTelemetry(store))
    cycle = SimpleNamespace(root=tmp_path/'cycle', services=services, store=store,
        config=dict(project_configs={original.project_id: asdict(previous)}),
        scheduler=SimpleNamespace(state={'memory': 'retained-child-memory'}, _save=lambda: None))
    recovery = Phase1Recovery(cycle)
    # A failed original's packet resolves to its actually leased source/config.
    recovery.parent = lambda row: (ProjectSpec.from_dict(initial['spec']),
        dict(config=asdict(previous), binding=configuration))
    operation = uuid4().hex
    retry = recovery.original_retry(original, {}, {'operation_id': operation})
    assert retry.project_id == original.project_id and retry.goal == original.goal
    assert retry.budgets == original.budgets
    entry = cycle.scheduler.state['original_retry_inputs'][original.project_id]
    retry_config = ProjectConfig.from_dict(store.read_json(entry['config_ref']))
    assert case_identity(retry_config) == original_cases
    assert (Path(retry_config.source)/'candidate.py').read_bytes() == latest.read_bytes()
    assert entry['binding'] != configuration
    table = json.loads(Path(entry['binding']['manifest']).read_text())['projects'][original.project_id]
    assert set(table['variants']) == {'0', '1'}
    assert table['logical_source_identity'] == source_identity(retry_config.source,
        verifier=retry_config.verifier, generated_paths=retry_config.generated_paths)
    admission = dict(admissions[0], operation_id=operation, spec=retry.to_dict(), config=asdict(retry_config))
    resolver = recovery.executor_factory(retry)
    resolved = resolver.bind_project(devices[next_device]['executor_identity'],
        request('retry-'+next_device, next_device, operation), admission, remaining=lambda: 30.)
    validate_resolution(admission, resolved, device=next_device,
                        executor_identity=devices[next_device]['executor_identity'])
    assert case_identity(ProjectConfig.from_dict(resolved['config'])) == original_cases
    assert Path(resolved['config']['source'], 'candidate.py').read_bytes() == latest.read_bytes()
    assert recovery.original_retry(original, {}, {'operation_id': operation}) == retry
