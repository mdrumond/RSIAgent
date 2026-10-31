"""Host variant construction and post-lease binding; no dispatch or device access.

The canonical wave leases devices. This hook only selects a fully frozen variant
for that actual lease before ProjectRunner captures its source/command contract.
"""
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

from core.artifacts import canonical_bytes

from rsi.contracts import ProjectSpec
from rsi.project_runner import ProjectConfig
from rsi.project_binding import admission_hash, source_identity, validate_resolution, PUBLIC_RUNTIME
from rsi.python_execution import atomic_json
from rsi.triton_project_config import project_config, trusted_resources


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def logical(spec):
    value = spec.to_dict() if isinstance(spec, ProjectSpec) else ProjectSpec.from_dict(spec).to_dict()
    value.pop('command')
    return value


def case_identity(config):
    """Hash workload/cases/oracle bytes, never physical argv or machine paths."""
    argv = config.commands['test']
    path = Path(argv[argv.index('--cases')+1])
    path = path if path.is_absolute() else Path(config.source)/path
    if digest(path) != argv[argv.index('--cases-sha256')+1]:
        raise ValueError('Frozen correctness input changed')
    contract = json.loads(path.read_text())
    # Reuse the shared canonical hashing helper on a distinct case payload,
    # not on logical_admission; these two identities have different semantics.
    return admission_hash({'operation': contract['operation'], 'cases': contract['cases'],
                           'oracle_sha256': digest(config.verifier)})


def initial_source(config):
    return source_identity(config.source, verifier=config.verifier,
                           generated_paths=config.generated_paths)


def prepare_worker_bindings(projects, devices, destination):
    """Pre-generate approved variants; the manifest is host-owned operational data.

    projects: rows containing spec, source, cases and optional exercise_profile.
    devices: physical-ID mapping to runtime_manifest, executor_identity, python,
    and optional exercise_profile/health_guard. IDs are actual physical IDs, not
    project ordinals. No variant is selected or device permission granted here.
    """
    if not projects or not devices:
        raise ValueError('Logical projects and concrete approved device profiles are required')
    manifest = {'schema_version': 1, 'projects': {}}
    for row in projects:
        template = row['spec'] if isinstance(row['spec'], ProjectSpec) else ProjectSpec.from_dict(row['spec'])
        source = Path(row['source']).resolve()
        template = replace(template, starting_folder=str(source))
        variants = {}
        for device, options in devices.items():
            physical = str(int(device))
            metadata = options['metadata']
            if (str(metadata['physical_device']) != physical
                    or metadata.get('executor_identity') != options['executor_identity']):
                raise ValueError('Device profile metadata differs from its concrete variant')
            config = project_config(source, row['cases'], options['runtime_manifest'],
                executor_identity=options['executor_identity'], python=options['python'],
                physical_device=int(physical), health_guard=options.get('health_guard'),
                exercise_profile=row.get('exercise_profile', options.get('exercise_profile')))
            spec = replace(template, command=tuple(config.commands['test']))
            resources = [{'path': str(path), 'sha256': digest(path)} for path in trusted_resources(config)]
            runtime = Path(options['runtime_manifest']).resolve()
            variants[physical] = {'spec': spec.to_dict(), 'config': asdict(config),
                'metadata': metadata,
                'physical_device': physical, 'executor_identity': config.executor_identity,
                'runtime_manifest': str(runtime), 'runtime_sha256': digest(runtime),
                'trusted_resources': resources}
        if template.project_id in manifest['projects']:
            raise ValueError('Duplicate logical project')
        manifest['projects'][template.project_id] = {'logical': logical(template),
            'logical_source_identity': initial_source(config), 'logical_case_identity': case_identity(config),
            'variants': variants}
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError('Worker-binding manifests are immutable; choose a new revision path')
    atomic_json(destination, manifest)
    return {'manifest': str(destination), 'sha256': digest(destination)}


def profile_fields(configuration):
    """Ignored ExecutorProfiles fields referencing this shipped target resolver."""
    return {'project_binder': 'rsi.triton_worker_binding:bind_project',
            'project_configuration': {'manifest': configuration['manifest'], 'sha256': configuration['sha256']}}


def bind_project(request, logical_admission, *, metadata, configuration, remaining):
    """Resolve only in the owning child before Runner.create; never on resume.

    Write only request.root.parent/lease-inputs. Runtime/cases/producers become
    read-only execution inputs; the verifier remains external. Generic CPL can
    stage the selected project without mutable per-worker transport profiles.
    """
    def check_remaining():
        value = remaining()
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError('Lease binding exhausted remaining elapsed allocation')

    check_remaining()
    root = Path(request.root)
    owner = json.loads((root.parent/'worker.json').read_text())
    if (owner['pid'] != os.getpid() or owner['dispatch_id'] != request.dispatch_id
            or getattr(request, 'reopen', False)):
        raise ValueError('Lease binding requires the owning initial worker child')
    path = Path(configuration['manifest'])
    if digest(path) != configuration['sha256']:
        raise ValueError('Frozen worker-binding manifest changed')
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported worker-binding manifest')
    original = ProjectSpec.from_dict(logical_admission['spec'])
    if request.project_id != original.project_id:
        raise ValueError('Lease request and logical project differ')
    device = str(request.device)
    if str(metadata['physical_device']) != device:
        raise ValueError('Profile metadata differs from the actual physical lease')
    project = manifest['projects'][request.project_id]
    if logical(original) != project['logical']:
        raise ValueError('Binding would change the frozen logical project')
    variant = project['variants'][device]
    for key in ('target', 'runtime', 'physical_device', 'executor_identity'):
        expected, actual = variant['metadata'][key], metadata[key]
        matches = str(expected) == str(actual) if key == 'physical_device' else expected == actual
        if not matches:
            raise ValueError('Resolved variant differs from leased target/runtime metadata')
    resolved = ProjectSpec.from_dict(variant['spec'])
    resolved_config = ProjectConfig.from_dict(variant['config'])
    if (logical(resolved) != project['logical'] or resolved.command != tuple(resolved_config.commands['test'])
            or resolved_config.executor_identity != metadata.get('executor_identity')
            or variant['physical_device'] != device):
        raise ValueError('Resolved variant does not match the frozen project and lease')
    before = ProjectConfig.from_dict(logical_admission['config'])
    for field in ('source', 'verifier', 'editable_files', 'generated_paths', 'retained_paths', 'performance_seconds'):
        if getattr(before, field) != getattr(resolved_config, field):
            raise ValueError('Device variant changes non-device project configuration')
    if initial_source(before) != project['logical_source_identity']:
        raise ValueError('Accepted source/resources changed before Runner freeze')
    if case_identity(before) != project['logical_case_identity']:
        raise ValueError('Accepted logical cases or oracle changed before lease')
    for resource in variant['trusted_resources']:
        if digest(resource['path']) != resource['sha256']:
            raise ValueError('Resolved trusted resource changed')
        check_remaining()
    runtime = json.loads(Path(variant['runtime_manifest']).read_text())
    if str(runtime['physical_device']) != device or digest(variant['runtime_manifest']) != variant['runtime_sha256']:
        raise ValueError('Runtime manifest differs from the actual leased physical device')
    lease_base = root.parent/'lease-inputs'
    lease = lease_base/request.dispatch_id
    source = Path(resolved_config.source)
    if lease.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(lease.resolve()):
        raise ValueError('Lease inputs must be outside the accepted source tree')
    lease_base.mkdir(exist_ok=True)
    lease.mkdir()

    def ignore(directory, names):
        selected = []
        for name in names:
            relative = (Path(directory)/name).relative_to(source).as_posix()
            if (name in ('.git', '__pycache__', '.pytest_cache')
                    or (Path(directory)/name).resolve() == Path(resolved_config.verifier).resolve()
                    or any(relative == p or relative.startswith(p.rstrip('/')+'/')
                           for p in resolved_config.generated_paths)):
                selected.append(name)
        return selected

    shutil.copytree(source, lease/'project', ignore=ignore)
    if source_identity(lease/'project', verifier=resolved_config.verifier,
                       generated_paths=resolved_config.generated_paths) != project['logical_source_identity']:
        raise ValueError('Accepted source/resources changed while creating lease inputs')
    check_remaining()
    inputs = lease/'project'/'.execution-inputs'
    inputs.mkdir()
    mapping = {}
    runtime_hash = None
    for resource in variant['trusted_resources']:
        origin = Path(resource['path'])
        data = origin.read_bytes()
        if hashlib.sha256(data).hexdigest() != resource['sha256']:
            raise ValueError('Trusted lease input changed during copy')
        if origin == Path(variant['runtime_manifest']):
            allowed = PUBLIC_RUNTIME | {'expected_versions', 'expected_compiler_target', 'device_health',
                                        'compiler_consumer_files', 'cann_version', 'cann_metadata'}
            data = canonical_bytes({key: value for key, value in runtime.items() if key in allowed})
            runtime_hash = hashlib.sha256(data).hexdigest()
        destination = inputs/(hashlib.sha256(data).hexdigest()+origin.suffix)
        destination.write_bytes(data)
        mapping[str(origin)] = destination.relative_to(lease/'project').as_posix()
        check_remaining()
    shutil.copyfile(resolved_config.verifier, lease/'verify.py')
    if digest(lease/'verify.py') != digest(before.verifier):
        raise ValueError('Resolved verifier changed the logical oracle')

    def remap(value):
        if value in mapping:
            return mapping[value]
        if '=' in value:
            key, argument = value.split('=', 1)
            if argument in mapping:
                return key+'='+mapping[argument]
        return value

    commands = {}
    for key, argv in resolved_config.commands.items():
        values = [remap(value) for value in argv]
        values[values.index('--runtime-sha256')+1] = runtime_hash
        commands[key] = values
    resolved_config = replace(resolved_config, source=str(lease/'project'), verifier=str(lease/'verify.py'),
                              commands=commands)
    resolved = replace(resolved, starting_folder=resolved_config.source,
                       command=tuple(resolved_config.commands['test']))
    resolved_case = case_identity(resolved_config)
    if resolved_case != project['logical_case_identity']:
        raise ValueError('Physical lease resolution changed workload/cases/oracle')
    public = {key: value for key, value in {**metadata, **runtime}.items()
              if key in PUBLIC_RUNTIME and (value is None or isinstance(value, (str, int, float, bool)))}
    public.update(physical_device=int(device), logical_device=0)
    evidence = {'logical_admission_hash': admission_hash(logical_admission),
                'logical_project_id': request.project_id, 'physical_device': device,
                'logical_case_identity': project['logical_case_identity'],
                'resolved_case_identity': resolved_case,
                'logical_source_identity': initial_source(before),
                'resolved_source_identity': initial_source(resolved_config),
                'public_runtime': public}
    check_remaining()
    return validate_resolution(logical_admission,
        {'spec': resolved.to_dict(), 'config': asdict(resolved_config), 'evidence': evidence},
        device=device, executor_identity=metadata.get('executor_identity'))
