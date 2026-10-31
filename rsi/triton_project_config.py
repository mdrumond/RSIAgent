"""Host-owned Triton command configuration for the canonical ProjectRunner.

The common rsi.cpl_execution executor owns all remote transport. This module
freezes standalone verifier argv and host input hashes; it never dispatches work.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rsi.contracts import BudgetLimits, ProjectSpec
from rsi.device_health_probe import standalone_hash, standalone_source
from rsi.project_runner import ProjectConfig


ACTOR_SECONDS = 7200.0
HARNESS_RECOVERY_SECONDS = 7200.0


def budgets():
    """Recovery is a separate once-only host grant, never an automatic reset."""
    return BudgetLimits(design_iterations=None, implementation_iterations=None,
                        execution_tries=None, actor_seconds=ACTOR_SECONDS, wall_seconds=None)


def validate_fixture_closure(source, contract):
    """Frozen case fixtures travel inside the snapshotted read-only project."""
    for field in ('compiler_failure', 'ascendc_reference', 'feature_probe', 'historical_environment'):
        fixture = contract.get(field)
        if fixture is None:
            continue
        name = fixture.get('project_path')
        if not isinstance(name, str) or not name:
            raise ValueError('Frozen fixture requires a project-relative path: ' + field)
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Frozen fixture must be portable and project-relative: ' + field)
        path = (source / relative).resolve()
        if (not path.is_relative_to(source) or not path.is_file() or relative == Path('candidate.py')
                or relative.parts[0] in ('outputs', 'caches')):
            raise ValueError('Frozen fixture must be an immutable source input: ' + field)
        if hashlib.sha256(path.read_bytes()).hexdigest() != fixture.get('sha256'):
            raise ValueError('Frozen fixture content differs from its declared hash: ' + field)


def project_config(source, cases, runtime_manifest, *, executor_identity, python, physical_device,
                   health_guard=None, exercise_profile=None):
    """Build executable compile/test/validation commands against a frozen verifier.

    The caller includes trusted_resources(config) in CplConfig.trusted_resources.
    The exercise profile freezes tool commands, not candidate attestations.
    Only candidate.py is actor editable.
    Python is the selected registered runtime's interpreter, supplied by the host.
    """
    source, cases, runtime_manifest = (Path(p).resolve() for p in (source, cases, runtime_manifest))
    if not source.is_dir() or not (source/'candidate.py').is_file():
        raise ValueError('Project source requires candidate.py')
    if cases.is_relative_to(source) or runtime_manifest.is_relative_to(source):
        raise ValueError('Frozen cases and runtime identity must be outside the actor project')
    contract = json.loads(cases.read_text())
    validate_fixture_closure(source, contract)
    identity = json.loads(runtime_manifest.read_text())
    if not contract.get('cases') or str(identity['physical_device']) != str(physical_device):
        raise ValueError('Cases or runtime identity do not match the binding')
    if not executor_identity or not python:
        raise ValueError('Frozen executor identity and target interpreter are required')
    probe_source = standalone_source().encode('utf-8')
    probe_hash = standalone_hash()
    if hashlib.sha256(probe_source).hexdigest() != probe_hash:
        raise ValueError('Shared health source changed while building frozen inputs')
    health_guard = Path(health_guard).resolve() if health_guard else cases.parent/'health-probes'/(probe_hash+'.py')
    if health_guard.is_relative_to(source) or not isinstance(identity.get('device_health'), dict):
        raise ValueError('Frozen common health probe and runtime device-health configuration are required')
    health_guard.parent.mkdir(parents=True, exist_ok=True)
    if health_guard.exists() and health_guard.read_bytes() != probe_source:
        raise ValueError('Existing health resource differs from the current shared export')
    if not health_guard.exists():
        with health_guard.open('xb') as stream:
            stream.write(probe_source)
    case_hash = hashlib.sha256(cases.read_bytes()).hexdigest()
    runtime_hash = hashlib.sha256(runtime_manifest.read_bytes()).hexdigest()
    common = ['--candidate', 'candidate.py', '--cases', str(cases), '--cases-sha256', case_hash,
              '--runtime-manifest', str(runtime_manifest), '--runtime-sha256', runtime_hash,
              '--physical-device', str(physical_device), '--artifacts', 'outputs/triton',
              '--cache', 'caches/triton', '--health-guard', str(health_guard),
              '--health-guard-sha256', probe_hash]
    commands = {mode: [python, '{verifier}', '--mode', mode, *common]
                for mode in ('compile', 'test', 'validation')}
    producer = Path(__file__).with_name('triton_exercises.py')
    commands['validation'] += ['--exercise-producer', str(producer),
                               '--exercise-producer-sha256', hashlib.sha256(producer.read_bytes()).hexdigest()]
    if exercise_profile is not None:
        exercise_profile = Path(exercise_profile).resolve()
        if exercise_profile.is_relative_to(source):
            raise ValueError('Exercise command profile must be outside the actor project')
        commands['validation'] += ['--exercise-profile', str(exercise_profile),
                                   '--exercise-profile-sha256', hashlib.sha256(exercise_profile.read_bytes()).hexdigest()]
    return ProjectConfig(source=str(source), verifier=str(Path(__file__).with_name('triton_verifier.py')),
                         commands=commands, editable_files=('candidate.py',),
                         generated_paths=('outputs', 'caches'), retained_paths=('outputs',),
                         executor_identity=executor_identity)


def project_spec(project_id, goal, config, *, correctness_ref):
    """Freeze an executable ProjectSpec after the host has resolved runtime inputs."""
    return ProjectSpec(project_id=project_id, goal=goal, command=tuple(config.commands['test']),
                       budgets=budgets(), starting_folder=config.source,
                       correctness_ref=correctness_ref)


def trusted_resources(config):
    """Files required by the shared transport, including the standalone producer."""
    flags = {'--cases', '--runtime-manifest', '--exercise-producer', '--exercise-profile', '--health-guard'}
    def resolve(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (Path(config.source)/path).resolve()
    return tuple(sorted({resolve(argv[i+1]) for argv in config.commands.values()
                         for i, token in enumerate(argv[:-1]) if token in flags}))
