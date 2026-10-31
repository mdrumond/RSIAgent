"""Host-owned Triton command configuration for the canonical ProjectRunner.

The common rsi.cpl_execution executor owns all remote transport. This module
freezes standalone verifier argv and host input hashes; it never dispatches work.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rsi.contracts import BudgetLimits, ProjectSpec
from rsi.project_runner import ProjectConfig


ACTOR_SECONDS = 7200.0
HARNESS_RECOVERY_SECONDS = 7200.0


def budgets():
    """Recovery is a separate once-only host grant, never an automatic reset."""
    return BudgetLimits(design_iterations=None, implementation_iterations=None,
                        execution_tries=None, actor_seconds=ACTOR_SECONDS, wall_seconds=None)


def project_config(source, cases, runtime_manifest, *, executor_identity, python, physical_device,
                   exercise_evidence=None):
    """Build executable compile/test/validation commands against a frozen verifier.

    The caller includes the immutable cases and runtime_manifest files in the
    common CplConfig.trusted_resources. Only candidate.py is actor editable.
    Python is the selected registered runtime's interpreter, supplied by the host.
    """
    source, cases, runtime_manifest = (Path(p).resolve() for p in (source, cases, runtime_manifest))
    if not source.is_dir() or not (source/'candidate.py').is_file():
        raise ValueError('Project source requires candidate.py')
    if cases.is_relative_to(source) or runtime_manifest.is_relative_to(source):
        raise ValueError('Frozen cases and runtime identity must be outside the actor project')
    contract = json.loads(cases.read_text())
    identity = json.loads(runtime_manifest.read_text())
    if not contract.get('cases') or str(identity['physical_device']) != str(physical_device):
        raise ValueError('Cases or runtime identity do not match the binding')
    if not executor_identity or not python:
        raise ValueError('Frozen executor identity and target interpreter are required')
    case_hash = hashlib.sha256(cases.read_bytes()).hexdigest()
    runtime_hash = hashlib.sha256(runtime_manifest.read_bytes()).hexdigest()
    common = ['--candidate', 'candidate.py', '--cases', str(cases), '--cases-sha256', case_hash,
              '--runtime-manifest', str(runtime_manifest), '--runtime-sha256', runtime_hash,
              '--physical-device', str(physical_device), '--artifacts', 'artifacts/triton']
    commands = {mode: [python, '{verifier}', '--mode', mode, *common]
                for mode in ('compile', 'test', 'validation')}
    if exercise_evidence is not None:
        exercise_evidence = Path(exercise_evidence).resolve()
        if exercise_evidence.is_relative_to(source):
            raise ValueError('Exercise evidence must be outside the actor project')
        commands['validation'] += ['--exercise-evidence', str(exercise_evidence),
                                   '--exercise-evidence-sha256', hashlib.sha256(exercise_evidence.read_bytes()).hexdigest()]
    return ProjectConfig(source=str(source), verifier=str(Path(__file__).with_name('triton_verifier.py')),
                         commands=commands, editable_files=('candidate.py',),
                         generated_paths=('artifacts/',), executor_identity=executor_identity)


def executor_factory(bindings):
    """Bind opaque identities to the one shared CPL implementation, without dispatch.

    bindings maps host-approved identity strings to common CplConfig objects.
    Supply this factory both when creating and reopening a ProjectRunner snapshot.
    """
    frozen = dict(bindings)

    def create(identity):
        from rsi.cpl_execution import CplExecutor
        try:
            config = frozen[identity]
        except KeyError as exc:
            raise ValueError('Unknown trusted Triton executor identity') from exc
        return CplExecutor(identity, config)

    return create


def project_spec(project_id, goal, config, *, correctness_ref):
    """Freeze an executable ProjectSpec after the host has resolved runtime inputs."""
    return ProjectSpec(project_id=project_id, goal=goal, command=tuple(config.commands['test']),
                       budgets=budgets(), starting_folder=config.source,
                       correctness_ref=correctness_ref)
