"""Trusted lease resolution and public context, independent of target imports."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

from core.artifacts import canonical_bytes


def admission_hash(logical_admission):
    return hashlib.sha256(canonical_bytes(logical_admission)).hexdigest()


def source_identity(source, *, verifier, generated_paths=()):
    """Initial snapshot digest using PythonExecution.project_hash's file contract."""
    source, verifier = Path(source).resolve(), Path(verifier).resolve()
    entries = []
    for path in sorted(source.rglob('*')):
        name = path.relative_to(source).as_posix()
        if '.git' in path.relative_to(source).parts:
            continue
        if path.is_symlink():
            raise ValueError('lease inputs must not contain symlinks')
        if (path.resolve() == verifier or any(p in ('__pycache__', '.pytest_cache') for p in Path(name).parts)
                or any(name == p or name.startswith(p.rstrip('/') + '/') for p in generated_paths)):
            continue
        if path.is_file():
            entries.append([name, hashlib.sha256(path.read_bytes()).hexdigest()])
    return hashlib.sha256(json.dumps(sorted(entries), separators=(',', ':')).encode()).hexdigest()


def validate_resolution(logical, resolution, *, device, executor_identity):
    """Authenticate logical scope and invariants; only host binding fields vary."""
    from rsi.contracts import ProjectSpec
    from rsi.project_runner import ProjectConfig
    result = json.loads(canonical_bytes(resolution))
    spec = ProjectSpec.from_dict(result['spec'])
    config = ProjectConfig.from_dict(result['config'])
    evidence = result['evidence']
    if (evidence['logical_admission_hash'] != admission_hash(logical)
            or evidence['logical_project_id'] != logical['spec']['project_id']
            or str(evidence['physical_device']) != str(device)
            or evidence['logical_case_identity'] != evidence['resolved_case_identity']):
        raise ValueError('resolved binding differs from logical project/cases or actual lease')
    for field in ('logical_case_identity', 'resolved_case_identity',
                  'logical_source_identity', 'resolved_source_identity'):
        if not isinstance(evidence[field], str) or not re.fullmatch(r'[0-9a-f]{64}', evidence[field]):
            raise ValueError('host binding requires exact case and source identities')
    wire = spec.to_dict()
    if any(wire[k] != v for k, v in logical['spec'].items()
           if k not in ('command', 'starting_folder', 'correctness_ref')):
        raise ValueError('lease binding changed logical goal, project or frozen budget')
    frozen = asdict(ProjectConfig.from_dict(logical['config']))
    current = asdict(config)
    if any(current[k] != v for k, v in frozen.items()
           if k not in ('source', 'verifier', 'commands', 'executor_identity')):
        raise ValueError('lease binding changed editable, generated or retained input policy')
    if (set(current['commands']) != set(frozen['commands'])
            or tuple(config.commands['test']) != spec.command
            or config.executor_identity != executor_identity
            or spec.starting_folder and spec.starting_folder != config.source):
        raise ValueError('resolved commands.test, source or executor contract differs')
    result.update(spec=wire, config=json.loads(canonical_bytes(current)))
    return result


PUBLIC_RUNTIME = {'target', 'runtime', 'soc', 'soc_version', 'cann_version', 'source_revision',
                  'dependency_revision', 'physical_device', 'logical_device', 'outer_handle',
                  'controller_placement'}
PUBLIC_MEASUREMENTS = {'health', 'degraded', 'hardware_error_code', 'fault_event_count',
                       'memory_percent', 'memory_capacity_mb', 'utilization_percent', 'idle'}


def attach_execution_context(runner, binding, resolution, admission=None):
    """Extend the canonical context wrapper with actual, explicitly public facts."""
    context = runner.context
    metadata = binding.get('executor_provenance') or {}
    evidence = resolution['evidence'] if resolution else {}
    declared = evidence.get('public_runtime', {})
    facts = {k: v for k, v in {**metadata, **declared}.items()
             if k in PUBLIC_RUNTIME and (v is None or isinstance(v, (str, int, float, bool)))}
    if evidence:
        facts['physical_device'] = evidence['physical_device']
        facts['case_identity'] = evidence['resolved_case_identity']
        facts['source_identity'] = evidence['resolved_source_identity']
    def wrapped():
        guard = admission
        last = runner.latest('command-results')
        for row in last.get('receipts', []) if last else ():
            guard = row.get('guard_receipt', guard)
            for observation in row.get('observations', []):
                guard = observation.get('guard_receipt', guard)
        public = dict(facts)
        if guard:
            public['admission'] = {k: guard[k] for k in ('admitted', 'degraded', 'physical_device', 'observation')
                                   if k in guard}
            public['admission']['measurements'] = {
                k: v for k, v in guard.get('measurements', {}).items() if k in PUBLIC_MEASUREMENTS}
        return dict(context(), execution_context=public)
    runner.context = wrapped
