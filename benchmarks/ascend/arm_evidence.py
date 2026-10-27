"""Portable partial arm evidence and overlap-aware elapsed telemetry."""
from collections import defaultdict
import json
import math
from pathlib import Path
import re

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from .subject_execution import SubjectOutcome
from benchmarks.a5kernels.phase1_project_bz import ProjectInfrastructureError
from benchmarks.a5kernels.phase1_provider import ProviderRejectionError
from coding.pi.rpc import PiCancelled, PiProviderError


def interruption_reason(error):
    """Prefer structural transport/stop metadata, including wrapped causes."""
    chain = []
    current = error
    while current is not None:
        chain.append(current)
        current = current.__cause__
    # A wrapping exception must not erase the authoritative stop provenance.
    for current in chain:
        reason = getattr(current, 'interruption_reason', None)
        if reason:
            return reason
    if any(isinstance(current, PiCancelled) for current in chain):
        return 'cancelled'
    if any(isinstance(current, (PiProviderError, ProviderRejectionError)) for current in chain):
        return 'provider-failure'
    if any(isinstance(current, TimeoutError) for current in chain):
        return 'deadline-exhausted'
    return ('transport-unresolved' if any(isinstance(current, (OSError, ProjectInfrastructureError))
            for current in chain) else 'harness-error')


def arm_evidence(directory, store, *, project_root=None, binding_root=None, project_id=None):
    """Read only owned evidence paths; never dispatch or infer job completion."""
    directory = Path(directory)
    refs, source, missing, spans = {}, None, [], []
    provider_seconds = 0.0
    project = Path(project_root) if project_root is not None else directory / 'project'
    source_store = ContentAddressedArtifacts(project / 'artifacts')
    ledger = project / 'evidence.jsonl'
    if ledger.exists():
        refs['log:evidence.jsonl'] = store.put_bytes(ledger.read_bytes())
        provider_ids = set()
        for line in ledger.read_bytes().splitlines():
            row = json.loads(line).get('payload', {})
            digest = row.get('sha256')
            if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                continue
            try:
                data = source_store.read_bytes(digest)
            except FileNotFoundError:
                missing.append(digest)
                continue
            refs['retained:'+digest] = store.put_bytes(data)
            if row.get('role') not in {'pi-provider', 'host-observation'}:
                continue
            value = json.loads(data)
            if row['role'] == 'host-observation':
                host = value.get('host', {})
                if host.get('event') == 'candidate-written':
                    source = host.get('source_sha256')
            elif value.get('reservation_id', digest) not in provider_ids:
                provider_ids.add(value.get('reservation_id', digest))
                provider_seconds += value.get('seconds', 0)
                spans.append(dict(value, stage='provider'))
    binding = Path(binding_root) if binding_root is not None else directory / 'runtime' / 'binding'
    transport = binding / 'transport-state'
    owned = None if project_id is None else {path.parent
        for path in transport.rglob('candidate.json')
        if json.loads(path.read_bytes()).get('project_id') == project_id}
    operations = []
    for path in sorted((binding / 'transport-state').rglob('operation.json')):
        if owned is not None and not any(candidate in path.parents for candidate in owned):
            continue
        record = json.loads(path.read_bytes())
        refs['operation:'+path.relative_to(binding).as_posix()] = store.put_bytes(path.read_bytes())
        operations.append(dict(handle=record.get('handle'), state=record.get('state'),
            stage=path.parent.name, expired=record.get('expired', False)))
    for path in sorted(binding.rglob('timing.jsonl')):
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        if owned is not None:
            rows = [row for row in rows if path.parent / str(row.get('key', '')).split('/')[0] in owned]
        if rows:
            data = path.read_bytes() if owned is None else ''.join(json.dumps(row)+'\n' for row in rows).encode()
            refs['timing:'+path.relative_to(binding).as_posix()] = store.put_bytes(data)
            spans.extend(rows)
    for path in sorted((binding / 'environment-checks').rglob('report.json')):
        if owned is not None:
            continue  # Campaign environment probes are not project-owned operations.
        report = json.loads(path.read_bytes())
        refs['environment:'+path.relative_to(binding).as_posix()] = store.put_bytes(path.read_bytes())
        spans.append(dict(report, stage='environment'))
    if source is not None:
        try:
            refs['source'] = store.put_bytes(source_store.read_bytes(source))
        except FileNotFoundError:
            missing.append(source)
            source = None
    return refs, source, operations, missing, spans, provider_seconds


def timing_summary(spans, started_epoch, elapsed_seconds, provider_seconds=0):
    """Stage totals may overlap. Only timestamped interval union is attributed."""
    intervals, stages = [], defaultdict(float)
    for row in spans:
        start, end = row.get('started_epoch'), row.get('ended_epoch')
        if not all(type(x) in (int, float) and math.isfinite(x) for x in (start, end)):
            continue
        start, end = max(started_epoch, start), min(started_epoch+elapsed_seconds, end)
        if end < start:
            continue
        stages[row['stage']] += end-start
        intervals.append((start, end))
    covered, boundary = 0.0, started_epoch
    for start, end in sorted(intervals):
        covered += max(0, end-max(start, boundary))
        boundary = max(boundary, end)
    return dict(wall_seconds=elapsed_seconds, by_stage_seconds=dict(stages),
        overlap_seconds=max(0, sum(stages.values())-covered),
        attributed_seconds=covered, unattributed_seconds=max(0, elapsed_seconds-covered),
        provider_recorded_seconds=provider_seconds)


def interrupted_outcome(directory, error, store, *, started_epoch, elapsed_seconds, project_root=None, binding_root=None, project_id=None):
    refs, source, operations, missing, spans, provider_seconds = arm_evidence(directory, store, project_root=project_root,
        binding_root=binding_root, project_id=project_id)
    partial = getattr(error, 'partial_evidence', {})
    project = Path(project_root) if project_root is not None else Path(directory) / 'project'
    source_store = ContentAddressedArtifacts(project / 'artifacts')
    # The action can be interrupted before its observation is published.
    for digest in [partial.get('source_sha256'), *partial.get('admitted_artifact_sha256', [])]:
        if digest is None:
            continue
        try:
            refs['retained:'+digest] = store.put_bytes(source_store.read_bytes(digest))
            if digest == partial.get('source_sha256'):
                source = digest
                refs['source'] = digest
        except FileNotFoundError:
            missing.append(digest)
    observations = dict(classification='interrupted', interruption_reason=interruption_reason(error),
        error_type=type(error).__name__, summary=str(error), evidence=partial,
        remote_operations=operations, missing_artifacts=sorted(set(missing)),
        timing=timing_summary(spans, started_epoch, elapsed_seconds, provider_seconds))
    refs['result:execution'] = store.put_json(observations)
    return SubjectOutcome(False, source, refs, observations, store.root)
