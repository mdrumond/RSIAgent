"""Pin published native learning using authenticated production readers."""
from __future__ import annotations

import base64
import json
from pathlib import Path, PurePosixPath

from .jobs import canonical, sha


def freeze_phase1_source(source_root, initial_memory, review_path, *, snapshot_root):
    """Admit reviewed Phase 1 directly.

    The existing admission reader authenticates publication, review, treatment,
    and every retained source/evidence object. snapshot_root is the destination
    for that immutable portable admission, not a legacy campaign state root.
    """
    from benchmarks.ascend.reviewed_memory import admit_native_snapshot
    admitted = admit_native_snapshot(source_root, initial_memory, review_path, snapshot_root)
    return freeze_reviewed_phase1(admitted)


def freeze_reviewed_phase1(admitted):
    """Export an already admitted native Phase 1 seed in optimizer wire format."""
    admitted.verify()
    destination = admitted.destination.as_dict()
    if destination.get('language') != 'ascend-c' or destination.get('target') != 'a3':
        raise ValueError('optimizer seed requires reviewed A3 Ascend C memory')
    entries = list(admitted.memory_entries())
    files = {}
    for name in [*admitted.manifest['files'], 'snapshot.json']:
        data = _read(admitted.root, name)
        if name == 'snapshot.json' and json.loads(data) != admitted.manifest:
            raise ValueError('reviewed Phase 1 manifest changed during snapshot')
        if name in admitted.manifest['files'] and sha(data) != admitted.manifest['files'][name]:
            raise ValueError('reviewed Phase 1 changed during snapshot')
        files[f'reviewed-seed/{name}'] = dict(sha256=sha(data), base64=base64.b64encode(data).decode())
    admitted.verify()
    body = dict(schema='optimizer-memory-snapshot-v1', phase='reviewed-phase1',
                source_identity=destination['source_identity'], terminal=None,
                phase1_snapshot_id=admitted.snapshot_id, memory_entries=entries, files=files)
    return {**body, 'snapshot_id': sha(canonical(body))}


def _read(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('memory reference must be campaign-relative')
    resolved = root.joinpath(*path.parts)
    if any(parent.is_symlink() for parent in [resolved, *resolved.parents] if parent != root.parent):
        raise ValueError('memory reference contains symlink')
    return resolved.read_bytes()




def freeze_source(root, *, snapshot_roots=()):
    """Freeze a completed current native Phase 1 source with its exact seed."""
    root = Path(root).absolute()
    state = json.loads(_read(root, 'state.json'))
    if state.get('schema') != 'a5-phase1-adaptive-state-v2' or state.get('phase'):
        raise ValueError('unsupported source; current committed Phase 1 memory required')
    if (root / 'pending.json').exists():
        raise ValueError('memory source has pending work')
    if not state.get('terminal') and len(state.get('prior_outcomes', [])) < state.get('project_target', float('inf')):
        raise ValueError('memory source has not completed its project target')
    return freeze_committed_phase1(root, _read(root, 'initial-memory.bin'), snapshot_roots=snapshot_roots)


def verify_snapshot(snapshot):
    if snapshot.get('phase') not in {None, 'reviewed-phase1', 'committed-phase1', 'subject-curriculum'}:
        raise ValueError('unsupported optimizer memory source')
    body = {key: value for key, value in snapshot.items() if key != 'snapshot_id'}
    if sha(canonical(body)) != snapshot.get('snapshot_id'):
        raise ValueError('optimizer memory snapshot digest mismatch')
    for name, record in snapshot['files'].items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or sha(base64.b64decode(
                record['base64'], validate=True)) != record['sha256']:
            raise ValueError('optimizer memory snapshot file digest mismatch')


def discover_sources(configured_roots, *, snapshot_roots=()):
    """Return explicit availability for exactly the configured roots."""
    result = []
    for configured in configured_roots:
        root = Path(configured).absolute()
        try:
            snapshot = freeze_source(root, snapshot_roots=snapshot_roots)
            result.append(dict(root=str(root), available=True,
                               snapshot_id=snapshot['snapshot_id'], phase=snapshot['phase'],
                               memory_count=len(snapshot['memory_entries']),
                               source_identity=snapshot['source_identity']))
        except (OSError, ValueError, KeyError, TypeError) as error:
            result.append(dict(root=str(root), available=False, reason=str(error)))
    return result


def freeze_committed_phase1(root, initial_memory, *, snapshot_roots=()):
    """Export committed host-authenticated learning without claiming human review."""
    from benchmarks.ascend.reviewed_memory import _native_identity
    from benchmarks.ascend.training_annex import _verify_adaptive_state, _verify_wave, _project_memory
    root = Path(root).absolute()
    state_bytes = _read(root, 'state.json')
    state = json.loads(state_bytes)
    identity = json.loads(_read(root, 'campaign-identity.json'))
    _native_identity(identity)
    if state.get('schema') != 'a5-phase1-adaptive-state-v2' or state.get('phase'):
        raise ValueError('committed Phase 1 source required')
    if state['initial_memory_sha256'] != sha(initial_memory):
        raise ValueError('committed Phase 1 initial memory differs')
    # Use the production state reader and host publication verifier, including
    # failed branches; a human review verdict is deliberately not manufactured.
    import tempfile
    with tempfile.NamedTemporaryFile() as seed:
        seed.write(initial_memory)
        seed.flush()
        _verify_adaptive_state(root, state, external_initial_memory=seed.name)
    from benchmarks.ascend.reviewed_memory import _published_evidence
    _published_evidence(root, state, initial_memory)
    files = {'state.json': state_bytes, 'campaign-identity.json': canonical(identity),
             'initial-memory.bin': initial_memory}
    entries = []
    try:
        inherited = json.loads(initial_memory)
    except (ValueError, UnicodeDecodeError):
        inherited = None
    if isinstance(inherited, dict) and inherited.get('schema') == 'ascend-reviewed-memory-seed-v1':
        from benchmarks.ascend.reviewed_memory import reviewed_seed
        candidates = list(dict.fromkeys([root, *(Path(path) for path in snapshot_roots)]))
        matching = [candidate for candidate in candidates
                    if any(json.loads(path.read_bytes()).get('snapshot_id') == inherited['snapshot_id']
                           for path in candidate.glob('cells/*/snapshot/snapshot.json'))]
        if len(matching) != 1:
            raise ValueError('reviewed seed missing or ambiguous')
        authenticated, directory = reviewed_seed(matching[0], inherited['snapshot_id'])
        if json.loads(authenticated) != inherited:
            raise ValueError('reviewed seed differs from committed initial memory')
        entries.extend(inherited['phase1_memory'])
        manifest = json.loads(_read(directory, 'snapshot.json'))
        for name in [*manifest['files'], 'snapshot.json']:
            data = _read(directory, name)
            if name in manifest['files'] and sha(data) != manifest['files'][name]:
                raise ValueError('inherited reviewed snapshot changed')
            files['reviewed-seed/' + name] = data
        from benchmarks.a5kernels.phase1_memory import Phase1ProjectMemory
        for entry in entries:
            memory = Phase1ProjectMemory.from_mapping(entry)
            for digest in {memory.source_revision, *memory.admitted_artifact_sha256,
                           *(fact.evidence_sha256 for fact in memory.host_facts)}:
                data = _read(directory, f'artifacts/objects/{digest[:2]}/{digest[2:]}')
                if sha(data) != digest:
                    raise ValueError('inherited reviewed memory CAS mismatch')
    for wave_id in state['completed_waves']:
        wave = root / 'waves' / wave_id
        _verify_wave(wave)
        for name in ('plan.json', 'published.json', 'consolidated-memory.json'):
            relative = f'waves/{wave_id}/{name}'
            files[relative] = _read(root, relative)
        plan = json.loads(files[f'waves/{wave_id}/plan.json'])
        for project in plan['wave']['projects']:
            branch = wave / 'branches' / project['spec']['project_id']
            if not (branch / 'outcome.json').exists():
                # Both publication readers above authenticated this skipped
                # assignment and its exact retained interruption record.
                relative = (branch / 'interruption.json').relative_to(root).as_posix()
                files[relative] = _read(root, relative)
                continue
            for name in ('outcome.json', 'memory-delta.bin', 'execution-evidence.bin'):
                relative = (branch / name).relative_to(root).as_posix()
                files[relative] = _read(root, relative)
            if not json.loads((branch / 'outcome.json').read_bytes())['verified']:
                continue
            _, memory = _project_memory(branch)
            for digest in {memory.source_revision, *memory.admitted_artifact_sha256,
                           *(fact.evidence_sha256 for fact in memory.host_facts)}:
                path = branch / (memory.artifact_store or 'artifacts') / 'objects' / digest[:2] / digest[2:]
                relative = path.relative_to(root).as_posix()
                data = _read(root, relative)
                if sha(data) != digest:
                    raise ValueError('committed Phase 1 CAS mismatch')
                files[relative] = data
    if _read(root, 'state.json') != state_bytes:
        raise ValueError('Phase 1 changed during committed snapshot')
    body = dict(schema='optimizer-memory-snapshot-v1', phase='committed-phase1',
        source_identity=identity, terminal=state.get('terminal'),
        phase1_snapshot_id=sha(state_bytes),
        memory_entries=entries + [json.loads(base64.b64decode(raw, validate=True))
                                  for raw in state['memory_bytes_base64']],
        files={name: dict(sha256=sha(data), base64=base64.b64encode(data).decode())
               for name, data in sorted(files.items())})
    return {**body, 'snapshot_id': sha(canonical(body))}
