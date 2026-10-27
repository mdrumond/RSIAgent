"""Pin published native Phase 2 learning using the campaign's production readers."""
from __future__ import annotations

import base64
import json
from pathlib import Path, PurePosixPath

from .jobs import canonical, sha


def freeze_phase1_source(source_root, initial_memory, review_path, *, snapshot_root):
    """Admit reviewed Phase 1 directly, without fabricating a completed Phase 2.

    The existing admission reader authenticates publication, review, treatment,
    and every retained source/evidence object. snapshot_root is the destination
    for that immutable portable admission, not a legacy campaign state root.
    """
    from benchmarks.ascend.phase2 import admit_native_snapshot
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
    """Authenticate and return a self-contained snapshot, never just a terminal flag.

    Only configured roots are passed here; discovery never scans parent folders.
    Profiler report trees are not needed to consume learning and are not copied.
    """
    from benchmarks.ascend.phase2_curriculum import reviewed_seed
    from benchmarks.ascend.phase2 import _native_identity
    from benchmarks.ascend.training_annex import _verify_adaptive_state, _verify_wave
    from benchmarks.a5kernels.phase1_memory import Phase1ProjectMemory

    root = Path(root).absolute()
    state_bytes = _read(root, 'state.json')
    state = json.loads(state_bytes)
    phase = state.get('phase')
    if phase == 'optimizer-phase2':
        from benchmarks.ascend.optimizer_phase2_artifacts import freeze_campaign
        return freeze_campaign(root)
    if phase not in {'adaptive-phase2', 'target-driven-phase2'}:
        raise ValueError('unsupported legacy Phase 2 format; native reviewed adaptive lineage required')
    if (root / 'pending.json').exists():
        raise ValueError('Phase 2 has a pending attempt')
    terminal = (state.get('terminal') or {}).get('status')
    if phase == 'target-driven-phase2':
        if (terminal != 'READY' or not state.get('goal_identity')
                or not state.get('attempts') or state['attempts'][-1].get('kind') != 'target'
                or state['attempts'][-1].get('verified') is not True):
            raise ValueError('target-driven Phase 2 has not reached READY')
    elif (not state.get('prior_outcomes') or not (
            len(state['prior_outcomes']) >= state.get('project_target', float('inf'))
            or terminal == 'SATURATED')):
        raise ValueError('adaptive Phase 2 has no compatible completed boundary')

    # Validate all snapshot references before invoking the production reader.
    matches = []
    for seed_root in dict.fromkeys([root, *(Path(path).absolute() for path in snapshot_roots)]):
        for path in seed_root.glob('cells/*/snapshot/snapshot.json'):
            relative = path.relative_to(seed_root).as_posix()
            manifest = json.loads(_read(seed_root, relative))
            if manifest.get('snapshot_id') == state['snapshot_id']:
                for name in manifest['files']:
                    _read(path.parent, name)
                matches.append((seed_root, path.parent, manifest))
    if len(matches) != 1:
        raise ValueError('reviewed seed missing or ambiguous')
    seed_root, directory, manifest = matches[0]
    destination = manifest['destination_identity']
    identity = destination['source_identity']
    _native_identity(identity)
    if destination.get('language') != 'ascend-c' or destination.get('target') != 'a3':
        raise ValueError('Phase 2 reviewed source is not compatible A3 Ascend C memory')
    seed, checked_directory = reviewed_seed(seed_root, state['snapshot_id'])
    assert directory == checked_directory
    _verify_adaptive_state(root, state, reviewed_seed_root=seed_root)
    files = {'state.json': state_bytes}
    prefix = 'reviewed-seed'
    for name in [*manifest['files'], 'snapshot.json']:
        data = _read(directory, name)
        if name in manifest['files'] and sha(data) != manifest['files'][name]:
            raise ValueError('reviewed memory changed during snapshot')
        files[f'{prefix}/{name}'] = data

    entries = json.loads(seed)['phase1_memory']
    for wave_id in state['completed_waves']:
        wave_root = root / 'waves' / wave_id
        _verify_wave(wave_root)
        plan = json.loads(_read(root, f'waves/{wave_id}/plan.json'))
        for name in ('plan.json', 'published.json', 'consolidated-memory.json'):
            relative = f'waves/{wave_id}/{name}'
            files[relative] = _read(root, relative)
        for project in plan['wave']['projects']:
            # WaveProject's wire form nests the project spec.
            project_id = project['spec']['project_id']
            branch = f'waves/{wave_id}/branches/{project_id}'
            for name in ('outcome.json', 'memory-delta.bin', 'execution-evidence.bin'):
                relative = f'{branch}/{name}'
                files[relative] = _read(root, relative)
            outcome = json.loads(files[f'{branch}/outcome.json'])
            if not outcome['verified']:
                continue
            delta = json.loads(files[f'{branch}/memory-delta.bin'])
            entry = json.loads(base64.b64decode(delta['memory_base64'], validate=True))
            memory = Phase1ProjectMemory.from_mapping(entry)
            digests = {memory.source_revision, *memory.admitted_artifact_sha256,
                       *(fact.evidence_sha256 for fact in memory.host_facts)}
            for digest in digests:
                relative = f'{branch}/{memory.artifact_store or "artifacts"}/objects/{digest[:2]}/{digest[2:]}'
                data = _read(root, relative)
                if sha(data) != digest:
                    raise ValueError('published memory artifact digest mismatch')
                files[relative] = data
    entries += [json.loads(base64.b64decode(payload, validate=True))
                for payload in state['memory_bytes_base64']]
    # Authenticate seed CAS as well as manifest hashes; all memory references travel.
    for entry in json.loads(seed)['phase1_memory']:
        memory = Phase1ProjectMemory.from_mapping(entry)
        for digest in {memory.source_revision, *memory.admitted_artifact_sha256,
                       *(fact.evidence_sha256 for fact in memory.host_facts)}:
            data = _read(directory, f'artifacts/objects/{digest[:2]}/{digest[2:]}')
            if sha(data) != digest:
                raise ValueError('reviewed memory artifact digest mismatch')
    if _read(root, 'state.json') != state_bytes or (root / 'pending.json').exists():
        raise ValueError('Phase 2 changed during snapshot')
    body = dict(schema='optimizer-memory-snapshot-v1', source_identity=identity,
                phase=phase, terminal=state.get('terminal'),
                phase1_snapshot_id=state['snapshot_id'], memory_entries=entries,
                files={name: {'sha256': sha(data), 'base64': base64.b64encode(data).decode()}
                       for name, data in sorted(files.items())})
    return {**body, 'snapshot_id': sha(canonical(body))}


def verify_snapshot(snapshot):
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


def freeze_committed_phase1(root, initial_memory):
    """Export committed host-authenticated learning without claiming human review."""
    from benchmarks.ascend.phase2 import _native_identity
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
    from benchmarks.a5kernels.phase2_adaptive_memory import _published_evidence
    _published_evidence(root, state, initial_memory)
    files = {'state.json': state_bytes, 'campaign-identity.json': canonical(identity),
             'initial-memory.bin': initial_memory}
    entries = []
    try:
        inherited = json.loads(initial_memory)
    except (ValueError, UnicodeDecodeError):
        inherited = None
    if isinstance(inherited, dict) and inherited.get('schema') == 'ascend-phase2-curriculum-seed-v1':
        from benchmarks.ascend.phase2_curriculum import reviewed_seed
        authenticated, directory = reviewed_seed(root, inherited['snapshot_id'])
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
