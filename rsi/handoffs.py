"""Immutable accepted-input handoffs; no providers, replay, or parent writes.

capture_parent(..., inheritance='auto'|'parent'|'baseline', overrides=())
defaults to Phase1 for Phase2, Phase2 for inference, baseline for Phase1.
Explicit revisits accept any phase. Auto refuses edited baseline
inputs that differ from the selected parent's admitted configuration. Explicit
parent acknowledges that choice; baseline ignores parents. overrides names
memory/skills/kdb to take from CURRENT configuration, independently of parent.
from_cycle selects the CURRENT published accepted heads of that retained cycle,
not an arbitrary historical task. Returned CAS references never follow later
parent changes. Services copy these bytes into fresh mutable child heads; the
existing fork_inference must still isolate inference. CLI --from-cycle maps to
from_cycle; --inherit-inputs maps to inheritance. Neither implies resume.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
from pathlib import Path
import re

from core.artifacts import ContentAddressedArtifacts
from rsi.configuration import checked_snapshot, show_configuration
from rsi.contracts import CycleSpec, Mode, Stage
from rsi.cycles import IsolationSnapshot
from rsi.skills import SkillBundle


@dataclass(frozen=True)
class ParentSnapshot:
    memory_ref: str
    skills_ref: str
    kdb_ref: str
    parent_cycle: str | None
    source_config_ref: str
    provenance: str


@dataclass(frozen=True)
class PublishedSkills:
    bundle_ref: str
    bundle: SkillBundle
    provenance: str


_KEYS = {'memory': ('memory_snapshot', 'memory_sha256'),
         'skills': ('baseline_skills', 'skills', 'skill_map_ref', 'skill_map'),
         'kdb': ('baseline_kdb', 'kdb', 'kdb_sha256', 'kdb_filename', 'kdb_ingestion_refs')}


def _json(path):
    return json.loads(path.read_bytes())


@contextmanager
def _locked(root):
    # Scheduler owns this file already. Do not create or mutate a parent's files.
    with (root / 'cycle.lock').open('rb') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('parent cycle busy; retry at a published boundary') from None
        if (root / 'revision.pending.json').exists():
            raise ValueError('parent configuration adoption requires reconciliation')
        yield


def _memory(store, reference):
    snapshot = store.read_json(reference)
    if (snapshot.get('schema') != 'subject-memory-snapshot-v1'
            or not isinstance(snapshot.get('initial_memory'), str)
            or not isinstance(snapshot.get('memories'), list)):
        raise ValueError('accepted SubjectMemory snapshot required')
    return snapshot


def _parent(root, store, expected=None, *, require_memory_boundary=True):
    with _locked(root):
        binding, state = _json(root / 'service.json'), _json(root / 'cycle.json')
        spec = CycleSpec.from_dict(binding['spec'])
        if state.get('active_wave'):
            raise ValueError('parent wave is unsettled or its publication requires reconciliation')
        if (expected is not None and spec.mode != expected) or spec.cycle_id != root.name:
            raise ValueError('parent cycle identity/mode differs')
        configuration = checked_snapshot(binding['config'])
        if (store.read_json(spec.configuration_ref) != configuration
                or state['config']['spec'] != spec.to_dict()):
            raise ValueError('parent configuration differs from admitted state')
        # A constructed cycle or completed project is not yet a memory boundary.
        if require_memory_boundary and 'memory' not in state['completed_operations'].values():
            return None
        memory = state['memory']
        accepted = _memory(store, memory)
        from benchmarks.ascend.subject_memory import digest
        branch = root
        if spec.mode == Mode.INFERENCE:
            isolation = state.get('isolation')
            if not isolation:
                raise ValueError('inference published heads require verified isolation')
            branch = Path(isolation['branch_root']).resolve(strict=True)
            if not branch.is_relative_to(root.resolve()) or branch == root.resolve():
                raise ValueError('inference branch must be independent within cycle')
            proof = _json(branch / 'isolation.json')
            IsolationSnapshot(state['namespace'], proof['memory'], proof['skills'],
                              str(branch), isolation['receipt_ref']).verify()
        memory_root = branch / 'memory'
        heads = _json(memory_root / 'active.json')
        if hashlib.sha256(accepted['initial_memory'].encode()).hexdigest() != heads['initial_sha256']:
            raise ValueError('accepted memory initial identity differs')
        admitted = {row['version'] for row in heads['admissions'].values()}
        # Accepted baseline records are copied into independent mutable heads;
        # their original admission identities need not be copied by the fork.
        admitted.update(digest(record) for record in _memory(store, spec.baseline_kdb)['memories'])
        for record in accepted['memories']:
            version = digest(record)
            if (version not in admitted or
                    _json(memory_root / 'versions' / (version + '.json')) != record):
                raise ValueError('memory lacks accepted SubjectMemory admission')
        skills = state['skills']
        SkillBundle.from_dict(store.read_json(skills))
        if spec.mode != Mode.PHASE1:
            learning = _json(branch / 'learning.json')
            if learning['active'] != skills:
                raise ValueError('published skill and Learning heads differ')
            record = learning['candidates'].get(skills)
            if record and record.get('quarantined'):
                raise ValueError('published skills are quarantined')
            if skills != spec.baseline_skills:
                if (not record or record.get('quarantined')
                        or record.get('validation', {}).get('workflow') is not True
                        or record.get('validation', {}).get('correctness') is not True
                        or skills not in learning['versions']):
                    raise ValueError('published skills lack successful validation')
        elif skills != spec.baseline_skills:
            raise ValueError('Phase1 changed its fixed skills')
        kdb = binding['kdb']
        if hashlib.sha256((root / 'knowledge.sqlite3').read_bytes()).hexdigest() != kdb:
            raise ValueError('parent frozen KDB checksum differs')
        store.read_bytes(kdb)
        # CAS is the scheduler's last accepted boundary even if a later operation
        # is unresolved; do not replace it with a possibly-ahead mutable head.
        return dict(memory=memory, skills=skills, kdb=kdb, config=configuration,
                    config_ref=spec.configuration_ref, parent=root.name,
                    created=state['created'], pending=state['pending'],
                    state_ref=store.put_json(state), mode=spec.mode.value)


def _baseline(root, config, store, needed):
    def path(value):
        p = Path(value).expanduser()
        return (p if p.is_absolute() else root / p).resolve(strict=True)
    values = {}
    if 'kdb' in needed:
        kdb = path(config.get('baseline_kdb', config.get('kdb')))
        if kdb.is_dir():
            kdb = kdb / config.get('kdb_filename', 'knowledge.sqlite3')
        data = kdb.read_bytes()
        manifest = _json(root / 'instance.json')
        relative = str(kdb.relative_to(root)) if kdb.is_relative_to(root) else None
        expected = config.get('kdb_sha256') or manifest.get('files', {}).get(relative)
        if not expected or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('verified current baseline KDB required')
        values['kdb'] = store.put_bytes(data)
    if 'skills' in needed:
        mapping = _json(path(config.get('skill_map_ref', config.get('skill_map'))))
        bundle = SkillBundle.capture(path(config.get('baseline_skills', config.get('skills'))),
                                    {Stage(k): tuple(v) for k, v in mapping.items()})
        values['skills'] = store.put_json(bundle.to_dict())
    if 'memory' in needed:
        if config.get('memory_snapshot'):
            initial = path(config['memory_snapshot']).read_bytes()
            if hashlib.sha256(initial).hexdigest() != config.get('memory_sha256'):
                raise ValueError('verified current baseline memory required')
            values['memory'] = store.put_bytes(initial)
            _memory(store, values['memory'])
        else:
            from benchmarks.ascend.subject_memory import canonical
            values['memory'] = store.put_bytes(canonical(dict(schema='subject-memory-snapshot-v1',
                                                   initial_memory='', memories=[])))
    return values


def capture_parent(instance, mode, from_cycle=None, *, inheritance='auto',
                   overrides=(), configuration=None):
    """Freeze accepted heads into the instance CAS, returning immutable provenance.

    No parent -> verified explicit baseline. Explicit missing/unready parent fails;
    auto skips cycles without a completed memory boundary. Busy/corrupt candidates
    fail closed, rather than silently falling back to older or empty inputs.
    overrides is a tuple of component names; full baseline mode conflicts with
    from_cycle. The current requested configuration is independently authenticated.
    """
    root = Path(instance).resolve(strict=True)
    mode = Mode(mode)
    overrides = tuple(overrides)
    if inheritance not in ('auto', 'parent', 'baseline') or set(overrides) - set(_KEYS):
        raise ValueError('explicit inheritance and memory/skills/kdb overrides required')
    if from_cycle is not None and (not isinstance(from_cycle, str)
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', from_cycle)):
        raise ValueError('simple parent cycle identity required')
    if from_cycle and inheritance == 'baseline':
        raise ValueError('baseline cannot select a parent')
    current = checked_snapshot(configuration if configuration is not None else show_configuration(root))
    store = ContentAddressedArtifacts(root / 'artifacts')
    selected = None
    expected = Mode.PHASE1 if mode == Mode.PHASE2 else Mode.PHASE2
    if inheritance != 'baseline' and (mode != Mode.PHASE1 or from_cycle or inheritance == 'parent'):
        paths = [root / 'runs' / from_cycle] if from_cycle else sorted((root / 'runs').glob('*'))
        candidates = []
        for folder in paths:
            if not (folder / 'service.json').exists():
                if from_cycle:
                    raise ValueError('selected parent cycle does not exist')
                continue
            # Default curriculum progression is mode-specific; explicit revisits are not.
            if (not from_cycle and mode != Mode.PHASE1
                    and _json(folder / 'service.json')['spec']['mode'] != expected.value):
                continue
            candidate = _parent(folder, store, None if from_cycle or mode == Mode.PHASE1 else expected)
            if candidate:
                candidates.append(candidate)
        if candidates:
            selected = max(candidates, key=lambda c: (c['created'], c['parent']))
        elif from_cycle or inheritance == 'parent':
            raise ValueError('selected parent has no accepted memory boundary')
    if selected and inheritance == 'auto':
        old, new = selected['config']['payload'], current['payload']
        conflicts = [component for component, keys in _KEYS.items() if component not in overrides
                     and {k: old.get(k) for k in keys} != {k: new.get(k) for k in keys}]
        if conflicts:
            raise ValueError('edited baselines require explicit parent/baseline or overrides: ' + ', '.join(conflicts))
    values = dict(selected) if selected else {}
    needed = set(overrides) if selected else set(_KEYS)
    if needed:
        baseline = _baseline(root, current['payload'], store, needed)
        values.update({key: baseline[key] for key in needed})
    config_ref = selected['config_ref'] if selected else store.put_json(current)
    proof = store.put_json(dict(schema='rsi-parent-snapshot-v1', mode=mode.value,
        parent_cycle=selected['parent'] if selected else None,
        source_config_ref=config_ref, requested_config_ref=store.put_json(current),
        inheritance=inheritance, overrides=sorted(overrides),
        accepted_state_ref=selected['state_ref'] if selected else None,
        later_pending=selected['pending'] if selected else None,
        boundary='last-published-accepted-heads' if selected else 'explicit-baseline',
        memory_ref=values['memory'], skills_ref=values['skills'], kdb_ref=values['kdb']))
    return ParentSnapshot(values['memory'], values['skills'], values['kdb'],
                          selected['parent'] if selected else None, config_ref, proof)


def published_skills(instance, cycle_id):
    """Export only a locked published head, never an unvalidated candidate.

    The inference head is read from its authenticated isolated Learning branch.
    A published baseline can be exported before a new memory update; inference
    must already have completed isolation. The returned receipt freezes scope,
    source configuration, scheduler state, and any later unresolved operation.
    """
    if not isinstance(cycle_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', cycle_id):
        raise ValueError('simple cycle identity required')
    root = Path(instance).resolve(strict=True)
    store = ContentAddressedArtifacts(root / 'artifacts')
    accepted = _parent(root / 'runs' / cycle_id, store, require_memory_boundary=False)
    bundle = SkillBundle.from_dict(store.read_json(accepted['skills']))
    proof = store.put_json(dict(schema='rsi-published-skills-v1', cycle_id=cycle_id,
        mode=accepted['mode'], bundle_ref=accepted['skills'],
        source_config_ref=accepted['config_ref'], accepted_state_ref=accepted['state_ref'],
        later_pending=accepted['pending']))
    return PublishedSkills(accepted['skills'], bundle, proof)
