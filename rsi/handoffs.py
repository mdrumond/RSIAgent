"""Immutable accepted-input handoffs; no providers, replay, or parent writes.

capture_parent(..., inheritance='auto'|'parent'|'baseline', overrides=())
selects Phase1 for Phase2, Phase2 for inference. Auto refuses edited baseline
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
from rsi.skills import SkillBundle


@dataclass(frozen=True)
class ParentSnapshot:
    memory_ref: str
    skills_ref: str
    kdb_ref: str
    parent_cycle: str | None
    source_config_ref: str
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


def _parent(root, store, expected):
    with _locked(root):
        binding, state = _json(root / 'service.json'), _json(root / 'cycle.json')
        spec = CycleSpec.from_dict(binding['spec'])
        if spec.mode != expected or spec.cycle_id != root.name:
            raise ValueError('parent cycle identity/mode differs')
        configuration = checked_snapshot(binding['config'])
        if (store.read_json(spec.configuration_ref) != configuration
                or state['config']['spec'] != spec.to_dict()):
            raise ValueError('parent configuration differs from admitted state')
        # A constructed cycle or completed project is not yet a memory boundary.
        if 'memory' not in state['completed_operations'].values():
            return None
        memory = state['memory']
        accepted = _memory(store, memory)
        from benchmarks.ascend.subject_memory import digest
        memory_root = root / 'memory'
        heads = _json(memory_root / 'active.json')
        if hashlib.sha256(accepted['initial_memory'].encode()).hexdigest() != heads['initial_sha256']:
            raise ValueError('accepted memory initial identity differs')
        admitted = {row['version'] for row in heads['admissions'].values()}
        for record in accepted['memories']:
            version = digest(record)
            if (version not in admitted or
                    _json(memory_root / 'versions' / (version + '.json')) != record):
                raise ValueError('memory lacks accepted SubjectMemory admission')
        skills = state['skills']
        SkillBundle.from_dict(store.read_json(skills))
        if expected != Mode.PHASE1:
            learning = _json(root / 'learning.json')
            if learning['active'] != skills:
                raise ValueError('published skill and Learning heads differ')
            if skills != spec.baseline_skills:
                record = learning['candidates'].get(skills)
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
                    state_ref=store.put_json(state))


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
    if from_cycle and (inheritance == 'baseline' or mode == Mode.PHASE1):
        raise ValueError('baseline/Phase1 cannot select a parent')
    current = checked_snapshot(configuration if configuration is not None else show_configuration(root))
    store = ContentAddressedArtifacts(root / 'artifacts')
    selected = None
    expected = Mode.PHASE1 if mode == Mode.PHASE2 else Mode.PHASE2
    if mode != Mode.PHASE1 and inheritance != 'baseline':
        paths = [root / 'runs' / from_cycle] if from_cycle else sorted((root / 'runs').glob('*'))
        candidates = []
        for folder in paths:
            if not (folder / 'service.json').exists():
                if from_cycle:
                    raise ValueError('selected parent cycle does not exist')
                continue
            # Other modes are not candidates; selected explicit mode must match.
            if not from_cycle and _json(folder / 'service.json')['spec']['mode'] != expected.value:
                continue
            candidate = _parent(folder, store, expected)
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
