"""Real scheduler publications and SubjectMemory admissions, without actors."""
import hashlib
import json

import pytest

from core.artifacts import ContentAddressedArtifacts
from benchmarks.ascend.subject_memory import SubjectMemory
from rsi.configuration import _snapshot
from rsi.contracts import CycleSpec, Mode, ProjectSpec
from rsi.cycles import CycleScheduler, HostTaskResult, SnapshotUpdate
from rsi.handoffs import capture_parent
from rsi.python_execution import atomic_json
from rsi.skills import builtin_bundle


def setup(tmp_path):
    root = tmp_path / 'instance'
    root.mkdir()
    store = ContentAddressedArtifacts(root / 'artifacts')
    bundle = builtin_bundle()
    skills = store.put_json(bundle.to_dict())
    kdb = b'frozen KDB bytes'
    config = _snapshot({'kdb': 'baseline.sqlite3', 'skills': 'skills', 'skill_map': 'map.json'})
    (root / 'baseline.sqlite3').write_bytes(kdb)
    atomic_json(root / 'instance.json', {'files': {'baseline.sqlite3': hashlib.sha256(kdb).hexdigest()}})
    # Export a real bundle for explicit baseline fallback.
    for document in bundle.documents:
        folder = root / 'skills' / document.name
        folder.mkdir(parents=True)
        (folder / 'SKILL.md').write_text(document.content)
    atomic_json(root / 'map.json', {k.value: list(v) for k, v in bundle.mapping})
    return root, store, skills, kdb, config


def parent(inputs, name='research', mode=Mode.PHASE1, created=1, accepted=True):
    root, store, skills, kdb, config = inputs
    folder = root / 'runs' / name
    folder.mkdir(parents=True)
    memory = SubjectMemory(folder / 'memory')
    initial = store.put_bytes(memory.snapshot())
    spec = CycleSpec(name, mode, (ProjectSpec('tiny', 'Prove sum', ('python', 'verify.py')),),
                     initial, skills, configuration_ref=store.put_json(config))
    def update(previous, rows, context):
        memory.accept(context['operation_id'],
            {'base_versions': [], 'kind': 'correctness', 'summary': name}, {'source': 'host'})
        return SnapshotUpdate(store.put_bytes(memory.snapshot()))
    scheduler = CycleScheduler(folder / 'cycle.json', spec,
        lambda *a: HostTaskResult('accepted', store.put_json({'verified': True}), 1),
        update, skill_update=lambda *a: SnapshotUpdate(skills), recover=False, clock=lambda: created)
    atomic_json(folder / 'service.json', {'spec': spec.to_dict(), 'config': config,
                                        'kdb': store.put_bytes(kdb)})
    (folder / 'knowledge.sqlite3').write_bytes(kdb)
    if accepted:
        scheduler.step()
        scheduler.step()  # Actual memory hook and scheduler reconciliation.
    if mode == Mode.PHASE2:
        atomic_json(folder / 'learning.json', {'active': skills, 'versions': [skills], 'candidates': {}})
    scheduler.close()
    return folder, scheduler, memory


def test_auto_parent_and_frozen_refs_survive_new_task(tmp_path):
    inputs = setup(tmp_path)
    root, store, _, _, config = inputs
    folder, scheduler, memory = parent(inputs)
    before = (folder / 'cycle.json').read_bytes()
    snapshot = capture_parent(root, 'phase2', configuration=config)
    assert snapshot.parent_cycle == 'research'
    assert store.read_json(snapshot.memory_ref)['memories'][0]['proposal']['summary'] == 'research'
    assert (folder / 'cycle.json').read_bytes() == before
    memory.accept('later', {'base_versions': [], 'kind': 'failure', 'summary': 'later'}, {})
    assert len(store.read_json(snapshot.memory_ref)['memories']) == 1
    assert len(json.loads(memory.snapshot())['memories']) == 2
    assert store.read_json(snapshot.provenance)['boundary'] == 'last-published-accepted-heads'


def test_newest_accepted_boundary_and_explicit_revisit(tmp_path):
    inputs = setup(tmp_path)
    root, _, _, _, config = inputs
    parent(inputs, 'older', created=1)
    parent(inputs, 'newer', created=2)
    parent(inputs, 'unfinished', created=3, accepted=False)
    assert capture_parent(root, 'phase2', configuration=config).parent_cycle == 'newer'
    assert capture_parent(root, 'phase2', 'older', configuration=config).parent_cycle == 'older'
    with pytest.raises(ValueError, match='no accepted'):
        capture_parent(root, 'phase2', 'unfinished', configuration=config)


def test_inference_picks_phase2_not_candidate(tmp_path):
    inputs = setup(tmp_path)
    root, store, skills, _, config = inputs
    folder, _, _ = parent(inputs, 'end-to-end', Mode.PHASE2)
    learning = json.loads((folder / 'learning.json').read_bytes())
    learning['candidates']['pending'] = {'validation': None}
    atomic_json(folder / 'learning.json', learning)
    snapshot = capture_parent(root, 'inference', configuration=config)
    assert snapshot.skills_ref == skills and snapshot.parent_cycle == 'end-to-end'
    learning['active'] = 'pending'
    atomic_json(folder / 'learning.json', learning)
    with pytest.raises(ValueError, match='heads differ'):
        capture_parent(root, 'inference', configuration=config)


def test_edited_inputs_require_explicit_choice(tmp_path):
    inputs = setup(tmp_path)
    root, _, _, _, config = inputs
    parent(inputs)
    edited = _snapshot(dict(config['payload'], baseline_skills='edited-skills'))
    with pytest.raises(ValueError, match='edited baselines'):
        capture_parent(root, 'phase2', configuration=edited)
    assert capture_parent(root, 'phase2', inheritance='parent', configuration=edited).parent_cycle == 'research'


def test_corrupt_adoption_and_kdb_refused(tmp_path):
    inputs = setup(tmp_path)
    root, _, _, _, config = inputs
    folder, _, _ = parent(inputs)
    atomic_json(folder / 'revision.pending.json', {'unknown': True})
    with pytest.raises(ValueError, match='adoption'):
        capture_parent(root, 'phase2', configuration=config)
    (folder / 'revision.pending.json').unlink()
    (folder / 'knowledge.sqlite3').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        capture_parent(root, 'phase2', configuration=config)


def test_unknown_pending_is_retained_and_labelled(tmp_path):
    inputs = setup(tmp_path)
    root, store, _, _, config = inputs
    folder, scheduler, _ = parent(inputs)
    scheduler.state['pending'] = {'kind': 'original', 'id': 'unobserved'}
    scheduler._save()
    before = (folder / 'cycle.json').read_bytes()
    snapshot = capture_parent(root, 'phase2', configuration=config)
    assert store.read_json(snapshot.provenance)['later_pending']['id'] == 'unobserved'
    assert (folder / 'cycle.json').read_bytes() == before


def test_no_parent_baseline_and_component_override(tmp_path):
    inputs = setup(tmp_path)
    root, store, skills, _, config = inputs
    baseline = capture_parent(root, 'phase2', configuration=config)
    assert baseline.parent_cycle is None and baseline.skills_ref == skills
    assert store.read_json(baseline.memory_ref)['memories'] == []
    parent(inputs)
    inherited = capture_parent(root, 'phase2', configuration=config, overrides=('memory',))
    assert inherited.parent_cycle == 'research'
    assert inherited.memory_ref == baseline.memory_ref
    assert store.read_json(inherited.provenance)['overrides'] == ['memory']


def test_busy_parent_does_not_fall_back(tmp_path):
    import fcntl
    inputs = setup(tmp_path)
    root, _, _, _, config = inputs
    folder, _, _ = parent(inputs)
    with (folder / 'cycle.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='busy'):
            capture_parent(root, 'phase2', configuration=config)


def test_mismatched_configuration_refused(tmp_path):
    inputs = setup(tmp_path)
    root, _, _, _, config = inputs
    folder, _, _ = parent(inputs)
    binding = json.loads((folder / 'service.json').read_bytes())
    binding['config'] = _snapshot(dict(config['payload'], prompts={'actor': 'changed'}))
    atomic_json(folder / 'service.json', binding)
    with pytest.raises(ValueError, match='configuration differs'):
        capture_parent(root, 'phase2', configuration=config)
