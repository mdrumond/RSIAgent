"""Exported folders preserve exact captured bundles and bootstrap directly."""
from hashlib import sha256
import json
from pathlib import Path

import pytest

from rsi.contracts import Stage
from rsi.instances import create_instance
from rsi.skill_delivery import export_bundle
from rsi.skills import SkillBundle, builtin_bundle


def capture(folder):
    mapping = json.loads((folder / 'map.json').read_text())
    return SkillBundle.capture(folder, {Stage(k): tuple(v) for k, v in mapping.items()})


def test_export_preserves_content_map_hash_and_can_bootstrap_fresh_instance(tmp_path):
    original = builtin_bundle()
    destination = tmp_path / 'accepted'
    result = export_bundle(original, destination, source={'cycle': 'validated'})
    assert capture(destination) == original
    assert capture(destination).digest == result['bundle_hash'] == original.digest
    payload = (destination / 'bundle.json').read_bytes()
    assert sha256(payload).hexdigest() == result['bundle_ref']
    assert SkillBundle.from_dict(json.loads(payload)) == original
    manifest = json.loads((destination / 'export.json').read_text())
    assert manifest['source'] == {'cycle': 'validated'}
    for relative, digest in manifest['files'].items():
        assert sha256((destination / relative).read_bytes()).hexdigest() == digest
    kdb = tmp_path / 'baseline-kdb'
    kdb.mkdir()
    (kdb / 'README.md').write_text('Baseline data.\n')
    fresh = tmp_path / 'fresh-instance'
    create_instance(fresh, baseline_kdb=kdb, baseline_skills=destination,
                    skill_map=destination / 'map.json')
    configured = json.loads((fresh / 'config/instance.json').read_text())
    mapping = json.loads((fresh / configured['skill_map']).read_text())
    restored = SkillBundle.capture(fresh / configured['skills'], {Stage(k): tuple(v) for k, v in mapping.items()})
    assert restored == original
    assert not (fresh / 'runtime/setup.json').exists()


def test_export_never_overwrites_destination_or_publishes_partial_tree(tmp_path, monkeypatch):
    import rsi.skill_delivery as delivery
    destination = tmp_path / 'output'
    destination.mkdir()
    (destination / 'owned').write_text('keep')
    with pytest.raises(FileExistsError):
        export_bundle(builtin_bundle(), destination)
    assert (destination / 'owned').read_text() == 'keep'
    destination = tmp_path / 'racing'
    def race(staging, target):
        target.mkdir()
        (target / 'winner').write_text('other publication')
        from rsi.instances import _publish
        _publish(staging, target)
    monkeypatch.setattr(delivery, 'publish_directory', race)
    with pytest.raises(FileExistsError):
        export_bundle(builtin_bundle(), destination)
    assert [p.name for p in destination.iterdir()] == ['winner']
    assert not list(tmp_path.glob('.racing-*'))


def test_write_failure_retains_no_delivery_and_cleans_only_own_staging(tmp_path, monkeypatch):
    import rsi.skill_delivery as delivery
    destination = tmp_path / 'failed'
    writer = delivery.publish_file
    def fail(path, content, **options):
        if path.name == 'map.json':
            raise OSError('disk unavailable')
        return writer(path, content, **options)
    monkeypatch.setattr(delivery, 'publish_file', fail)
    with pytest.raises(OSError):
        export_bundle(builtin_bundle(), destination)
    assert not destination.exists()
    assert not list(tmp_path.glob('.failed-*'))


def test_builtin_cli_delivery_needs_no_instance_or_model_setup(tmp_path, capsys):
    from rsi.cli import main
    destination = tmp_path / 'starter-baseline'
    assert main(['skills', 'export', '--builtin', '--destination', str(destination)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['bundle_hash'] == builtin_bundle().digest
    assert capture(destination) == builtin_bundle()


def test_published_cycle_delivery_ignores_baseline_and_pending_candidates_and_bootstraps(tmp_path, capsys):
    from dataclasses import replace
    import shutil
    from test_rsi_handoffs import setup, parent
    from rsi.cli import main
    from rsi.contracts import Mode
    from rsi.python_execution import atomic_json
    inputs = setup(tmp_path)
    root, store, baseline, _, _ = inputs
    folder, scheduler, _ = parent(inputs, 'validated', Mode.PHASE2)
    original = builtin_bundle()
    published = replace(original, documents=tuple(
        replace(d, content=d.content + '\n## Accepted recovery\nUse independent numeric checks.\n')
        if d.name == 'debug' else d for d in original.documents))
    accepted = store.put_json(published.to_dict())
    pending = store.put_json(replace(original, documents=tuple(
        replace(d, content=d.content + '\nUnvalidated idea.\n') if d.name == 'debug' else d
        for d in original.documents)).to_dict())
    learning = dict(active=accepted, versions=[baseline, accepted], candidates={
        accepted: dict(quarantined=False, validation=dict(workflow=True, correctness=True)),
        pending: dict(quarantined=False, validation=None)})
    atomic_json(folder / 'learning.json', learning)
    scheduler.publish_validated_skills(baseline, accepted)
    scheduler.close()
    before = {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    # Published CAS is the authority even if the original baseline folder is gone.
    shutil.rmtree(root / 'skills')
    assert not (root / 'runtime/setup.json').exists()
    assert main(['--instance', str(root), 'skills', 'get', '--cycle', 'validated']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['bundle_ref'] == accepted and result['bundle_hash'] == published.digest
    assert SkillBundle.from_dict({k: result[k] for k in ('schema_version', 'documents', 'mapping')}) == published
    proof = store.read_json(result['provenance_ref'])
    assert proof['cycle_id'] == 'validated' and proof['bundle_ref'] == accepted
    destination = tmp_path / 'accepted-delivery'
    assert main(['--instance', str(root), 'skills', 'export', '--cycle', 'validated',
                 '--destination', str(destination)]) == 0
    exported = json.loads(capsys.readouterr().out)
    assert exported['bundle_ref'] == accepted and capture(destination) == published
    kdb = tmp_path / 'bootstrap-memory'
    kdb.mkdir()
    (kdb / 'README.md').write_text('Independent baseline input.\n')
    fresh = tmp_path / 'fresh-from-accepted'
    create_instance(fresh, baseline_kdb=kdb, baseline_skills=destination, skill_map=destination / 'map.json')
    mapping = json.loads((fresh / 'config/skill-map.json').read_text())
    assert SkillBundle.capture(fresh / 'skills', {Stage(k): tuple(v) for k, v in mapping.items()}) == published
    assert before == {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    learning['candidates'][accepted]['quarantined'] = True
    atomic_json(folder / 'learning.json', learning)
    rejected = tmp_path / 'quarantined-delivery'
    assert main(['--instance', str(root), 'skills', 'export', '--cycle', 'validated',
                 '--destination', str(rejected)]) == 2
    assert not rejected.exists()
