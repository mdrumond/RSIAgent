import json
from dataclasses import FrozenInstanceError

import pytest

from rsi.contracts import Stage
from rsi.skills import SkillBundle, SkillUsage, builtin_bundle


def baseline(tmp_path):
    for name in ('design', 'kdb'):
        folder = tmp_path / name
        folder.mkdir()
        (folder / 'SKILL.md').write_text('# Guide\n\n## Contract\nDo work.\n\n## Evidence\nRecord it.\n')
    return SkillBundle.capture(tmp_path, {stage: ('design', 'kdb') for stage in Stage})


def test_snapshot_detaches_inputs_and_hashes_mapping(tmp_path):
    bundle = baseline(tmp_path)
    before = bundle.digest
    (tmp_path / 'design/SKILL.md').write_text('changed')
    assert bundle.digest == before
    assert bundle.document('design').sections == ('contract', 'evidence')
    assert SkillBundle.from_dict(json.loads(json.dumps(bundle.to_dict()))) == bundle
    changed = bundle.to_dict()
    changed['mapping']['design'] = ['kdb']
    assert SkillBundle.from_dict(changed).digest != before


def test_corrupt_snapshot_is_rejected(tmp_path):
    data = baseline(tmp_path).to_dict()
    data['documents'][0]['content'] += 'tampered'
    with pytest.raises(ValueError, match='hash'):
        SkillBundle.from_dict(data)


def test_missing_stage_or_skill_rejected(tmp_path):
    baseline(tmp_path)
    with pytest.raises(ValueError):
        SkillBundle.capture(tmp_path, {Stage.DESIGN: ('design',)})
    with pytest.raises(ValueError):
        SkillBundle.capture(tmp_path, {stage: ('absent',) for stage in Stage})


def test_usage_distinguishes_exposure_read_citation_application(tmp_path):
    bundle = baseline(tmp_path)
    events = []
    usage = SkillUsage(bundle, 'op-1', Stage.DESIGN, events.append)
    usage.expose()
    assert [e['kind'] for e in events] == ['exposed', 'exposed']
    with pytest.raises(ValueError, match='read'):
        usage.cite('design', 'contract', 'decision-1')
    text = usage.read('design', 'contract')
    assert text == 'Do work.\n\n'
    usage.cite('design', 'contract', 'decision-1')
    usage.apply('design', 'contract', 'tool-receipt-1')
    assert [e['kind'] for e in events][-3:] == ['read', 'cited', 'applied']
    assert events[-1]['evidence_ref'] == 'tool-receipt-1'
    assert all(e['bundle_hash'] == bundle.digest for e in events)
    assert all(e['operation_id'] == 'op-1' for e in events)
    with pytest.raises(ValueError):
        usage.read('design', 'missing')
    with pytest.raises(ValueError):
        usage.apply('design', 'evidence', 'receipt')


def test_full_read_authorises_section_citation(tmp_path):
    usage = SkillUsage(baseline(tmp_path), 'op', Stage.TEST, lambda event: None)
    usage.expose()
    usage.read('design')
    usage.cite('design', 'evidence', 'decision')


def test_sink_failure_does_not_establish_read(tmp_path):
    events = []
    def sink(event):
        if event['kind'] == 'read':
            raise OSError('unavailable')
        events.append(event)
    usage = SkillUsage(baseline(tmp_path), 'op', Stage.DESIGN, sink)
    usage.expose()
    with pytest.raises(OSError):
        usage.read('design', 'contract')
    with pytest.raises(ValueError):
        usage.cite('design', 'contract', 'receipt')


def test_builtins_cover_stages_and_global_skills():
    bundle = builtin_bundle()
    for stage in Stage:
        assigned = bundle.skills_for(stage)
        assert assigned
        for name in assigned:
            document = bundle.document(name)
            assert document.content
        assert 'kdb' in assigned
    for name in ('memory-update', 'recovery', 'skill-writing', 'skill-evaluation'):
        assert bundle.document(name).content


def test_resolver_uses_host_receipts_and_checks_stage_section(tmp_path):
    events = []
    usage = SkillUsage(baseline(tmp_path), 'op', Stage.DESIGN, events.append)
    usage.expose()
    with pytest.raises(ValueError):
        usage.resolve_read(events[0]['receipt_id'], stage=Stage.DESIGN, skill='design')
    usage.read('design', 'contract')
    receipt = usage.last_read_receipt
    resolved = usage.resolve_read(receipt['receipt_id'], stage=Stage.DESIGN,
                                  skill='design', section='contract')
    assert resolved['content_hash'] == receipt['content_hash']
    events[-1]['kind'] = 'applied'  # Sink/caller mutation cannot alter host authority.
    assert usage.resolve_read(receipt['receipt_id'], stage=Stage.DESIGN,
                              skill='design', section='contract')['kind'] == 'read'
    for stage, section in ((Stage.TEST, 'contract'), (Stage.DESIGN, 'evidence')):
        with pytest.raises(ValueError):
            usage.resolve_read(receipt['receipt_id'], stage=stage, skill='design', section=section)
    with pytest.raises(ValueError):
        usage.resolve_read('agent-invented-id', stage=Stage.DESIGN, skill='design')


def test_read_requires_stage_exposure(tmp_path):
    usage = SkillUsage(baseline(tmp_path), 'op', Stage.DESIGN, lambda event: None)
    with pytest.raises(ValueError, match='exposed'):
        usage.read('design')


def test_mapping_detaches_mutable_inputs(tmp_path):
    bundle = baseline(tmp_path)
    assigned = ['design']
    mapping = [(stage, assigned) for stage in Stage]
    snapshot = SkillBundle(list(bundle.documents), mapping)
    assigned.append('kdb')
    mapping.clear()
    assert snapshot.skills_for(Stage.DESIGN) == ('design',)
    with pytest.raises(FrozenInstanceError):
        snapshot.documents[0].content = 'changed'
