import hashlib
import json

import pytest

from benchmarks.ascend.runtime_allocation import admit_identity, permitted_devices


def raw(device, **extra):
    return json.dumps(dict(physical_device=device, runtime='named', compiler='pinned', **extra)).encode()


def test_device_change_preserves_semantics_and_records_allocation(tmp_path):
    first = admit_identity(tmp_path, {'target':'a3'}, {'config':raw(0)})
    assert admit_identity(tmp_path, {'target':'a3'}, {'config':raw(2)}) == first
    history = json.loads((tmp_path/'execution-allocations.json').read_bytes())
    assert [row['allocation']['config']['physical_device'] for row in history] == [0, 2]
    with pytest.raises(ValueError, match='settings or target'):
        admit_identity(tmp_path, {'target':'a3'}, {'config':raw(2, arch='other')})


def test_legacy_migration_authenticates_exact_prior_bytes(tmp_path):
    before = raw(0)
    legacy = {'target':'a3', 'referenced_json_sha256':{'config':hashlib.sha256(before).hexdigest()}}
    (tmp_path/'subject-runtime.json').write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match='settings or target'):
        admit_identity(tmp_path, {'target':'a3'}, {'config':raw(1)})
    result = admit_identity(tmp_path, {'target':'a3'}, {'config':raw(1)}, previous={'config':before})
    assert result['schema'] == 'subject-runtime-v2'
    assert json.loads(next((tmp_path/'identity-history').glob('*.json')).read_bytes()) == legacy


def test_unresolved_allocation_cannot_move_until_terminal_observation(tmp_path):
    admit_identity(tmp_path, {'target':'a3'}, {'config':raw(0)})
    op = tmp_path/'transport/operation.json';op.parent.mkdir()
    op.write_text(json.dumps({'state':'cancelling','handle':'remote:t:job:owned'}))
    with pytest.raises(ValueError, match='unsettled'):
        admit_identity(tmp_path, {'target':'a3'}, {'config':raw(1)})
    (tmp_path/'remote-observations.json').write_text(json.dumps([{'state':'cancelled','handle':'remote:t:job:owned'}]))
    admit_identity(tmp_path, {'target':'a3'}, {'config':raw(1)})
    assert json.loads((tmp_path/'execution-allocations.json').read_bytes())[-1]['allocation']['config']['physical_device'] == 1


def test_campaign_permissions():
    assert permitted_devices([0, 2, 3, 7], [0, 1, 2]) == [0, 2]
    assert permitted_devices([0, 2, 3, 7]) == [0, 2, 3, 7]


def test_subject_campaign_recomposes_after_permitted_device_only_migration(tmp_path):
    from benchmarks.ascend.subject_runtime import build_subject_campaign
    from benchmarks.ascend.subjects import Subject
    from test_subject_runtime import bound
    path=tmp_path/'native.json';path.write_bytes(raw(0))
    settings={'phase1_arguments':['--config',str(path)]};root=tmp_path/'campaign'
    kwargs=dict(target='a3',root=root,subjects=[Subject('s','copies')],initial_memory=b'',
                compose=lambda settings,**options:bound(options['root']))
    first = build_subject_campaign(settings,**kwargs)
    first.memory_gate.close()
    path.write_bytes(raw(1))
    second = build_subject_campaign(settings,**kwargs)
    second.memory_gate.close()
    assert json.loads((root/'execution-allocations.json').read_bytes())[-1]['allocation']['config']['physical_device']==1


def test_health_workspace_and_device_permissions_are_recorded_allocation(tmp_path):
    first=admit_identity(tmp_path,{'target':'a3'},{'config':raw(0)})
    next_config=raw(0,health_workspace='/owned/incoming',allowed_devices=[0,1,2])
    assert admit_identity(tmp_path,{'target':'a3'},{'config':next_config})==first
    allocation=json.loads((tmp_path/'execution-allocations.json').read_bytes())[-1]['allocation']['config']
    assert allocation['health_workspace']=='/owned/incoming' and allocation['allowed_devices']==[0,1,2]
