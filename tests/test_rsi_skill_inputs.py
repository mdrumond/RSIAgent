"""Immutable baseline edits and actual request propagation; no provider calls."""
import json
from pathlib import Path

import pytest

from rsi.cli import main
from rsi.configuration import set_configuration, show_configuration
from rsi.skill_inputs import inspect_skills, update_skills
from rsi.skills import SkillBundle, builtin_bundle
from core.artifacts import ContentAddressedArtifacts


def test_learning_metadata_uses_shared_contract_and_preserves_valid_bundle():
    from dataclasses import replace
    from rsi.learning import _skill_metadata
    bundle = builtin_bundle()
    assert _skill_metadata(bundle) is bundle
    invalid = replace(bundle, documents=(replace(bundle.documents[0], content='No metadata'),
                                         *bundle.documents[1:]))
    with pytest.raises(ValueError, match='metadata'):
        _skill_metadata(invalid)


def test_learning_missing_yaml_remains_host_failure_not_proposal_rejection(monkeypatch):
    import builtins
    from rsi.learning import ProposalRejected, validate_skill_proposal
    bundle = builtin_bundle()
    original_import = builtins.__import__
    def without_yaml(name, *args, **kwargs):
        if name == 'yaml':
            raise ModuleNotFoundError('host YAML unavailable', name='yaml')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_yaml)
    wire = {'rationale': 'Retain validated baseline', 'evidence_ids': ['outcome'], 'edits': []}
    with pytest.raises(ModuleNotFoundError) as caught:
        validate_skill_proposal(bundle, wire, {'outcome': {'kind': 'correctness'}}, {})
    assert not isinstance(caught.value, ProposalRejected)


@pytest.fixture
def baseline(tmp_path):
    root = tmp_path / 'instance'
    (root / 'config').mkdir(parents=True)
    bundle = builtin_bundle()
    for doc in bundle.documents:
        path = root / 'skills' / doc.name / 'SKILL.md'
        path.parent.mkdir(parents=True)
        path.write_text(doc.content)
    (root / 'config/map.json').write_text(json.dumps(bundle.to_dict()['mapping']))
    (root / 'config/instance.json').write_text(json.dumps(dict(
        machinery='machinery', kdb='memory/kdb.sqlite', skills='skills', skill_map='config/map.json')))
    (root / 'instance.json').write_text('{}')
    return root, bundle


def test_edit_publishes_new_baseline_and_preserves_prior_capture(baseline):
    root, original = baseline
    before = inspect_skills(root)
    content = original.document('debug').content + '\n## Numeric checks\nCheck error against Decimal.\n'
    result = update_skills(root, name='debug', content=content, expect_bundle=before['bundle_ref'])
    assert result['bundle_ref'] != before['bundle_ref']
    assert original.document('debug').content != content
    assert (root / 'skills/debug/SKILL.md').read_text() == original.document('debug').content
    assert inspect_skills(root, 'debug')['documents'][0]['content'] == content
    frozen = SkillBundle.from_dict(ContentAddressedArtifacts(root / 'artifacts').read_json(result['bundle_ref']))
    assert frozen.document('debug').content == content
    payload = show_configuration(root)['payload']
    assert str(Path(payload['baseline_skills']).parent).endswith('skill-bundles')


def test_invalid_content_and_mapping_never_change_configuration(baseline):
    root, original = baseline
    before = show_configuration(root)
    with pytest.raises(ValueError, match='metadata'):
        update_skills(root, name='debug', content='## Contract\nNo metadata\n')
    mapping = original.to_dict()['mapping']
    mapping['design'] = ['missing-skill']
    with pytest.raises(ValueError):
        update_skills(root, mapping=mapping)
    assert show_configuration(root) == before
    with pytest.raises(ValueError, match='conflict'):
        update_skills(root, name='debug', content=original.document('debug').content, expect_bundle='stale')


def test_map_edit_validates_whole_bundle_and_captures_new_snapshot(baseline):
    root, original = baseline
    mapping = original.to_dict()['mapping']
    mapping['design'] = ['design', 'debug', 'kdb']
    result = update_skills(root, mapping=mapping)
    after = inspect_skills(root)
    assert after['mapping']['design'] == ['design', 'debug', 'kdb']
    assert after['bundle_ref'] == result['bundle_ref']
    assert original.skills_for('design') == ('design', 'kdb')


def test_revision_race_preserves_concurrent_editor_without_partial_pointer(baseline, monkeypatch):
    root, original = baseline
    import rsi.skill_inputs as module
    setter = module.set_configuration
    def race(instance, payload, *, expect_revision):
        current = show_configuration(instance)['payload']
        current['prompts']['actor'] = 'concurrent steering'
        setter(instance, current)
        return setter(instance, payload, expect_revision=expect_revision)
    monkeypatch.setattr(module, 'set_configuration', race)
    with pytest.raises(ValueError, match='revision conflict'):
        update_skills(root, name='debug', content=original.document('debug').content + '\nReviewed.\n')
    after = show_configuration(root)['payload']
    assert after['prompts']['actor'] == 'concurrent steering'
    assert 'baseline_skills' not in after


def test_cli_get_map_and_builtin_are_real_bundle_data(baseline, capsys):
    root, original = baseline
    assert main(['--instance', str(root), 'skills', 'get', 'design']) == 0
    assert json.loads(capsys.readouterr().out)['documents'][0]['content'] == original.document('design').content
    assert main(['--instance', str(root), 'skills', 'map', 'inspect']) == 0
    assert json.loads(capsys.readouterr().out)['mapping'] == original.to_dict()['mapping']


def test_manual_tune_resolves_retained_host_project_and_guidance(baseline, monkeypatch):
    root, _ = baseline
    from rsi import jobs
    store = ContentAddressedArtifacts(root / 'artifacts')
    task = root / 'runs/application/projects/op1'
    task.mkdir(parents=True)
    reference = store.put_json(dict(root=str(task), operation_id='op1'))
    (task / 'service-result.json').write_text(json.dumps({'reference': reference}))
    (root / 'runs/application/service.json').write_text(json.dumps({'spec': {'mode': 'phase2'}}))
    captured = []
    monkeypatch.setattr(jobs, 'start', lambda instance, request: captured.append(request) or {})
    for value in ('application/op1', reference):
        assert main(['--instance', str(root), 'skills', 'tune', '--project', value,
                     '--guidance', 'Avoid previous numeric cancellation']) == 0
    assert captured == [dict(action='tune', mode='phase2', cycle_id='application',
        steering={'project_ref': reference, 'user_guidance': 'Avoid previous numeric cancellation'})] * 2
    wrong = store.put_json(dict(root=str(task), operation_id='some-other-operation'))
    assert main(['--instance', str(root), 'skills', 'tune', '--project', wrong,
                 '--guidance', 'no invented receipts']) == 2
    assert len(captured) == 2


def test_new_skill_can_be_added_then_mapped_without_changing_original(baseline):
    root, original = baseline
    content = '---\nname: numeric-proof\ndescription: Independent numeric evidence.\n---\n## Contract\nCheck Decimal.\n'
    update_skills(root, name='numeric-proof', content=content)
    mapping = original.to_dict()['mapping']
    mapping['debug'].append('numeric-proof')
    update_skills(root, mapping=mapping)
    assert inspect_skills(root)['mapping']['debug'][-1] == 'numeric-proof'
    assert len(original.documents) + 1 == len(inspect_skills(root)['documents'])


def test_capability_discovery_enables_implemented_editor_and_tuning_routes(baseline, capsys):
    root, _ = baseline
    assert main(['--instance', str(root), 'capabilities']) == 0
    capabilities = json.loads(capsys.readouterr().out)
    assert capabilities['manual_tuning'] and capabilities['skill_content_editing']
    assert capabilities['job_accounting']


def test_resuming_manual_tuning_preserves_scope_and_durable_request_identity(baseline, monkeypatch):
    import uuid
    root, _ = baseline
    from rsi import jobs
    from rsi.configuration import _publish, _bytes
    _, directory = jobs._directory(root)
    identity = str(uuid.uuid4())
    value = dict(job_id=identity, state='failed', process=None, configuration=show_configuration(root),
        request=dict(action='tune', mode='phase2', cycle_id='application', request_id=identity,
                     steering={'project_ref': 'retained', 'user_guidance': 'recover'}))
    _publish(jobs._path(directory, identity), _bytes(value))
    requests = []
    monkeypatch.setattr(jobs, 'start', lambda instance, request: requests.append(request) or {})
    assert main(['--instance', str(root), 'phase2', 'resume', '--job', identity]) == 0
    assert requests[0]['request_id'] == identity
    assert requests[0]['origin_action'] == 'tune'
    assert requests[0]['cycle_id'] == 'application'
