"""Production composition reaches real SDK role tools without direct completion."""
import hashlib
import json
from types import SimpleNamespace

from benchmarks.ascend.production import compose_adaptive_controller
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumContext
from optimizer.jobs import canonical
from test_native_pi_filesystem import scripted_provider
from test_native_pi_programming_roles import registered_profile, redirect


class Credentials:
    def coding_environment(self):
        return {'DEEPSEEK_API_KEY': 'fixture-secret'}

    def complete(self, *args, **kwargs):
        raise AssertionError('production composition must not call direct completion')


def test_adaptive_production_retrieves_and_validates_inside_native_role(tmp_path, monkeypatch, scripted_provider):
    url, calls, requests = scripted_provider
    redirect(monkeypatch, url)
    queries = []
    citation = SimpleNamespace(as_dict=lambda:dict(path='api.md', start_line=1, end_line=2))
    def query(action, *, check_remaining):
        assert check_remaining() > 0
        queries.append(action.query)
        return [SimpleNamespace(citation=citation, text='Observed API limitation', score=1.)]
    _, coordinator = compose_adaptive_controller(root=tmp_path, objective='Investigate observed failure',
        initial_memory=b'', executor=None, actor=None, profile=registered_profile(), provider=Credentials(),
        immutable_context=b'', project_actor=lambda project:None, curriculum_query_knowledge=query)
    calls.extend([('knowledge_query', dict(query='actual limitation')),
        ('submit_curriculum_proposal', dict(decision='STALLED', payload=dict(reason='Observed runtime unavailable')))])
    context = CurriculumContext('Investigate observed failure', 1, 'a'*64, (), (),
        feedback=({'error':'runtime unavailable'},), remaining_seconds=30)
    result = coordinator.curriculum.select(context)
    assert result.status == 'STALLED'
    assert queries == ['actual limitation']
    assert any(message['role']=='tool' and 'Observed API limitation' in message['content']
               for message in requests[-1]['messages'])
    assert list((tmp_path/'native-curriculum').glob('*/closure.json'))


def test_dashboard_default_subject_binding_uses_native_role_and_retains_evidence(tmp_path, monkeypatch, scripted_provider):
    from dashboard import subject_services
    from benchmarks.ascend import subject_runtime
    from core.artifacts import ContentAddressedArtifacts
    url, calls, requests = scripted_provider
    redirect(monkeypatch, url)
    settings = {}
    config = tmp_path/'settings.json'
    config.write_text(json.dumps(settings))
    initial = tmp_path/'seed.json'
    initial.write_text('{}')
    entry = dict(target='a3', config=str(config), initial_memory=str(initial))
    (tmp_path/'subject-runtime.json').write_text(json.dumps(dict(target='a3', max_seconds=30,
        settings_sha256=hashlib.sha256(canonical(settings)).hexdigest(), referenced_json_sha256={})))
    monkeypatch.setattr(subject_services, 'configured_subject', lambda *args:(entry, tmp_path))
    monkeypatch.setattr(subject_runtime, 'compose_production', lambda *args, **kwargs:dict(
        provider=Credentials(), profile=registered_profile(), coordinator=SimpleNamespace(
            curriculum=SimpleNamespace(target_policy=None))))
    value = dict(answer='Retained documentation supports this answer', sources=[1])
    calls.append(('submit_curriculum_proposal', dict(value=value)))
    with subject_services.bind_subject('ignored', 'campaign', max_seconds=30) as services:
        response = services.complete('librarian', [{'role':'user', 'content':'Explain observed failure'}], remaining_seconds=20)
    assert json.loads(response) == value
    assert len(requests) == 1
    role_root = next((tmp_path/'dashboard-native-roles'/'librarian').iterdir())
    closure = json.loads((role_root/'closure.json').read_text())
    assert closure['provider_settled'] and closure['process_exit'] == 0
    retained = ContentAddressedArtifacts(tmp_path/'artifacts')
    assert closure['artifact_refs']
    for reference in closure['artifact_refs']:
        assert retained.read_bytes(reference)
