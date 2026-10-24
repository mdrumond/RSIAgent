import json
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.knowledge import KnowledgeDB
from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
from benchmarks.a5kernels.phase1_provider import direct_profile
from benchmarks.ascend.subject_runtime import main, build_subject_campaign
from benchmarks.ascend.subjects import Subject
from test_a5_knowledge import FakeEmbeddings
from test_project_verifier_agent import memory, SOURCE, RESULT


def bound(root, *, kdb=None, embeddings=None):
    actor = SimpleNamespace(actor=SimpleNamespace(provider=object()), profile=direct_profile(MODEL_IDENTITIES[0]),
                            executor=SimpleNamespace(runtime_identity=None))
    return dict(actor=actor, provider=actor.actor.provider, profile=actor.profile,
        coordinator=SimpleNamespace(curriculum=SimpleNamespace(target_policy=None, query_knowledge=None,
                                                              unavailable_capabilities=())),
        paths=SimpleNamespace(kdb=kdb, collection='reference', root=root, arch='arch'), embeddings=embeddings)


@pytest.mark.parametrize('action', ['run', 'resume'])
def test_cli_dispatches_requested_action_and_subject_config(tmp_path, capsys, action):
    subjects, settings, seed = (tmp_path / name for name in ('subjects.json', 'settings.json', 'seed'))
    subjects.write_text(json.dumps([{'subject_id': 'copy', 'objective': 'Correct copies', 'min_waves': 1}]))
    settings.write_text(json.dumps({'phase1_arguments': ['--model', 'deepseek-flash']}))
    seed.write_bytes(b'initial memory')
    calls = []
    def builder(config, **options):
        calls.append((config, options))
        return SimpleNamespace(**{action: lambda: {'terminal': 'COMPLETE'}})
    assert main([action, '--root', str(tmp_path / 'run'), '--subjects', str(subjects),
                 '--config', str(settings), '--target', 'a3', '--initial-memory', str(seed),
                 '--max-seconds', '90'], builder=builder) == {'terminal': 'COMPLETE'}
    assert calls[0][0]['phase1_arguments'] == ['--model', 'deepseek-flash']
    assert calls[0][1]['subjects'] == (Subject('copy', 'Correct copies', 1),)
    assert calls[0][1]['initial_memory'] == b'initial memory'
    assert calls[0][1]['max_seconds'] == 90
    assert json.loads(capsys.readouterr().out)['terminal'] == 'COMPLETE'


def test_report_needs_no_provider_configuration_or_runtime(tmp_path, capsys):
    (tmp_path / 'subject-campaign.json').write_text(json.dumps({'terminal': 'COMPLETE'}))
    (tmp_path / 'subjects.json').write_text(json.dumps({'frontier': 1}))
    result = main(['report', '--root', str(tmp_path)], builder=lambda *a, **k: pytest.fail('runtime composed'))
    assert result['subjects']['frontier'] == 1
    assert json.loads(capsys.readouterr().out)['terminal'] == 'COMPLETE'


def test_runtime_freezes_configuration_before_recomposing(tmp_path):
    calls = []
    def compose(settings, **options):
        calls.append(settings)
        return bound(tmp_path)
    kwargs = dict(target='a3', root=tmp_path, subjects=[Subject('s', 'copies')],
                  initial_memory=b'', compose=compose)
    build_subject_campaign({}, **kwargs)
    with pytest.raises(ValueError, match='settings or target changed'):
        build_subject_campaign({'verifier_model': 'deepseek-flash'}, **kwargs)
    assert calls == [{}]


def test_accepted_memory_index_selects_only_active_versions(tmp_path):
    database_path = tmp_path / 'knowledge.sqlite'
    source = tmp_path / 'reference.txt'
    source.write_text('Vector increments preserve shape.\n')
    embeddings = FakeEmbeddings()
    with KnowledgeDB(database_path, embeddings) as database:
        database.index(tmp_path, [source], collection='reference', language='ascend-c')
    campaign = build_subject_campaign({}, target='a3', root=tmp_path / 'run',
        subjects=[Subject('s', 'copies')], initial_memory=b'',
        compose=lambda settings, **options: bound(options['root'], kdb=database_path, embeddings=embeddings))
    store = ContentAddressedArtifacts(campaign.root / 'artifacts')
    store.put_bytes(SOURCE); store.put_bytes(RESULT)
    proposal = dict(memory(), summary='Vector increments preserve shape')
    first = campaign.memory.accept('one', proposal, {})
    before = campaign.verifier.query_knowledge({'query': 'vector increments'}, remaining_seconds=60)
    assert [row['version'] for row in before['memories']] == [first['version']]
    replacement = dict(proposal, operation='append', base_versions=[first], summary='Vector increments and overflow')
    second = campaign.memory.accept('two', replacement, {})
    result = campaign.verifier.query_knowledge({'query': 'vector increments'}, remaining_seconds=60)
    assert [row['version'] for row in result['memories']] == [second['version']]
    collections = {citation['collection'] for citation in result['citations']}
    assert 'subject-memory-' + second['version'] in collections
    assert 'subject-memory-' + first['version'] not in collections
    assert 'reference' in collections


def test_real_direct_provider_role_callback_uses_private_elapsed_and_http_bounds(tmp_path, monkeypatch):
    import sys
    from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider, DirectCompletion
    from benchmarks.ascend.subject_runtime import bounded_completion
    profile = direct_profile(MODEL_IDENTITIES[0])
    now, requests = [0.0], []
    def transport(**options):
        requests.append(options['request'])
        return DirectCompletion('{"action":"continue"}', 3)
    provider = DirectCompletionProvider({profile.identity.credential_env: 'fixture-only'},
        models=[profile.identity.backend_model], transport=transport, clock=lambda: now[0])
    config = bound(tmp_path)
    config.update(provider=provider)
    campaign = build_subject_campaign({}, target='a3', root=tmp_path,
        subjects=[Subject('s', 'copies')], initial_memory=b'', compose=lambda *a, **k: config)
    reply = campaign.verifier.complete(({'role': 'user', 'content': 'Inspect'},), remaining_seconds=5)
    assert json.loads(reply)['action'] == 'continue'
    assert requests[0]['model'] == profile.model
    assert provider.retry_max_seconds == 480
    assert provider._transport is transport
    def late(**options):
        now[0] += 6
        return DirectCompletion('{}', 1)
    provider._transport = late
    with pytest.raises(TimeoutError):
        bounded_completion(provider, profile, (), 5)

    timeouts = []
    def client(**options):
        timeouts.append((options['timeout'], options['max_retries']))
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **request:
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))],
                            usage=SimpleNamespace(completion_tokens=1)))))
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=client))
    default_provider = DirectCompletionProvider({profile.identity.credential_env: 'fixture-only'},
        models=[profile.identity.backend_model], clock=lambda: 0)
    assert bounded_completion(default_provider, profile, (), 7) == '{}'
    assert timeouts == [(7, 0)]
