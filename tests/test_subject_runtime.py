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


def test_project_plan_ordinal_survives_other_projects_and_rejects_plan_change(tmp_path, monkeypatch):
    from benchmarks.ascend import subject_runtime
    from test_a5_phase1_adaptive import _project
    requests = []
    class Execution:
        def __init__(self, *args):
            pass
        def graph(self, request, **kwargs):
            requests.append(request)
            return request
    monkeypatch.setattr(subject_runtime, 'SubjectExecutor', Execution)
    campaign = build_subject_campaign({}, target='a3', root=tmp_path,
        subjects=[Subject('s', 'copies')], initial_memory=b'',
        compose=lambda *a, **k: bound(tmp_path))
    first = dict(project_id='first', spec=_project('first', 'gm-ub-movement'))
    second = dict(project_id='second', spec=_project('second', 'gm-ub-movement'))
    for item in (first, second, first):
        campaign.execute(item, campaign.memory.snapshot(), tmp_path / item['project_id'], remaining_seconds=30)
    assert [request.authored_ordinal for request in requests] == [1, 2, 1]
    with pytest.raises(ValueError, match='plan changed'):
        campaign.execute(dict(first, repairs=['different']), campaign.memory.snapshot(),
                         tmp_path / 'first', remaining_seconds=30)


def test_trial_arms_get_fresh_bindings_and_isolated_worker_workspaces(tmp_path, monkeypatch):
    from benchmarks.ascend import subject_runtime
    from test_a5_phase1_adaptive import _project
    worker = tmp_path / 'worker.json'
    worker.write_text(json.dumps(dict(execution='local', workspace=str(tmp_path / 'device'), physical_device=0)))
    composed = []
    def compose(settings, **options):
        composed.append((settings, options))
        return bound(options['root'])
    class Execution:
        def __init__(self, *args):
            pass
        def graph(self, request, **kwargs):
            return request
    monkeypatch.setattr(subject_runtime, 'SubjectExecutor', Execution)
    campaign = build_subject_campaign({'phase1_arguments': ['--worker-config', str(worker)]}, target='a3',
        root=tmp_path / 'campaign', subjects=[Subject('s', 'copies')], initial_memory=b'', compose=compose)
    item = dict(project_id='transfer', spec=_project('transfer', 'gm-ub-movement'))
    snapshot = campaign.memory.snapshot()
    a = campaign.execute_trial_arm(item, snapshot, tmp_path / 'a', remaining_seconds=60)
    b = campaign.execute_trial_arm(item, snapshot, tmp_path / 'b', remaining_seconds=60)
    replay = campaign.execute_trial_arm(item, snapshot, tmp_path / 'a', remaining_seconds=30)
    assert len(composed) == 4
    assert a.branch_root != b.branch_root and replay.branch_root == a.branch_root
    workers = [json.loads((tmp_path / arm / 'worker.json').read_bytes()) for arm in ('a', 'b')]
    assert workers[0]['workspace'] != workers[1]['workspace']
    assert workers[0]['physical_device'] == workers[1]['physical_device'] == 0
    assert a.pre_wave_memory == b.pre_wave_memory == snapshot
    assert composed[-1][1]['max_seconds'] == 60  # resume keeps the admitted allocation identity


def test_a5_catlass_optimizer_is_rejected_before_composition(tmp_path):
    calls = []
    with pytest.raises(ValueError, match='A5 Catlass optimizer subjects are unsupported'):
        build_subject_campaign({}, target='a5', root=tmp_path / 'run',
            subjects=[Subject('opt', 'optimize', task={'kind': 'optimizer', 'task': {}})],
            initial_memory=b'', compose=lambda *a, **k: calls.append(a))
    assert calls == []
    assert not (tmp_path / 'run').exists()


def test_wave_author_gets_reused_id_feedback_without_changing_frozen_context(tmp_path):
    from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider, DirectCompletion
    from test_a5_phase1_adaptive import _project, _wave
    requests = []
    profile = direct_profile(MODEL_IDENTITIES[0])
    def transport(**options):
        messages = options['request']['messages']
        requests.append(messages)
        project_id = 'repeat' if len(requests) == 1 else 'fresh'
        return DirectCompletion(_wave(messages[1]['content'], [_project(project_id, 'gm-ub-movement')]), 1)
    provider = DirectCompletionProvider({profile.identity.credential_env: 'fixture-only'},
        models=[profile.identity.backend_model], transport=transport)
    binding = bound(tmp_path)
    binding['provider'] = provider
    campaign = build_subject_campaign({}, target='a3', root=tmp_path,
        subjects=[Subject('s', 'copies')], initial_memory=b'', compose=lambda *a, **k: binding)
    items = campaign.select_wave(campaign.scheduler.context(), campaign.memory.snapshot(),
                                 {'repeat': {}}, remaining_seconds=30)
    assert [item['project_id'] for item in items] == ['fresh']
    assert len(requests) == 2
    assert requests[0][1] == requests[1][1]
    assert [row['role'] for row in requests[1][-2:]] == ['assistant', 'user']


def test_arm_composition_consumes_allocation_and_late_preflight_never_starts_actor(tmp_path, monkeypatch):
    from benchmarks.ascend import subject_runtime
    from test_a5_phase1_adaptive import _project
    now, requests = [0.0], []
    monkeypatch.setattr(subject_runtime.time, 'monotonic', lambda: now[0])
    worker = tmp_path / 'worker.json'
    worker.write_text(json.dumps(dict(execution='local', workspace=str(tmp_path / 'device'), physical_device=0)))
    parent = tmp_path / 'parent'
    def compose(settings, **options):
        if options['root'] != parent:
            now[0] += 4
        return bound(options['root'])
    class Execution:
        def __init__(self, *args):
            pass
        def graph(self, request, **kwargs):
            requests.append(request)
            return request
    monkeypatch.setattr(subject_runtime, 'SubjectExecutor', Execution)
    campaign = build_subject_campaign({'phase1_arguments': ['--worker-config', str(worker)]}, target='a3',
        root=parent, subjects=[Subject('s', 'copies')], initial_memory=b'', compose=compose)
    item = dict(project_id='transfer', spec=_project('transfer', 'gm-ub-movement'))
    campaign.execute_trial_arm(item, campaign.memory.snapshot(), tmp_path / 'valid', remaining_seconds=10)
    assert requests[0].remaining_seconds == 6
    with pytest.raises(TimeoutError):
        campaign.execute_trial_arm(item, campaign.memory.snapshot(), tmp_path / 'late', remaining_seconds=3)
    assert len(requests) == 1


@pytest.mark.parametrize('kind', ['config', 'worker-config', 'optimizer_worker_config'])
def test_same_path_runtime_json_mutation_is_rejected_before_recomposition(tmp_path, kind):
    referenced = tmp_path / 'runtime.json'
    referenced.write_text(json.dumps({'physical_device': 0, 'workspace': '/old'}))
    settings = ({'optimizer_worker_config': str(referenced)} if kind == 'optimizer_worker_config'
                else {'phase1_arguments': ['--' + kind, str(referenced)]})
    calls = []
    def compose(*args, **options):
        calls.append(options)
        return bound(options['root'])
    options = dict(target='a3', root=tmp_path / 'run', subjects=[Subject('s', 'copies')],
                   initial_memory=b'', compose=compose)
    build_subject_campaign(settings, **options)
    referenced.write_text(json.dumps({'physical_device': 1, 'workspace': '/new'}))
    with pytest.raises(ValueError, match='settings or target changed'):
        build_subject_campaign(settings, **options)
    assert len(calls) == 1


def test_optimizer_reads_initial_seed_and_published_source_evidence_via_real_tools(tmp_path, monkeypatch):
    from benchmarks.ascend import subject_runtime
    from optimizer.jobs import JobStore
    from optimizer.loop import OptimizationLoop
    observed = []
    class Execution:
        def __init__(self, *args):
            pass
        def optimizer(self, directory, task, snapshot, **kwargs):
            jobs = JobStore(directory / 'jobs')
            job = jobs.create('def f(x): return x', 'f', {'args': []}, snapshot)
            loop = OptimizationLoop(jobs, job['job_id'], None, None)
            for action in [dict(action='read-memory', index=0), dict(action='read-memory', index=1),
                           dict(action='read-artifact', sha256=snapshot['memory_entries'][0]['artifact_sha256'], offset=0),
                           dict(action='read-artifact', sha256=snapshot['memory_entries'][1]['proposal']['after']['sha256'], offset=0),
                           dict(action='read-artifact', sha256=snapshot['memory_entries'][1]['proposal']['evidence'][0], offset=0)]:
                loop._tool(action, snapshot)
                observed.append(loop.state['history'][-1]['content'])
            return snapshot
    monkeypatch.setattr(subject_runtime, 'SubjectExecutor', Execution)
    seed = b'Initial host guidance\n'
    campaign = build_subject_campaign({}, target='a3', root=tmp_path / 'run',
        subjects=[Subject('s', 'optimization')], initial_memory=seed,
        compose=lambda *args, **options: bound(options['root']))
    campaign.artifacts.put_bytes(SOURCE); campaign.artifacts.put_bytes(RESULT)
    campaign.memory.accept('lesson', memory(), {})
    item = dict(project_id='optimizer', task=dict(kind='optimizer', task={}))
    snapshot = campaign.execute(item, campaign.memory.snapshot(), tmp_path / 'execution', remaining_seconds=30)
    assert observed[0]['text'] == seed.decode()
    assert observed[1]['proposal']['after']['text'] == SOURCE.decode()
    assert observed[2]['text'] == seed.decode()
    assert observed[3]['text'] == SOURCE.decode()
    assert observed[4]['text'] == RESULT.decode()


def selection_context(messages):
    for line in messages[1]['content'].splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and 'next_wave_id' in value:
            return value
    raise AssertionError('No frozen selection context received')


def test_generation_bounds_rich_memories_and_preserves_initial_seed_identity(tmp_path):
    import hashlib
    from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider, DirectCompletion
    from benchmarks.ascend.agent_context import ContextBudget, request_bytes
    from benchmarks.ascend.subject_memory import digest
    from test_a5_phase1_adaptive import _project, _wave
    requests = []
    profile = direct_profile(next(row for row in MODEL_IDENTITIES if row.backend_model.value == 'deepseek-flash'))
    def transport(**options):
        requests.append(options['request'])
        payload = selection_context(options['request']['messages'])
        return DirectCompletion(_wave(json.dumps(payload), [_project('fresh', 'gm-ub-movement')]), 1)
    provider = DirectCompletionProvider({profile.identity.credential_env: 'fixture-only'},
        models=[profile.identity.backend_model], transport=transport,
        context_budgets={'deepseek-flash': ContextBudget(40000, 32768, 49152)})
    binding = bound(tmp_path)
    binding.update(provider=provider, profile=profile)
    seed = b'Configured initial guidance\n' * 5000
    campaign = build_subject_campaign({}, target='a3', root=tmp_path / 'run',
        subjects=[Subject('s', 'copies')], initial_memory=seed, compose=lambda *a, **k: binding)
    code = b'// retained example ' + b'x' * 20000 + b'\n'
    code_sha = campaign.artifacts.put_bytes(code)
    campaign.artifacts.put_bytes(RESULT)
    for index in range(12):
        learned = dict(memory(), summary='Vector lesson ' + str(index))
        learned['after'] = dict(sha256=code_sha, start_line=1, end_line=1, text=code.decode())
        campaign.memory.accept('lesson-' + str(index), learned,
            {'subject_id': 's' if index % 2 == 0 else 'earlier', 'long_audit': 'z' * 30000})
    snapshot = campaign.memory.snapshot()
    items = campaign.select_wave(campaign.scheduler.context(), snapshot, {}, remaining_seconds=60)
    assert [item['project_id'] for item in items] == ['fresh']
    assert request_bytes(requests[0]) < 49152
    payload = selection_context(requests[0]['messages'])
    header, *synopsis = payload['curriculum_memory']
    assert header['snapshot_sha256'] == payload['pre_wave_memory_sha256'] == hashlib.sha256(snapshot).hexdigest()
    assert header['initial_memory']['sha256'] == hashlib.sha256(seed).hexdigest()
    assert seed.startswith(header['initial_memory']['preview'].encode())
    assert campaign.artifacts.read_bytes(header['snapshot_sha256']) == snapshot
    assert campaign.artifacts.read_bytes(header['initial_memory']['sha256']) == seed
    assert synopsis and synopsis[0]['subject_id'] == 's'
    records = {record['memory_id']: record for record in campaign.memory.records()}
    for row in synopsis:
        original = records[row['memory_id']]
        assert row['version'] == digest(original)
        assert row['summary'] == original['proposal']['summary']
        assert row['preconditions'] == original['proposal']['preconditions']
        assert row['expectations'] == original['proposal']['expectations']


def test_generation_retrieval_pins_snapshot_and_bounds_full_code_results(tmp_path):
    from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider, DirectCompletion
    from test_a5_phase1_adaptive import _project, _wave
    database_path = tmp_path / 'knowledge.sqlite'
    reference = tmp_path / 'reference.txt'
    reference.write_text('Vector increment reference\n')
    embeddings = FakeEmbeddings()
    with KnowledgeDB(database_path, embeddings) as database:
        database.index(tmp_path, [reference], collection='reference', language='ascend-c')
    requests, payloads, versions = [], [], {}
    profile = direct_profile(MODEL_IDENTITIES[0])
    def transport(**options):
        messages = options['request']['messages']
        requests.append(messages)
        if len(requests) == 1:
            payloads.append(selection_context(messages))
            versions['new'] = campaign.memory.accept('two', dict(learned, operation='append',
                base_versions=[versions['old']], summary='Vector new accepted replacement'), {})
            return DirectCompletion(json.dumps({'knowledge_query': {'query': 'vector', 'limit': 5}}), 1)
        page = json.loads(messages[-1]['content'])['knowledge_query']
        payloads.append(page)
        return DirectCompletion(_wave(json.dumps(payloads[0]), [_project('fresh', 'gm-ub-movement')]), 1)
    provider = DirectCompletionProvider({profile.identity.credential_env: 'fixture-only'},
        models=[profile.identity.backend_model], transport=transport)
    binding = bound(tmp_path, kdb=database_path, embeddings=embeddings)
    binding.update(provider=provider, profile=profile)
    campaign = build_subject_campaign({}, target='a3', root=tmp_path / 'run', subjects=[Subject('s', 'vectors')],
        initial_memory=b'initial guidance', compose=lambda *a, **k: binding)
    code = b'// retained example ' + b'x' * 20000 + b'\n'
    campaign.artifacts.put_bytes(RESULT)
    learned = dict(memory(), summary='Vector original accepted lesson')
    learned['after'] = dict(sha256=campaign.artifacts.put_bytes(code), start_line=1, end_line=1, text=code.decode())
    versions['old'] = campaign.memory.accept('one', learned, {})
    snapshot = campaign.memory.snapshot()
    campaign.select_wave(campaign.scheduler.context(), snapshot, {}, remaining_seconds=60)
    page = payloads[1]
    collections = {row['citation']['collection'] for row in page['results']}
    old_collection = 'subject-memory-' + versions['old']['version']
    assert old_collection in collections
    assert 'subject-memory-' + versions['new']['version'] not in collections
    position = next(index for index, row in enumerate(page['results']) if row['citation']['collection'] == old_collection)
    pointer = page['search_details']['artifacts'][position]
    assert pointer['truncated'] is True
    full = campaign.artifacts.read_bytes(pointer['sha256'])
    assert json.loads(full)['proposal']['summary'] == learned['summary']
    assert len(full) > len(page['results'][position]['text'].encode())
    assert page['search_details']['snapshot_sha256'] == payloads[0]['pre_wave_memory_sha256']
