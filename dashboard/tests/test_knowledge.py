import json
from types import SimpleNamespace

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.knowledge import KnowledgeDB, DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_REVISION
from benchmarks.ascend.subject_knowledge import SubjectKnowledge
from benchmarks.ascend.subject_memory import SubjectMemory
from dashboard.knowledge import answer_question


class Embeddings:
    model, revision = DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_REVISION
    def embed(self, texts):
        return [[float(text.lower().count(word)) for word in ('vector', 'matrix', 'pipeline')] for text in texts]


def knowledge(tmp_path):
    embeddings = Embeddings()
    path = tmp_path / 'knowledge.sqlite'
    reference = tmp_path / 'reference.txt'
    reference.write_text('Vector copies preserve all logical values.\n')
    with KnowledgeDB(path, embeddings) as database:
        database.index(tmp_path, [reference], collection='reference', language='ascend-c')
    paths = SimpleNamespace(kdb=path, collection='reference', root=tmp_path)
    store = ContentAddressedArtifacts(tmp_path / 'artifacts')
    source = store.put_bytes(b'copy all vector values\n')
    service = SubjectKnowledge(tmp_path, store, lambda: memory, paths=paths, embeddings=embeddings)
    memory = SubjectMemory(tmp_path / 'memory', b'initial', index=service.index)
    proposal = dict(operation='create', base_versions=[], summary='Vector copies preserve tails',
        applicability='Vector tails', preconditions='Bounded size', expectations='Equal values', limitations='One shape',
        before=None, before_unavailable_reason='First copy',
        after=dict(sha256=source, start_line=1, end_line=1, text='copy all vector values\n'), evidence=[source], citations=[])
    memory.accept('accepted', proposal, {})
    return service, memory, path


def test_standalone_question_queries_real_reference_and_accepted_catalog(tmp_path):
    service, memory, database_path = knowledge(tmp_path)
    calls = []
    def complete(role, messages, **options):
        context = json.loads(messages[1]['content'])
        calls.append((role, context))
        return json.dumps(dict(answer='Copy each logical value and handle the tail.',
                               sources=list(range(1, len(context['sources']) + 1))))
    before = memory.snapshot()
    result = answer_question(SimpleNamespace(knowledge=service, complete=complete), 'vector copies', max_seconds=10)
    collections = {row['citation']['collection'] for row in result['sources']}
    assert 'reference' in collections
    assert any(name.startswith('subject-memory-') for name in collections)
    with KnowledgeDB.open_read_only(database_path, Embeddings()) as database:
        for row in result['sources']:
            citation = row['citation']
            database.validate_citation(citation['collection'], chunk_id=citation['chunk_hash'],
                path=citation['path'], start_line=citation['start_line'], end_line=citation['end_line'])
    assert memory.snapshot() == before
    assert calls[0][0] == 'librarian' and not (tmp_path / 'actor').exists()


def test_answer_cannot_invent_source_identifiers(tmp_path):
    service, _, _ = knowledge(tmp_path)
    responses = iter([dict(answer='Unsupported', sources=[999]), dict(answer='Vector evidence', sources=[1])])
    calls = []
    def complete(role, messages, **options):
        calls.append(messages.copy())
        return json.dumps(next(responses))
    result = answer_question(SimpleNamespace(knowledge=service, complete=complete), 'vector', max_seconds=10)
    assert len(calls) == 2 and result['sources'][0]['index'] == 1


def test_no_matching_evidence_does_not_generate_an_unsupported_answer(tmp_path):
    memory = SubjectMemory(tmp_path / 'memory', b'initial')
    service = SubjectKnowledge(tmp_path, ContentAddressedArtifacts(tmp_path / 'artifacts'), lambda: memory)
    def forbidden(*args, **kwargs):
        raise AssertionError('No evidence must not invoke model answer generation')
    assert answer_question(SimpleNamespace(knowledge=service, complete=forbidden), 'vector')['sources'] == []


def test_question_route_uses_async_operations_with_idempotent_replay(tmp_path, monkeypatch):
    import time
    from contextlib import contextmanager
    from dashboard.app import create_app
    import dashboard.subject_services as binding
    service, _, _ = knowledge(tmp_path)
    calls = []
    def complete(role, messages, **options):
        calls.append(role)
        return json.dumps(dict(answer='Vector copies preserve values.', sources=[1]))
    @contextmanager
    def bind(*args, **kwargs):
        yield SimpleNamespace(knowledge=service, complete=complete)
    monkeypatch.setattr(binding, 'bind_subject', bind)
    config = tmp_path / 'dashboard.json'
    config.write_text(json.dumps(dict(controls=dict(enabled=True), subject_campaigns=[dict(
        id='research', target='a3', root=str(tmp_path))])))
    app = create_app(config, tmp_path / 'cache.sqlite')
    client = app.test_client()
    payload = dict(campaign_id='research', question='vector copies', max_seconds=10)
    try:
        assert client.post('/api/knowledge/ask', json=payload).status_code == 403
        token = client.get('/api/csrf').get_json()['token']
        headers = {'X-CSRF-Token': token, 'Idempotency-Key': 'knowledge-one'}
        response = client.post('/api/knowledge/ask', json=payload, headers=headers)
        assert response.status_code == 202
        operation_id = response.get_json()['operation_id']
        assert client.post('/api/knowledge/ask', json=payload, headers=headers).get_json()['operation_id'] == operation_id
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            operation = client.get('/api/operations/' + operation_id).get_json()
            if operation['status'] not in {'queued', 'running'}:
                break
            time.sleep(.01)
        assert operation['status'] == 'succeeded', operation
        assert operation['result']['sources'][0]['citation']['collection'].startswith(('reference', 'subject-memory-'))
        assert calls == ['librarian']
        assert not (tmp_path / 'actor').exists()
    finally:
        app.extensions['dashboard_controls'].close()
