"""Real KDB and content-addressed artifacts across accepted/frozen memory views."""
import json
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.knowledge import KnowledgeDB
from benchmarks.ascend.subject_knowledge import SubjectKnowledge, memory_artifacts
from benchmarks.ascend.subject_memory import SubjectMemory, digest
from test_a5_knowledge import FakeEmbeddings


SOURCE = b'def kernel(x):\n    return x + 1\n'
RESULT = b'{"verified":true,"cases":3}'


def proposal(store, summary='Vector increments preserve shape'):
    source, result = store.put_bytes(SOURCE), store.put_bytes(RESULT)
    return dict(operation='create', base_versions=[], summary=summary,
        applicability='Integer vector increment', preconditions='No overflow',
        expectations='Input plus one', limitations='No overflow proof', before=None,
        before_unavailable_reason='First source',
        after=dict(sha256=source, start_line=1, end_line=2, text=SOURCE.decode()),
        evidence=[result], citations=[])


def setup(tmp_path, *, enabled=True, embeddings=None, reference_query=None, clock=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = ContentAddressedArtifacts(tmp_path / 'artifacts')
    embeddings = embeddings or FakeEmbeddings()
    database_path = tmp_path / 'knowledge.sqlite'
    if enabled:
        source = tmp_path / 'reference.txt'
        source.write_text('Vector increment reference\n')
        with KnowledgeDB(database_path, embeddings) as database:
            database.index(tmp_path, [source], collection='reference', language='ascend-c')
    paths = SimpleNamespace(kdb=database_path if enabled else None, collection='reference', root=tmp_path)
    service = SubjectKnowledge(tmp_path, store, lambda: memory, paths=paths, embeddings=embeddings,
        reference_query=reference_query, **({'clock': clock} if clock else {}))
    memory = SubjectMemory(tmp_path / 'memory', b'initial guidance', index=service.index, ledger=service.ledger)
    return service, memory, store, database_path


def context(tmp_path, snapshot):
    return SimpleNamespace(request=SimpleNamespace(pre_wave_memory=snapshot, branch_root=tmp_path,
                           spec=SimpleNamespace(project_id='project')),
                           artifacts=ContentAddressedArtifacts(tmp_path / 'artifacts'),
                           access_event_id='query-1', actor_id=str(tmp_path), actor_role='writer')


def test_accepted_index_queries_exact_active_versions_and_validated_citations(tmp_path):
    service, memory, store, path = setup(tmp_path)
    first_proposal = proposal(store)
    first = memory.accept('one', first_proposal, {})
    initial = service.query({'query': 'vector increments'}, remaining_seconds=60)
    assert [row['version'] for row in initial['memories']] == [first['version']]
    second = memory.accept('two', dict(first_proposal, operation='append', base_versions=[first],
                                     summary='Vector increments and overflow'), {})
    result = service.query({'query': 'vector increments'}, remaining_seconds=60)
    assert [row['version'] for row in result['memories']] == [second['version']]
    collections = {citation['collection'] for citation in result['citations']}
    assert 'subject-memory-' + second['version'] in collections
    assert 'subject-memory-' + first['version'] not in collections
    assert 'reference' in collections
    with KnowledgeDB.open_read_only(path, FakeEmbeddings()) as database:
        for citation in result['citations']:
            database.validate_citation(citation['collection'], chunk_id=citation['chunk_hash'],
                path=citation['path'], start_line=citation['start_line'], end_line=citation['end_line'])


def test_actor_view_remains_frozen_after_live_catalog_replacement_and_retains_results(tmp_path):
    service, memory, store, _ = setup(tmp_path)
    original = proposal(store)
    first = memory.accept('one', original, {})
    frozen = memory.snapshot()
    second = memory.accept('two', dict(original, operation='append', base_versions=[first]), {})
    actor = context(tmp_path / 'actor', frozen)
    result = service.actor_query({'query': 'vector increments'}, actor, remaining_seconds=60)
    collections = {citation.collection for citation in result.citations}
    assert 'subject-memory-' + first['version'] in collections
    assert 'subject-memory-' + second['version'] not in collections
    assert result.admitted_artifacts
    for artifact in result.admitted_artifacts:
        assert actor.artifacts.read_bytes(artifact)
    assert service.read_artifact(original['after']['sha256'], remaining_seconds=60) == SOURCE



def test_kdb_off_preserves_reachable_code_and_evidence_without_embedding(tmp_path):
    class NoEmbeddings:
        def embed(self, texts):
            pytest.fail('disabled KDB requested embeddings')
    service, memory, store, database = setup(tmp_path, enabled=False, embeddings=NoEmbeddings())
    learned = proposal(store)
    before = store.put_bytes(b'def kernel(x):\n    return x\n')
    learned.update(before=dict(sha256=before, start_line=1, end_line=2, text='def kernel(x):\n    return x\n'),
                   before_unavailable_reason=None)
    log, review = store.put_bytes(b'actor log'), store.put_bytes(b'verifier review')
    provenance = dict(request=dict(artifacts={'log': log}), review_sha256=review)
    memory.accept('one', learned, provenance)
    result = service.query({'query': 'vector increments'}, remaining_seconds=60)
    reachable = set(result['memories'][0]['artifacts'].values())
    record = memory.records()[0]
    document = result['memories'][0]['artifacts']['memory-document']
    assert document == digest(record)
    assert json.loads(service.read_artifact(document, remaining_seconds=60)) == record
    assert reachable == {learned['after']['sha256'], learned['evidence'][0], before, log, review, document}
    assert memory_artifacts(record) == reachable - {document}
    assert result['hits'] == result['citations'] == []
    assert not database.exists()


def test_query_forwards_elapsed_check_to_reference_callback(tmp_path):
    now, calls = [0.0], []
    def reference(action, *, check_remaining):
        calls.append(action.query)
        check_remaining()
        now[0] += 4
        check_remaining()
        return ()
    service, memory, _, _ = setup(tmp_path, enabled=False, reference_query=reference, clock=lambda: now[0])
    before = memory.snapshot()
    with pytest.raises(TimeoutError):
        service.query({'query': 'vector'}, remaining_seconds=3)
    assert calls == ['vector']
    assert memory.snapshot() == before


def test_indexing_the_same_version_reuses_existing_embeddings(tmp_path):
    class Recording(FakeEmbeddings):
        calls = 0
        def embed(self, texts):
            self.calls += 1
            return super().embed(texts)
    embeddings = Recording()
    service, memory, store, _ = setup(tmp_path, embeddings=embeddings)
    accepted = memory.accept('one', proposal(store), {})
    count = embeddings.calls
    service.index(accepted['version'], memory.records()[0])
    assert embeddings.calls == count


@pytest.mark.parametrize('frozen', [False, True])
def test_semantic_memory_candidate_is_selected_without_query_token_overlap(tmp_path, frozen):
    class Semantic(FakeEmbeddings):
        def embed(self, texts):
            return [[1., 0.] if text == 'prismatic' or 'Stateful relocation' in text else [0., 1.]
                    for text in texts]
    service, memory, store, _ = setup(tmp_path, embeddings=Semantic())
    for index in range(6):
        memory.accept('noise-' + str(index), proposal(store, 'Distractor ' + str(index)), {})
    selected = memory.accept('target', proposal(store, 'Stateful relocation'), {})
    record = memory.read(selected['version'])
    assert 'prismatic' not in json.dumps(record).casefold()
    result = service.query({'query': 'prismatic', 'limit': 1}, remaining_seconds=60,
        **({'snapshot': memory.snapshot(), 'branch_root': tmp_path / 'actor'} if frozen else {}))
    assert [row['memory_id'] for row in result['memories']] == ['target']
    assert result['hits'][0]['citation']['collection'] == 'subject-memory-' + selected['version']


def test_reference_prefix_cannot_starve_memory_with_actor_limit_five(tmp_path):
    from benchmarks.a5kernels.knowledge_agent import validate_knowledge_hit
    def reference(action, *, check_remaining):
        hits = []
        with KnowledgeDB.open_read_only(database_path, FakeEmbeddings()) as database:
            for index in range(6):
                check_remaining()
                collection = 'ref-' + str(index)
                hits.extend(validate_knowledge_hit(database, collection, hit)
                    for hit in database.query(collection, action.query, limit=1))
        return hits
    service, memory, store, database_path = setup(tmp_path, reference_query=reference)
    with KnowledgeDB(database_path, FakeEmbeddings()) as database:
        for index in range(6):
            document = tmp_path / ('reference-' + str(index) + '.txt')
            document.write_text('Vector reference topic ' + str(index))
            database.index(tmp_path, [document], collection='ref-' + str(index), language='ascend-c')
    accepted = memory.accept('learned', proposal(store), {})
    result = service.actor_query({'query': 'vector', 'limit': 5}, context(tmp_path / 'actor', memory.snapshot()),
                                 remaining_seconds=60)
    assert len(result.citations) == 5
    assert any(citation.collection == 'subject-memory-' + accepted['version'] for citation in result.citations)
    assert any(citation.collection.startswith('ref-') for citation in result.citations)


@pytest.mark.parametrize('frozen', [False, True])
def test_large_memory_document_pointer_preserves_exact_original_record(tmp_path, frozen):
    from optimizer.jobs import canonical
    service, memory, store, _ = setup(tmp_path, enabled=False)
    learned = proposal(store, 'largevector relocation lesson')
    long_source = b'def kernel(x):\n' + b'    x = x + 1\n' * 1200 + b'    return x\n'
    source = store.put_bytes(long_source)
    learned['after'] = dict(sha256=source, start_line=1, end_line=1202, text=long_source.decode())
    log = store.put_bytes(b'full retained Actor trajectory')
    provenance = dict(request=dict(artifacts={'actor-log': log}),
                      notes='Retained independent evidence and applicability. ' * 400)
    admitted = memory.accept('large', learned, provenance)
    original = memory.read(admitted['version'])
    options = dict(snapshot=memory.snapshot(), branch_root=tmp_path / 'actor') if frozen else {}
    result = service.query({'query': 'largevector'}, remaining_seconds=60, **options)
    selected = result['memories'][0]
    document = selected['artifacts']['memory-document']
    exact = service.read_artifact(document, remaining_seconds=60)
    assert document == selected['version'] == admitted['version']
    assert exact == canonical(original)
    assert len(exact) > 10000
    assert json.loads(exact)['proposal']['after']['text'] == long_source.decode()
    assert json.loads(exact)['provenance'] == provenance
    assert 'version' not in json.loads(exact) and 'artifacts' not in json.loads(exact)
    assert memory_artifacts(original) <= set(selected['artifacts'].values())
    assert service.read_artifact(selected['proposal']['after']['sha256'], remaining_seconds=60) == long_source


def test_actor_reader_follows_selected_frozen_memory_code_and_log_links(tmp_path):
    from benchmarks.ascend.agent_presentation import MemoryArtifactReader, artifact_page_presentation
    service, memory, store, _ = setup(tmp_path)
    learned = proposal(store)
    log = store.put_bytes(b'Verified three negative and positive input cases.\n')
    first = memory.accept('one', learned, {'request': {'artifacts': {'actor-log': log}}})
    frozen = memory.snapshot()
    # A later accepted version and an unaccepted pending artifact must stay out
    # of the independently empty Actor branch store.
    later_log = store.put_bytes(b'Later replacement execution log\n')
    memory.accept('two', dict(learned, operation='append', base_versions=[first]),
                  {'request': {'artifacts': {'actor-log': later_log}}})
    pending = store.put_bytes(b'Pending unrelated source\n')
    actor = context(tmp_path/'standalone-actor', frozen)
    result = service.actor_query({'query': 'vector increments', 'limit': 1}, actor, remaining_seconds=60)
    reader = MemoryArtifactReader(actor.request, actor.artifacts)
    entry = result.observation['results'][0]
    document = json.loads(artifact_page_presentation({'sha256': entry['artifact_sha256']}, reader)['text'])
    source = document['proposal']['after']['sha256']
    linked_log = document['provenance']['request']['artifacts']['actor-log']
    assert artifact_page_presentation({'sha256': source}, reader)['text'] == SOURCE.decode()
    assert artifact_page_presentation({'sha256': linked_log}, reader)['text'].startswith('Verified three')
    assert {source, linked_log, learned['evidence'][0], first['version']} <= set(result.admitted_artifacts)
    for excluded in (later_log, pending):
        assert excluded not in result.admitted_artifacts
        with pytest.raises(FileNotFoundError):
            reader.read_bytes(excluded)


def test_reference_callback_receives_positive_lock_timeout_and_elapsed_expiry(tmp_path):
    import threading
    now, allocations = [10.0], []
    lock = threading.Lock()
    def reference(action, *, check_remaining):
        allocations.append(check_remaining())
        assert lock.acquire(timeout=allocations[-1])
        lock.release()
        now[0] += 1
        allocations.append(check_remaining())
        return []
    service, _, _, _ = setup(tmp_path, reference_query=reference, clock=lambda: now[0])
    assert service.query({'query': 'vector'}, remaining_seconds=3)['hits'] == []
    assert allocations == [3.0, 2.0]
    with pytest.raises(TimeoutError):
        service.query({'query': 'vector'}, remaining_seconds=0.5)
