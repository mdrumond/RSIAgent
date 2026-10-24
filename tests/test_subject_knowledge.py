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
    memory = SubjectMemory(tmp_path / 'memory', b'initial guidance', index=service.index)
    return service, memory, store, database_path


def context(tmp_path, snapshot):
    return SimpleNamespace(request=SimpleNamespace(pre_wave_memory=snapshot, branch_root=tmp_path),
                           artifacts=ContentAddressedArtifacts(tmp_path / 'artifacts'))


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


def test_pending_treatment_memory_is_indexed_only_inside_its_frozen_arm(tmp_path):
    service, memory, store, shared = setup(tmp_path)
    draft = dict(memory_id='pending', proposal=proposal(store, 'Vector pendingmemoryvector trick'), provenance={})
    baseline = memory.snapshot()
    treatment = json.dumps(dict(schema='subject-memory-snapshot-v1', initial_memory='', memories=[draft])).encode()
    version = digest(draft)
    def query(snapshot, arm):
        return service.actor_query({'query': 'pendingmemoryvector'}, context(tmp_path / arm, snapshot),
                                   remaining_seconds=60)
    assert 'subject-memory-' + version in {row.collection for row in query(treatment, 'treatment').citations}
    assert 'subject-memory-' + version not in {row.collection for row in query(baseline, 'baseline').citations}
    with KnowledgeDB.open_read_only(shared, FakeEmbeddings()) as database:
        with pytest.raises(KeyError):
            database.manifest('subject-memory-' + version)
    assert memory.records() == []
    assert list((tmp_path / 'treatment' / 'frozen-memory').glob('*/knowledge.sqlite'))


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
    assert reachable == {learned['after']['sha256'], learned['evidence'][0], before, log, review}
    assert memory_artifacts(memory.records()[0]) == reachable
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
