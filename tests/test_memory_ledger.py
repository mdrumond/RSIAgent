"""Shared memory delivery and lifecycle decisions survive resumes and revisions."""
from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from benchmarks.ascend.memory_ledger import MemoryLedger
from benchmarks.ascend.subject_memory import SubjectMemory, digest
from test_subject_knowledge import setup, proposal, context


EVIDENCE = 'e' * 64


def record(memory_id='lesson', bases=(), summary='Useful relocation rule'):
    return dict(memory_id=memory_id, proposal=dict(base_versions=list(bases), summary=summary,
        after={'sha256': 'a'*64}, before=None, evidence=[EVIDENCE]), provenance={})


def delivery(ledger, version, event='event', project='project', actor='writer'):
    return ledger.access(event, memory_id=ledger.record(version)['memory_id'], version=version,
        campaign_id='campaign', project_id=project, actor_id=actor, actor_role=actor)


def test_shared_deliveries_deduplicate_replay_but_count_real_repeated_reads(tmp_path):
    first = MemoryLedger.for_kdb(tmp_path/'knowledge.sqlite3')
    version = first.publish(record())
    second = MemoryLedger.for_kdb(tmp_path/'knowledge.sqlite3')
    exposure = delivery(first, version)
    assert delivery(second, version) == exposure
    assert delivery(second, version, event='read-2') == exposure
    other = delivery(second, version, event='analyzer-read', actor='analyzer')
    assert other != exposure
    rows = second.exposures('campaign', 'project')
    assert sorted(row['accesses'] for row in rows) == [1, 2]
    assert first.stats('lesson')['accesses'] == 3
    assert len(first.stats('lesson')['by_actor']) == 2
    with pytest.raises(ValueError, match='different delivery'):
        delivery(first, version, project='different-project')


def test_concurrent_clients_record_exactly_one_event(tmp_path):
    path = tmp_path/'ledger.sqlite3'
    version = MemoryLedger(path).publish(record())
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(lambda _: delivery(MemoryLedger(path), version), range(16)))
    assert len(set(ids)) == 1
    assert MemoryLedger(path).stats('lesson')['accesses'] == 1


@pytest.mark.parametrize(('useful', 'decided', 'unknown', 'rewrite'), [(2,9,3,False),(2,10,3,True),(3,10,0,False)])
def test_hot_rule_uses_distinct_decided_exposures_not_repeated_accesses(tmp_path, useful, decided, unknown, rewrite):
    ledger = MemoryLedger(tmp_path/'ledger.sqlite3'); version = ledger.publish(record())
    for i in range(decided + unknown):
        exposure = delivery(ledger, version, event=str(i), project=str(i))
        judgment = dict(usefulness='useful' if i < useful else 'not-useful' if i < decided else 'unknown',
                        correctness='supported', reason='Reviewed project evidence', evidence=[EVIDENCE])
        ledger.judge(exposure, **judgment); ledger.judge(exposure, **judgment)
    for i in range(20):
        delivery(ledger, version, event='repeat-'+str(i), project='0')
    stats = ledger.stats('lesson', version)
    assert stats['accesses'] == decided + unknown + 20
    assert stats['decided'] == decided and stats['unknown'] == unknown
    assert stats['hot'] is True and stats['reviewed_hot'] is (decided >= 10)
    assert stats['needs_rewrite'] is rewrite
    assert stats['usefulness_rate'] == useful/decided


def test_revision_and_merge_preserve_lifetime_without_copying_version_counts(tmp_path):
    ledger = MemoryLedger(tmp_path/'ledger.sqlite3')
    first = ledger.publish(record('first')); second = ledger.publish(record('second'))
    delivery(ledger, first, event='first'); delivery(ledger, second, event='second')
    merged = ledger.publish(record('first', [{'memory_id':'first','version':first}, {'memory_id':'second','version':second}]))
    assert ledger.stats('first', merged)['accesses'] == 0
    assert ledger.stats('first')['accesses'] == 2
    assert ledger.status(first) == ledger.status(second) == 'superseded'
    assert ledger.available(first)  # Ordinary frozen views retain their historical version.
    ledger.transition(first, 'quarantined', event_id='contradiction', reason='Original assumption contradicted', evidence=[EVIDENCE])
    assert not ledger.available(first) and ledger.available(merged)
    assert ledger.stats('first')['accesses'] == 2


def test_retirement_history_survives_restore_and_index_replay(tmp_path):
    ledger = MemoryLedger(tmp_path/'ledger.sqlite3'); version = ledger.publish(record())
    ledger.transition(version, 'quarantined', event_id='q', reason='Contradiction', evidence=[EVIDENCE])
    ledger.transition(version, 'retired', event_id='retirement', reason='Rule cannot be supported', evidence=[EVIDENCE], expected_status='quarantined')
    assert ledger.publish(record()) == version
    assert ledger.status(version) == 'retired'
    with pytest.raises(ValueError, match='withdrawn'):
        ledger.require_actor_artifact(version)
    with pytest.raises(ValueError, match='unavailable'):
        delivery(ledger, version)
    tombstone = ledger.retired('relocation')[0]
    assert tombstone['tombstone_id'] == 'retirement' and tombstone['record'] == record()
    ledger.transition(version, 'active', event_id='restore', reason='New evidence resolves original contradiction', evidence=['b'*64], expected_status='retired')
    assert ledger.available(version) and ledger.retired()[0] == tombstone
    assert ledger.require_actor_artifact(version) is None
    with pytest.raises(ValueError, match='changed'):
        ledger.transition(version, 'quarantined', event_id='stale-review', reason='Old suspicion', evidence=[EVIDENCE], expected_status='retired')


def test_shared_artifact_is_readable_from_another_available_memory(tmp_path):
    ledger = MemoryLedger(tmp_path/'ledger.sqlite3')
    first = ledger.publish(record()); second = ledger.publish(record('other'))
    ledger.transition(first, 'quarantined', event_id='q', reason='Bad lesson', evidence=[EVIDENCE])
    assert ledger.require_actor_artifact('a'*64) is None
    assert len(ledger.artifact_versions('a'*64)) == 2
    with pytest.raises(ValueError, match='withdrawn'):
        ledger.require_actor_artifact('a'*64, version=first)
    assert ledger.require_actor_artifact('a'*64, version=second) is None
    assert ledger.require_actor_artifact('f'*64) is None  # Non-memory bytes remain a caller authorization concern.


def test_quarantine_filters_live_and_frozen_queries_and_keeps_repair_base(tmp_path):
    service, memory, store, _ = setup(tmp_path)
    admitted = memory.accept('lesson', proposal(store), {})
    frozen = memory.snapshot()
    service.ledger.transition(admitted['version'], 'quarantined', event_id='q', reason='Conflicting result', evidence=[EVIDENCE])
    assert memory.records() == [] and json.loads(memory.snapshot())['memories'] == []
    assert memory.read(admitted['version'])['memory_id'] == 'lesson'
    assert service.query({'query':'vector'}, remaining_seconds=60, snapshot=frozen, branch_root=tmp_path/'old-actor')['memories'] == []
    replacement = memory.accept('fixed', dict(proposal(store), operation='append', base_versions=[admitted]), {})
    assert [digest(row) for row in memory.records()] == [replacement['version']]
    assert not service.ledger.available(admitted['version'])
    assert [row['version'] for row in service.ledger.replacements(admitted['version'])] == [replacement['version']]


def test_only_selected_actor_delivery_is_counted_and_replay_is_idempotent(tmp_path):
    service, memory, store, _ = setup(tmp_path)
    admitted = memory.accept('lesson', proposal(store), {})
    memory.accept('other', proposal(store, 'Vector other increment'), {})
    service.query({'query':'vector', 'limit':1}, remaining_seconds=60)
    assert service.ledger.stats('lesson')['accesses'] == 0
    actor = context(tmp_path/'actor', memory.snapshot())
    reply = service.actor_query({'query':'vector', 'limit':1}, actor, remaining_seconds=60)
    service.actor_query({'query':'vector', 'limit':1}, actor, remaining_seconds=60)
    versions = [citation.collection.removeprefix('subject-memory-') for citation in reply.citations if citation.collection.startswith('subject-memory-')]
    rows = service.ledger.exposures(str(tmp_path.resolve()), 'project')
    assert [row['version'] for row in rows] == versions
    assert [row['accesses'] for row in rows] == [1]
    actor.access_event_id = 'query-2'
    service.actor_query({'query':'vector', 'limit':1}, actor, remaining_seconds=60)
    assert service.ledger.exposures(str(tmp_path.resolve()), 'project')[0]['accesses'] == 2


def test_judgments_need_real_exposure_and_contradiction_evidence(tmp_path):
    ledger = MemoryLedger(tmp_path/'ledger.sqlite3'); version = ledger.publish(record())
    with pytest.raises(KeyError):
        ledger.judge('missing', usefulness='unknown', correctness='unknown', reason='No exposure', evidence=[])
    exposure = delivery(ledger, version)
    with pytest.raises(ValueError, match='contradiction evidence'):
        ledger.judge(exposure, usefulness='useful', correctness='contradicted', reason='Claim', evidence=[])
    ledger.judge(exposure, usefulness='unknown', correctness='unknown', reason='Incomplete result', evidence=[])
    with pytest.raises(ValueError, match='different judgment'):
        ledger.judge(exposure, usefulness='useful', correctness='supported', reason='Changed result', evidence=[EVIDENCE])


def test_elapsed_expiry_does_not_publish_an_indexed_draft(tmp_path):
    service, memory, store, _ = setup(tmp_path)
    def expired():
        raise TimeoutError('elapsed allocation exhausted')
    learned = proposal(store)
    with pytest.raises(TimeoutError):
        memory.accept('late', learned, {}, check_remaining=expired)
    version = digest(dict(memory_id='late', proposal=learned, provenance={}))
    assert service.ledger.status(version) is None and memory.records() == []
    assert service.query({'query':'vector'}, remaining_seconds=60,
        snapshot=json.dumps({'memories':[dict(memory_id='late',proposal=learned,provenance={})]}).encode(),
        branch_root=tmp_path/'actor')['memories'] == []


def test_existing_accepted_catalog_initializes_zero_counters_without_restoring_retired(tmp_path):
    service, memory, store, _ = setup(tmp_path)
    admitted = memory.accept('lesson', proposal(store), {})
    service.ledger.transition(admitted['version'], 'retired', event_id='r', reason='Unsupported claim', evidence=[EVIDENCE])
    reopened = SubjectMemory(tmp_path/'memory', b'initial guidance', ledger=service.ledger)
    assert reopened.records() == []
    assert service.ledger.stats('lesson')['accesses'] == 0
    assert service.ledger.status(admitted['version']) == 'retired'


def test_frequently_accessed_unknown_memory_is_hot_without_rewrite(tmp_path):
    ledger = MemoryLedger(tmp_path/'ledger.sqlite3'); version = ledger.publish(record())
    for i in range(10):
        exposure = delivery(ledger, version, event=str(i))
    ledger.judge(exposure, usefulness='unknown', correctness='unknown', reason='Missing outcome', evidence=[])
    stats = ledger.stats('lesson', version)
    assert stats['hot'] and not stats['reviewed_hot'] and not stats['needs_rewrite']
    assert stats['accesses'] == 10 and stats['exposures'] == stats['unknown'] == 1


def test_second_campaign_discovers_and_replaces_shared_catalog(tmp_path):
    from types import SimpleNamespace
    from benchmarks.ascend.subject_knowledge import SubjectKnowledge
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from test_a5_knowledge import FakeEmbeddings
    service, first, store, kdb = setup(tmp_path/'first')
    original = first.accept('lesson', proposal(store), {})
    second = SubjectMemory(tmp_path/'second/memory', ledger=MemoryLedger.for_kdb(kdb))
    other = SubjectKnowledge(tmp_path/'second', ContentAddressedArtifacts(tmp_path/'second/artifacts'),
        lambda: second, paths=SimpleNamespace(kdb=kdb), embeddings=FakeEmbeddings(),
        reference_query=lambda *args, **kwargs: ())
    assert [digest(row) for row in second.records()] == [original['version']]
    assert second.read(original['version']) == first.read(original['version'])
    found = other.query({'query':'vector'}, remaining_seconds=60)
    assert [row['version'] for row in found['memories']] == [original['version']]
    service.ledger.transition(original['version'], 'quarantined', event_id='conflict', reason='New contradictory execution', evidence=[EVIDENCE])
    assert first.records() == second.records() == []
    fixed = second.accept('correction', dict(proposal(store), operation='append', base_versions=[original], summary='Corrected claim'), {})
    assert [digest(row) for row in first.records()] == [fixed['version']]
    assert first.read(fixed['version']) == second.read(fixed['version'])
    assert not service.ledger.available(original['version'])
    from benchmarks.ascend.subject_memory import StaleMemory
    with pytest.raises(StaleMemory):
        first.accept('stale', dict(proposal(store), operation='append', base_versions=[original]), {})


def test_concurrent_cross_campaign_replacements_compare_global_heads_atomically(tmp_path):
    from threading import Barrier
    from benchmarks.ascend.subject_memory import StaleMemory
    ledger = MemoryLedger(tmp_path/'ledger.sqlite3')
    first = SubjectMemory(tmp_path/'a', ledger=ledger)
    base = first.accept('lesson', record()['proposal'], {})
    barrier = Barrier(2)
    first.index = lambda *args: barrier.wait(timeout=10)
    second = SubjectMemory(tmp_path/'b', ledger=MemoryLedger(ledger.path), index=first.index)
    def replace(pair):
        memory, summary = pair
        try:
            return memory.accept(summary, dict(record()['proposal'], base_versions=[base], summary=summary), {})
        except StaleMemory:
            return 'stale'
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(replace, [(first,'one'),(second,'two')]))
    assert results.count('stale') == 1
    winner = next(result for result in results if isinstance(result, dict))
    assert [digest(row) for row in ledger.records()] == [winner['version']]
    assert not ledger.bases_current([base])


def test_retired_global_base_requires_new_source_review_resolution(tmp_path):
    ledger = MemoryLedger(tmp_path/'ledger.sqlite3')
    first = SubjectMemory(tmp_path/'a', ledger=ledger)
    base = first.accept('lesson', record()['proposal'], {})
    ledger.transition(base['version'], 'retired', event_id='retirement', reason='Unsupported old claim', evidence=[EVIDENCE])
    second = SubjectMemory(tmp_path/'b', ledger=MemoryLedger(ledger.path))
    changed = dict(record()['proposal'], base_versions=[base], summary='Corrected supported claim')
    with pytest.raises(ValueError, match='source-reviewed resolving evidence'):
        second.accept('correction', changed, {})
    correction = second.accept('correction', changed, dict(retirement_resolutions=[dict(
        tombstone_id='retirement', evidence=['f'*64], reason='New project resolves old counterexample')]))
    assert [digest(row) for row in first.records()] == [correction['version']]
    assert ledger.retired()[0]['version'] == base['version'] and not ledger.available(base['version'])


def test_lifecycle_change_during_indexing_rejects_publication(tmp_path):
    from benchmarks.ascend.subject_memory import StaleMemory
    ledger = MemoryLedger(tmp_path/'ledger.sqlite3'); memory = SubjectMemory(tmp_path/'memory', ledger=ledger)
    base = memory.accept('lesson', record()['proposal'], {})
    memory.index = lambda *args: ledger.transition(base['version'], 'quarantined', event_id='conflict', reason='New evidence', evidence=[EVIDENCE])
    with pytest.raises(StaleMemory, match='lifecycle changed'):
        memory.accept('correction', dict(record()['proposal'], base_versions=[base], summary='New version'), {})
    assert ledger.status(base['version']) == 'quarantined' and ledger.records() == []
