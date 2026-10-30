from canonical_optimizer_fixtures import canonical_optimizer_backend

"""Production actor and optimizer delivery enforces shared usage and withdrawal."""
import json

import pytest

def test_linked_memory_reads_count_only_returned_content_and_enforce_revocation(tmp_path):
    from test_subject_knowledge import setup as knowledge_setup, proposal as knowledge_proposal, context
    service,memory,store,_=knowledge_setup(tmp_path/'knowledge')
    base=memory.accept('lesson',knowledge_proposal(store),{})
    actor=context(tmp_path/'actor',memory.snapshot())
    service.actor_query({'query':'vector','limit':1},actor,remaining_seconds=60)
    source=memory.read(base['version'])['proposal']['after']['sha256']
    actor.access_event_id='linked-read'
    page={'text_available':True,'page_bytes':10}
    service.actor_read({'sha256':source},actor,page)
    service.actor_read({'sha256':source},actor,page)
    assert service.ledger.stats(base['memory_id'],base['version'])['accesses']==2
    assert service.ledger.stats(base['memory_id'],base['version'])['exposures']==1
    service.ledger.transition(base['version'],'quarantined',event_id='conflict',reason='Conflicting evidence',evidence=['e'*64])
    with pytest.raises(ValueError,match='withdrawn'):
        service.actor_read({'sha256':source},actor,page)
    unrelated=store.put_bytes(b'unrelated reference')
    service.actor_read({'sha256':unrelated},actor,page)
    assert service.ledger.stats(base['memory_id'],base['version'])['accesses']==2


def test_project_source_read_keeps_independent_authority_when_memory_is_withdrawn(tmp_path):
    from test_actor_compiler_diagnostics import actions,host
    from test_a5_phase1_adaptive_actor import SourceBackend
    from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActionResult
    current=actions(tmp_path,SourceBackend())
    source=host(current,'write',content='fixed source')['source_sha256']
    document=current.artifacts.put_bytes(b'withdrawn memory document')
    current.query_knowledge=lambda *a:AdaptiveActionResult({'event':'query'},admitted_artifacts={source:'knowledge',document:'knowledge'})
    host(current,'query-knowledge',query='source')
    def deny(*args):raise ValueError('memory withdrawn')
    current.observe_knowledge_read=deny
    assert current.admitted[source]=='candidate-source'
    assert host(current,'read-artifact',sha256=source)['text']=='fixed source'
    assert host(current,'read-artifact',sha256=document)['error']=='memory withdrawn'


def test_indexed_prompt_delivery_counts_only_shown_available_memories(tmp_path):
    from dataclasses import replace
    from test_subject_knowledge import setup as knowledge_setup, proposal as knowledge_proposal
    from test_a5_phase1_adaptive_actor import SourceBackend,ScriptedActor,_request
    from benchmarks.a5kernels.evidence import EvidenceLedger
    from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActorProjectExecutor
    from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
    service,memory,store,_=knowledge_setup(tmp_path/'knowledge')
    base=memory.accept('lesson',knowledge_proposal(store),{})
    frozen=memory.snapshot()
    executor=ProjectExecutor(SourceBackend(),EvidenceLedger(tmp_path/'evidence.jsonl'))
    request=replace(_request(tmp_path,executor),pre_wave_memory=frozen,publish_learning=False)
    actor=ScriptedActor(['{"write":{"slot":"kernel","content":"fixed source"}}','{"run":{}}','{"submit":{}}'],memory=frozen)
    driver=AdaptiveActorProjectExecutor(executor,actor,object())
    driver.prepare_memory=service.actor_snapshot
    driver.observe_memory_delivery=service.actor_delivery
    driver.observe_knowledge_read=service.actor_read
    driver(request)
    assert service.ledger.stats(base['memory_id'])['accesses']==1
    service.ledger.transition(base['version'],'quarantined',event_id='bad',reason='Conflict',evidence=['e'*64])
    filtered=service.actor_snapshot(frozen)
    assert json.loads(filtered)['memories']==[]
    actor.expected_memory=filtered
    driver(request)
    assert service.ledger.stats(base['memory_id'])['accesses']==1


def exercise_optimizer_memory_delivery(tmp_path, monkeypatch):
    from optimizer.jobs import JobStore,canonical,sha
    from optimizer.loop import OptimizationLoop
    from test_optimizer_loop import Agent,Backend,candidate
    from test_subject_knowledge import setup as knowledge_setup,proposal as knowledge_proposal
    service,memory,store,_=knowledge_setup(tmp_path/'knowledge')
    base=memory.accept('lesson',knowledge_proposal(store),{})
    snapshot={'files':{},'memory_entries':memory.records()}
    snapshot['snapshot_id']=sha(canonical(snapshot))
    jobs=JobStore(tmp_path/'jobs')
    key=jobs.create('def forward(x): return x','forward',{'args':[{'shape':[5],'dtype':'float32'}]},snapshot,max_seconds=60)['job_id']
    observed = []
    actual_observer = service.optimizer_observer(tmp_path/'optimizer-project')
    def observer(kind, snapshot, payload, event_id):
        result = actual_observer(kind, snapshot, payload, event_id)
        if kind == 'read-memory' or kind == 'presentation' and result.get('memory_summary'):
            observed.append((kind, event_id))
        return result
    observer.filter_presentation = actual_observer.filter_presentation
    from canonical_optimizer_fixtures import scripted_optimizer_http
    delivered = []
    class DeliveryAgent(Agent):
        def complete(self, stage, context, record):
            delivered.extend(row['result'] for row in context['evidence'] if row['name'] == 'read_memory')
            return super().complete(stage, context, record)
    agent = DeliveryAgent([{'action':'cases','signatures':[],'summary':''},
        {'action':'read-memory','index':0},candidate('fast')])
    agent.canonical_fixture = True
    state = scripted_optimizer_http(monkeypatch, agent)
    loop=OptimizationLoop(jobs,key,agent,Backend(),memory_observer=observer)
    result = loop.run(single_evaluation=True)
    assert result['status'] == 'running'
    assert loop.state['canonical_active']['status'] == 'accepted'
    assert state.requests
    assert len(delivered) == 1
    assert all(delivered[0][key] == value for key, value in snapshot['memory_entries'][0].items())
    requests = [row for row in loop.fixture_runner.events if row['type'] == 'pi-provider-request']
    assert len(requests) == len(state.requests)
    reservations = [row['reservation_id'] for row in requests]
    assert len(reservations) == len(set(reservations))
    rows=service.ledger.exposures(str(service.root.resolve()),'optimizer-project')
    assert len(rows)==1 and rows[0]['actor_role']=='optimizer'
    assert any(kind == 'presentation' for kind, _ in observed)
    assert sum(kind == 'read-memory' for kind, _ in observed) == 1
    accesses = len(set(observed))
    assert rows[0]['accesses'] == accesses
    service.ledger.transition(base['version'],'quarantined',event_id='bad',reason='Conflict',evidence=['e'*64])
    result=service.optimizer_observer(tmp_path/'optimizer-project')('read-memory',snapshot,
        {'action':{'index':0},'result':memory.read(base['version'])},'withdrawn-read')
    assert result['status']=='tool-error'
    assert service.ledger.stats(base['memory_id'])['accesses'] == accesses


def test_optimizer_counts_actual_memory_presentation_and_read(tmp_path, monkeypatch):
    exercise_optimizer_memory_delivery(tmp_path, monkeypatch)


@pytest.mark.parametrize('withdraw_first',[False,True])
def test_omitted_knowledge_index_access_counts_and_checks_unexposed_memory(tmp_path,withdraw_first):
    from test_subject_knowledge import setup as knowledge_setup,proposal as knowledge_proposal,context
    service,memory,store,_=knowledge_setup(tmp_path/'knowledge')
    for number in range(8):memory.accept('lesson-'+str(number),knowledge_proposal(store,'Vector increments '+str(number)),{})
    actor=context(tmp_path/'actor',memory.snapshot())
    actor.request.actor_memory_delivery='kdb-only'
    result=service.actor_query({'query':'vector','limit':10},actor,remaining_seconds=60)
    observation=result.observation
    assert observation['omitted_results']>0
    index_sha=observation['results_index_sha256']
    raw=actor.artifacts.read_bytes(index_sha)
    entries=json.loads(raw)['results']
    inline={row['citation']['collection'] for row in observation['results']}
    target=next(row for row in entries if row['citation']['collection'] not in inline)
    version=target['citation']['collection'].removeprefix('subject-memory-')
    memory_id=service.ledger.record(version)['memory_id']
    assert service.ledger.stats(memory_id,version)['accesses']==0
    encoded=json.dumps(target,ensure_ascii=True,separators=(',',':')).encode('ascii')
    page=dict(offset=raw.find(encoded),page_bytes=len(encoded),text_available=True)
    actor.access_event_id='read-omitted-index'
    if withdraw_first:
        service.ledger.transition(version,'quarantined',event_id='conflict',reason='Contradictory project',evidence=['e'*64])
        with pytest.raises(ValueError,match='withdrawn'):service.actor_read({'sha256':index_sha},actor,page)
        assert service.ledger.stats(memory_id,version)['accesses']==0
    else:
        service.actor_read({'sha256':index_sha},actor,page)
        service.actor_read({'sha256':index_sha},actor,page)
        assert service.ledger.stats(memory_id,version)['accesses']==1


def test_cross_campaign_default_indexed_execute_materializes_shared_evidence(tmp_path):
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.a5kernels.evidence import EvidenceLedger
    from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActorProjectExecutor
    from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
    from benchmarks.ascend.subject_runtime import build_subject_campaign
    from benchmarks.ascend.subjects import Subject
    from test_a5_phase1_adaptive_actor import SourceBackend, ScriptedActor, _spec
    from test_subject_knowledge import setup, proposal
    from test_subject_runtime import bound
    origin, memory, store, kdb = setup(tmp_path/'origin')
    base = memory.accept('lesson', proposal(store), {})
    snapshot = memory.snapshot()
    store.root.rename(tmp_path/'unavailable-origin-artifacts')
    consumer_root = tmp_path/'consumer'
    binding = bound(consumer_root, kdb=kdb, embeddings=origin.embeddings)
    actor = ScriptedActor(['{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}', '{"submit":{}}'], memory=snapshot)
    actor.provider = binding['provider']
    driver = AdaptiveActorProjectExecutor(ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path/'evidence.jsonl')),
        actor, binding['profile'])
    binding['actor'] = driver
    campaign = build_subject_campaign({}, target='a3', root=consumer_root,
        subjects=[Subject('s','learn tails')], initial_memory=b'initial guidance', compose=lambda *a,**kw:binding)
    outcome = campaign.execute(dict(project_id=_spec().project_id, spec=_spec().as_dict()),
        campaign.memory.snapshot(), consumer_root/'project', remaining_seconds=60)
    assert outcome.host_verified
    assert campaign.memory.ledger.stats(base['memory_id'])['accesses'] == 1
    copied = ContentAddressedArtifacts(consumer_root/'project/artifacts')
    assert copied.read_bytes(memory.read(base['version'])['proposal']['after']['sha256'])


def test_indexed_without_reference_kdb_guards_linked_reads(tmp_path):
    from dataclasses import replace
    from benchmarks.ascend.subject_runtime import build_subject_campaign
    from benchmarks.ascend.subjects import Subject
    from test_subject_runtime import bound
    from test_actor_compiler_diagnostics import actions, host
    from test_a5_phase1_adaptive_actor import SourceBackend
    from test_subject_knowledge import proposal
    binding = bound(tmp_path)
    campaign = build_subject_campaign({}, target='a3', root=tmp_path/'campaign',
        subjects=[Subject('s','learn tails')], initial_memory=b'', compose=lambda *a,**kw:binding)
    base = campaign.memory.accept('lesson', proposal(campaign.artifacts), {})
    current = actions(tmp_path/'actor', SourceBackend())
    current.request = replace(current.request, pre_wave_memory=campaign.memory.snapshot())
    current.observe_knowledge_read = binding['actor'].observe_knowledge_read
    assert current.observe_knowledge_read.__self__.database_path is None
    source = campaign.memory.read(base['version'])['proposal']['after']['sha256']
    current.artifacts.put_bytes(campaign.artifacts.read_bytes(source))
    assert host(current, 'read-artifact', sha256=source)['text']
    ledger = campaign.memory.ledger
    assert ledger.stats(base['memory_id'])['accesses'] == 1
    ledger.transition(base['version'],'quarantined',event_id='conflict',reason='Counterexample',evidence=['e'*64])
    assert 'withdrawn' in host(current,'read-artifact',sha256=source)['error']
    assert ledger.stats(base['memory_id'])['accesses'] == 1


def test_manual_episode_has_new_exposure_while_durable_replay_deduplicates(tmp_path):
    from test_actor_compiler_diagnostics import actions
    from test_a5_phase1_adaptive_actor import SourceBackend
    from test_subject_knowledge import setup, proposal
    service, memory, store, _ = setup(tmp_path/'knowledge')
    base = memory.accept('lesson', proposal(store), {})
    original = actions(tmp_path/'actor', SourceBackend()); original.action_event_turn=43
    first = original._action_context({'opening_memory':'same'})
    service.actor_delivery({'memories':[base]}, first)
    marker = original.request.branch_root/'manual-execution.json'
    marker.parent.mkdir(parents=True,exist_ok=True)
    marker.write_text(json.dumps({'attempt_id':'continued-attempt'}))
    resumed = actions(tmp_path/'actor', SourceBackend()); resumed.action_event_turn=43
    second = resumed._action_context({'opening_memory':'same'})
    service.actor_delivery({'memories':[base]}, second)
    again = actions(tmp_path/'actor', SourceBackend()); again.action_event_turn=43
    replay = again._action_context({'opening_memory':'same'})
    service.actor_delivery({'memories':[base]}, replay)
    assert first.actor_id != second.actor_id and replay.actor_id == second.actor_id
    assert replay.access_event_id == second.access_event_id
    rows = service.ledger.exposures(str(service.root.resolve()), first.request.spec.project_id)
    assert len(rows) == 2 and sum(row['accesses'] for row in rows) == 2
