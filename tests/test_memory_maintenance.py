from canonical_optimizer_fixtures import canonical_optimizer_backend

"""Ordinary verified projects correct or retire accessed memories without trials."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from benchmarks.ascend.memory_ledger import MemoryLedger
from benchmarks.ascend.project_verifier import ProjectReview
from benchmarks.ascend.subject_campaign import SubjectCampaign
from benchmarks.ascend.subjects import Subject
from test_subject_campaign import proposal


def setup(tmp_path, *, ledger=None, name='campaign'):
    ledger = ledger or MemoryLedger(tmp_path / 'shared.sqlite')
    campaign = SubjectCampaign(tmp_path / name, [Subject('s','learn tails',1)], b'',
        select_wave=lambda *a, **kw: [], execute=lambda *a, **kw: None,
        verifier=None, complete=lambda *a, **kw: None, memory_ledger=ledger)
    return campaign, ledger


def seed(campaign):
    source = campaign.artifacts.put_bytes(b'old source\n')
    return campaign.memory.accept('seed', proposal(source), {})


def observe(campaign, ledger, base, *, project_id='consumer', number=0,
            usefulness='not-useful', correctness='contradicted', investigate=False, result_bytes=b'new host result\n', objective='learn tails'):
    source = campaign.artifacts.put_bytes(b'new source\n')
    result = campaign.artifacts.put_bytes(result_bytes)
    exposure = ledger.access('access-' + str(number), **base, campaign_id=str(campaign.root.resolve()),
        project_id=project_id, actor_id=project_id, actor_role='writer')
    review = dict(decision='accept', memory=None, memory_judgments=[dict(exposure_id=exposure,
        usefulness=usefulness, correctness=correctness, investigate=investigate,
        reason='Current project provides independent evidence', evidence=[result])],
        memory_resolutions=[], retirement_checks=[])
    record = dict(objective=objective, review=review, review_sha256=campaign.artifacts.put_json(review),
        outcome=dict(host_verified=True,source_sha256=source,artifacts={'source':source,'result':result},observations={}))
    campaign.state['projects'][project_id] = record
    campaign.memory_maintenance.observe(project_id,record)
    return record


def test_contradiction_quarantines_and_mandatory_project_publishes_correction(tmp_path):
    campaign, ledger = setup(tmp_path)
    base = seed(campaign)
    observe(campaign, ledger, base)
    assert campaign.memory.records() == []
    repair = campaign.memory_maintenance.prioritize()
    assert repair['repair_id'].startswith('memory-correct-')
    new_source = campaign.artifacts.put_bytes(b'corrected source\n')
    request = campaign.memory_maintenance.prepare(ProjectReview('correction',new_source,True,{'source':new_source}), [repair['repair_id']])
    assert request.memory_repairs[0]['version'] == base['version']
    replacement = dict(proposal(new_source),operation='append',base_versions=[base])
    campaign.memory.accept('correction',replacement,{})
    review = dict(decision='accept',memory=replacement,memory_resolutions=[dict(
        repair_id=repair['repair_id'],action='replace',reason='Corrected precondition',evidence=[new_source])])
    record = dict(review=review,review_request=asdict(request),review_sha256=campaign.artifacts.put_json(review))
    campaign.memory_maintenance.resolve('correction',record)
    assert campaign.memory_maintenance.obligations() == []
    assert campaign.scheduler.state()['repairs'][repair['repair_id']]['status'] == 'resolved'
    assert ledger.status(base['version']) == 'quarantined'  # old snapshots remain revoked
    assert campaign.memory.records()[0]['proposal']['after']['sha256'] == new_source
    new_version = ledger.replacements(base['version'])[0]['version']
    assert ledger.stats(base['memory_id'],new_version)['accesses'] == 0
    assert ledger.stats(base['memory_id'])['accesses'] == 1


def test_hot_unhelpful_memory_stays_active_and_generates_one_global_rewrite(tmp_path):
    campaign, ledger = setup(tmp_path)
    base = seed(campaign)
    for number in range(10):
        observe(campaign,ledger,base,project_id='consumer-'+str(number),number=number,correctness='supported')
    assert ledger.available(base['version'])
    obligations = campaign.memory_maintenance.obligations()
    assert [row['kind'] for row in obligations] == ['rewrite']
    campaign.memory_maintenance.reconcile()
    assert len(campaign.memory_maintenance.obligations()) == 1
    other, _ = setup(tmp_path,ledger=ledger,name='other-campaign')
    other.memory_maintenance.reconcile()
    assert other.memory_maintenance.prioritize()['repair_id'] == obligations[0]['repair_id']


def test_unknown_does_not_impose_investigation_without_explicit_concern(tmp_path):
    campaign,ledger=setup(tmp_path);base=seed(campaign)
    observe(campaign,ledger,base,usefulness='unknown',correctness='unknown')
    assert campaign.memory_maintenance.obligations() == []
    observe(campaign,ledger,base,project_id='concern',number=1,usefulness='unknown',correctness='unknown',investigate=True)
    assert campaign.memory_maintenance.obligations()[0]['kind'] == 'investigate'
    assert ledger.available(base['version'])


def test_passing_project_without_explicit_resolution_leaves_repair_open(tmp_path):
    campaign,ledger=setup(tmp_path);base=seed(campaign)
    observe(campaign,ledger,base)
    repair=campaign.memory_maintenance.prioritize()
    source=campaign.artifacts.put_bytes(b'passing unrelated source\n')
    record=dict(review=dict(decision='accept',memory=None,reason='Passes',memory_resolutions=[]),
        review_sha256=campaign.artifacts.put_bytes(b'passing review'),repairs=[repair['repair_id']])
    campaign.state['projects']['passing']=record
    campaign.scheduler.start_wave('wave','a'*64)
    campaign._publish_wave(dict(wave_id='wave',subject={'subject_id':'s'},projects=['passing']))
    assert campaign.scheduler.state()['repairs'][repair['repair_id']]['status']=='open'
    assert ledger.status(base['version'])=='quarantined'


def test_retirement_evidence_survives_origin_campaign_and_is_globally_visible(tmp_path):
    campaign,ledger=setup(tmp_path);base=seed(campaign)
    record=observe(campaign,ledger,base)
    repair=campaign.memory_maintenance.prioritize()
    source=record['outcome']['source_sha256']
    request=campaign.memory_maintenance.prepare(ProjectReview('retire',source,True,{'source':source}),[repair['repair_id']])
    evidence=record['outcome']['artifacts']['result']
    review=dict(decision='accept',memory=None,memory_resolutions=[dict(repair_id=repair['repair_id'],
        action='retire',reason='Claim contradicted with no reusable replacement',evidence=[evidence])])
    campaign.memory_maintenance.resolve('retire',dict(review=review,review_request=asdict(request),
        review_sha256=campaign.artifacts.put_json(review)))
    campaign.artifacts.root.rename(campaign.root/'archived-artifacts')
    other,_=setup(tmp_path,ledger=ledger,name='other')
    src=other.artifacts.put_bytes(b'independent project\n')
    prepared=other.memory_maintenance.prepare(ProjectReview('p',src,True,{'source':src}))
    tombstone=prepared.retired_memories[0]
    assert tombstone['version']==base['version']
    assert other.memory_maintenance.read_artifact(evidence,remaining_seconds=1)==b'new host result\n'


def test_maintenance_cannot_be_bypassed_by_unrelated_wave(tmp_path):
    campaign,ledger=setup(tmp_path);base=seed(campaign)
    observe(campaign,ledger,base)
    attempts=[]
    def select(context,*args,**kw):
        attempts.append(context['repairs'])
        if len(attempts)==2:raise TimeoutError('test boundary')
        return [dict(project_id='unrelated',repairs=[])]
    campaign.select_wave=select
    campaign.policy.decide=lambda *a,**kw:pytest.fail('ordinary curriculum policy bypassed imposed repair')
    campaign.run()
    assert len(attempts)==2
    assert attempts[0][0]['repair_id'].startswith('memory-correct-')
    assert 'unrelated' not in campaign.state['projects']
    assert campaign.memory_maintenance.obligations()


def test_another_campaign_recovers_crash_between_judgment_and_quarantine(tmp_path, monkeypatch):
    campaign,ledger=setup(tmp_path);base=seed(campaign)
    apply=campaign.memory_maintenance._apply_pending
    monkeypatch.setattr(campaign.memory_maintenance,'_apply_pending',lambda:None)
    observe(campaign,ledger,base)
    assert ledger.status(base['version'])=='active'
    # The origin never resumes; the durable assessment and shared evidence suffice.
    other,_=setup(tmp_path,ledger=ledger,name='recovery')
    other.memory_maintenance.reconcile()
    assert ledger.status(base['version'])=='quarantined'
    assert other.memory_maintenance.prioritize()['repair_id'].startswith('memory-correct-')


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


def test_cross_campaign_imposed_repair_publishes_against_global_base(tmp_path):
    origin,ledger=setup(tmp_path);base=seed(origin)
    observe(origin,ledger,base)
    other,_=setup(tmp_path,ledger=ledger,name='repair-campaign')
    other.memory_maintenance.reconcile()
    repair=other.memory_maintenance.prioritize()
    source=other.artifacts.put_bytes(b'new independently verified correction\n')
    request=other.memory_maintenance.prepare(ProjectReview('fix',source,True,{'source':source}),[repair['repair_id']])
    fixed=dict(proposal(source),operation='append',base_versions=[base])
    published=other.memory.accept('fix',fixed,{})
    review=dict(decision='accept',memory=fixed,memory_resolutions=[dict(repair_id=repair['repair_id'],
        action='replace',reason='Independent corrected example',evidence=[source])])
    other.memory_maintenance.resolve('fix',dict(review=review,review_request=asdict(request),
        review_sha256=other.artifacts.put_json(review)))
    assert [row['memory_id'] for row in origin.memory.records()]==[base['memory_id']]
    assert origin.memory.read(published['version'])['proposal']['after']['sha256']==source
    assert other.memory_maintenance.obligations()==[]


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


def test_optimizer_counts_actual_memory_presentation_and_read(tmp_path):
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
    loop=OptimizationLoop(jobs,key,Agent([{'action':'cases','signatures':[],'summary':''},
        {'action':'read-memory','index':0},candidate('fast')]),Backend(),
        memory_observer=service.optimizer_observer(tmp_path/'optimizer-project'))
    loop.run(single_evaluation=True)
    rows=service.ledger.exposures(str(service.root.resolve()),'optimizer-project')
    assert len(rows)==1 and rows[0]['actor_role']=='optimizer'
    assert rows[0]['accesses']==3  # prompt, full read, next prompt
    service.ledger.transition(base['version'],'quarantined',event_id='bad',reason='Conflict',evidence=['e'*64])
    result=service.optimizer_observer(tmp_path/'optimizer-project')('read-memory',snapshot,
        {'action':{'index':0},'result':memory.read(base['version'])},'withdrawn-read')
    assert result['status']=='tool-error'
    assert service.ledger.stats(base['memory_id'])['accesses']==3


@pytest.mark.parametrize('action',['restore','retire'])
def test_resolution_replays_after_lifecycle_committed_before_obligation_checkpoint(tmp_path,monkeypatch,action):
    campaign,ledger=setup(tmp_path);base=seed(campaign)
    current=observe(campaign,ledger,base)
    repair=campaign.memory_maintenance.prioritize()
    source=current['outcome']['source_sha256']
    request=campaign.memory_maintenance.prepare(ProjectReview('resolve',source,True,{'source':source}),[repair['repair_id']])
    review=dict(decision='accept',memory=None,memory_resolutions=[dict(repair_id=repair['repair_id'],
        action=action,reason='New project resolves dispute',evidence=[source])])
    record=dict(review=review,review_request=asdict(request),review_sha256=campaign.artifacts.put_json(review))
    transition=ledger.transition
    def interrupted(*args,**kwargs):
        transition(*args,**kwargs)
        raise KeyboardInterrupt('simulated death after lifecycle commit')
    monkeypatch.setattr(ledger,'transition',interrupted)
    with pytest.raises(KeyboardInterrupt):campaign.memory_maintenance.resolve('resolve',record)
    assert campaign.memory_maintenance.obligations()
    monkeypatch.setattr(ledger,'transition',transition)
    campaign.memory_maintenance.resolve('resolve',record)
    assert campaign.memory_maintenance.obligations()==[]
    assert ledger.status(base['version'])==('active' if action=='restore' else 'retired')


def test_new_contradiction_after_restore_creates_new_repair_episode(tmp_path):
    campaign,ledger=setup(tmp_path);base=seed(campaign)
    current=observe(campaign,ledger,base)
    old=campaign.memory_maintenance.prioritize()
    source=current['outcome']['source_sha256']
    request=campaign.memory_maintenance.prepare(ProjectReview('restore',source,True,{'source':source}),[old['repair_id']])
    review=dict(decision='accept',memory=None,memory_resolutions=[dict(repair_id=old['repair_id'],
        action='restore',reason='Counterexample outside documented applicability',evidence=[source])])
    campaign.memory_maintenance.resolve('restore',dict(review=review,review_request=asdict(request),review_sha256=campaign.artifacts.put_json(review)))
    observe(campaign,ledger,base,project_id='new-consumer',number=1)
    fresh=campaign.memory_maintenance.prioritize()
    assert fresh['repair_id']!=old['repair_id']
    assert ledger.status(base['version'])=='quarantined'
    history=campaign.memory_maintenance.obligations(open_only=False)
    assert len(history)==2 and sum(row['resolution'] is None for row in history)==1
    campaign.memory_maintenance.reconcile()
    assert campaign.memory_maintenance.obligations()[0]['repair_id']==fresh['repair_id']


def test_stale_source_publication_rereview_preserves_original_exposure_judgment(tmp_path):
    from benchmarks.ascend.subject_memory import digest
    campaign,ledger=setup(tmp_path);base=seed(campaign)
    record=observe(campaign,ledger,base,correctness='supported',usefulness='useful')
    source=record['outcome']['source_sha256']
    record.update(repairs=[],review_request={})
    record['review']['memory']=dict(proposal(source),operation='append',base_versions=[base])
    campaign.memory.accept('concurrent',dict(proposal(source),operation='append',base_versions=[base]),{})
    calls=[]
    class Reviewer:
        last_trace=[]
        def review(self,request,**kwargs):
            calls.append(request)
            current=ledger.records()[0]
            return dict(decision='accept',reason='Reconcile current base',evidence=[source],repair=None,
                memory=dict(proposal(source),operation='append',base_versions=[dict(memory_id=current['memory_id'],version=digest(current))]),
                memory_judgments=[],memory_resolutions=[],retirement_checks=[])
    campaign.verifier=Reviewer()
    campaign.budget=SimpleNamespace(deadline=campaign.clock()+30)
    campaign.scheduler.start_wave('wave','a'*64)
    campaign._publish_wave(dict(wave_id='wave',subject={'subject_id':'s'},projects=['consumer']))
    assert len(calls)==1 and calls[0].memory_exposures==()
    assert ledger.stats(base['memory_id'])['decided']==1
    assert ledger.stats(base['memory_id'])['useful']==1


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


def test_investigation_can_confirm_correct_memory_without_version_or_counter_reset(tmp_path):
    campaign,ledger=setup(tmp_path);base=seed(campaign)
    current=observe(campaign,ledger,base,usefulness='unknown',correctness='unknown',investigate=True)
    repair=campaign.memory_maintenance.prioritize()
    source=current['outcome']['source_sha256']
    request=campaign.memory_maintenance.prepare(ProjectReview('confirm',source,True,{'source':source}),[repair['repair_id']])
    review=dict(decision='accept',memory=None,memory_resolutions=[dict(repair_id=repair['repair_id'],
        action='confirm',reason='Observed behavior is consistent with the documented preconditions',evidence=[source])])
    campaign.memory_maintenance.resolve('confirm',dict(review=review,review_request=asdict(request),review_sha256=campaign.artifacts.put_json(review)))
    assert campaign.memory_maintenance.obligations()==[]
    assert ledger.status(base['version'])=='active'
    assert ledger.stats(base['memory_id'],base['version'])['accesses']==1
    assert ledger.records()[0]==ledger.record(base['version'])


def test_ordinary_memory_named_project_repair_still_resolves(tmp_path):
    campaign,_=setup(tmp_path)
    campaign.scheduler.add_repair('memory-copy-001','s','original','Repair the project',['a'*64])
    review=dict(decision='accept',memory=None,reason='Repair passed',memory_resolutions=[])
    campaign.state['projects']['fixed']=dict(review=review,review_sha256='b'*64,repairs=['memory-copy-001'])
    campaign.scheduler.start_wave('wave','c'*64)
    campaign._publish_wave(dict(wave_id='wave',subject={'subject_id':'s'},projects=['fixed']))
    assert campaign.scheduler.state()['repairs']['memory-copy-001']['status']=='resolved'


@pytest.mark.parametrize('action',['replace','retire','confirm'])
def test_manual_continuation_applies_explicit_memory_resolution(tmp_path,action):
    from test_subject_interaction import setup as interaction_setup
    from benchmarks.ascend import subject_interaction as interaction
    from benchmarks.ascend.subject_execution import SubjectOutcome
    _,root,directory,scheduler,artifacts=interaction_setup(tmp_path)
    ledger=MemoryLedger(tmp_path/'shared.sqlite')
    campaign=SubjectCampaign(root,[Subject('study','study')],b'',select_wave=None,execute=None,
        verifier=None,complete=None,memory_ledger=ledger)
    campaign.budget=SimpleNamespace(deadline=campaign.clock()+30)
    base=seed(campaign)
    observe(campaign,ledger,base,project_id='one',correctness='unknown' if action=='confirm' else 'contradicted',
            usefulness='unknown',investigate=action=='confirm')
    repair=campaign.memory_maintenance.prioritize()
    original=campaign.state['projects']['one']
    original.update(plan=dict(project_id='one',spec={'project_id':'one'},repairs=[repair['repair_id']]),
        wave_index=1,origin_wave_id='wave-1',subject_id='study',repairs=[repair['repair_id']])
    source=artifacts.put_bytes(b'manually corrected source\n')
    result=artifacts.put_bytes(b'manually observed host result\n')
    campaign.execute=lambda *a,**kw:SubjectOutcome(True,source,{'source':source,'result':result},{},artifacts.root)
    def review(pid,record):
        request=campaign.memory_maintenance.prepare(ProjectReview(pid,source,True,{'source':source,'result':result}),record['repairs'])
        proposed=dict(proposal(source),operation='append',base_versions=[base]) if action=='replace' else None
        decision=dict(decision='accept',memory=proposed,memory_judgments=[],retirement_checks=[],
            memory_resolutions=[dict(repair_id=repair['repair_id'],action=action,reason='Manual project resolves dispute',evidence=[result])])
        record.update(review=decision,review_request=asdict(request),review_sha256=artifacts.put_json(decision))
    campaign._review=review
    request=dict(id='manual-resolution',kind='continue',project_id='one',feedback_key='repair',text='Resolve memory',additional_seconds=30)
    path=root/'request.json'
    interaction._execute(campaign,request,dict(original),directory,path)
    assert campaign.memory_maintenance.obligations()==[]
    assert scheduler.state()['repairs'][repair['repair_id']]['status']=='resolved'
    assert ledger.status(base['version'])==('quarantined' if action=='replace' else 'retired' if action=='retire' else 'active')


def test_cross_campaign_author_and_actor_can_read_only_assigned_disputed_context(tmp_path):
    from test_actor_compiler_diagnostics import actions,host
    from test_a5_phase1_adaptive_actor import SourceBackend
    from benchmarks.ascend.subject_knowledge import SubjectKnowledge
    origin,ledger=setup(tmp_path);base=seed(origin)
    observe(origin,ledger,base)
    other,_=setup(tmp_path,ledger=ledger,name='repair-project')
    other.memory_maintenance.reconcile()
    selected=other.memory_maintenance.prioritize()
    assigned=selected['memory_context']
    document=assigned['artifacts']['disputed-memory']
    assert json.loads(other.artifacts.read_bytes(document))==ledger.record(base['version'])
    assert any(other.artifacts.read_bytes(sha)==b'new host result\n' for sha in assigned['artifacts'].values())
    # The curriculum receives exact claim metadata and admitted artifact pointers.
    calls=[]
    def author(context,*args,**kwargs):
        assert context['repairs'][0]['memory_context']['claim']==assigned['claim']
        assert document in context['selection_artifacts'].values()
        return [dict(project_id='assigned-project',repairs=[selected['repair_id']])]
    def actor(item,snapshot,directory,**kwargs):
        assert item['memory_maintenance']==assigned
        current=actions(directory,SourceBackend())
        for sha in assigned['artifacts'].values():
            current.artifacts.put_bytes(other.artifacts.read_bytes(sha))
            current.admitted[sha]='memory-maintenance'
        current.observe_knowledge_read=lambda *a:pytest.fail('assigned repair evidence is not ordinary memory retrieval')
        page=host(current,'read-artifact',sha256=document)
        assert json.loads(page['text'])==ledger.record(base['version'])
        calls.append('read')
        raise TimeoutError('stop after inspecting actor repair context')
    other.select_wave=author;other.execute=actor
    other.run()
    assert calls==['read']
    assert ledger.status(base['version'])=='quarantined'
    assert ledger.stats(base['memory_id'])['accesses']==1  # only the original ordinary actor exposure


def test_manual_execution_episode_gets_new_exposure_but_resume_replays_it(tmp_path):
    from test_actor_compiler_diagnostics import actions
    from test_a5_phase1_adaptive_actor import SourceBackend
    from benchmarks.ascend.subject_knowledge import SubjectKnowledge
    ledger=MemoryLedger(tmp_path/'campaign/memory-ledger.sqlite3')
    campaign,_=setup(tmp_path,ledger=ledger);base=seed(campaign)
    knowledge=SubjectKnowledge(campaign.root,campaign.artifacts,lambda:campaign.memory)
    original=actions(tmp_path/'actor',SourceBackend());original.action_event_turn=43
    first=original._action_context({'opening_memory':'same'})
    presentation={'memories':[base]}
    knowledge.actor_delivery(presentation,first)
    project=first.request.spec.project_id
    old=ledger.exposures(str(campaign.root.resolve()),project)[0]
    ledger.judge(old['exposure_id'],usefulness='unknown',correctness='unknown',reason='Initial project interrupted',evidence=[])
    marker=original.request.branch_root/'manual-execution.json'
    marker.parent.mkdir(parents=True,exist_ok=True);marker.write_text(json.dumps({'attempt_id':'continued-attempt'}))
    resumed=actions(tmp_path/'actor',SourceBackend());resumed.action_event_turn=43
    second=resumed._action_context({'opening_memory':'same'})
    assert second.actor_id!=first.actor_id and second.access_event_id!=first.access_event_id
    knowledge.actor_delivery(presentation,second)
    again=actions(tmp_path/'actor',SourceBackend());again.action_event_turn=43
    replay=again._action_context({'opening_memory':'same'})
    assert replay.actor_id==second.actor_id and replay.access_event_id==second.access_event_id
    knowledge.actor_delivery(presentation,replay)
    source=campaign.artifacts.put_bytes(b'continued source\n');result=campaign.artifacts.put_bytes(b'continued conflicting result\n')
    request=campaign.memory_maintenance.prepare(ProjectReview(project,source,True,{'source':source,'result':result}))
    assert len(request.memory_exposures)==1 and request.memory_exposures[0]['actor_id']==second.actor_id
    judgment=dict(exposure_id=request.memory_exposures[0]['exposure_id'],usefulness='not-useful',correctness='contradicted',
                  investigate=False,reason='Continuation found a counterexample',evidence=[result])
    review=dict(decision='accept',memory=None,memory_judgments=[judgment],memory_resolutions=[],retirement_checks=[])
    record=dict(objective='learn tails',review=review,review_request=asdict(request),review_sha256=campaign.artifacts.put_json(review))
    campaign.memory_maintenance.observe(project,record)
    assert ledger.status(base['version'])=='quarantined'
    rows=ledger.exposures(str(campaign.root.resolve()),project)
    assert len(rows)==2 and sum(row['accesses'] for row in rows)==2
    assert next(row for row in rows if row['actor_id']==first.actor_id)['judgment']['correctness']=='unknown'
    assert campaign.memory_maintenance.prepare(ProjectReview(project,source,True,{'source':source})).memory_exposures==()


def test_indexed_without_reference_kdb_counts_reads_and_blocks_stale_memory(tmp_path):
    from dataclasses import replace
    from test_subject_runtime import bound
    from benchmarks.ascend.subject_runtime import build_subject_campaign
    from test_actor_compiler_diagnostics import actions,host
    from test_a5_phase1_adaptive_actor import SourceBackend
    binding=bound(tmp_path)
    campaign=build_subject_campaign({},target='a3',root=tmp_path/'campaign',subjects=[Subject('s','learn tails')],
        initial_memory=b'',compose=lambda *a,**kw:binding)
    base=seed(campaign)
    current=actions(tmp_path/'actor',SourceBackend())
    current.request=replace(current.request,pre_wave_memory=campaign.memory.snapshot())
    callback=binding['actor'].observe_knowledge_read
    assert callback.__self__.database_path is None
    current.observe_knowledge_read=callback
    source=campaign.memory.read(base['version'])['proposal']['after']['sha256']
    current.artifacts.put_bytes(campaign.artifacts.read_bytes(source))
    assert host(current,'read-artifact',sha256=source)['text']=='old source\n'
    ledger=campaign.memory_maintenance.ledger
    assert ledger.stats(base['memory_id'])['accesses']==1
    ledger.transition(base['version'],'quarantined',event_id='conflict',reason='Counterexample',evidence=['e'*64])
    assert 'withdrawn' in host(current,'read-artifact',sha256=source)['error']
    assert ledger.stats(base['memory_id'])['accesses']==1


def test_new_counterevidence_during_indexing_aborts_global_publication(tmp_path):
    from benchmarks.ascend.subject_memory import StaleMemory
    campaign,ledger=setup(tmp_path);base=seed(campaign)
    ledger.access('access-1', **base, campaign_id=str(campaign.root.resolve()),
        project_id='second-consumer', actor_id='second-consumer', actor_role='writer')
    first=observe(campaign,ledger,base)
    repair=campaign.memory_maintenance.prioritize()
    source=campaign.artifacts.put_bytes(b'corrected source\n')
    request=campaign.memory_maintenance.prepare(ProjectReview('fix',source,True,{'source':source}),[repair['repair_id']])
    corrected=dict(proposal(source),operation='append',base_versions=[base])
    review=dict(decision='accept',memory=corrected,memory_resolutions=[dict(repair_id=repair['repair_id'],
        action='replace',reason='Corrected example',evidence=[source])])
    record=dict(review=review,review_request=asdict(request),review_sha256=campaign.artifacts.put_json(review))
    provenance=dict(memory_maintenance_revisions=campaign.memory_maintenance.validate_resolution(record))
    index=campaign.memory.index
    def interrupted_by_new_evidence(version,value):
        index(version,value)
        observe(campaign,ledger,base,project_id='second-consumer',number=1,result_bytes=b'second distinct failure\n')
    campaign.memory.index=interrupted_by_new_evidence
    with pytest.raises(StaleMemory,match='counterevidence changed'):
        campaign.memory.accept('fix',corrected,provenance)
    assert ledger.status(base['version'])=='quarantined' and ledger.replacements(base['version'])==[]
    fresh=campaign.memory_maintenance.prepare(ProjectReview('fix',source,True,{'source':source}),[repair['repair_id']])
    assert fresh.memory_repairs[0]['episode_revision']!=request.memory_repairs[0]['episode_revision']
    assert len(fresh.memory_repairs[0]['concerns'])==2
    assert first['review']['memory_judgments'][0]['evidence'][0] in fresh.memory_repairs[0]['evidence']
    assert campaign.artifacts.put_bytes(b'second distinct failure\n') in fresh.memory_repairs[0]['evidence']
    assert campaign.memory_maintenance.resolve('fix',record) is False
    assert campaign.memory_maintenance.obligations()


def test_counterevidence_revision_refreshes_source_review_instead_of_publishing_cached_decision(tmp_path):
    campaign,ledger=setup(tmp_path);base=seed(campaign)
    ledger.access('access-1',**base,campaign_id=str(campaign.root.resolve()),project_id='late',actor_id='late',actor_role='writer')
    observe(campaign,ledger,base)
    repair=campaign.memory_maintenance.prioritize()
    source=campaign.artifacts.put_bytes(b'candidate correction\n')
    request=campaign.memory_maintenance.prepare(ProjectReview('fix',source,True,{'source':source}),[repair['repair_id']])
    proposal_before=dict(proposal(source),operation='append',base_versions=[base])
    review=dict(decision='accept',memory=proposal_before,memory_judgments=[],retirement_checks=[],
        memory_resolutions=[dict(repair_id=repair['repair_id'],action='replace',reason='Initial correction',evidence=[source])])
    record=dict(objective='learn tails',repairs=[repair['repair_id']],review=review,review_request=asdict(request),
        review_sha256=campaign.artifacts.put_json(review),outcome=dict(host_verified=True,source_sha256=source,
        artifacts={'source':source},observations={}))
    campaign.state['projects']['fix']=record
    observe(campaign,ledger,base,project_id='late',number=1,result_bytes=b'new distinct counterexample\n')
    calls=[]
    class Reviewer:
        last_trace=[]
        def review(self,request,**kwargs):
            calls.append(request)
            assert len(request.memory_repairs[0]['concerns'])==2
            return dict(decision='repair',reason='New case needs a project',memory=None,
                evidence=[source],repair={'instructions':'Test the new case','expected_evidence':'Correct output'},
                memory_judgments=[],memory_resolutions=[],retirement_checks=[])
    campaign.verifier=Reviewer();campaign.budget=SimpleNamespace(deadline=campaign.clock()+30)
    campaign.scheduler.start_wave('wave','a'*64)
    campaign._publish_wave(dict(wave_id='wave',subject={'subject_id':'s'},projects=['fix']))
    assert len(calls)==1
    assert ledger.replacements(base['version'])==[] and ledger.status(base['version'])=='quarantined'
    assert campaign.memory_maintenance.obligations()


def test_retirement_between_source_review_and_publication_requires_new_review(tmp_path):
    from benchmarks.ascend.subject_memory import StaleMemory
    campaign,ledger=setup(tmp_path);base=seed(campaign)
    source=campaign.artifacts.put_bytes(b'new source\n')
    request=campaign.memory_maintenance.prepare(ProjectReview('new',source,True,{'source':source}))
    assert request.retired_memories==()
    ledger.transition(base['version'],'retired',event_id='new-retirement',reason='New contradictory evidence',evidence=[source])
    with pytest.raises(StaleMemory,match='retirement history changed'):
        campaign.memory.accept('new',proposal(source),{'observed_retirements':[]})
    refreshed=campaign.memory_maintenance.prepare(ProjectReview('new',source,True,{'source':source}))
    assert [row['tombstone_id'] for row in refreshed.retired_memories]==['new-retirement']
    assert ledger.records()==[]


def test_retirement_commit_retains_new_proof_before_origin_disappears(tmp_path,monkeypatch):
    campaign,ledger=setup(tmp_path);base=seed(campaign)
    observe(campaign,ledger,base)
    repair=campaign.memory_maintenance.prioritize()
    proof=campaign.artifacts.put_bytes(b'new retirement-only independent evidence\n')
    request=campaign.memory_maintenance.prepare(ProjectReview('retire',proof,True,{'source':proof}),[repair['repair_id']])
    review=dict(decision='accept',memory=None,memory_resolutions=[dict(repair_id=repair['repair_id'],
        action='retire',reason='Retired after new independent proof',evidence=[proof])])
    record=dict(review=review,review_request=asdict(request),review_sha256=campaign.artifacts.put_json(review))
    transition=ledger.transition
    def interrupted(*args,**kwargs):
        transition(*args,**kwargs)
        raise KeyboardInterrupt('origin lost immediately after retirement commit')
    monkeypatch.setattr(ledger,'transition',interrupted)
    with pytest.raises(KeyboardInterrupt):campaign.memory_maintenance.resolve('retire',record)
    campaign.artifacts.root.rename(campaign.root/'offline-origin')
    other,_=setup(tmp_path,ledger=ledger,name='other')
    src=other.artifacts.put_bytes(b'next project\n')
    prepared=other.memory_maintenance.prepare(ProjectReview('next',src,True,{'source':src}))
    assert prepared.retired_memories[0]['evidence']==[proof]
    assert other.memory_maintenance.read_artifact(proof,remaining_seconds=1)==b'new retirement-only independent evidence\n'


def test_shared_later_subject_repair_runs_at_the_consumers_reachable_frontier(tmp_path):
    subjects=[Subject('early','learn tails',1),Subject('later','learn reductions',1)]
    ledger=MemoryLedger(tmp_path/'shared.sqlite3')
    def make(name):
        return SubjectCampaign(tmp_path/name,subjects,b'',select_wave=lambda *a,**kw:[],
            execute=lambda *a,**kw:None,verifier=None,complete=lambda *a,**kw:None,
            memory_ledger=ledger)
    origin=make('advanced');base=seed(origin)
    origin.scheduler.start_wave('early-wave','a'*64)
    origin.scheduler.complete_wave('early-wave',[])
    origin.scheduler.decide(dict(action='advance',reason='Initial subject complete',evidence=['a'*64]))
    observe(origin,ledger,base,objective='learn reductions')
    consumer=make('beginner')
    consumer.memory_maintenance.reconcile()
    local=consumer.memory_maintenance.prioritize()
    assert consumer.scheduler.state()['frontier']==0
    assert local['subject_id']=='early'
    assert consumer.scheduler.context()['current_subject']['subject_id']=='early'
    assert consumer.memory_maintenance.obligations()[0]['subject_id']=='later'
    consumer.memory_maintenance.reconcile()
    assert consumer.scheduler.state()['repairs'][local['repair_id']]['subject_id']=='early'
