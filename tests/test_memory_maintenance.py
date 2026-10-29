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
            usefulness='not-useful', correctness='contradicted', investigate=False):
    source = campaign.artifacts.put_bytes(b'new source\n')
    result = campaign.artifacts.put_bytes(b'new host result\n')
    exposure = ledger.access('access-' + str(number), **base, campaign_id=str(campaign.root.resolve()),
        project_id=project_id, actor_id=project_id, actor_role='writer')
    review = dict(decision='accept', memory=None, memory_judgments=[dict(exposure_id=exposure,
        usefulness=usefulness, correctness=correctness, investigate=investigate,
        reason='Current project provides independent evidence', evidence=[result])],
        memory_resolutions=[], retirement_checks=[])
    record = dict(objective='learn tails', review=review, review_sha256=campaign.artifacts.put_json(review),
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
    assert service.ledger.stats('lesson',base['version'])['accesses']==2
    assert service.ledger.stats('lesson',base['version'])['exposures']==1
    service.ledger.transition(base['version'],'quarantined',event_id='conflict',reason='Conflicting evidence',evidence=['e'*64])
    with pytest.raises(ValueError,match='withdrawn'):
        service.actor_read({'sha256':source},actor,page)
    unrelated=store.put_bytes(b'unrelated reference')
    service.actor_read({'sha256':unrelated},actor,page)
    assert service.ledger.stats('lesson',base['version'])['accesses']==2


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
