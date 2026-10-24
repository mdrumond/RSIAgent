import json
import threading

from benchmarks.ascend.memory_admission import MemoryAdmission, treatment_snapshot
from benchmarks.ascend.subject_memory import SubjectMemory
from test_subject_campaign import proposal


def draft():
    return proposal('a'*64)


def request():
    return dict(project_id='p',source_sha256='a'*64,host_verified=True,artifacts={'source':'a'*64})


class Evaluation:
    def __init__(self, decisions):
        self.decisions=iter(decisions)
        self.calls=[]
    def evaluate(self, proposed, baseline, treatment, directory, **kw):
        self.calls.append((proposed,baseline,treatment,kw))
        return dict(decision=next(self.decisions),reason='observed transfer',instructions='clarify scope',
            evidence=['a'*64],trial_task=kw.get('task') or {'frozen':'task'},arms={})


def test_two_rewrites_then_reject_without_publishing(tmp_path):
    memory=SubjectMemory(tmp_path/'memory',b'initial')
    evaluation=Evaluation(['rewrite','reject','rewrite'])
    rewrites=[]
    def rewrite(req,feedback,max_seconds):
        rewrites.append(feedback)
        return dict(decision='accept',memory=dict(draft(),summary=f'revision{len(rewrites)}'))
    gate=MemoryAdmission(tmp_path/'gate',memory,evaluation,rewrite)
    try:
        gate.submit('p',draft(),{},request(),remaining_seconds=100)
        gate.wait(remaining_seconds=5)
        result=gate.poll(remaining_seconds=5)
        assert result[0]['outcome']=='rejected'
        assert len(rewrites)==2 and len(evaluation.calls)==3
        assert memory.records()==[]
        assert evaluation.calls[1][3]['task']==evaluation.calls[2][3]['task']=={'frozen':'task'}
    finally:gate.close()


def test_pending_draft_is_invisible_then_rewrite_is_accepted(tmp_path):
    memory=SubjectMemory(tmp_path/'memory',b'initial')
    started, release=threading.Event(),threading.Event()
    class Paused(Evaluation):
        def evaluate(self,*args,**kw):
            if not self.calls:
                started.set()
                assert release.wait(3)
            return super().evaluate(*args,**kw)
    evaluator=Paused(['rewrite','accept'])
    rewritten=dict(draft(),summary='precise scope')
    gate=MemoryAdmission(tmp_path/'gate',memory,evaluator,
        lambda *a,**k:dict(decision='accept',memory=rewritten))
    old=memory.snapshot()
    try:
        gate.submit('p',draft(),{},request(),remaining_seconds=100)
        assert started.wait(3)
        assert memory.snapshot()==old and memory.search('tail')==[]
        release.set();gate.wait(remaining_seconds=5)
        assert gate.poll(remaining_seconds=5)[0]['outcome']=='accepted'
        assert memory.records()[0]['proposal']['summary']=='precise scope'
        assert evaluator.calls[0][1]==evaluator.calls[1][1]
        assert evaluator.calls[0][2]!=evaluator.calls[1][2]
    finally:release.set();gate.close()


def test_replacement_arm_uses_prior_memory_and_keeps_unrelated_memory(tmp_path):
    memory=SubjectMemory(tmp_path,b'initial')
    original=memory.accept('old',draft(),{})
    memory.accept('unrelated',dict(draft(),summary='other operation'),{})
    baseline=memory.snapshot()
    replacement=dict(draft(),operation='merge',base_versions=[original],summary='merged')
    treatment=treatment_snapshot(baseline,replacement,{},'pending')
    before=json.loads(baseline)['memories'];after=json.loads(treatment)['memories']
    assert len(before)==len(after)==2
    assert next(r for r in before if r['memory_id']=='unrelated')==next(r for r in after if r['memory_id']=='unrelated')
    assert next(r for r in after if r['memory_id']=='old')['proposal']['summary']=='merged'
    assert memory.snapshot()==baseline


def test_inconclusive_does_not_rewrite_or_admit(tmp_path):
    memory=SubjectMemory(tmp_path/'memory',b'')
    evaluation=Evaluation(['inconclusive'])
    def forbidden(*a,**k):raise AssertionError('invalid experiment must not rewrite')
    gate=MemoryAdmission(tmp_path/'gate',memory,evaluation,forbidden)
    try:
        gate.submit('p',draft(),{},request(),remaining_seconds=100)
        gate.wait(remaining_seconds=5)
        assert gate.poll(remaining_seconds=5)==[]
        job=json.loads(gate._path('p').read_bytes())
        assert job['status']=='inconclusive' and job['rewrites']==0
        assert memory.records()==[]
    finally:gate.close()


def test_stale_base_is_reconciled_without_consuming_rewrite_chances(tmp_path):
    from benchmarks.ascend.subject_memory import digest
    memory=SubjectMemory(tmp_path/'memory',b'')
    old=memory.accept('old',draft(),{})
    started,release=threading.Event(),threading.Event()
    class Paused(Evaluation):
        def evaluate(self,*a,**kw):
            if not self.calls:
                started.set();assert release.wait(3)
            return super().evaluate(*a,**kw)
    evaluation=Paused(['accept','accept'])
    proposed=dict(draft(),operation='append',base_versions=[old])
    def rewrite(req,feedback,max_seconds):
        record=memory.records()[0]
        return dict(decision='accept',memory=dict(proposed,base_versions=[dict(
            memory_id=record['memory_id'],version=digest(record))]))
    gate=MemoryAdmission(tmp_path/'gate',memory,evaluation,rewrite)
    try:
        gate.submit('pending',proposed,{},request(),remaining_seconds=100)
        assert started.wait(3)
        memory.accept('concurrent',dict(proposed,summary='concurrent accepted change'),{})
        release.set();gate.wait(remaining_seconds=5)
        assert gate.poll(remaining_seconds=5)[0]['outcome']=='accepted'
        job=json.loads(gate._path('pending').read_bytes())
        assert job['rewrites']==0 and job['revision']==1
        assert evaluation.calls[0][1]!=evaluation.calls[1][1]
    finally:release.set();gate.close()


def test_other_waves_run_on_stale_memory_while_verification_is_pending(tmp_path):
    from benchmarks.ascend.subject_campaign import SubjectCampaign
    from benchmarks.ascend.subjects import Subject
    from benchmarks.ascend.subject_execution import SubjectOutcome
    started,release=threading.Event(),threading.Event()
    class Paused(Evaluation):
        def evaluate(self,*a,**kw):
            if not self.calls:
                started.set();assert release.wait(3)
            return super().evaluate(*a,**kw)
    snapshots=[]
    class Review:
        last_trace=[]
        def review(self,req,**kw):return dict(decision='accept',reason='host checked',
            evidence=[req.source_sha256],repair=None,memory=draft())
    def execute(item,snapshot,*a,**kw):
        snapshots.append(snapshot)
        if len(snapshots)==2:
            assert started.wait(3);release.set()
        sha=c.artifacts.put_bytes(b'source\n')
        return SubjectOutcome(True,sha,{'source':sha},{},c.artifacts.root)
    c=SubjectCampaign(tmp_path,[Subject('one','one',1),Subject('two','two',1)],b'',
        select_wave=lambda context,*a,**kw:[dict(project_id=context['current_subject']['subject_id'])],
        execute=execute,verifier=Review(),complete=lambda *a,**kw:json.dumps(dict(
            action='advance',reason='complete wave',evidence=['retained'])))
    gate=MemoryAdmission(tmp_path/'gate',c.memory,Paused(['accept','accept']),None,parent_remaining=c._remaining)
    c.memory_gate=gate
    try:
        assert c.run()['terminal']=='COMPLETE'
        assert snapshots[0]==snapshots[1]
        assert len(c.memory.records())==2
        assert all(row['status']=='accepted' for row in c.scheduler.state()['reviews'].values())
    finally:release.set();gate.close()
