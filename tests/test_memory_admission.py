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
