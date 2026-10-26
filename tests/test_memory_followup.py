"""Durable targeted probes use actual callbacks, reads and admission state."""
import json
from pathlib import Path

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.ascend.memory_admission import MemoryAdmission
from benchmarks.ascend.memory_verifier import MemoryVerifier
from benchmarks.ascend.subject_execution import SubjectOutcome
from benchmarks.ascend.subject_memory import SubjectMemory
from test_memory_admission import draft, request


class Probes:
    def __init__(self, decisions=('probe', 'accept')):
        self.now = 0
        self.decisions = decisions
        self.executions, self.selections, self.reads = [], [], []
        self.failure = False
        self.verifier = MemoryVerifier(self.complete, self.select, self.execute,
            model_identity='model', runtime_identity='synthetic', max_seconds=100,
            arm_seconds=10, clock=lambda:self.now, wall_clock=lambda:self.now)

    def select(self, proposal, *, remaining_seconds, **kwargs):
        self.selections.append(kwargs)
        self.now += 1
        return dict(project_id='probe', inputs=[31+len(self.selections)])

    def execute(self, task, memory, directory, *, remaining_seconds):
        self.executions.append((Path(directory), task, memory, remaining_seconds))
        self.now += 1
        if self.failure:
            self.failure = False
            raise ConnectionError('unsettled operation retained by adapter')
        store = ContentAddressedArtifacts(Path(directory)/'artifacts')
        log = store.put_json(dict(task=task, arm=Path(directory).name,
            effect='unfavorable earlier evidence' if task['inputs']==[32] else 'new contrast'))
        result = store.put_json(dict(host_verified=True, score=1))
        return SubjectOutcome(True, None, {'log:actor':log,'result':result},
                              {'metrics':{'score':1}}, store.root)

    def complete(self, messages, *, remaining_seconds):
        self.now += .1
        context = json.loads(messages[1]['content'])
        observations = [json.loads(m['content']) for m in messages[2:] if m['role']=='user']
        listings = [o for o in observations if 'artifacts'in o and 'total'in o]
        if not listings:
            return json.dumps({'list_artifacts':{'offset':0,'limit':50}})
        registry = {k:v for o in listings for k,v in o['artifacts'].items()}
        if listings[-1]['next_offset'] is not None:
            return json.dumps({'list_artifacts':{'offset':listings[-1]['next_offset'],'limit':50}})
        read = {o['sha256']:o['text'] for o in observations if 'sha256'in o}
        required = {v for k,v in registry.items() if any(x in k for x in ('log','result','evaluation'))}
        for sha in sorted(required-read.keys()):
            return json.dumps({'read_artifact':{'sha256':sha}})
        self.reads.append(read)
        choice = self.decisions[min(context['history_count'],len(self.decisions)-1)]
        value = dict(decision='inconclusive'if choice=='probe'else choice,
            reason='Prior unfavorable trial and this contrast considered',
            evidence=sorted(read),instructions='Narrow the unsupported claim'if choice=='rewrite'else None)
        if choice=='probe':value['next_probe']={'question':'Does an odd extent expose a failure absent from the aligned workload?'}
        return json.dumps(value)


def test_real_admission_runs_fresh_targeted_pair_and_keeps_prior_evidence(tmp_path):
    h=Probes();memory=SubjectMemory(tmp_path/'memory',b'initial')
    gate=MemoryAdmission(tmp_path/'gate',memory,h.verifier,None,clock=lambda:h.now)
    try:
        gate.submit('p',draft(),{},request(),remaining_seconds=60)
        gate.wait(remaining_seconds=5)
        assert gate.poll(remaining_seconds=5)[0]['outcome']=='accepted'
        job=json.loads(gate._path('p').read_bytes())
        assert [r['decision']for r in job['evaluations']]==['inconclusive','accept']
        assert [p.name for p,*_ in h.executions]==['control','treatment','treatment','control']
        assert len({str(p)for p,*_ in h.executions})==4
        assert h.selections[1]['probe_request']['question']
        assert h.selections[1]['prior_task']['inputs']==[32]
        assert h.selections[1]['history']==job['evaluations'][:1]
        prior_log=job['evaluations'][0]['arms']['control']['artifacts']['log:actor']
        assert 'unfavorable earlier evidence'in h.reads[-1][prior_log]
        assert any(k.startswith('history:')for k in job['evaluations'][-1]['artifacts'])
        proof=memory.records()[0]['provenance']['memory_verification']
        assert len(proof['evaluations'])==2 and len(proof['traces'])==2
    finally:gate.close()


def test_cumulative_history_supports_reject_despite_both_host_passing(tmp_path):
    h=Probes(('probe','reject'))
    first=h.verifier.evaluate(draft(),b'old',b'new',tmp_path/'first',remaining_seconds=60)
    prior=ContentAddressedArtifacts(tmp_path/'first/artifacts')
    result=h.verifier.evaluate(draft(),b'old',b'new',tmp_path/'second',remaining_seconds=60,
        history=[first],probe_request=first['next_probe'],prior_task=first['trial_task'],
        read_history_artifact=prior.read_bytes,sequence=1)
    assert result['decision']=='reject' and all(x['host_verified']for x in result['arms'].values())
    assert h.selections[1]['history'][0]['reason']==first['reason']
    count=len(h.executions)
    assert h.verifier.evaluate(draft(),b'old',b'new',tmp_path/'first',remaining_seconds=60)==first
    assert len(h.executions)==count
    with pytest.raises(ValueError,match='identity changed'):
        h.verifier.evaluate(draft(),b'old',b'new',tmp_path/'second',remaining_seconds=60,
            history=[],probe_request=first['next_probe'],sequence=1)


def test_selector_cannot_rename_same_workload_into_additional_evidence(tmp_path):
    h=Probes();first=h.verifier.evaluate(draft(),b'a',b'b',tmp_path/'a',remaining_seconds=60)
    h.verifier.select_trial=lambda *a,**kw:dict(first['trial_task'],project_id='renamed')
    result=h.verifier.evaluate(draft(),b'a',b'b',tmp_path/'b',remaining_seconds=60,
        history=[first],probe_request=first['next_probe'],prior_task=first['trial_task'],
        read_history_artifact=ContentAddressedArtifacts(tmp_path/'a/artifacts').read_bytes)
    assert result['decision']=='inconclusive' and result['error_type']=='ValueError'
    assert len(h.executions)==2


@pytest.mark.parametrize('certified',[False,True])
def test_invalid_retry_requires_explicit_settlement_and_uses_new_directories(tmp_path,certified):
    h=Probes(('accept',));h.failure=True;memory=SubjectMemory(tmp_path/'memory',b'')
    certifications=[]
    def ready(result,directory,**kw):
        certifications.append((result,directory))
        return certified
    gate=MemoryAdmission(tmp_path/'gate',memory,h.verifier,None,retry_ready=ready,clock=lambda:h.now)
    try:
        gate.submit('p',draft(),{},request(),remaining_seconds=60);gate.wait(remaining_seconds=5)
        result=gate.poll(remaining_seconds=5)
        assert len(certifications)==1
        assert len(h.executions)==(4 if certified else 2)
        if certified:
            assert result[0]['outcome']=='accepted'
            assert h.executions[0][0]!=h.executions[3][0]
        else:
            assert not result and not memory.records()
            gate.close();gate.start_session();gate.poll(remaining_seconds=5);gate.wait(remaining_seconds=5)
            assert len(h.executions)==2
    finally:gate.close()


def test_elapsed_admission_not_probe_count_stops_continuation(tmp_path):
    h=Probes(('probe',));memory=SubjectMemory(tmp_path/'memory',b'')
    gate=MemoryAdmission(tmp_path/'gate',memory,h.verifier,None,max_seconds=9,clock=lambda:h.now)
    try:
        gate.submit('p',draft(),{},request(),remaining_seconds=9);gate.wait(remaining_seconds=5)
        gate.poll(remaining_seconds=5)
        job=json.loads(gate._path('p').read_bytes())
        assert job['status']=='inconclusive'and not memory.records()
        assert json.loads((gate._path('p').parent/'elapsed.json').read_text())['elapsed_seconds']>=9
        assert len({str(p)for p,*_ in h.executions})==len(h.executions)
    finally:gate.close()


def test_completed_uncertainty_is_not_rescheduled_on_worker_restart(tmp_path):
    h=Probes(('inconclusive',));memory=SubjectMemory(tmp_path/'memory',b'')
    gate=MemoryAdmission(tmp_path/'gate',memory,h.verifier,None,clock=lambda:h.now)
    try:
        gate.submit('p',draft(),{},request(),remaining_seconds=60);gate.wait(remaining_seconds=5)
        assert not gate.poll(remaining_seconds=5)
        elapsed=json.loads((gate._path('p').parent/'elapsed.json').read_text())
        gate.close();gate.start_session()
        assert not gate.poll(remaining_seconds=5)and gate.future is None
        assert len(h.executions)==2 and len(h.reads)==1
        assert json.loads((gate._path('p').parent/'elapsed.json').read_text())==elapsed
    finally:gate.close()


def test_restart_before_followup_dispatch_preserves_history_ordinal_and_budget(tmp_path):
    h=Probes();memory=SubjectMemory(tmp_path/'memory',b'')
    original=h.verifier.evaluate;paused=[]
    def evaluate(*args,**kwargs):
        if 'probe-1'in str(args[3])and not paused:
            paused.append(True);h.now+=3
            raise TimeoutError('parent session paused before new trial dispatch')
        return original(*args,**kwargs)
    h.verifier.evaluate=evaluate
    gate=MemoryAdmission(tmp_path/'gate',memory,h.verifier,None,clock=lambda:h.now)
    try:
        gate.submit('p',draft(),{},request(),remaining_seconds=60);gate.wait(remaining_seconds=5)
        gate.poll(remaining_seconds=5);before=json.loads(gate._path('p').read_text())
        assert before['trial_ordinal']==1 and before['trial_history_length']==1
        assert len(h.executions)==2 and not memory.records()
        spent=json.loads((gate._path('p').parent/'elapsed.json').read_text())['elapsed_seconds']
        gate.close();gate.start_session();gate.poll(remaining_seconds=5);gate.wait(remaining_seconds=5)
        assert gate.poll(remaining_seconds=5)[0]['outcome']=='accepted'
        after=json.loads(gate._path('p').read_text())
        assert after['trial_ordinal']==1 and len(after['evaluations'])==2
        assert [p.name for p,*_ in h.executions]==['control','treatment','treatment','control']
        assert json.loads((gate._path('p').parent/'elapsed.json').read_text())['elapsed_seconds']>spent
    finally:gate.close()


def test_missing_history_feedback_drives_exact_reads_then_finishes(tmp_path):
    from benchmarks.ascend.memory_verifier import MemoryVerifierAgent
    store=ContentAddressedArtifacts(tmp_path/'artifacts')
    current_log=store.put_bytes(b'current Actor log')
    current_result=store.put_bytes(b'current host result')
    prior_log=store.put_bytes(b'prior unfavorable Actor evidence')
    history=[dict(decision='inconclusive',reason='earlier uncertainty',artifacts={'log:actor':prior_log})]
    arms={name:SubjectOutcome(True,None,{'log:actor':current_log,'result':current_result},{},store.root)
          for name in ['control','treatment']}
    corrections=[]
    def complete(messages,**kwargs):
        observations=[json.loads(m['content'])for m in messages[2:]if m['role']=='user']
        read={o['sha256']for o in observations if 'sha256'in o}
        for sha in [current_log,current_result]:
            if sha not in read:return json.dumps({'read_artifact':{'sha256':sha}})
        feedback=[o for o in observations if 'missing_required'in o]
        if feedback:
            corrections.extend(feedback[-1:])
            for entry in feedback[-1]['missing_required']:
                if entry['sha256']not in read:
                    return json.dumps({'read_artifact':{'sha256':entry['sha256']}})
        return json.dumps(dict(decision='inconclusive',reason='Actual current and prior evidence compared',
                               evidence=sorted(read),instructions=None))
    judge=MemoryVerifierAgent(complete)
    result=judge.review(draft(),arms,lambda sha,**kw:store.read_bytes(sha),max_seconds=30,history=history)
    feedback=[e['observation']for e in judge.last_trace if e['kind']=='feedback']
    assert len(feedback)==1 and feedback[0]['missing_required_count']==2
    entries=feedback[0]['missing_required']
    assert next(e for e in entries if e['sha256']==prior_log)['roles']==['history:0:log:actor']
    assert any(e['roles']==['history:0:evaluation']for e in entries)
    assert feedback[0]['discovery']['list_artifacts']['prefix']=='history:'
    assert prior_log in result['evidence']and result['decision']=='inconclusive'
    assert {e['observation']['sha256']for e in judge.last_trace if e['kind']=='read'}>=set(result['evidence'])
