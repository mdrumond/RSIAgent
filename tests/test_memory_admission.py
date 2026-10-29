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
        assert result[0]['assessment']['status']=='rejected'
        assert gate.report()['p']['assessment']['reason_code']=='rewrite-opportunities-exhausted'
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
        assert gate.report()['p']['assessment']['status']=='benefit-unproven'
        # Legacy held records get diagnostics without changing cached decisions
        # or triggering another evaluator callback from a reporting operation.
        job.pop('assessment')
        gate._path('p').write_text(json.dumps(job))
        before = gate._path('p').read_bytes()
        assert gate.report()['p']['assessment']['status']=='benefit-unproven'
        assert gate._path('p').read_bytes()==before and len(evaluation.calls)==1
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


def test_rewrite_to_newly_accepted_merge_target_refreshes_control_memory(tmp_path):
    memory=SubjectMemory(tmp_path/'memory',b'')
    started,release=threading.Event(),threading.Event()
    class Paused(Evaluation):
        def evaluate(self,*args,**kw):
            if not self.calls:
                started.set();assert release.wait(3)
            return super().evaluate(*args,**kw)
    evaluator=Paused(['rewrite','accept'])
    target={}
    def rewrite(*a,**kw):
        return dict(decision='accept',memory=dict(draft(),operation='append',base_versions=[target['base']]))
    gate=MemoryAdmission(tmp_path/'gate',memory,evaluator,rewrite)
    try:
        gate.submit('p',draft(),{},request(),remaining_seconds=100)
        assert started.wait(3)
        target['base']=memory.accept('other',dict(draft(),summary='related accepted memory'),{})
        release.set();gate.wait(remaining_seconds=5)
        assert gate.poll(remaining_seconds=5)[0]['outcome']=='accepted'
        assert json.loads(evaluator.calls[0][1])['memories']==[]
        assert json.loads(evaluator.calls[1][1])['memories'][0]['memory_id']=='other'
    finally:release.set();gate.close()


def test_reopen_worker_preserves_elapsed_allocation(tmp_path):
    now=[0.0]
    class Timed(Evaluation):
        def evaluate(self,*a,**kw):
            now[0]+=3
            if now[0]==3:
                raise TimeoutError('paused before a judgment was completed')
            return super().evaluate(*a,**kw)
    memory=SubjectMemory(tmp_path/'memory',b'')
    gate=MemoryAdmission(tmp_path/'gate',memory,Timed(['accept']),None,clock=lambda:now[0])
    try:
        gate.submit('p',draft(),{},request(),remaining_seconds=100)
        gate.wait(remaining_seconds=5);gate.poll(remaining_seconds=5);gate.close()
        elapsed=gate._path('p').parent/'elapsed.json'
        assert json.loads(elapsed.read_text())['elapsed_seconds']==3
        gate.start_session();gate.poll(remaining_seconds=5);gate.wait(remaining_seconds=5)
        assert gate.poll(remaining_seconds=5)[0]['outcome']=='accepted'
        assert json.loads(elapsed.read_text())['elapsed_seconds']==6
    finally:gate.close()


def test_parent_budget_includes_background_drain_before_return(tmp_path):
    from benchmarks.ascend.subject_campaign import SubjectCampaign
    from benchmarks.ascend.subjects import Subject
    from benchmarks.ascend.subject_execution import SubjectOutcome
    now=[0.0];started=threading.Event()
    class Pending(Evaluation):
        def evaluate(self,*a,**kw):
            started.set();assert gate.stop_requested.wait(3)
            now[0]+=10
            return dict(decision='inconclusive',reason='paused',instructions=None,evidence=[],arms={})
    class Review:
        last_trace=[]
        calls=0
        def review(self,req,**kw):
            self.calls+=1
            if self.calls==2:
                assert started.wait(3)
                raise TimeoutError('main provider request timed out')
            return dict(decision='accept',reason='verified',evidence=[req.source_sha256],repair=None,memory=draft())
    def execute(*a,**kw):
        sha=c.artifacts.put_bytes(b'source\n')
        return SubjectOutcome(True,sha,{'source':sha},{},c.artifacts.root)
    c=SubjectCampaign(tmp_path,[Subject('one','one',1),Subject('two','two',1)],b'',
        select_wave=lambda context,*a,**kw:[dict(project_id=context['current_subject']['subject_id'])],
        execute=execute,verifier=Review(),complete=lambda *a,**kw:json.dumps(dict(
            action='advance',reason='complete',evidence=['retained'])),max_seconds=100,clock=lambda:now[0])
    gate=MemoryAdmission(tmp_path/'gate',c.memory,Pending([]),None,parent_remaining=c._remaining,clock=lambda:now[0])
    c.memory_gate=gate
    result=c.run()
    assert result['terminal']=='REVIEW_PENDING'
    assert c.memory.records()==[]
    assert json.loads((tmp_path/'elapsed.json').read_text())['elapsed_seconds']==10


def test_large_rewrite_uses_bounded_feedback_and_readable_exact_cas_documents(tmp_path):
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider, DirectCompletion, direct_profile
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.ascend.agent_context import ContextBudget, request_bytes
    from benchmarks.ascend.project_verifier import ProjectVerifierAgent
    from optimizer.jobs import canonical
    from test_project_verifier_agent import S, SOURCE, R, RESULT
    memory = SubjectMemory(tmp_path/'memory', b'')
    store = ContentAddressedArtifacts(tmp_path/'artifacts')
    store.put_bytes(SOURCE);store.put_bytes(RESULT)
    proposed = dict(draft(), summary='retain lesson scope')
    proposed['after']['text'] = 'large code\n' * 20000
    feedback = dict(decision='rewrite', reason='Observed limited transfer', instructions='Clarify preconditions',
                    arms={'treatment': dict(host_verified=True, source_sha256=S,
                        artifacts={f'retained:{n}': R for n in range(500)},
                        observations={'actor_logs': 'long trace\n' * 30000})}, evidence=[R])
    profile = direct_profile(next(row for row in MODEL_IDENTITIES if row.backend_model.value == 'deepseek-flash'))
    requests, phase = [], [0]
    def transport(**options):
        wire = options['request'];requests.append(wire)
        if phase[0] == 0:
            context = json.loads(wire['messages'][1]['content'])
            bounded = context['rewrite_feedback']
            assert bounded['previous_proposal']['summary'] == proposed['summary']
            assert bounded['usefulness']['instructions'] == feedback['instructions']
            assert store.read_bytes(bounded['previous_proposal']['sha256']) == canonical(proposed)
            assert store.read_bytes(bounded['usefulness']['sha256']) == canonical(feedback)
            phase[0] += 1
            return DirectCompletion(json.dumps({'read_artifact': {'sha256': bounded['usefulness']['sha256'], 'limit': 200}}), 1)
        observation = json.loads(wire['messages'][-1]['content'])
        assert observation['size'] == len(canonical(feedback))
        return DirectCompletion(json.dumps(dict(decision='repair', reason='Need stronger project evidence',
            evidence=[observation['sha256']], repair=dict(instructions='Run a control', expected_evidence='Independent outputs'), memory=None)), 1)
    provider = DirectCompletionProvider({profile.identity.credential_env: 'fixture-only'}, transport=transport,
        models=[profile.identity.backend_model], context_budgets={'deepseek-flash': ContextBudget(40000,32768,49152)})
    agent = ProjectVerifierAgent(lambda messages, **kw: provider.complete(profile, messages).text,
        lambda sha, **kw: store.read_bytes(sha))
    gate = MemoryAdmission(tmp_path/'gate', memory, Evaluation([]), agent.review)
    try:
        result = gate._rewrite(dict(request=dict(project_id='p', source_sha256=S, host_verified=True,
            artifacts={'source': S, 'result': R}), proposal=proposed), feedback, 60)
        assert result['decision'] == 'repair'
        assert all(request_bytes(wire) < 49152 for wire in requests)
    finally:
        gate.close()


def test_rewrite_provider_outage_retains_inconclusive_job_for_resume(tmp_path):
    from benchmarks.a5kernels.phase1_live import InfrastructureFailure
    memory = SubjectMemory(tmp_path/'memory', b'initial')
    def unavailable(*args, **kwargs):
        raise InfrastructureFailure('temporary provider outage')
    gate = MemoryAdmission(tmp_path/'gate', memory, Evaluation(['rewrite']), unavailable)
    try:
        gate.submit('p', draft(), {}, request(), remaining_seconds=100)
        gate.wait(remaining_seconds=5)
        assert gate.poll(remaining_seconds=5) == []
        retained = json.loads(gate._path('p').read_bytes())
        assert retained['status'] == 'inconclusive'
        assert retained['error_type'] == 'InfrastructureFailure'
        assert gate.report()['p']['assessment']['status'] == 'infrastructure-blocked'
        assert gate.report()['p']['reason'] == retained['error'] == 'temporary provider outage'
        assert retained['rewrites'] == 0 and memory.records() == []
    finally:
        gate.close()
    resumed = MemoryAdmission(tmp_path/'gate', memory, Evaluation(['accept']), unavailable)
    try:
        resumed.wait(remaining_seconds=5)
        assert resumed.poll(remaining_seconds=5)[0]['outcome'] == 'accepted'
        shown = resumed.report()['p']
        assert shown['assessment']['status'] == 'accepted'
        assert shown['reason'] == 'observed transfer' and shown['error_type'] is None
        assert 'error' not in json.loads(resumed._path('p').read_bytes())
        assert len(memory.records()) == 1
    finally:
        resumed.close()


def test_resumed_device_retries_follow_submission_order_not_hashed_directories(tmp_path):
    from benchmarks.ascend.subject_memory import digest
    memory = SubjectMemory(tmp_path / 'memory', b'')
    earlier, later = sorted(['reduce-axis', 'mask-positive'], key=digest, reverse=True)
    gate = MemoryAdmission(tmp_path / 'gate', memory, Evaluation([]), None)
    gate.stop_requested.set()
    try:
        for sequence, review_id in enumerate((earlier, later), start=2):
            gate.submit(review_id, dict(draft(), summary=review_id), {}, request(), remaining_seconds=100)
            path = gate._path(review_id)
            job = json.loads(path.read_bytes())
            job.update(sequence=sequence, status='inconclusive', error_type='InfrastructureFailure')
            path.write_text(json.dumps(job))
        assert str(gate._path(later)) < str(gate._path(earlier))
    finally:
        gate.close()
    evaluation = Evaluation(['accept', 'accept'])
    resumed = MemoryAdmission(tmp_path / 'gate', memory, evaluation, None)
    try:
        resumed.wait(remaining_seconds=5)
        assert evaluation.calls[0][0]['summary'] == earlier
        resumed.poll(remaining_seconds=5)
        resumed.wait(remaining_seconds=5)
        assert [call[0]['summary'] for call in evaluation.calls] == [earlier, later]
        assert {row['memory_id'] for row in memory.records()} == {earlier, later}
    finally:
        resumed.close()


def test_legacy_pending_jobs_without_sequence_have_stable_resume_order(tmp_path):
    from benchmarks.ascend.subject_memory import digest
    memory = SubjectMemory(tmp_path / 'memory', b'')
    gate = MemoryAdmission(tmp_path / 'gate', memory, Evaluation([]), None)
    gate.stop_requested.set()
    ids = ['legacy-first', 'legacy-second']
    try:
        for review_id in ids:
            gate.submit(review_id, dict(draft(), summary=review_id), {}, request(), remaining_seconds=100)
            path = gate._path(review_id)
            job = json.loads(path.read_bytes())
            job.pop('sequence')
            path.write_text(json.dumps(job))
    finally:
        gate.close()
    evaluation = Evaluation(['accept', 'accept'])
    resumed = MemoryAdmission(tmp_path / 'gate', memory, evaluation, None)
    try:
        resumed.wait(remaining_seconds=5)
        resumed.poll(remaining_seconds=5)
        resumed.wait(remaining_seconds=5)
        assert [call[0]['summary'] for call in evaluation.calls] == sorted(ids, key=digest)
    finally:
        resumed.close()
