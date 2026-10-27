from dataclasses import dataclass
import hashlib
import json

import pytest

from benchmarks.ascend.subjects import Subject
from benchmarks.ascend.subject_campaign import SubjectCampaign
from benchmarks.ascend.subject_memory import SubjectMemory, StaleMemory
from benchmarks.ascend.subject_execution import SubjectOutcome


def proposal(source):
    return dict(operation='create', base_versions=[], summary='tail zeros',
        applicability='tail kernels', preconditions='padding present', expectations='padding zero',
        limitations='one dtype', before=None, before_unavailable_reason='first source',
        after=dict(sha256=source, start_line=1, end_line=1,text='source\n'),
        evidence=[source], citations=[])


class Verifier:
    last_trace = []
    def __init__(self):
        self.calls = []
    def review(self, request, *, max_seconds, feedback=None):
        self.calls.append(request)
        repair = len(self.calls) == 1
        return dict(decision='repair' if repair else 'accept', reason='needs control' if repair else 'control passed',
            evidence=[request.source_sha256], repair=dict(instructions='add control',expected_evidence='control result') if repair else None,
            memory=None if repair else proposal(request.source_sha256))


def test_repair_then_advance_and_resume_without_reexecution(tmp_path):
    calls, snapshots = [], []
    verifier = Verifier()
    def select(context, memory, history, **kwargs):
        pid = 'p' + str(len(calls))
        return [dict(project_id=pid, repairs=[r['repair_id'] for r in context['repairs']])]
    def execute(item, memory, directory, **kwargs):
        calls.append(item['project_id']); snapshots.append(memory)
        sha = campaign.artifacts.put_bytes(b'source\n')
        result = campaign.artifacts.put_json(dict(passed=True))
        return SubjectOutcome(True, sha, {'source':sha,'result':result}, {},campaign.artifacts.root)
    decisions = iter([dict(action='repair',repair_id='p0',reason='need control',evidence=['p0']),
                      dict(action='advance',reason='repaired',evidence=['p1'])])
    campaign = SubjectCampaign(tmp_path, [Subject('basics','learn tails',1)], b'initial',
        select_wave=select,execute=execute,verifier=verifier,
        complete=lambda *a,**kw:json.dumps(next(decisions)))
    result = campaign.run()
    assert result['terminal']=='COMPLETE'
    assert result['subjects']['repairs']==[]
    assert len(result['accepted_memory'])==1
    assert snapshots[0]==snapshots[1]  # rejected project never enters memory
    assert campaign.resume()['terminal']=='COMPLETE'
    assert calls==['p0','p1']


def test_catalog_keeps_prior_version_until_accept_and_refuses_stale_merge(tmp_path):
    store = SubjectMemory(tmp_path,b'initial')
    source = 'a'*64
    first = store.accept('p0',proposal(source),{})
    before = store.snapshot()
    replacement = dict(proposal(source),operation='append',base_versions=[first],summary='new control')
    assert store.snapshot()==before  # constructing a draft cannot publish it
    second = store.accept('p1',replacement,{})
    assert second!=first
    assert store.read(first['version'])['proposal']['summary']=='tail zeros'
    with pytest.raises(StaleMemory):
        store.accept('p2',replacement,{})
    assert len(store.records())==1
    assert store.search('control')[0]['version']==second['version']


def test_no_new_knowledge_can_accept_without_publishing(tmp_path):
    class NoMemory:
        last_trace=[]
        def review(self, request, **kw):
            return dict(decision='accept',reason='already known',evidence=[request.source_sha256],repair=None,memory=None)
    def execute(*args, **kwargs):
        sha=c.artifacts.put_bytes(b'source\n')
        return SubjectOutcome(True,sha,{'source':sha},{},c.artifacts.root)
    c=SubjectCampaign(tmp_path,[Subject('s','subject',1)],b'',
        select_wave=lambda *a,**k:[dict(project_id='p')],execute=execute,verifier=NoMemory(),
        complete=lambda *a,**k:json.dumps(dict(action='advance',reason='covered',evidence=['p'])))
    assert c.run()['terminal']=='COMPLETE'
    assert c.memory.records()==[]


def test_sibling_appends_reconcile_without_executing_projects_again(tmp_path):
    executions=[]
    class Appender:
        last_trace=[]
        def review(self, request, *, feedback=None, **kw):
            base = seed if feedback is None else dict(memory_id=campaign.memory.records()[0]['memory_id'],
                version=__import__('benchmarks.ascend.subject_memory',fromlist=['digest']).digest(campaign.memory.records()[0]))
            memory=dict(proposal(request.source_sha256),operation='append',base_versions=[base])
            return dict(decision='accept',reason='combined evidence',evidence=[request.source_sha256],repair=None,memory=memory)
    def execute(item, *args, **kwargs):
        executions.append(item['project_id'])
        sha=campaign.artifacts.put_bytes(b'source\n')
        return SubjectOutcome(True,sha,{'source':sha},{},campaign.artifacts.root)
    campaign=SubjectCampaign(tmp_path,[Subject('s','s',1)],b'',
        select_wave=lambda *a,**k:[dict(project_id='one'),dict(project_id='two')],
        execute=execute,verifier=Appender(),complete=lambda *a,**k:json.dumps(
            dict(action='advance',reason='complete',evidence=['one','two'])))
    seed=campaign.memory.accept('seed',proposal('a'*64),{})
    assert campaign.run()['terminal']=='COMPLETE'
    assert executions==['one','two']
    assert len(campaign.state['projects']['two']['prior_reviews'])==1
    assert len(campaign.memory.records())==1


def test_probe_control_wave_resolves_shared_repair_once_with_original_requirements(tmp_path):
    class Inspecting(Verifier):
        def review(self, request, **kw):
            if self.calls:
                original=json.loads(request.objective)['repairs']['failed']
                assert original['repair']['instructions']=='add control'
                siblings=json.loads(request.objective)['sibling_evidence']
                assert len(siblings)==1
                assert any(role.startswith('sibling:') for role in request.artifacts)
            return super().review(request,**kw)
    verifier=Inspecting()
    def select(context, *args, **kwargs):
        return [dict(project_id='failed')] if not verifier.calls else [
            dict(project_id='probe',repairs=['failed']),dict(project_id='control',repairs=['failed'])]
    def execute(item,*args,**kw):
        sha=c.artifacts.put_bytes(b'source\n')
        return SubjectOutcome(True,sha,{'source':sha},{},c.artifacts.root)
    actions=iter([dict(action='repair',repair_id='failed'),dict(action='advance')])
    c=SubjectCampaign(tmp_path,[Subject('s','s',1)],b'',select_wave=select,execute=execute,
        verifier=verifier,complete=lambda *a,**k:json.dumps(dict(next(actions),reason='evidence',evidence=['source'])))
    assert c.run()['terminal']=='COMPLETE'
    repair=c.scheduler.state()['repairs']['failed']
    assert len(repair['resolution_evidence'])==2


def test_interrupted_stale_review_retains_rewrite_feedback(tmp_path):
    class Interrupted:
        last_trace=[]
        def review(self, request, *, feedback, **kw):
            assert feedback=={'reason':'stale base'}
            raise OSError('provider disconnected')
    c=SubjectCampaign(tmp_path,[Subject('s','s')],b'',select_wave=None,execute=None,
        verifier=Interrupted(),complete=lambda *a,**k:None)
    sha=c.artifacts.put_bytes(b'source\n')
    c.state['projects']['p']=dict(outcome=SubjectOutcome(True,sha,{'source':sha},{},c.artifacts.root).as_dict(),
        objective='s',rewrite_feedback={'reason':'stale base'})
    c._remaining=lambda:60
    with pytest.raises(OSError):c._review('p',c.state['projects']['p'])
    retained=json.loads(c.path.read_text())['projects']['p']
    assert retained['rewrite_feedback']=={'reason':'stale base'}
    assert retained.get('review') is None


def test_local_verifier_timeout_does_not_exhaust_campaign_allocation(tmp_path):
    class Slow:
        last_trace=[]
        def review(self,*a,**kw):raise TimeoutError('request timed out')
    def execute(*a,**kw):
        sha=c.artifacts.put_bytes(b'source\n')
        return SubjectOutcome(True,sha,{'source':sha},{},c.artifacts.root)
    c=SubjectCampaign(tmp_path,[Subject('s','s')],b'',select_wave=lambda *a,**k:[dict(project_id='p')],
        execute=execute,verifier=Slow(),complete=lambda *a,**k:None,max_seconds=100)
    result=c.run()
    assert result['terminal']=='REVIEW_PENDING'
    assert result['pending'] is not None
    assert json.loads((tmp_path/'elapsed.json').read_text())['elapsed_seconds']<100


def test_second_controller_cannot_start_while_original_work_is_active(tmp_path):
    import fcntl
    c=SubjectCampaign(tmp_path,[Subject('s','s')],b'',select_wave=None,execute=None,
        verifier=None,complete=None)
    with (tmp_path/'.campaign.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='already active'):c.resume()
    assert not (tmp_path/'elapsed.json').exists()


def test_late_memory_rejection_reaches_curriculum_before_completion(tmp_path):
    class Review:
        last_trace=[]
        def review(self,req,**kw):return dict(decision='accept',reason='verified',
            evidence=[req.source_sha256],repair=None,memory=proposal(req.source_sha256))
    class LateGate:
        pending=[]
        def submit(self,review_id,*a,**kw):self.pending.append(review_id)
        def poll(self,**kw):
            if c.scheduler.context()['current_subject'] is not None or not self.pending:return []
            pid=self.pending.pop(0)
            return [dict(review_id=pid,outcome='rejected' if pid=='first' else 'accepted',
                judgment=dict(reason='memory ignored'),rewrites=2,provenance=dict(subject_id='s'))]
        def wait(self,**kw):pass
        def close(self):pass
    actions=iter([dict(action='advance'),dict(action='revisit',subject_id='s'),dict(action='advance')])
    contexts=[]
    def decide(messages,**kw):
        contexts.append(json.loads(messages[1]['content']))
        return json.dumps(dict(next(actions),reason='review coverage',evidence=['retained']))
    def execute(*a,**kw):
        sha=c.artifacts.put_bytes(b'source\n')
        return SubjectOutcome(True,sha,{'source':sha},{},c.artifacts.root)
    c=SubjectCampaign(tmp_path,[Subject('s','s',1)],b'',
        select_wave=lambda *a,**kw:[dict(project_id='first' if not c.state['waves'] else 'revisit')],
        execute=execute,verifier=Review(),complete=decide,memory_gate=LateGate())
    assert c.run()['terminal']=='COMPLETE'
    assert len(c.state['waves'])==2
    assert contexts[1]['memory']['memory_reviews']['first']['judgment']['reason']=='memory ignored'


def test_progression_context_preserves_coverage_without_repeating_large_sources(tmp_path):
    c=SubjectCampaign(tmp_path,[Subject('s','s')],b'prior accepted guidance',
        select_wave=None,execute=None,verifier=None,complete=None)
    for i in range(30):
        learned=dict(proposal('a'*64),summary='lesson '+str(i),
            after=dict(sha256='a'*64,start_line=1,end_line=2000,text='long source\n'*2000))
        c.memory.accept('p'+str(i),learned,dict(subject_id='s',wave_index=i))
    context=c._decision_context()
    assert context['accepted_counts']=={'s':30}
    assert context['accepted'][0]['summary']=='lesson 29'
    assert context['initial_memory']=='prior accepted guidance'
    assert len(json.dumps(context).encode())<10000


def test_verifier_receives_frozen_individual_contract_for_subject_control(tmp_path):
    plan=dict(project_id='control',spec=dict(hypothesis='Baseline copy control',operation='copy'))
    class Review:
        last_trace=[]
        def review(self,request,**kw):
            objective=json.loads(request.objective)
            contract=c.artifacts.read_json(objective['project']['contract_sha256'])
            assert contract==plan
            assert objective['project']['contract_sha256'] in request.artifacts.values()
            return dict(decision='accept',reason='valid control for arithmetic study',
                evidence=[request.source_sha256],repair=None,memory=None)
    def execute(*a,**kw):
        sha=c.artifacts.put_bytes(b'source\n')
        return SubjectOutcome(True,sha,{'source':sha},{},c.artifacts.root)
    c=SubjectCampaign(tmp_path,[Subject('add','Learn elementwise addition',1)],b'',
        select_wave=lambda *a,**kw:[plan],execute=execute,verifier=Review(),
        complete=lambda *a,**kw:json.dumps(dict(action='advance',reason='control evidence complete',evidence=['control'])))
    assert c.run()['terminal']=='COMPLETE'


@pytest.mark.parametrize('interrupted_at', [0, 1])
def test_interrupted_execution_holds_automatic_wave_and_resume(tmp_path, interrupted_at):
    now = [0.0]
    calls, reviews, gate_calls = [], [], []
    class Gate:
        def start_session(self): gate_calls.append('start')
        def poll(self, **kw): gate_calls.append('poll'); return []
        def close(self): gate_calls.append('close')
    class NoReview:
        def review(self, *args, **kw): reviews.append(args); raise AssertionError('unsettled wave reviewed')
    def execute(item, memory, directory, **kw):
        calls.append(item['project_id'])
        now[0] += 3
        directory.mkdir(parents=True, exist_ok=True)
        receipt = tmp_path / 'binding' / 'shared' / 'operation.json'
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(dict(handle='remote:target:job:retained')))
        sha = c.artifacts.put_bytes(b'partial source')
        interrupted = len(calls) - 1 == interrupted_at
        return SubjectOutcome(False, sha, {'source': sha},
            dict(classification='interrupted' if interrupted else 'actor-unverified'), c.artifacts.root)
    c = SubjectCampaign(tmp_path, [Subject('s', 'subject', 1)], b'initial',
        select_wave=lambda *a, **kw: [dict(project_id=p) for p in ['one', 'two', 'three']],
        execute=execute, verifier=NoReview(), complete=lambda *a, **kw: None,
        memory_gate=Gate(), max_seconds=100, clock=lambda: now[0])
    first = c.run()
    assert first['terminal'] == 'OBSERVATION_REQUIRED'
    assert calls == ['one', 'two'][:interrupted_at + 1]
    assert not reviews and first['waves'] == [] and first['pending']
    retained = first['observation_required'][0]
    assert retained['memory_sha256'] == first['pending']['memory_sha256']
    assert json.loads(c.artifacts.read_bytes(retained['outcome_sha256'])) == first['projects'][calls[-1]]['outcome']
    inventory = json.loads(c.artifacts.read_bytes(retained['operations_inventory_sha256']))
    assert json.loads(c.artifacts.read_bytes(inventory[0]['artifact_sha256']))['handle'] == 'remote:target:job:retained'
    before = json.loads((tmp_path / 'elapsed.json').read_bytes())
    gate_calls.clear()
    assert c.resume()['terminal'] == 'OBSERVATION_REQUIRED'
    assert gate_calls == ['close']  # No background A/B polling/session or project work on resume.
    assert calls == ['one', 'two'][:interrupted_at + 1] and not reviews
    after = json.loads((tmp_path / 'elapsed.json').read_bytes())
    assert after['max_seconds'] == before['max_seconds'] == 100
    assert after['elapsed_seconds'] == before['elapsed_seconds'] == now[0]
    if interrupted_at:
        assert c.state['projects']['one']['outcome']['observations']['classification'] == 'actor-unverified'


def test_pending_interruption_scan_uses_effective_replacement_and_checks_all_entries(tmp_path):
    c = SubjectCampaign(tmp_path, [Subject('s', 'subject', 1)], b'',
        select_wave=None, execute=None, verifier=None, complete=None)
    interrupted = dict(outcome=dict(observations=dict(classification='interrupted')))
    c.state['projects'] = dict(old=dict(interrupted, superseded_by='replacement'),
                               replacement=dict(outcome=dict(observations=dict(classification='actor-unverified'))),
                               later=interrupted)
    c.state['pending'] = dict(wave_id='w', projects=['unstarted', 'old', 'later'], memory_sha256=c.artifacts.put_bytes(b'frozen'))
    from benchmarks.ascend.subject_campaign import ObservationRequired
    with pytest.raises(ObservationRequired):
        c._require_observed_pending()
    assert [r['project_id'] for r in c.state['observation_required']] == ['later']
    # A cached later interruption blocks resume before the unstarted first item.
    assert c.resume()['terminal'] == 'OBSERVATION_REQUIRED'
    assert 'unstarted' not in c.state['projects']
    c.state['projects']['later']['outcome']['observations']['classification'] = 'actor-unverified'
    c._require_observed_pending()
    assert 'observation_required' not in c.state


def test_interrupted_wave_at_elapsed_deadline_keeps_partial_outcome(tmp_path):
    now = [0.0]
    def execute(item, memory, directory, **kw):
        now[0] = 10.0
        sha = c.artifacts.put_bytes(b'partial')
        return SubjectOutcome(False, sha, {'source': sha}, dict(classification='interrupted'), c.artifacts.root)
    c = SubjectCampaign(tmp_path, [Subject('s', 'subject', 1)], b'', max_seconds=10,
        clock=lambda: now[0], select_wave=lambda *a, **kw: [dict(project_id='p'), dict(project_id='peer')],
        execute=execute, verifier=None, complete=None)
    report = c.run()
    assert report['terminal'] == 'TIME_BUDGET_EXHAUSTED'
    assert list(report['projects']) == ['p']
    assert report['pending']['projects'] == ['p', 'peer']
    assert report['observation_required'][0]['project_id'] == 'p'
    assert c.resume()['terminal'] == 'TIME_BUDGET_EXHAUSTED'
    assert json.loads((tmp_path / 'elapsed.json').read_bytes())['elapsed_seconds'] == 10
