"""Actual pinned Pi runs completed host tools and its own compaction."""
import json
import threading

import pytest

from coding.pi.schema import obj, STRING
from rsi.actor import EvidenceTool, PiSessionAdapter, StageResult, WorkflowActor
from rsi.contracts import Stage
from rsi.runtime import Workflow, WorkflowError
from rsi.skills import builtin_bundle


def response(actions, index, tokens=100, *, summary=None, finish=None, completion_tokens=10):
    from test_pi_runtime import stream
    result = stream(*(actions[0] if actions else ('unused', {})), index, finish=finish or ('tool_calls' if actions else 'stop'))
    original = result.iter_lines
    def lines():
        for line in original():
            if line.startswith(b'data: {'):
                item = json.loads(line[6:])
                delta = item['choices'][0]['delta']
                if delta.get('tool_calls'):
                    if actions:
                        delta['tool_calls'] = [dict(index=i,id=f'call_{index}_{i}',type='function',
                            function=dict(name=name,arguments=json.dumps(args))) for i,(name,args) in enumerate(actions)]
                    else:
                        delta.pop('tool_calls')
                        delta['content'] = summary or '# Retained context\nSkills read; retain actual receipts and host evidence. Continue current stage.'
                if 'usage' in item:
                    item['usage']['prompt_tokens'] = tokens
                    item['usage']['completion_tokens'] = completion_tokens
                line = b'data: '+json.dumps(item).encode()
            yield line
    result.iter_lines = lines
    return result


def native_actor(tmp_path, events, perform, tools=None, *, clock=None, limits=None):
    from coding.pi.session import runtime_preflight, RuntimeUnavailable
    try: runtime_preflight()
    except RuntimeUnavailable: pytest.skip('cached pinned Pi/Node22 required')
    workflow = Workflow(tmp_path/'workflow.json', lambda *_:False, source_hash='project-v1',
                        limits=limits, **({'clock':clock} if clock is not None else {}))
    actor = WorkflowActor(workflow, builtin_bundle(),
        lambda:PiSessionAdapter(model='deepseek-flash',environ={'DEEPSEEK_API_KEY':'fake-provider-secret'}),
        perform=perform,preflight=lambda *_:StageResult(),evidence_tools=tools,
        emit=lambda kind,data:events.append((kind,data)))
    return actor, workflow


def test_native_completed_tools_keep_conversation_and_main_thread(monkeypatch, tmp_path):
    events, requests, called = [], [], []
    names = builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    def provider(*_, **kwargs):
        request=json.loads(kwargs['data']); requests.append(request)
        if len(requests)==1:
            return response([('read_skill',{'name':name}) for name in names],len(requests))
        tools=[json.loads(m['content']) for m in request['messages'] if m['role']=='tool']
        assert len(tools)==2
        for result,name in zip(tools,names):
            assert result['observation']['executed'] is True
            assert result['observation']['content']==builtin_bundle().document(name).content
        return response([('perform',{'proposal':'construct Python task'})],len(requests))
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    def perform(*args):
        assert threading.current_thread() is threading.main_thread()
        called.append(args)
        return StageResult()
    actor, flow=native_actor(tmp_path,events,perform)
    try:
        actor.step({'goal':'native tool proof'})
        actor.step({'goal':'native tool proof'})
        assert actor.session.session.rpc.process.poll() is None
    finally:actor.close()
    assert flow.stage==Stage.DESIGN and len(called)==1 and len(requests)==2
    assert len([row for kind,row in events if kind=='skill-delivered'])==2
    ids=[row['reservation_id'] for kind,row in events if kind=='pi-provider']
    assert len(ids)==len(set(ids))==2
    assert any(kind=='pi-event' and row['event']['type']=='agent_settled' for kind,row in events)


def test_native_compaction_runs_through_same_gateway_without_context_rewrite(monkeypatch,tmp_path):
    events, requests, recalls, summaries = [], [], [], []
    visible_receipts = []
    names=builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    def provider(*_, **kwargs):
        request=json.loads(kwargs['data']);requests.append(request)
        assert len(kwargs['data'])<=49152
        if not request.get('tools'):
            summaries.append(request)
            return response([],len(requests),tokens=len(kwargs['data'])//4)
        for message in request['messages']:
            if message['role'] != 'tool':
                continue
            result = json.loads(message['content'])
            observation = result['observation']
            if observation.get('action') == 'recall':
                reference = observation['reference']
                assert {'name': 'recall', 'reference': reference} in result['state']['evidence_receipts']
                visible_receipts.append(reference)
        read=[row['skill'] for kind,row in events if kind=='skill-usage' and row['kind']=='read'
              and row['stage']==Stage.TASK_CONSTRUCTION.value]
        missing=[name for name in names if name not in read]
        actions=([('read_skill',{'name':missing[0]})] if missing else
                 [('recall',{'sample':str(len(recalls))})] if len(recalls)<9 else
                 [('perform',{'proposal':'finish after retained native context'})])
        return response(actions,len(requests),tokens=len(kwargs['data'])//4)
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    def recall(stage,args,operation,seconds):
        assert threading.current_thread() is threading.main_thread()
        recalls.append(args['sample'])
        return dict(reference='actual-'+args['sample'],evidence='retained evidence '*140)
    actor,flow=native_actor(tmp_path,events,lambda *_:StageResult(),
        {'recall':EvidenceTool(obj({'sample':STRING}),recall)})
    try:
        actor.step({'goal':'force Pi native compaction'})
        actor.step({'goal':'force Pi native compaction'})
    finally:actor.close()
    assert summaries and visible_receipts and len(recalls)==9 and flow.stage==Stage.DESIGN
    assert any(kind=='pi-event' and row['event']['type']=='compaction_end' and not row['event'].get('aborted') for kind,row in events), [row['event'] for kind,row in events if kind=='pi-event' and 'compact' in row['event']['type']]
    assert len([row for kind,row in events if kind=='pi-provider-request'])==len(requests)
    assert any('retained evidence' in json.dumps(request) for request in summaries)


def test_native_unknown_pending_does_not_start_session(tmp_path):
    flow=Workflow(tmp_path/'state.json',lambda *_:False,source_hash='v1')
    flow.begin()
    actor=WorkflowActor(flow,builtin_bundle(),lambda:pytest.fail('replayed native work'),
        perform=lambda *_:StageResult(),preflight=lambda *_:StageResult(),emit=lambda *_:None)
    with pytest.raises(WorkflowError):actor.step({})


def test_native_host_transfer_is_excluded_but_compile_execute_profile_are_charged(monkeypatch,tmp_path):
    from rsi.contracts import BudgetLimits
    class Clock:
        value=1000.0
        def __call__(self):return self.value
    clock, events, calls = Clock(), [], []
    names=builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    def provider(*_,**kwargs):
        calls.append(json.loads(kwargs['data']));clock.value+=1
        return response([('read_skill',{'name':name}) for name in names] if len(calls)==1 else
                        [('perform',{'proposal':'host work'})],len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    flow=Workflow(tmp_path/'workflow.json',lambda *_:False,source_hash='v1',
                  limits=BudgetLimits(actor_seconds=30),clock=clock)
    def perform(*args):
        for category,seconds in [('transfer',100),('compilation',2),('execution',3),('profiling',4)]:
            token=flow.accounting.start(category);clock.value+=seconds;flow.accounting.end(token)
        return StageResult()
    actor=WorkflowActor(flow,builtin_bundle(),
        lambda:PiSessionAdapter(model='deepseek-flash',environ={'DEEPSEEK_API_KEY':'fake-provider-secret'}),
        perform=perform,host_category=None,preflight=lambda *_:StageResult(),emit=lambda kind,data:events.append((kind,data)))
    try:
        actor.step({});actor.step({'goal':'timed native transfer'})
    finally:actor.close()
    assert flow.accounting.seconds('thinking')==2
    assert flow.accounting.seconds('transfer')==100 and flow.accounting.seconds()==11
    assert flow.remaining()['actor_seconds']==19


def test_native_interrupt_preserves_pending_and_cleans_transport(monkeypatch,tmp_path):
    events, calls = [], []
    names=builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    def provider(*_,**kwargs):
        calls.append(json.loads(kwargs['data']))
        return response([('read_skill',{'name':name}) for name in names] if len(calls)==1 else
                        [('perform',{'proposal':'interrupt'})],len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    def interrupt(*args):
        assert threading.current_thread() is threading.main_thread()
        raise KeyboardInterrupt()
    actor,flow=native_actor(tmp_path,events,interrupt)
    try:
        actor.step({})
        with pytest.raises(KeyboardInterrupt):actor.step({'goal':'interrupt native callback'})
        process=actor.session.session.rpc.process
    finally:actor.close()
    assert process.poll() is not None and flow.pending
    assert flow.stage==Stage.TASK_CONSTRUCTION and len(flow.state['history'])==1


def test_host_success_without_rpc_settlement_preserves_pending(monkeypatch,tmp_path):
    from coding.pi.rpc import PiError, PiRPC
    events, calls, performed = [], [], []
    names=builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    def provider(*_,**kwargs):
        calls.append(json.loads(kwargs['data']))
        return response([('read_skill',{'name':name}) for name in names] if len(calls)==1 else
                        [('perform',{'proposal':'host succeeds before disconnect'})],len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    actor,flow=native_actor(tmp_path,events,lambda *_:performed.append(True) or StageResult())
    original=PiRPC.prompt
    def disconnected(rpc,message,**kwargs):
        listener=kwargs['on_event']
        def observe(event):
            listener(event)
            if event['type']=='agent_end' and performed:
                assert performed==[True] and flow.pending and flow.stage==Stage.TASK_CONSTRUCTION
                raise PiError('disconnected before agent_settled')
        kwargs['on_event']=observe
        return original(rpc,message,**kwargs)
    monkeypatch.setattr(PiRPC,'prompt',disconnected)
    try:
        actor.step({})
        with pytest.raises(PiError,match='before agent_settled'):
            actor.step({'goal':'disconnect after actual host success'})
    finally:actor.close()
    assert performed==[True] and flow.pending and flow.stage==Stage.TASK_CONSTRUCTION
    assert len(flow.state['history'])==1
    assert any(kind=='stage-result-proposed' and row['stage']==Stage.TASK_CONSTRUCTION.value for kind,row in events)
    assert not any(kind=='stage-result' and row['stage']==Stage.TASK_CONSTRUCTION.value for kind,row in events)


def test_successful_native_settlement_commits_stage_once(monkeypatch,tmp_path):
    events,calls,performed=[],[],[]
    names=builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    def provider(*_,**kwargs):
        calls.append(json.loads(kwargs['data']))
        return response([('read_skill',{'name':name}) for name in names] if len(calls)==1 else
                        [('perform',{'proposal':'settle'})],len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    actor,flow=native_actor(tmp_path,events,lambda *_:performed.append(True) or StageResult())
    try:
        actor.step({});actor.step({'goal':'commit once after settlement'})
    finally:actor.close()
    settled=next(i for i,(kind,row) in enumerate(events) if kind=='pi-event' and row['event']['type']=='agent_settled')
    commits=[i for i,(kind,row) in enumerate(events) if kind=='stage-result' and row['stage']==Stage.TASK_CONSTRUCTION.value]
    assert performed==[True] and len(commits)==1 and commits[0]>settled
    assert flow.stage==Stage.DESIGN and flow.pending is None and len(flow.state['history'])==2


def test_native_mixed_evidence_and_perform_batch_has_no_post_success_request(monkeypatch,tmp_path):
    events,calls,reads=[],[],[]
    names=builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    def provider(*_,**kwargs):
        calls.append(json.loads(kwargs['data']))
        assert len(calls)<=2
        return response([('read_skill',{'name':name}) for name in names] if len(calls)==1 else
            [('recall',{}),('perform',{'proposal':'finish mixed batch'})],len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    actor,flow=native_actor(tmp_path,events,lambda *_:StageResult(),
        {'recall':EvidenceTool(obj(),lambda *_:reads.append(True) or {'reference':'actual-receipt'})})
    try:
        actor.step({});actor.step({'goal':'mixed batch'})
    finally:actor.close()
    assert reads==[True] and len(calls)==2 and flow.stage==Stage.DESIGN


def test_native_large_task_source_summary_are_paged_but_hashes_stay_visible(monkeypatch,tmp_path):
    from rsi.actor import SourceSnapshot
    events,calls,initial_prompts=[],[],[]
    names=builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    source=SourceSnapshot('large.py','x = 1\n'*12000,'project-v1')
    def provider(*_,**kwargs):
        request=json.loads(kwargs['data']);calls.append(request)
        assert len(kwargs['data'])<=49152
        if not request.get('tools'):
            return response([],len(calls))
        for message in request['messages']:
            if message['role']=='user':
                content=message['content']
                texts=[content] if isinstance(content,str) else [part['text'] for part in content if part.get('type')=='text']
                for text in texts:
                    try:prompt=json.loads(text)
                    except ValueError:continue
                    if 'task' not in prompt:continue
                    initial_prompts.append(prompt)
                    assert prompt['source']['sha256']==source.sha256
                    assert prompt['source']['text']['read_with']=='read_context'
                    assert 'full_observation' in prompt['task']
        count=sum(kind=='skill-usage' and row['kind']=='read' and row['stage']==Stage.TASK_CONSTRUCTION.value for kind,row in events)
        if count<len(names):actions=[('read_skill',{'name':names[count]})]
        elif not any(kind=='evidence-tool-result' for kind,_ in events):actions=[('large',{})]
        else:actions=[('perform',{'proposal':'finish bounded resources'})]
        return response(actions,len(calls),tokens=len(kwargs['data'])//4)
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    actor,flow=native_actor(tmp_path,events,lambda *_:StageResult(),
        {'large':EvidenceTool(obj(),lambda *_:dict(reference='actual-large',summary={'rows':['x'*500]*100},evidence='y'*70000))})
    actor.source_snapshot=lambda:source
    try:
        actor.step({});actor.step({'many':{str(i):'small'*100 for i in range(100)}})
    finally:actor.close()
    assert flow.stage==Stage.DESIGN and calls and initial_prompts


def test_native_exact_edit_maps_to_single_guarded_perform(monkeypatch,tmp_path):
    from rsi.actor import SourceSnapshot
    import hashlib
    events,calls,edits=[],[],[]
    source=SourceSnapshot('large.py','padding = 0\n'*1000+'answer = 1\n','project-v1')
    names=builtin_bundle().skills_for(Stage.IMPLEMENTATION)
    def provider(*_,**kwargs):
        request=json.loads(kwargs['data']);calls.append(request)
        if len(calls)==1:return response([('read_skill',{'name':name}) for name in names],len(calls))
        assert 'edit_source' in {t['function']['name'] for t in request['tools']}
        return response([('edit_source',dict(sha256=source.sha256,old_text='answer = 1',new_text='answer = 2'))],len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    def perform(stage,args,operation,seconds):
        assert stage==Stage.IMPLEMENTATION
        edits.append(args['edit'])
        return StageResult(new_source_hash='project-v2')
    actor,flow=native_actor(tmp_path,events,perform)
    flow.state['stage']=Stage.IMPLEMENTATION.value;flow._save()
    actor.source_snapshot=lambda:source
    try:actor.step({'goal':'repair larger file with exact edit'})
    finally:actor.close()
    assert len(edits)==1 and edits[0]['source']==source.text.replace('answer = 1','answer = 2')
    assert edits[0]['sha256']==hashlib.sha256(source.text.encode()).hexdigest()
    assert flow.stage==Stage.TEST and flow.source_hash=='project-v2'
    assert any(kind=='pi-action' and row['name']=='edit_source' for kind,row in events)


@pytest.mark.parametrize('budget', [10,5])
def test_post_perform_settlement_time_is_charged_before_commit(monkeypatch,tmp_path,budget):
    from coding.pi.rpc import PiRPC
    from rsi.contracts import BudgetLimits
    class Clock:
        value=1000.0
        def __call__(self):return self.value
    clock,events,calls,performed=Clock(),[],[],[]
    names=builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    def provider(*_,**kwargs):
        calls.append(json.loads(kwargs['data']));clock.value+=1
        return response([('read_skill',{'name':name}) for name in names] if len(calls)==1 else
                        [('perform',{'proposal':'wait for settlement'})],len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    flow=Workflow(tmp_path/'state.json',lambda *_:False,source_hash='v1',
                  limits=BudgetLimits(actor_seconds=budget),clock=clock)
    def perform(*args):
        performed.append(True);clock.value+=2
        return StageResult()
    actor=WorkflowActor(flow,builtin_bundle(),
        lambda:PiSessionAdapter(model='deepseek-flash',environ={'DEEPSEEK_API_KEY':'fake-provider-secret'}),
        perform=perform,preflight=lambda *_:StageResult(),emit=lambda kind,data:events.append((kind,data)))
    original=PiRPC.prompt
    def delayed(rpc,message,**kwargs):
        listener=kwargs['on_event']
        def observe(event):
            listener(event)
            if event['type']=='agent_end' and performed:
                clock.value+=5
        kwargs['on_event']=observe
        return original(rpc,message,**kwargs)
    monkeypatch.setattr(PiRPC,'prompt',delayed)
    try:
        actor.step({})
        if budget==5:
            with pytest.raises(WorkflowError,match='time budget'):actor.step({'goal':'settlement budget'})
        else:actor.step({'goal':'settlement budget'})
    finally:actor.close()
    assert performed==[True] and flow.accounting.seconds('thinking')==7
    assert flow.accounting.seconds()==9
    assert (flow.pending is not None)==(budget==5)
    assert flow.stage==(Stage.TASK_CONSTRUCTION if budget==5 else Stage.DESIGN)


def test_aborted_settlement_never_commits_completed_host_result(monkeypatch,tmp_path):
    from coding.pi.rpc import PiRPC, PiCancelled
    events,calls,performed=[],[],[]
    names=builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    def provider(*_,**kwargs):
        calls.append(json.loads(kwargs['data']))
        return response([('read_skill',{'name':name}) for name in names] if len(calls)==1 else
                        [('perform',{'proposal':'abort settlement'})],len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    actor,flow=native_actor(tmp_path,events,lambda *_:performed.append(True) or StageResult())
    original=PiRPC.prompt
    def abort_settlement(rpc,message,**kwargs):
        listener=kwargs['on_event']
        def observe(event):
            if event['type']=='agent_settled':event['aborted']=True
            listener(event)
        kwargs['on_event']=observe
        return original(rpc,message,**kwargs)
    monkeypatch.setattr(PiRPC,'prompt',abort_settlement)
    try:
        actor.step({})
        with pytest.raises(PiCancelled):actor.step({'goal':'abort before commit'})
    finally:actor.close()
    assert performed==[True] and flow.pending and flow.stage==Stage.TASK_CONSTRUCTION
    assert len(flow.state['history'])==1


@pytest.mark.parametrize('last,aborted,error', [('toolUse',False,None),('error',False,'provider'),('toolUse',True,'cancelled')])
def test_rpc_settlement_uses_recovered_final_assistant_outcome(last,aborted,error):
    import queue
    from coding.pi.rpc import PiRPC,PiProviderError,PiCancelled
    rpc=object.__new__(PiRPC)
    rpc.timeout=1;rpc.events=[];rpc.records=queue.Queue()
    def send(command):
        events=[dict(type='response',id=command['id'],success=True),
                dict(type='message_end',message=dict(role='assistant',stopReason='error')),
                dict(type='compaction_start'),dict(type='compaction_end'),
                dict(type='message_end',message=dict(role='assistant',stopReason=last)),
                dict(type='agent_settled',aborted=aborted)]
        for event in events:rpc.records.put(('stdout',json.dumps(event).encode()+b'\n'))
    rpc.send=send
    if error:
        with pytest.raises(PiProviderError if error=='provider' else PiCancelled):rpc.prompt('recover')
    else:
        rpc.prompt('recover')
        assert rpc.events[-1]['type']=='agent_settled'


@pytest.mark.parametrize('settles',[True,False])
def test_native_waits_for_late_provider_ledger_before_commit(monkeypatch,tmp_path,settles):
    from coding.pi.rpc import PiRPC
    from rsi.contracts import BudgetLimits
    class Clock:
        value=1000.0
        def __call__(self):return self.value
    clock=Clock()
    events,calls,workers=[],[],[]
    names=builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    def provider(*_,**kwargs):
        calls.append(json.loads(kwargs['data']));clock.value+=1
        return response([('read_skill',{'name':name}) for name in names] if len(calls)==1 else
                        [('perform',{'proposal':'wait for final ledger'})],len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    actor,flow=native_actor(tmp_path,events,lambda *_:StageResult(),clock=clock,
        limits=BudgetLimits(actor_seconds=20 if settles else 3))
    original=PiRPC.prompt
    def late_ledger(rpc,message,**kwargs):
        result=original(rpc,message,**kwargs)
        session=actor.session.session
        while not session.gateway.finished.is_set() or not session.requests.empty():
            kwargs['pump']()
            session.gateway.finished.wait(.01)
        finished=threading.Event()
        class FinalLedger:
            started=False
            def is_set(self):
                if not self.started and settles:
                    self.started=True
                    def publish():
                        session.record('pi-provider',dict(reservation_id='late-ledger',status='completed',usage={'completion_tokens':7}))
                        finished.set()
                    worker=threading.Thread(target=publish);workers.append(worker);worker.start()
                return finished.is_set()
            def wait(self,timeout):
                if not settles:clock.value+=2
                return finished.wait(timeout)
        session.gateway.finished=FinalLedger()
        return result
    monkeypatch.setattr(PiRPC,'prompt',late_ledger)
    try:
        actor.step({})
        if settles:actor.step({'goal':'late usage settlement'})
        else:
            with pytest.raises(WorkflowError,match='time budget'):actor.step({'goal':'late usage settlement'})
    finally:
        actor.close()
        for worker in workers:worker.join(timeout=2)
    if not settles:
        assert flow.pending and flow.stage==Stage.TASK_CONSTRUCTION and len(flow.state['history'])==1
        assert flow.accounting.seconds('thinking')==4
        return
    ledger=next(i for i,(kind,row) in enumerate(events) if kind=='pi-provider' and row.get('reservation_id')=='late-ledger')
    commit=next(i for i,(kind,row) in enumerate(events) if kind=='stage-result' and row['stage']==Stage.TASK_CONSTRUCTION.value)
    assert ledger<commit and flow.stage==Stage.DESIGN and not any(worker.is_alive() for worker in workers)


def test_existing_registered_route_transfers_to_real_native_controller(monkeypatch):
    import time
    import coding.pi.session as module
    from coding.pi.controller import ControllerSession
    monkeypatch.setattr(module,'runtime_preflight',lambda:None)
    raw=module.PiCodingSession(model='deepseek-flash',schemas={},environ={'DEEPSEEK_API_KEY':'fake-provider-secret'},
                               deadline=time.monotonic()+30,own_process_group=False)
    gateway,profile,deadline=raw.gateway,raw.profile,raw.deadline
    adapter=PiSessionAdapter.from_existing_session(raw)
    try:
        assert isinstance(adapter.session,ControllerSession)
        assert raw.closed and not adapter.session.closed and adapter.session.rpc is None
        assert adapter.session.gateway is gateway and adapter.session.profile is profile
        assert adapter.session.deadline==deadline and adapter.session.enclosing_deadline is None
        assert adapter.session.request_deadline_provenance==dict(session=deadline,gateway=deadline)
        assert adapter.session.own_process_group is False
        raw.close()
        assert not adapter.session.closed
    finally:adapter.close()
    assert adapter.session.closed


@pytest.mark.parametrize('state',['closed','started'])
def test_existing_closed_or_started_route_is_not_reopened(monkeypatch,state):
    import coding.pi.session as module
    monkeypatch.setattr(module,'runtime_preflight',lambda:None)
    raw=module.PiCodingSession(model='deepseek-flash',schemas={},environ={'DEEPSEEK_API_KEY':'fake-provider-secret'})
    if state=='closed':raw.close()
    else:raw.rpc=object()
    with pytest.raises(ValueError,match='unstarted'):PiSessionAdapter.from_existing_session(raw)
    raw.rpc=None;raw.close()


def test_adopted_registered_route_runs_native_tools_without_new_provider_client(monkeypatch,tmp_path):
    from coding.pi.session import PiCodingSession, runtime_preflight, RuntimeUnavailable
    try:runtime_preflight()
    except RuntimeUnavailable:pytest.skip('cached pinned Pi/Node22 required')
    events,calls=[],[]
    names=builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    def provider(*_,**kwargs):
        calls.append(json.loads(kwargs['data']))
        return response([('read_skill',{'name':name}) for name in names] if len(calls)==1 else
                        [('perform',{'proposal':'use original route'})],len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    raw=PiCodingSession(model='deepseek-flash',schemas={},environ={'DEEPSEEK_API_KEY':'fake-provider-secret'})
    import time
    raw.deadline=raw.gateway.deadline=time.monotonic()-1
    gateway=raw.gateway
    flow=Workflow(tmp_path/'state.json',lambda *_:False,source_hash='v1')
    actor=WorkflowActor(flow,builtin_bundle(),lambda:PiSessionAdapter.from_existing_session(raw),
        perform=lambda *_:StageResult(),preflight=lambda *_:StageResult(),emit=lambda kind,data:events.append((kind,data)))
    try:
        actor.step({});actor.step({'goal':'adopted registered native route'})
        assert actor.session.session.gateway is gateway
    finally:actor.close()
    assert raw.closed and flow.stage==Stage.DESIGN and len(calls)==2


def test_only_explicit_wall_deadline_bounds_adopted_route(monkeypatch):
    import time
    import coding.pi.session as module
    monkeypatch.setattr(module,'runtime_preflight',lambda:None)
    raw=module.PiCodingSession(model='deepseek-flash',schemas={},environ={'DEEPSEEK_API_KEY':'fake-provider-secret'})
    wall=time.monotonic()+30
    adapter=PiSessionAdapter.from_existing_session(raw,wall_deadline=wall)
    try:assert adapter.session.enclosing_deadline==wall
    finally:adapter.close()


def test_real_project_runner_durably_records_native_pi_event_envelopes(monkeypatch,tmp_path):
    from coding.pi.session import runtime_preflight,RuntimeUnavailable
    from test_rsi_project_runner import make
    try:runtime_preflight()
    except RuntimeUnavailable:pytest.skip('cached pinned Pi/Node22 required')
    calls=[]
    names=builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    def provider(*_,**kwargs):
        calls.append(json.loads(kwargs['data']))
        return response([('read_skill',{'name':name}) for name in names] if len(calls)==1 else
                        [('perform',{'proposal':'freeze actual Python task contract'})],len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    runner,_,database=make(tmp_path,session_factory=lambda:PiSessionAdapter(
        model='deepseek-flash',environ={'DEEPSEEK_API_KEY':'fake-provider-secret'}))
    try:
        runner.actor.step(runner.context())
        runner.actor.step(runner.context())
        persisted=[runner.store.read_json(ref) for ref in json.loads(runner.event_path.read_text())]
        pi_events=[row for row in persisted if row['type']=='pi-event']
        assert pi_events and any(row['event']['type']=='agent_settled' for row in pi_events)
        assert runner.workflow.stage==Stage.DESIGN and runner.workflow.pending is None
        assert len(runner.workflow.state['history'])==2 and len(calls)==2
        from core.artifacts import canonical_bytes
        assert canonical_bytes(persisted)==canonical_bytes(runner.events)
        operation=runner.workflow.state['history'][-1]['evidence']['operation_id']
        assert all(row['operation_id']==operation for row in pi_events)
    finally:
        runner.actor.close()
        database.close()


def test_native_concise_summary_covers_long_history_and_recovers_length_failure(monkeypatch,tmp_path):
    events,requests,recalls,summaries=[],[],[],[]
    names=builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    long_summary='# Actual checkpoint\n'+('Retain current stage, source hash and current receipts. '*45)
    def provider(*_,**kwargs):
        request=json.loads(kwargs['data']);requests.append(request)
        assert len(kwargs['data'])<=49152
        if not request.get('tools'):
            summaries.append(request)
            assert request['max_tokens']==1536
            if len(summaries)==1:
                return response([],len(requests),len(kwargs['data'])//4,summary=long_summary[:7000],
                                finish='length',completion_tokens=request['max_tokens'])
            return response([],len(requests),len(kwargs['data'])//4,summary=long_summary,completion_tokens=900)
        read=[row['skill'] for kind,row in events if kind=='skill-usage' and row['kind']=='read'
              and row['stage']==Stage.TASK_CONSTRUCTION.value]
        missing=[name for name in names if name not in read]
        actions=([('read_skill',{'name':missing[0]})] if missing else
                 [('recall',{'sample':str(len(recalls))})] if len(recalls)<18 else
                 [('perform',{'proposal':'finish after complete long summary'})])
        return response(actions,len(requests),len(kwargs['data'])//4)
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    def recall(stage,args,operation,seconds):
        recalls.append(args['sample'])
        return dict(reference='actual-'+args['sample'],evidence='retained evidence '*140)
    actor,flow=native_actor(tmp_path,events,lambda *_:StageResult(),
        {'recall':EvidenceTool(obj({'sample':STRING}),recall)})
    try:
        actor.step({});actor.step({'goal':'long native summaries under original wire cap'})
    finally:actor.close()
    compactions=[row['event'] for kind,row in events if kind=='pi-event' and row['event']['type']=='compaction_end']
    attempts=[row['event'] for kind,row in events if kind=='pi-native-event' and row['event']['kind']=='compaction-summary']
    assert any(not row['successful'] and row.get('stopReason')=='length' for row in attempts)
    assert any(not row.get('result') for row in compactions)
    successful=[row for row in compactions if row.get('result')]
    assert successful and all(row['result']['usage']['output']<=1000 for row in successful)
    assert all(len(row['result']['summary'].encode())<=6144 for row in successful)
    assert any(long_summary in row['result']['summary'] for row in successful)
    assert flow.stage==Stage.DESIGN and len(recalls)==18


def test_summary_requested_budget_is_honored_within_profile_ceiling():
    from coding.pi.gateway import Gateway
    from coding.pi.session import RequestAccounting
    gateway=Gateway('deepseek','fake',RequestAccounting(),'test',max_output_tokens=8192)
    payload=dict(model='deepseek-flash',messages=[dict(role='user',content='summarize')],max_tokens=6553)
    assert gateway.prepare(payload)['max_tokens']==6553
    sdk=dict(payload);sdk.pop('max_tokens');sdk['max_completion_tokens']=4096
    assert gateway.prepare(sdk)['max_tokens']==4096
    assert gateway.prepare(dict(payload,max_tokens=20000))['max_tokens']==8192


def test_rejected_request_retains_actual_byte_attempt_before_dispatch(monkeypatch):
    import requests
    from coding.pi.gateway import Gateway
    from coding.pi.session import RequestAccounting
    records=[]
    gateway=Gateway('deepseek','fake',RequestAccounting(),'test',hard_bytes=700,
                    on_record=lambda kind,data:records.append((kind,data)))
    original=requests.post
    def route(url,*args,**kwargs):
        assert url.startswith('http://127.0.0.1:')
        return original(url,*args,**kwargs)
    monkeypatch.setattr('coding.pi.gateway.requests.post',route)
    with gateway, requests.Session() as client:
        client.trust_env=False
        result=client.post(gateway.url+'/chat/completions',json=dict(model='deepseek-flash',
            messages=[dict(role='user',content='x'*1000)],max_tokens=6553),
            headers={'Authorization':'Bearer '+gateway.token},timeout=3)
    assert result.status_code==413 and gateway.calls==0 and gateway.records==[]
    assert result.json()['error']['code']=='context_length_exceeded'
    rejected=next(data for kind,data in records if kind=='pi-context-limit')
    assert rejected['request_bytes']>rejected['hard_bytes']==700
    assert rejected['messages_bytes']>1000 and rejected['message_count']==1
    assert rejected['summary_request'] is True and len(rejected['request_sha256'])==64
