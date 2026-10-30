"""Actual pinned Pi runs completed host tools and its own compaction."""
import json
import threading

import pytest

from coding.pi.schema import obj, STRING
from rsi.actor import EvidenceTool, PiSessionAdapter, StageResult, WorkflowActor
from rsi.contracts import Stage
from rsi.runtime import Workflow, WorkflowError
from rsi.skills import builtin_bundle


def response(actions, index, tokens=100):
    from test_pi_runtime import stream
    result = stream(*(actions[0] if actions else ('unused', {})), index, finish='tool_calls' if actions else 'stop')
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
                        delta['content'] = '# Retained context\nSkills read; retain actual receipts and host evidence. Continue current stage.'
                if 'usage' in item:
                    item['usage']['prompt_tokens'] = tokens
                line = b'data: '+json.dumps(item).encode()
            yield line
    result.iter_lines = lines
    return result


def native_actor(tmp_path, events, perform, tools=None):
    from coding.pi.session import runtime_preflight, RuntimeUnavailable
    try: runtime_preflight()
    except RuntimeUnavailable: pytest.skip('cached pinned Pi/Node22 required')
    workflow = Workflow(tmp_path/'workflow.json', lambda *_:False, source_hash='project-v1')
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
    assert any(kind=='pi-event' and row['type']=='agent_settled' for kind,row in events)


def test_native_compaction_runs_through_same_gateway_without_context_rewrite(monkeypatch,tmp_path):
    events, requests, recalls, summaries = [], [], [], []
    names=builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    def provider(*_, **kwargs):
        request=json.loads(kwargs['data']);requests.append(request)
        assert len(kwargs['data'])<=49152
        if not request.get('tools'):
            summaries.append(request)
            return response([],len(requests),tokens=len(kwargs['data'])//4)
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
    assert summaries and len(recalls)==9 and flow.stage==Stage.DESIGN
    assert any(kind=='pi-event' and row['type']=='compaction_end' and not row.get('aborted') for kind,row in events), [row for kind,row in events if kind=='pi-event' and 'compact' in row['type']]
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
    settled=next(i for i,(kind,row) in enumerate(events) if kind=='pi-event' and row['type']=='agent_settled')
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
