"""Actual native summary/overflow/close recovery, never Python history editing."""
import json
import threading
import time

import pytest

from rsi.actor import EvidenceTool,PiSessionAdapter,StageResult
from coding.pi.schema import STRING,obj
from rsi.contracts import Stage
from test_rsi_pi_controller import native_actor,response


def test_actual_pi_recovers_gateway_overflow_without_replaying_host_effects(monkeypatch,tmp_path):
    events,requests,effects,summaries=[],[],[],[]
    from rsi.skills import builtin_bundle
    names=builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    def provider(*_,**kwargs):
        request=json.loads(kwargs['data']);requests.append(request)
        assert len(kwargs['data'])<=49152
        if not request.get('tools'):
            summaries.append(request)
            return response([],len(requests),tokens=100,summary='Current task is pending. Retain guarded actions and actual receipt IDs.',completion_tokens=50)
        read=[row['skill'] for kind,row in events if kind=='skill-usage' and row['kind']=='read'
              and row['stage']==Stage.TASK_CONSTRUCTION.value]
        missing=[name for name in names if name not in read]
        actions=([('read_skill',{'name':missing[0]})] if missing else
                 [('evidence',{'sample':str(len(effects))})] if len(effects)<20 else
                 [('perform',{'proposal':'complete after native overflow recovery'})])
        # Real providers can underreport byte-heavy context; force the actual
        # byte guard before the SDK's token threshold, then native recovery.
        return response(actions,len(requests),tokens=1)
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    def evidence(stage,args,operation,seconds):
        effects.append(args['sample'])
        return dict(reference='actual-'+args['sample'],evidence='evidence payload '*150)
    actor,flow=native_actor(tmp_path,events,lambda *_:StageResult(),
        {'evidence':EvidenceTool(obj({'sample':STRING}),evidence)})
    try:actor.step({});actor.step({'goal':'recover a real gateway overflow'})
    finally:actor.close()
    limits=[row for kind,row in events if kind=='pi-context-limit']
    assert limits and all(row['request_bytes']>49152 for row in limits)
    assert summaries and effects==[str(i) for i in range(20)]
    assert any(kind=='pi-event' and row['event']['type']=='compaction_end'
               and row['event']['reason']=='overflow' and row['event'].get('result') for kind,row in events)
    assert flow.stage==Stage.DESIGN and len(flow.state['history'])==2
    intents=[row['reservation_id'] for kind,row in events if kind=='pi-provider-request']
    ledgers=[row['reservation_id'] for kind,row in events if kind=='pi-provider']
    assert set(intents)==set(ledgers) and len(ledgers)==len(set(ledgers))==len(requests)


def test_actual_project_runner_multistage_concise_summaries(monkeypatch,tmp_path):
    from test_rsi_project_runner import make,Session
    from dataclasses import asdict
    from coding.pi.session import runtime_preflight,RuntimeUnavailable
    try:runtime_preflight()
    except RuntimeUnavailable:pytest.skip('cached pinned Pi/Node22 required')
    requests,summaries=[],[]
    planner=Session()
    runner=None
    def provider(*_,**kwargs):
        request=json.loads(kwargs['data']);requests.append(request)
        assert len(kwargs['data'])<=49152
        if not request.get('tools'):
            summaries.append(request)
            return response([],len(requests),tokens=len(kwargs['data'])//4,
                summary='Current Python repair task: preserve source, current receipts, numeric verifier and outstanding failures.',completion_tokens=50)
        operation=runner.workflow.pending['id']
        reads=[dict(skill=row['skill'],section=row['section'],receipt_id=runner.service.read_receipt(row['receipt_id']))
               for row in runner.events if row['type']=='skill-usage' and row['kind']=='read' and row['operation_id']==operation]
        evidence=[dict(name=row['name'],result=dict(reference=row['reference'],evidence=runner.store.read_json(row['reference'])))
                  for row in runner.events if row['type']=='project-tool' and row['operation_id']==operation]
        source=runner.snapshot()
        context=dict(skills=runner.bundle.skills_for(runner.workflow.stage),read_skills=reads,
            evidence=evidence,stage=runner.workflow.stage.value,task=runner.context(),
            source=dict(path=source.path,sha256=source.sha256,text=source.text,project_hash=source.project_hash))
        action=planner.next_action(context,[])
        return response([action],len(requests),tokens=len(kwargs['data'])//4)
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    runner,_,database=make(tmp_path,session_factory=lambda:PiSessionAdapter(
        model='deepseek-flash',environ={'DEEPSEEK_API_KEY':'fake-provider-secret'}))
    try:
        result=runner.run()
        assert result['status']=='accepted',result
        assert result['correctness'] is True and summaries
        native=[row['event'] for row in runner.events if row['type']=='pi-native-event']
        assert native and all(len(row['summary'].encode())<=6144 for row in native if row['successful'])
        assert set(Stage(row['evidence']['stage']) for row in result['history'])==set(Stage)
    finally:runner.actor.close();database.close()


def test_actual_project_runner_cancel_retains_late_actual_ledger_once(monkeypatch,tmp_path):
    from test_rsi_project_runner import make
    from coding.pi.rpc import PiRPC,PiCancelled
    from rsi.skills import builtin_bundle
    from coding.pi.session import runtime_preflight,RuntimeUnavailable
    try:runtime_preflight()
    except RuntimeUnavailable:pytest.skip('cached pinned Pi/Node22 required')
    calls,holders=[],[]
    names=builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    def provider(*_,**kwargs):
        calls.append(json.loads(kwargs['data']))
        if len(calls)==1:return response([('read_skill',{'name':name}) for name in names],len(calls))
        result=response([],len(calls));original=result.iter_lines
        def lines():
            for line in original():
                if line.startswith(b'data: {'):
                    value=json.loads(line[6:]);value.pop('usage',None)
                    line=b'data: '+json.dumps(value).encode()
                yield line
        result.iter_lines=lines
        return result
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    def factory():
        adapter=PiSessionAdapter(model='deepseek-flash',environ={'DEEPSEEK_API_KEY':'fake-provider-secret'})
        session=adapter.session;holders.append(session);original=session.gateway.on_record
        def late(kind,data):
            rows=session.gateway.records
            if kind=='pi-provider' and len(rows)>1 and data['reservation_id']==rows[1]['reservation_id']:
                end=time.monotonic()+5
                while not session.closed and time.monotonic()<end:time.sleep(.01)
            original(kind,data)
        session.gateway.on_record=late
        return adapter
    runner,_,database=make(tmp_path,session_factory=factory)
    original_prompt=PiRPC.prompt
    def aborted(rpc,message,**kwargs):
        listener=kwargs['on_event']
        def observe(event):
            listener(event)
            if len(calls)>=2 and event['type']=='message_end' and event.get('message',{}).get('role')=='assistant':
                rpc.send({'type':'abort'})
                raise PiCancelled('test operator abort')
        kwargs['on_event']=observe
        return original_prompt(rpc,message,**kwargs)
    monkeypatch.setattr(PiRPC,'prompt',aborted)
    try:
        runner.actor.step(runner.context())
        with pytest.raises(PiCancelled):runner.actor.step(runner.context())
        runner.actor.close()
        intents=[row for row in runner.events if row['type']=='pi-provider-request']
        ledger=[row for row in runner.events if row['type']=='pi-provider']
        # A reserved row may be cancelled before durable intent admission.
        reserved=holders[0].gateway.records
        dispatched=[row for row in ledger if row['dispatch_attempted']]
        assert len(calls)==len(dispatched)>=2
        assert {row['reservation_id'] for row in dispatched} <= {row['reservation_id'] for row in intents}
        assert {row['reservation_id'] for row in reserved}=={row['reservation_id'] for row in ledger}
        assert len({row['reservation_id'] for row in ledger})==len(ledger)
        assert any(row['usage'] is None and row.get('close_snapshot') for row in ledger)
        assert runner.workflow.pending and runner.workflow.stage==Stage.TASK_CONSTRUCTION
        assert holders[0].gateway.finished.is_set()
    finally:runner.actor.close();database.close()


def test_actual_pi_close_reaps_child_when_host_leaves_accounting_unknown(monkeypatch,tmp_path):
    from rsi.runtime import WorkflowError
    from rsi.skills import builtin_bundle
    events,calls=[],[]
    names=builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    def provider(*_,**kwargs):
        calls.append(json.loads(kwargs['data']))
        actions=([('read_skill',{'name':name}) for name in names] if len(calls)==1 else
                 [('perform',{'proposal':'host failure leaves timing unresolved'})])
        return response(actions,len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    def failed_host(*args):
        flow.accounting.uncertainties['host-failure']={'reason':'unknown host interval'}
        flow.accounting.save()
        raise RuntimeError('host callback failed with unknown timing')
    actor,flow=native_actor(tmp_path,events,failed_host)
    actor.step({'goal':'retain unknown host failure'})
    with pytest.raises(Exception):
        actor.step({'goal':'retain unknown host failure'})
    session=actor.session.session
    process=session.rpc.process
    assert process.poll() is None
    with pytest.raises(WorkflowError):actor.close()
    assert session.closed and process.poll() is not None
    assert flow.pending and flow.stage==Stage.TASK_CONSTRUCTION
    assert 'host-failure' in flow.accounting.uncertainties
    closure=[row for kind,row in events if kind=='pi-ledger-closure']
    assert closure and closure[-1]['accounting_unknown'] is True
    assert {row['reservation_id'] for row in session.gateway.records}=={
        row['reservation_id'] for kind,row in events if kind=='pi-provider'}
    actor.close()  # Cleanup remains idempotent without reconciling unknown clocks.


def test_actual_pi_text_only_turn_continues_pending_stage_and_commits_once(monkeypatch,tmp_path):
    from rsi.skills import builtin_bundle
    events,calls,effects=[],[],[]
    names=builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    def provider(*_,**kwargs):
        request=json.loads(kwargs['data']);calls.append(request)
        if len(calls)==1:
            return response([],len(calls),summary='I will construct the task now.')
        if len(calls)==2:
            # The official boundary draft is actual provider-visible context.
            pending=[m for m in request['messages'] if m['role']=='user' and
                     flow.pending['id'] in json.dumps(m.get('content'))]
            assert pending
            assert all(name in json.dumps(pending[-1]) for name in names)
            return response([('read_skill',{'name':name}) for name in names],len(calls))
        assert len(calls)==3
        return response([('perform',{'proposal':'construct after actual skill reads'})],len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    def perform(*args):effects.append(args);return StageResult()
    actor,flow=native_actor(tmp_path,events,perform)
    try:
        actor.step({'goal':'text alone cannot finish host stage'})
        actor.step({'goal':'text alone cannot finish host stage'})
        assert flow.stage==Stage.DESIGN and flow.pending is None
        assert len(effects)==1 and len(calls)==3
        assert len(flow.state['history'])==2
        assert len([1 for kind,row in events if kind=='pi-event' and row['event']['type']=='agent_settled'])==1
    finally:actor.close()


def test_actual_pi_cancel_before_third_intent_retains_unadmitted_reservation(monkeypatch,tmp_path):
    from test_rsi_project_runner import make
    from coding.pi.rpc import PiCancelled
    from rsi.skills import builtin_bundle
    from coding.pi.session import runtime_preflight,RuntimeUnavailable
    try:runtime_preflight()
    except RuntimeUnavailable:pytest.skip('cached pinned Pi/Node22 required')
    calls,holders,cancelled=[],[],[]
    names=builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    def provider(*_,**kwargs):
        calls.append(json.loads(kwargs['data']))
        assert len(calls)<=2  # The third local reservation must never reach HTTP.
        return response([('read_skill',{'name':name}) for name in names] if len(calls)==1 else [],len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    def factory():
        adapter=PiSessionAdapter(model='deepseek-flash',environ={'DEEPSEEK_API_KEY':'fake-provider-secret'})
        session=adapter.session;holders.append(session)
        original=session.requests.get_nowait
        def dequeue():
            # Observe the actual queued intent BEFORE the main-thread pump
            # admits it. Leave it queued so close rejects the waiting callback.
            with session.requests.mutex:
                front=session.requests.queue[0] if session.requests.queue else None
            if not cancelled and front and front[0][:2]==('record','pi-provider-request') and len(session.gateway.records)==3:
                cancelled.append(front[0][2]['reservation_id'])
                session.rpc.send({'type':'abort'})
                raise PiCancelled('operator stops before durable third intent')
            return original()
        session.requests.get_nowait=dequeue
        return adapter
    runner,_,database=make(tmp_path,session_factory=factory)
    try:
        runner.actor.step(runner.context())
        with pytest.raises(PiCancelled):runner.actor.step(runner.context())
        runner.actor.close()
        session=holders[0]
        reserved=session.gateway.records
        ledger=[row for row in runner.events if row['type']=='pi-provider']
        intents=[row for row in runner.events if row['type']=='pi-provider-request']
        assert len(reserved)==len(ledger)==3 and len(calls)==len(intents)==2
        assert len({row['reservation_id'] for row in ledger})==3
        assert {row['reservation_id'] for row in reserved}=={row['reservation_id'] for row in ledger}
        assert {row['reservation_id'] for row in ledger if row['dispatch_attempted']}=={
            row['reservation_id'] for row in intents}
        unadmitted=[row for row in ledger if not row['dispatch_attempted']]
        assert len(unadmitted)==1 and unadmitted[0]['reservation_id']==cancelled[0]
        assert unadmitted[0]['usage'] is None and unadmitted[0]['status']=='transport_error'
        assert runner.workflow.pending and runner.workflow.stage==Stage.TASK_CONSTRUCTION
        assert session.closed and session.rpc.process.poll() is not None and session.gateway.finished.is_set()
    finally:runner.actor.close();database.close()
