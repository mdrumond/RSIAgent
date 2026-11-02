"""Public A3 usage derives from the consuming SDK session and durable receipts."""
import json
from types import SimpleNamespace

import pytest

from benchmarks.a3_experiments import BackendModel
from coding.pi.session import PiCodingSession
from tests.canonical_a3_fixtures import A3Session, make_trial, recording_candidate
from test_a3_trial_loop import _cell, SOURCE
from test_native_pi_programming_roles import redirect
from test_pi_runtime import _native_wire_objects


@pytest.mark.parametrize('mode', ['known', 'missing_usage', 'unknown'])
def test_a3_public_usage_survives_adoption_and_reopen(tmp_path, monkeypatch, scripted_provider, mode):
    import coding.pi.native as native
    url, calls, requests = scripted_provider
    redirect(monkeypatch, url)
    calls.include_usage = mode != 'missing_usage'
    original_record = native.NativeSession._record
    interrupted=[]
    def record(session,event):
        original_record(session,event)
        if mode=='unknown' and event['type']=='message_start' and event.get('message',{}).get('role')=='assistant' and not interrupted:
            interrupted.append(True)
            with session.events.mutex:
                session.events.queue.appendleft(dict(type='protocol_error',message='Lost native response before usage receipt'))
    monkeypatch.setattr(native.NativeSession,'_record',record)
    binding=PiCodingSession(model='deepseek-flash',environ={'DEEPSEEK_API_KEY':'fixture-secret'})
    assert binding.completion_tokens == 0
    owner,_=make_trial(tmp_path,session=binding,cell=_cell(model=BackendModel.DEEPSEEK_FLASH),candidate=recording_candidate())
    model=A3Session(initial=SOURCE,repaired=SOURCE+'\n// repaired')
    queued=[]
    task_reads=set()
    def choose(wire):
        prompt=next(value for role,value in reversed(list(_native_wire_objects([wire])))
                    if role=='user' and 'mapped_skills' in value)
        if queued:return queued.pop(0)
        if prompt['operation_id'] not in task_reads:
            task_reads.add(prompt['operation_id'])
            return 'read',dict(path=prompt['task']['path'])
        controller=binding.usage_source
        name,args=model.next_action(controller.current[0],list(controller.current[1]),seconds=60,record=lambda *_:None)
        if name=='read_skill':
            return 'read',dict(path=next(row['path'] for row in prompt['mapped_skills'] if row['name']==args['name']))
        if name=='perform' and 'edit' in args:
            edit=args.pop('edit')
            queued.append(('perform',args))
            return 'write',dict(path=prompt['editable_files'][0],content=edit['source'])
        return name,args
    calls.responder=choose
    result=owner.run()
    assert requests and binding.closed
    expected=2*len(requests) if mode=='known' else None
    assert result.tokens == binding.completion_tokens == expected
    if mode!='unknown':
        assert result.status=='passed'
    else:
        assert result.status!='passed'
    if mode=='unknown':assert interrupted and owner.candidate.calls==[]
    before=len(requests)
    fresh=PiCodingSession(model='deepseek-flash',environ={'DEEPSEEK_API_KEY':'fixture-secret'})
    assert fresh.completion_tokens == 0
    resumed,_=make_trial(tmp_path,session=fresh,cell=_cell(model=BackendModel.DEEPSEEK_FLASH),candidate=recording_candidate())
    # The CPU fixture resolver is in-memory; retain its actual prior authority.
    resumed.memory, resumed.candidate, resumed.profiler = owner.memory, owner.candidate, owner.profiler
    effects=list(owner.candidate.calls)
    assert resumed.run().tokens == expected
    assert owner.candidate.calls==effects
    assert len(requests)==before  # Accepted/unknown retained operations never replay.


@pytest.mark.parametrize('reported', [9, None])
def test_a3_historical_usage_receipts_override_an_unused_binding(tmp_path, reported):
    class Historical(A3Session):
        completion_tokens=0
        recorded=False
        def next_action(self,context,allowed,**kwargs):
            if not self.recorded:
                self.recorded=True
                kwargs['record']('pi-provider',dict(usage={'completion_tokens':reported}))
            return super().next_action(context,allowed,**kwargs)
    owner,_=make_trial(tmp_path,session=Historical(initial=SOURCE,repaired=SOURCE+'\n// repaired'))
    assert owner.run().tokens == reported
    runner=owner.canonical.build(owner)
    assert sum(event['type']=='pi-provider' for event in runner.events)==1
    unused=SimpleNamespace(completion_tokens=0)
    from rsi.domain_bindings import _retained_programming_tokens
    assert _retained_programming_tokens(runner,unused)==reported
