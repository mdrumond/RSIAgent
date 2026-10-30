"""Model receipt typos are feedback; delivered host evidence stays authoritative."""
from copy import deepcopy

import pytest

from rsi.contracts import Stage
from rsi.runtime import WorkflowError
from rsi.strategy import QueryUnavailable
from test_rsi_project_runner import Session, make


@pytest.mark.parametrize('field', ['query_ref', 'retrospective_ref'])
@pytest.mark.parametrize('truncated', [False, True])
def test_old_operation_reference_is_corrected_with_fresh_tool(tmp_path,field,truncated):
    class CorrectReceipt(Session):
        old = invalid = feedback = None
        def next_action(self,context,allowed,**kwargs):
            evidence={row['name']:row['result'] for row in context['evidence']}
            if self.invalid and self.feedback is None:
                self.feedback=deepcopy(context)
                observation=context['observation']
                assert observation['event']=='invalid-action' and observation['executed'] is False
                tool='query_kdb' if field=='query_ref' else 'retrospect'
                assert observation['required_tools'][field]==tool
                assert not any(e['type']=='project-tool' and e['name']=='select_strategy' and
                    e['operation_id']==context['operation_id'] for e in runner.events)
                return (tool,{'query':'stable sum'} if tool=='query_kdb' else {'supplement':'Review current operation'})
            name,args=super().next_action(context,allowed,**kwargs)
            if context['stage']==Stage.DESIGN.value and name=='select_strategy':
                self.old=deepcopy(args)
            if context['stage']==Stage.OPTIMISATION.value and self.invalid is None and (
                    name=='select_strategy' or field=='retrospective_ref' and name=='retrospect'):
                args=deepcopy(self.old)
                args['query_ref']=evidence['query_kdb']['reference']
                args['skill_refs']=[row['receipt_id'] for row in context['read_skills']]
                if 'retrospect' in evidence:args['retrospective_ref']=evidence['retrospect']['reference']
                args[field]=self.old[field][:8] if truncated else self.old[field]
                self.invalid=deepcopy(args)
                return 'select_strategy',args
            return name,args
    session=CorrectReceipt()
    runner,_,database=make(tmp_path,session,
        source_text='from math import fsum\ndef solve(xs): return fsum(xs)\n')
    loaded=[]
    original=runner.service._load
    def load(kind,reference,context):
        loaded.append((context['operation'],reference))
        return original(kind,reference,context)
    runner.service._load=load
    try:
        result=runner.run()
        assert result['status']=='accepted',result
        failures=[row for row in runner.events if row['type']=='actor-invalid-action']
        assert len(failures)==1
        operation=failures[0]['operation_id']
        assert (operation,session.invalid[field]) not in loaded
        tools=[row for row in runner.events if row['type']=='project-tool' and row['operation_id']==operation]
        assert len([row for row in tools if row['name']=='select_strategy'])==1
        assert len([row for row in result['history'] if row['evidence']['operation_id']==operation])==1
        if field=='retrospective_ref':
            assert not any(row['name']=='retrospect' for row in session.feedback['evidence'])
        assert session.feedback['observation']['evidence_receipts']==session.feedback['evidence_receipts']
    finally:database.close()


@pytest.mark.parametrize('fault', ['missing','corrupt','workflow','query'])
def test_delivered_reference_host_fault_stays_fatal(tmp_path,fault):
    class Fault(Session):
        def next_action(self,context,allowed,**kwargs):
            name,args=super().next_action(context,allowed,**kwargs)
            if name=='select_strategy':
                if fault in ('missing','corrupt'):
                    path=runner.store._path(args['query_ref'])
                    if fault=='missing':path.unlink()
                    else:path.write_bytes(b'corrupt actual delivered artifact')
                else:
                    def fail(*a,**kw):
                        raise (QueryUnavailable if fault=='query' else WorkflowError)('host evidence unavailable')
                    runner.service.select=fail
            return name,args
    runner,_,database=make(tmp_path,Fault())
    try:
        runner.actor.step(runner.context());runner.actor.step(runner.context())
        expected={'missing':FileNotFoundError,'corrupt':ValueError,'workflow':WorkflowError,'query':QueryUnavailable}[fault]
        with pytest.raises(expected):runner.actor.step(runner.context())
        assert runner.workflow.stage==Stage.DESIGN and runner.workflow.pending
        assert not any(row['type']=='actor-invalid-action' for row in runner.events)
        assert not any(row['type']=='project-tool' and row['name']=='select_strategy' for row in runner.events)
    finally:runner.actor.close();database.close()


def test_all_delivered_receipts_are_eligible_even_when_inventory_has_newer_refs(tmp_path):
    class EarlierReceipt(Session):
        first=None
        repeated=0
        def next_action(self,context,allowed,**kwargs):
            name,args=super().next_action(context,allowed,**kwargs)
            if context['stage']==Stage.DESIGN.value and name=='select_strategy':
                if self.first is None:
                    self.first=deepcopy(args)
                    return 'query_kdb',{'query':'stable sum again'}
                if not self.repeated:
                    self.repeated=1
                    return 'retrospect',{'supplement':'Another current-operation observation'}
                latest={row['name']:row['reference'] for row in context['evidence_receipts']}
                assert latest['query_kdb']!=self.first['query_ref']
                assert latest['retrospect']!=self.first['retrospective_ref']
                return name,self.first
            return name,args
    session=EarlierReceipt()
    runner,_,database=make(tmp_path,session,
        source_text='from math import fsum\ndef solve(xs): return fsum(xs)\n')
    try:
        result=runner.run()
        assert result['status']=='accepted',result
        assert not any(row['type']=='actor-invalid-action' for row in runner.events)
        selected=next(row for row in runner.events if row['type']=='project-tool' and row['name']=='select_strategy')
        assert runner.store.read_json(selected['reference'])['payload']==session.first
    finally:database.close()


def test_strict_actor_preserves_original_unknown_receipt_error(tmp_path):
    class Strict(Session):
        def next_action(self,context,allowed,**kwargs):
            name,args=super().next_action(context,allowed,**kwargs)
            if name=='select_strategy':args['query_ref']=args['query_ref'][:8]
            return name,args
    runner,_,database=make(tmp_path,Strict())
    runner.actor.recover_invalid_actions=False
    try:
        runner.actor.step(runner.context());runner.actor.step(runner.context())
        with pytest.raises(WorkflowError):runner.actor.step(runner.context())
        assert runner.workflow.pending and not any(row['type']=='actor-invalid-action' for row in runner.events)
    finally:runner.actor.close();database.close()
