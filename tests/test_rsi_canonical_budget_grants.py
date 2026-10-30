"""Explicit elapsed grants on the one canonical workflow; CPU fixtures only."""
import json
from types import SimpleNamespace

import rsi.domain_bindings as bindings
from test_rsi_production_domain_bindings import CanonicalSession


def test_explicit_job_allocation_resumes_genuine_canonical_exhaustion_without_refund_or_duplicate_grants(tmp_path):
    from optimizer.loop import OptimizationLoop
    from test_optimizer_loop import fixture, Backend
    from test_rsi_domain_execution import Clock
    store,key=fixture(tmp_path,seconds=3);backend=Backend();clock=Clock()
    class LateDecision(CanonicalSession):
        expired=False
        def next_action(self,context,allowed,**kwargs):
            if context['stage']=='implementation' and not context['unread_skills'] and not self.expired:
                self.expired=True;clock.value+=4
            return super().next_action(context,allowed,**kwargs)
    model=LateDecision(initial=json.dumps(dict(source='fast',block_count=1)),
                       repaired=json.dumps(dict(source='fast',block_count=1)))
    adoption=bindings.optimizer_adoption(session_factory=lambda:model,test_backend=True,clock=clock)
    agent=SimpleNamespace(identity={'model':'CPU decisions'})
    def owner():
        return OptimizationLoop(store,key,agent,backend,knowledge=lambda _:[],canonical=adoption)
    first=owner()
    assert first.run()['status']=='budget-exhausted'
    path=first.directory/'canonical-1/workflow.json'
    original=json.loads(path.read_text());pending=original['pending']
    assert pending['stage']=='implementation' and backend.evaluations==[]
    assert store.get(key)['elapsed_seconds']==4 and original['limits']['actor_seconds']==3
    before=len(model.contexts)
    assert owner().run()['status']=='budget-exhausted'
    assert len(model.contexts)==before and backend.evaluations==[]
    store.resume(key,additional_seconds=10)
    second=owner()
    assert second.run()['status']=='completed'
    state=json.loads(path.read_text())
    assert state['limits']==original['limits']
    assert all(state['intervals'][key]==value for key,value in original['intervals'].items())
    assert len(state['actor_grants'])==1 and sum(row['seconds'] for row in state['actor_grants'].values())==10
    assert any(row.get('operation')==pending['id'] and row.get('rejected_operation') for row in state['history'])
    assert store.get(key)['elapsed_seconds']==4 and store.get(key)['max_seconds']==13
    assert backend.prepares==1 and backend.evaluations==['fast'] and store.get(key)['attempts']==1
    runner=second.canonical.build(second)
    assert runner.workflow.actor_allocation()==13 and runner.workflow.remaining()['actor_seconds']==9
    assert runner._completed_effects()['actor_seconds_total']==4
    calls=(backend.prepares,list(backend.evaluations),len(model.contexts))
    reopened=second.canonical.build(second)
    assert reopened.workflow.state['actor_grants']==state['actor_grants']
    assert (backend.prepares,backend.evaluations,len(model.contexts))==calls


def _allocation_owner(tmp_path, *, unknown=False):
    from optimizer.loop import OptimizationLoop
    from optimizer.control import WorkerStopped
    from test_optimizer_loop import fixture, Backend
    from test_rsi_domain_execution import Clock
    store,key=fixture(tmp_path,seconds=3);clock=Clock()
    class Host(Backend):
        def evaluate(self,source,*args,**kwargs):
            if unknown:
                self.evaluations.append(source);clock.value+=4
                raise WorkerStopped('Native result has no authoritative observer')
            return super().evaluate(source,*args,**kwargs)
    class Model(CanonicalSession):
        expired=False
        def next_action(self,context,allowed,**kwargs):
            if not unknown and context['stage']=='implementation' and not context['unread_skills'] and not self.expired:
                self.expired=True;clock.value+=4
            return super().next_action(context,allowed,**kwargs)
    backend=Host();model=Model(initial=json.dumps(dict(source='fast',block_count=1)))
    adoption=bindings.optimizer_adoption(session_factory=lambda:model,test_backend=True,clock=clock)
    def owner():
        return OptimizationLoop(store,key,SimpleNamespace(identity={'model':'CPU decisions'}),backend,
            knowledge=lambda _:[],canonical=adoption)
    return store,key,backend,model,owner


def test_additional_allocation_cannot_clear_unknown_native_work_or_replay_model(tmp_path):
    import pytest
    from optimizer.control import WorkerStopped
    from rsi.runtime import WorkflowError
    store,key,backend,model,owner=_allocation_owner(tmp_path,unknown=True)
    first=owner()
    with pytest.raises(WorkerStopped):first.run()
    path=first.directory/'canonical-1/workflow.json'
    before=json.loads(path.read_text())
    assert before['pending']['stage']=='test' and any(row[2] is None for row in before['intervals'].values())
    calls=(backend.prepares,list(backend.evaluations),len(model.contexts),store.get(key)['attempts'])
    store.resume(key,additional_seconds=10)
    with pytest.raises(WorkflowError):owner().run()
    after=json.loads(path.read_text())
    assert after['pending']==before['pending'] and after['intervals']==before['intervals']
    assert after['timing_uncertainties']==before['timing_uncertainties'] and after['outcome'] is None
    assert (backend.prepares,backend.evaluations,len(model.contexts),store.get(key)['attempts'])==calls


# The authority key is the actual committed JobStore resume event SHA.
def test_reusing_job_grant_authority_with_changed_amount_is_rejected_without_refund(tmp_path):
    import pytest
    from rsi.runtime import WorkflowError
    store,key,_,_,owner=_allocation_owner(tmp_path)
    first=owner();assert first.run()['status']=='budget-exhausted'
    store.resume(key,additional_seconds=10)
    current=owner();runner=current.canonical.build(current)
    grant_key,grant=next(iter(runner.workflow.state['actor_grants'].items()))
    before=json.loads(runner.workflow.path.read_text())
    with pytest.raises(WorkflowError):
        runner.workflow.grant_actor_seconds(grant_key,11,grant['receipt'])
    assert json.loads(runner.workflow.path.read_text())==before
    assert runner.workflow.accounting.seconds()==4 and runner.workflow.actor_allocation()==13


def test_unbacked_workflow_grant_cannot_become_reopen_authority(tmp_path):
    import pytest
    from rsi.runtime import WorkflowError
    store,key,backend,model,owner=_allocation_owner(tmp_path)
    first=owner();assert first.run()['status']=='budget-exhausted'
    store.resume(key,additional_seconds=10)
    current=owner();runner=current.canonical.build(current)
    state=json.loads(runner.workflow.path.read_text())
    state['actor_grants']['not-a-job-grant']=dict(seconds=50,receipt='0'*64)
    runner.workflow.path.write_text(json.dumps(state))
    raw=runner.workflow.path.read_bytes();calls=(backend.prepares,list(backend.evaluations),len(model.contexts))
    fresh=owner()
    with pytest.raises(WorkflowError):fresh.canonical.build(fresh)
    assert runner.workflow.path.read_bytes()==raw
    assert (backend.prepares,backend.evaluations,len(model.contexts))==calls


def test_missing_retained_grant_is_fatal_before_provider_or_host_dispatch(tmp_path):
    import pytest
    store,key,backend,model,owner=_allocation_owner(tmp_path)
    first=owner();assert first.run()['status']=='budget-exhausted'
    store.resume(key,additional_seconds=10)
    current=owner();runner=current.canonical.build(current)
    receipt=next(iter(runner.workflow.state['actor_grants'].values()))['receipt']
    (runner.store.objects/receipt[:2]/receipt[2:]).unlink()
    raw=runner.workflow.path.read_bytes();calls=(backend.prepares,list(backend.evaluations),len(model.contexts))
    fresh=owner()
    with pytest.raises(FileNotFoundError):fresh.canonical.build(fresh)
    assert runner.workflow.path.read_bytes()==raw
    assert (backend.prepares,backend.evaluations,len(model.contexts))==calls


def test_interrupted_grant_cache_save_recovers_committed_authority_once_without_replay(tmp_path,monkeypatch):
    import pytest
    from rsi.runtime import Workflow
    store,key,backend,model,owner=_allocation_owner(tmp_path)
    first=owner();assert first.run()['status']=='budget-exhausted'
    path=first.directory/'canonical-1/workflow.json'
    original=json.loads(path.read_text())
    calls=(backend.prepares,list(backend.evaluations),len(model.contexts),store.get(key)['attempts'])
    store.resume(key,additional_seconds=10)
    save=Workflow._save;crashed=[]
    def crash_cache(flow):
        if flow.state.get('actor_grants') and not crashed:
            crashed.append(True)
            raise RuntimeError('Controller grant event committed before workflow cache save')
        return save(flow)
    with monkeypatch.context() as scoped:
        scoped.setattr(Workflow,'_save',crash_cache)
        pending=owner()
        with pytest.raises(RuntimeError):pending.canonical.build(pending)
    assert not json.loads(path.read_text()).get('actor_grants')
    recovered=owner();runner=recovered.canonical.build(recovered)
    assert len(runner.workflow.state['actor_grants'])==1
    assert len([row for row in runner.events if row['type']=='actor-allocation-grant'])==1
    assert runner.workflow.actor_allocation()==13 and runner.workflow.remaining()['actor_seconds']==9
    assert all(runner.workflow.state['intervals'][key]==value for key,value in original['intervals'].items())
    assert (backend.prepares,backend.evaluations,len(model.contexts),store.get(key)['attempts'])==calls
    again=recovered.canonical.build(recovered)
    assert again.workflow.state['actor_grants']==runner.workflow.state['actor_grants']
    assert len([row for row in again.events if row['type']=='actor-allocation-grant'])==1
    assert (backend.prepares,backend.evaluations,len(model.contexts),store.get(key)['attempts'])==calls
