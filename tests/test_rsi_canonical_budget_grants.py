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
