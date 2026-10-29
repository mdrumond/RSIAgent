"""Subject optimizer tasks use the same actor-only budget as graph tasks."""
import json

from benchmarks.ascend.actor_time import ActorWorkBudget
from benchmarks.ascend.subject_runtime import build_subject_campaign
from benchmarks.ascend.subjects import Subject
from benchmarks.ascend.phase2_optimizer_execution import OptimizerTaskExecutor
from tests.test_subject_runtime import bound
from tests.test_phase2_optimizer_execution import Agent, Backend, candidate


def test_actual_subject_optimizer_branch_excludes_setup_and_stops_next_project(tmp_path, monkeypatch):
    now, created, reviewed = [0.], [], []
    class SlowBackend(Backend):
        def prepare(self, *args, **kwargs):
            now[0] += 1000
            return super().prepare(*args, **kwargs)
        def evaluate(self, *args, **kwargs):
            now[0] += 6
            return super().evaluate(*args, **kwargs)
    def factory(directory, task, snapshot, config, **kwargs):
        now[0] += 1000
        execution = OptimizerTaskExecutor(directory, task, snapshot, config,
            agent=Agent([{'action':'cases','signatures':[],'summary':'fixed'}, candidate('good')]),
            backend=SlowBackend(), **kwargs)
        created.append(execution)
        return execution
    monkeypatch.setattr('benchmarks.ascend.phase2_optimizer_execution.OptimizerTaskExecutor', factory)
    runtime = bound(tmp_path)
    runtime['actor'].executor.runtime_identity = {'architecture':'arch','device':'0'}
    config = {'arch':'arch','target':'registered-a3','physical_device':0,
        'validation_wrapper':str(tmp_path/'validation.sh'),'max_cases':1,'phase2_attempt_seconds':6}
    campaign = build_subject_campaign({'optimizer_worker_config':config},target='a3',root=tmp_path/'campaign',
        subjects=[Subject('s','optimizer',1)],initial_memory=b'',max_seconds=6,
        compose=lambda *a,**kw:runtime)
    campaign.clock = lambda:now[0]
    task = {'task_id':'relu','submission':{'source':'def f(x): return x.relu()',
        'entrypoint':'f','signatures':[{'args':[{'shape':[4],'dtype':'float32'}]}]}}
    campaign.select_wave = lambda *a,**kw:[dict(project_id=pid,task=dict(kind='optimizer',task=task)) for pid in ['one','two']]
    class Verifier:
        last_trace = []
        def review(self, request, **kwargs):
            reviewed.append(request.project_id)
            return dict(decision='accept',reason='verified',memory=None,repair=None,evidence=[request.source_sha256])
    campaign.verifier = Verifier()
    report = campaign.run()
    assert report['terminal'] == 'TIME_BUDGET_EXHAUSTED'
    assert len(created) == 1 and reviewed == ['one']
    assert report['projects']['one']['outcome']['host_verified']
    assert report['timing']['elapsed_seconds'] == 6
    assert report['timing']['preparation_seconds'] == 2000
    assert 'outcome' not in report['projects']['two']
    assert created[0].store.get(created[0].job_id)['elapsed_seconds'] == 6
