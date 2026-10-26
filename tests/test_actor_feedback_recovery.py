from types import SimpleNamespace

from benchmarks.ascend.actor_control import ActorControl
from optimizer.jobs import JobStore, canonical, sha
from optimizer.loop import OptimizationLoop


def test_cached_optimizer_action_is_superseded_before_replay(tmp_path):
    from optimizer.jobs import JobStore, sha
    from optimizer.loop import OptimizationLoop
    body = {'files': {}, 'memory_entries': []}
    store = JobStore(tmp_path/'jobs')
    job = store.create('def forward(x): return x', 'forward',
        {'args': [{'shape': [5], 'dtype': 'float32'}]}, {**body, 'snapshot_id': sha(canonical(body))}, max_seconds=100)
    seen = []
    class Agent:
        identity = {}
        def complete(self, stage, context, record):
            seen.extend(context['user_feedback'])
            return {'action': 'finish', 'summary': 'redirected'}
    store.transition(job['job_id'], 'running')
    loop = OptimizationLoop(store, job['job_id'], Agent(), SimpleNamespace())
    loop.state['model_action'] = loop._artifact('optimize', {'action': 'candidate', 'source': 'stale'})
    loop.context, loop.single_evaluation, loop.prepare_only = {}, False, False
    ActorControl(store.root/job['job_id']).send('redirection', 'discard stale candidate')
    assert loop._model('optimize', {})['action'] == 'finish'
    assert seen[0]['content'] == 'discard stale candidate'



def test_optimizer_stop_settles_manifest_before_return_and_can_resume(tmp_path):
    body = {'files': {}, 'memory_entries': []}
    store = JobStore(tmp_path/'jobs')
    job = store.create('def forward(x): return x', 'forward',
        {'args': [{'shape': [5], 'dtype': 'float32'}]}, {**body, 'snapshot_id': sha(canonical(body))}, max_seconds=100)
    class Agent:
        identity = {}
        def complete(self, *args):
            raise AssertionError('stopped actor must not issue a provider request')
    store.transition(job['job_id'], 'running')
    loop = OptimizationLoop(store, job['job_id'], Agent(), SimpleNamespace())
    loop.context, loop.single_evaluation, loop.prepare_only = {}, False, False
    ActorControl(store.root/job['job_id']).stop()
    assert loop._model('optimize', {}) is None
    assert store.get(job['job_id'])['status'] == 'stopped'
    assert store.get(job['job_id'])['stop_requested'] is True
    assert store.resume(job['job_id'], additional_seconds=15)['status'] == 'queued'
