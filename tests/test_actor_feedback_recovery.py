from types import SimpleNamespace

import json
import pytest

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


@pytest.mark.parametrize('single_evaluation,prepare_only', [(True, False), (False, True)])
def test_inflight_feedback_preserves_compiler_diagnostics_on_reissued_request(
        tmp_path, single_evaluation, prepare_only):
    body = {'files': {}, 'memory_entries': []}
    store = JobStore(tmp_path / 'jobs')
    job = store.create('def forward(x): return x', 'forward',
        {'args': [{'shape': [5], 'dtype': 'float32'}]},
        {**body, 'snapshot_id': sha(canonical(body))}, max_seconds=100)
    key = job['job_id']
    control = ActorControl(store.root / key)
    diagnostic = {'status': 'compile-error', 'stderr': 'kernel.cpp:17: error: unknown vector intrinsic'}
    seen = []
    class Agent:
        identity = {}
        def complete(self, stage, context, record):
            seen.append(json.loads(context['history'][-1]['content_preview']))
            if len(seen) == 1:
                control.send('fix-intrinsic', 'Use the supported intrinsic and keep the compiler diagnostic.')
                return {'action': 'candidate', 'source': 'stale', 'block_count': 1, 'summary': 'stale'}
            assert context['user_feedback'][0]['content'] == 'Use the supported intrinsic and keep the compiler diagnostic.'
            return {'action': 'finish', 'summary': 'reviewed diagnostic'}
    store.transition(key, 'running')
    loop = OptimizationLoop(store, key, Agent(), SimpleNamespace())
    loop.context, loop.single_evaluation, loop.prepare_only = {}, single_evaluation, prepare_only
    loop.state['history'] = [{'tool': {'action': 'compile'}, 'content': diagnostic}]
    loop._save()
    result = loop._model('optimize', {'history': loop.state['history']})
    assert result == {'action': 'finish', 'summary': 'reviewed diagnostic'}
    assert seen == [diagnostic, diagnostic]
    assert loop.state['history'][-1]['content'] == diagnostic
