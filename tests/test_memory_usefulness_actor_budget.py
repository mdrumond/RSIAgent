"""Exercise production actor/provider deadlines used by record-only arms."""
from dataclasses import replace
import json
import sys
import time
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.evidence import EvidenceLedger
from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActorProjectExecutor
from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
from benchmarks.a5kernels.phase1_provider import (
    DirectActorDriver, DirectCompletion, DirectCompletionProvider,
    _ProviderEvidenceRecorder, direct_profile,
)
from benchmarks.ascend.agent_context import ContextBudget, WorkingContext
from benchmarks.ascend.training_traces import recorder_factory
from test_a5_phase1_adaptive_actor import SourceBackend, _request


ENV = {'OPENAI_API_KEY': 'fake-openai', 'DEEPSEEK_API_KEY': 'fake-deepseek'}


def driver_type(kind):
    if kind == 'direct':
        return DirectActorDriver
    from coding.pi.actor import PiActorDriver
    from coding.pi.session import runtime_preflight
    try:
        runtime_preflight()
    except RuntimeError:
        pytest.skip('Pinned Pi and Node 22 are required')
    return PiActorDriver


@pytest.mark.parametrize('kind', ['direct', 'pi'])
def test_production_record_only_actor_caps_transport_without_mutating_parent(tmp_path, monkeypatch, kind):
    Driver = driver_type(kind)
    now, timeouts = [0.0], []
    actions = [('write', {'slot': 'kernel', 'content': 'fixed source'}), ('run', {}), ('submit', {})]
    def transport(**options):
        timeouts.append(options['request']['timeout'])
        name, value = actions[len(timeouts)-1]
        now[0] += 1
        return DirectCompletion(json.dumps({name: value}), 1)
    if kind == 'pi':
        from test_pi_runtime import stream
        def http(*args, **options):
            timeouts.append(options['timeout'][1])
            name, value = actions[len(timeouts)-1]
            return stream(name, value, len(timeouts))
        monkeypatch.setattr('coding.pi.gateway.requests.post', http)
    provider = DirectCompletionProvider(ENV, transport=transport, clock=lambda: now[0])
    parent = Driver(provider, max_seconds=100, clock=(time.monotonic if kind == 'pi' else lambda: now[0]))
    identity = next(row for row in MODEL_IDENTITIES if row.backend_model.value == 'deepseek-flash')
    executor = ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path/'events.jsonl'))
    adapter = AdaptiveActorProjectExecutor(executor, parent, direct_profile(identity),
        trace_recorder_factory=recorder_factory(phase='phase1', round_id='usefulness',
            model='fixture', source_manifest_sha256='a'*64))
    result = adapter(replace(_request(tmp_path, executor), remaining_seconds=10, publish_learning=False))
    assert result.evidence.verified and result.memory.lessons == ()
    assert len(timeouts) == 3 and 0 < timeouts[0] <= 10
    assert all(0 < after < before for before, after in zip(timeouts, timeouts[1:]))
    assert parent.max_seconds == 100 and not hasattr(parent, '_provider_deadline')
    assert parent.provider is provider


@pytest.mark.parametrize('kind', ['direct', 'pi'])
def test_bound_actor_compaction_and_completion_shrink_actual_http_timeout(tmp_path, monkeypatch, kind):
    Driver = driver_type(kind)
    now, timeouts = [0.0], []
    class OpenAI:
        def __init__(self, **options):
            timeouts.append(options['timeout'])
            def create(**request):
                now[0] += 1
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='retained work log'))],
                                       usage=SimpleNamespace(completion_tokens=1))
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=OpenAI))
    profile = direct_profile(MODEL_IDENTITIES[0])
    provider = DirectCompletionProvider(ENV, clock=lambda: now[0], context_budgets={
        profile.identity.backend_model.value: ContextBudget(5000, 3500, 7000)})
    parent = Driver(provider, clock=lambda: now[0])
    actor = parent.with_time_budget(10)
    working = WorkingContext([dict(role='system', content='system'), dict(role='user', content='task')])
    working.recent = [dict(role='user', content='x'*900) for _ in range(8)]
    recorder = _ProviderEvidenceRecorder(tmp_path/'actor')
    actor._prepare_context(working, profile, recorder, turn=1)
    compacted = len(timeouts)
    assert compacted > 0
    actor._complete(recorder, profile, working.messages(), stage='action', turn=2)
    assert len(timeouts) == compacted + 1
    assert timeouts == [10-index for index in range(len(timeouts))]
    now[0] = 10
    with pytest.raises(TimeoutError):
        actor._complete(recorder, profile, [], stage='action', turn=3)
    assert len(timeouts) == compacted + 1 and not hasattr(parent, '_provider_deadline')


def test_bound_injected_provider_keeps_its_existing_completion_contract():
    calls = []
    class Provider:
        def complete(self, profile, messages, *, attempt_observer):
            calls.append((profile, messages))
            return DirectCompletion('legacy result', 1)
    actor = DirectActorDriver(Provider(), clock=lambda: 1).with_time_budget(5)
    result = actor._complete(_ProviderEvidenceRecorder(None), object(), [], stage='action', turn=1)
    assert result.text == 'legacy result' and len(calls) == 1


def test_expired_arm_setup_retains_abort_trace_without_provider_dispatch(tmp_path, monkeypatch):
    import benchmarks.a5kernels.phase1_adaptive_actor as adaptive
    from benchmarks.a5kernels.sft_dataset import _read_trace
    now, calls = [0.0], []
    provider = DirectCompletionProvider(ENV, transport=lambda **kw: calls.append(kw), clock=lambda: now[0])
    parent = DirectActorDriver(provider, clock=lambda: now[0])
    executor = ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path/'events.jsonl'))
    request = replace(_request(tmp_path, executor), remaining_seconds=1, publish_learning=False)
    prepare = executor.prepare
    def slow_prepare(spec):
        prepared = prepare(spec)
        now[0] = 2
        return prepared
    monkeypatch.setattr(executor, 'prepare', slow_prepare)
    monkeypatch.setattr(adaptive.time, 'monotonic', lambda: now[0])
    adapter = AdaptiveActorProjectExecutor(executor, parent, direct_profile(MODEL_IDENTITIES[0]),
        trace_recorder_factory=recorder_factory(phase='phase1', round_id='usefulness',
            model='fixture', source_manifest_sha256='a'*64))
    with pytest.raises(TimeoutError):
        adapter(request)
    assert calls == []
    rows = _read_trace(next((request.branch_root/'sft-traces').glob('*.jsonl')))
    assert rows[-1]['kind'] == 'terminal'
    assert any(row['kind'] == 'validation' and row['payload']['status'] == 'infrastructure-failure'
               for row in rows)


def test_request_binding_cannot_extend_a_shorter_actor_limit():
    calls = []
    provider = DirectCompletionProvider(ENV, transport=lambda **options:
        calls.append(options['request']['timeout']) or DirectCompletion('bounded result', 1), clock=lambda: 1)
    actor = DirectActorDriver(provider, max_seconds=3, clock=lambda: 1).with_time_budget(10)
    actor._complete(_ProviderEvidenceRecorder(None), direct_profile(MODEL_IDENTITIES[0]), [], stage='action', turn=1)
    assert calls == [3] and actor.max_seconds == 3


@pytest.mark.parametrize('uncertain', [False, True])
def test_native_pool_reuses_only_host_admission_timeout(tmp_path, monkeypatch, uncertain):
    import benchmarks.a5kernels.phase1_adaptive_actor as adaptive
    from benchmarks.ascend.production import NativeProjectPool
    from benchmarks.a5kernels.phase1_wave_runtime import ProjectAdmissionTimeout
    now, calls = [0.0], []
    backend = SourceBackend()
    actions = [dict(write=dict(slot='kernel', content='fixed source')), dict(run={}), dict(submit={}),
               dict(distillation=dict(no_new_knowledge=True, lessons=[]))]
    def transport(**options):
        calls.append(options)
        if uncertain:
            raise TimeoutError('remote response lost after dispatch')
        return DirectCompletion(json.dumps(actions[len(calls)-1]), 1)
    provider = DirectCompletionProvider(ENV, transport=transport, clock=lambda: now[0])
    actor = DirectActorDriver(provider, clock=lambda: now[0])
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path/'events.jsonl'))
    request = replace(_request(tmp_path, executor), remaining_seconds=1, publish_learning=False)
    prepare = executor.prepare
    def setup(spec):
        prepared = prepare(spec)
        if not uncertain:
            now[0] += 2
        return prepared
    monkeypatch.setattr(executor, 'prepare', setup)
    monkeypatch.setattr(adaptive.time, 'monotonic', lambda: now[0])
    pool = NativeProjectPool([AdaptiveActorProjectExecutor(executor, actor, direct_profile(MODEL_IDENTITIES[0]))])
    from benchmarks.a5kernels.phase1_live import InfrastructureFailure
    with pytest.raises(InfrastructureFailure if uncertain else ProjectAdmissionTimeout) as failure:
        pool(request)
    main_request = replace(request, publish_learning=True, remaining_seconds=10,
                           branch_root=tmp_path/'ordinary')
    if uncertain:
        assert not isinstance(failure.value, ProjectAdmissionTimeout)
        with pytest.raises(RuntimeError, match='settle retained operations'):
            pool(main_request)
        assert len(calls) == 1 and pool.failure is failure.value
    else:
        assert isinstance(failure.value, ProjectAdmissionTimeout)
        assert calls == [] and backend.requests == [] and backend.run_requests == []
        assert pool.failure is None
        result = pool(main_request)
        assert result.evidence.verified and len(backend.run_requests) == 1
        assert len(calls) == 4 and pool.failure is None
