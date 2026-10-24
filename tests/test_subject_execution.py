"""Real production adapters with injected compiler/provider boundaries."""
from dataclasses import replace
import json

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import EvidenceLedger
from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActorProjectExecutor
from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
from benchmarks.ascend.phase2_optimizer_execution import OptimizerTaskExecutor
from benchmarks.ascend.subject_execution import SubjectExecutor, optimizer_snapshot
from test_a5_phase1_adaptive_actor import SourceBackend, ScriptedActor, _request
from test_ascend_phase2_targets import qualified, RUNTIME
from test_phase2_optimizer_execution import Agent, Backend, candidate


def graph_actor(tmp_path, replies):
    actor = ScriptedActor(replies)
    executor = ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path / 'events.jsonl'))
    return AdaptiveActorProjectExecutor(executor, actor, object()), actor


def test_graph_skips_self_distillation_retains_sources_and_resumes(tmp_path):
    execution, actor = graph_actor(tmp_path, [
        '{"write":{"slot":"kernel","content":"wrong source"}}',
        '{"run":{}}',
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}', '{"submit":{}}',
    ])
    request = _request(tmp_path, execution.executor)
    shared = ContentAddressedArtifacts(tmp_path / 'shared')
    branch_store = ContentAddressedArtifacts(request.branch_root / 'artifacts')
    historical = branch_store.put_bytes(b'unrelated historical memory')
    prewave = branch_store.put_bytes(request.pre_wave_memory)
    outcome = SubjectExecutor(execution).graph(request, artifacts=shared)
    assert outcome.host_verified and actor.distill_calls == []
    assert historical not in outcome.artifacts.values()
    assert prewave not in outcome.artifacts.values()
    result = json.loads(outcome.read_artifact(outcome.artifacts['result:execution']))
    assert result['evidence']['classification'] == 'host-verified'
    assert outcome.read_artifact(outcome.source_sha256) == b'fixed source'
    retained = [outcome.read_artifact(h) for h in outcome.artifacts.values()]
    assert b'wrong source' in retained
    assert outcome.observations['host_facts']
    assert all(shared.read_bytes(h) for h in outcome.artifacts.values())
    assert SubjectExecutor(execution).graph(request, artifacts=shared).as_dict() == outcome.as_dict()
    assert len(actor.observations) == 5
    with pytest.raises(ValueError, match='frozen inputs'):
        SubjectExecutor(execution).graph(replace(request, pre_wave_memory=b'changed'), artifacts=shared)


@pytest.mark.parametrize('changed', ['inputs', 'input-storage', 'oracle'])
def test_cached_graph_checks_complete_inputs_and_oracle(tmp_path, changed):
    execution, actor = graph_actor(tmp_path, [
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}', '{"submit":{}}',
    ])
    adapter = SubjectExecutor(execution)
    request = _request(tmp_path, execution.executor)
    passed = adapter.graph(request)
    assert passed.host_verified
    if changed == 'inputs':
        invalid = replace(request, inputs=())
    elif changed == 'input-storage':
        tensor = request.inputs[0]
        invalid = replace(request, inputs=(replace(tensor,
            storage=(tensor.storage[0] + 1, *tensor.storage[1:])), *request.inputs[1:]))
    else:
        invalid = replace(request, oracle=replace(request.oracle,
            values=tuple(value + 1 for value in request.oracle.values),
            storage=tuple(value + 1 for value in request.oracle.storage)))
    with pytest.raises(ValueError, match='rematerialized project'):
        adapter.graph(invalid)
    assert len(actor.observations) == 3
    assert adapter.graph(request).as_dict() == passed.as_dict()
    assert len(actor.observations) == 3


def test_sourceless_failed_actor_retains_trace_validation_evidence_in_shared_store(tmp_path):
    from benchmarks.ascend.training_traces import recorder_factory
    from test_a5_phase1_adaptive_actor import TracedScriptedActor
    executor = ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path / 'events.jsonl'))
    actor = TracedScriptedActor([])
    driver = AdaptiveActorProjectExecutor(executor, actor, object(),
        trace_recorder_factory=recorder_factory(phase='phase1', round_id='fixture',
            model='fixture', source_manifest_sha256='a' * 64))
    shared = ContentAddressedArtifacts(tmp_path / 'shared')
    outcome = SubjectExecutor(driver).graph(_request(tmp_path, executor), artifacts=shared)
    assert not outcome.host_verified and outcome.source_sha256 is None
    trace = next(digest for role, digest in outcome.artifacts.items() if role.startswith('log:sft-traces/'))
    rows = [json.loads(line) for line in outcome.read_artifact(trace).splitlines()]
    terminal = next(row['payload'] for row in rows
                    if row['kind'] == 'validation' and row['payload']['validator'] == 'actor-terminal')
    evidence = outcome.read_artifact(terminal['evidence_sha256'])
    assert json.loads(evidence)['status'] == 'turn-budget-exhausted'
    assert shared.read_bytes(terminal['evidence_sha256']) == evidence
    assert terminal['candidate_sha256'] not in outcome.artifacts.values()
    assert rows[0]['payload']['source_manifest_sha256'] not in outcome.artifacts.values()


def test_graph_retains_semantic_failure_without_publishing_memory(tmp_path):
    execution, actor = graph_actor(tmp_path, [
        '{"write":{"slot":"kernel","content":"wrong source"}}', '{"run":{}}',
    ])
    outcome = SubjectExecutor(execution).graph(_request(tmp_path, execution.executor))
    assert not outcome.host_verified and actor.distill_calls == []
    assert outcome.read_artifact(outcome.source_sha256) == b'wrong source'
    assert outcome.observations['evidence']['action_counts']['run'] == 1
    assert not (tmp_path / 'state.json').exists()


def test_graph_target_checks_observed_runtime_before_invoking_actor(tmp_path):
    target, store, _ = qualified(tmp_path)
    execution, actor = graph_actor(tmp_path, [])
    request = replace(_request(tmp_path, execution.executor), spec=target.spec)
    execution.executor.runtime_identity = dict(RUNTIME)
    with pytest.raises(ValueError, match='observed composed'):
        SubjectExecutor(execution).graph(request, target=target, artifacts=store,
                                        runtime={**RUNTIME, 'compiler': 'changed'})
    assert actor.instruction is None
    # Mutating the reference's observed output cannot be admitted using its old
    # saved verified classification; TargetContract recalculates the host oracle.
    reference = store.read_json(target.reference_sha256)
    observed = store.read_json(reference['artifacts']['observed'])
    observed['output'][0] += 999
    reference['artifacts']['observed'] = store.put_json(observed)
    corrupt = replace(target, reference_sha256=store.put_json(reference))
    with pytest.raises(ValueError, match='not host-verified'):
        SubjectExecutor(execution).graph(request, target=corrupt, artifacts=store)
    assert actor.instruction is None


def test_legacy_actor_request_still_distills(tmp_path):
    execution, actor = graph_actor(tmp_path, [
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}', '{"submit":{}}',
    ])
    result = execution(_request(tmp_path, execution.executor))
    assert len(actor.distill_calls) == 1 and result.memory.lessons


def optimizer(tmp_path, *, times=None):
    task = {'task_id': 'relu', 'submission': {'source': 'def f(x): return x.relu()',
        'entrypoint': 'f', 'signatures': [{'args': [{'shape': [4], 'dtype': 'float32'}]}]}}
    class RecordingAgent(Agent):
        def complete(self, stage, context, record):
            record('request', {'stage': stage, 'context': context})
            result = super().complete(stage, context, record)
            record('response', result)
            return result
    model = RecordingAgent([{'action': 'cases', 'signatures': [], 'summary': 'representative'},
                   candidate('bad'), candidate('good')])
    backend = Backend()
    config = {'target': 'registered-a3', 'physical_device': 0,
              'validation_wrapper': str(tmp_path / 'validation.sh'), 'max_cases': 1}
    def factory(directory, task, snapshot):
        return OptimizerTaskExecutor(directory, task, snapshot, config, agent=model, backend=backend)
    clock = (lambda: next(times)) if times is not None else __import__('time').monotonic
    return SubjectExecutor(optimizer_factory=factory, clock=clock), task, backend, model


def test_optimizer_reuses_real_job_baseline_and_publishes_no_learning(tmp_path):
    execution, task, backend, actor = optimizer(tmp_path)
    snapshot = optimizer_snapshot([{'claim': 'preserve tails'}], files={'code.c': b'reference'})
    shared = ContentAddressedArtifacts(tmp_path / 'shared')
    options = dict(directory=tmp_path / 'task', task=task, memory_snapshot=snapshot,
                   remaining_seconds=60, artifacts=shared)
    failed = execution.optimizer(**options, attempt_id='attempt-1')
    assert not failed.host_verified
    assert execution.optimizer(**options, attempt_id='attempt-1').as_dict() == failed.as_dict()
    passed = execution.optimizer(**options, attempt_id='attempt-2', feedback={'repair': 'tails'})
    assert passed.host_verified and passed.observations['speedup'] == 2
    assert passed.read_artifact(passed.source_sha256) == b'good'
    events = json.loads(passed.read_artifact(passed.artifacts['log:optimizer-events']))
    assert any(event['kind'] == 'model-request' for event in events)
    assert backend.prepares == 1 and [s for s, _ in backend.evaluations] == ['bad', 'good']
    assert actor.contexts[-1][1]['phase2']['feedback'] == {'repair': 'tails'}
    assert not (tmp_path / 'state.json').exists()


def test_optimizer_memory_and_elapsed_allocation_checked_before_attempt(tmp_path):
    execution, task, backend, _ = optimizer(tmp_path, times=iter([0., 1., 61.]))
    with pytest.raises(TimeoutError, match='elapsed allocation'):
        execution.optimizer(tmp_path / 'task', task, optimizer_snapshot(),
                            attempt_id='once', remaining_seconds=60)
    assert backend.prepares == 1 and backend.evaluations == []
    broken = optimizer_snapshot()
    broken['memory_entries'] = ['changed']
    with pytest.raises(ValueError, match='snapshot digest'):
        execution.optimizer(tmp_path / 'other', task, broken, attempt_id='once', remaining_seconds=60)
