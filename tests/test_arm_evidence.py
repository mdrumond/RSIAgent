"""Partial evidence is portable, factual, and never a completed paired arm."""
import json

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.phase1_project_bz import ProjectInfrastructureError
from benchmarks.ascend.arm_evidence import interrupted_outcome, interruption_reason, timing_summary
from benchmarks.ascend.memory_verifier import MemoryVerifier
from test_memory_verifier_agent import Harness, evaluate


def evidence(tmp_path):
    store = ContentAddressedArtifacts(tmp_path/'project/artifacts')
    source = store.put_bytes(b'kernel with repair in progress')
    observation = store.put_json({'host': {'event': 'candidate-written', 'source_sha256': source}})
    provider = store.put_json(dict(reservation_id='one', seconds=3, started_epoch=101, ended_epoch=104))
    rows = [dict(payload=dict(role=role, sha256=digest)) for role, digest in
        [('host-observation', observation), ('pi-provider', provider), ('pi-provider', provider)]]
    (tmp_path/'project/evidence.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    binding = tmp_path/'runtime/binding/transport-state'
    operation = binding/'candidate/compile/operation.json'
    operation.parent.mkdir(parents=True)
    record = dict(handle='remote:fixture:job:one', state='cancelling', expired=True,
        terminal_result=dict(returncode=1, stderr='candidate.cpp:4: error: unknown type'))
    operation.write_text(json.dumps(record))
    (binding/'timing.jsonl').write_text(json.dumps(dict(stage='compile', started_epoch=103, ended_epoch=108))+'\n')
    return source


def test_partial_deadline_retains_source_diagnostics_handles_and_timing(tmp_path):
    arm = tmp_path/'arm'
    source = evidence(arm)
    target = ContentAddressedArtifacts(tmp_path/'portable')
    cause = TimeoutError('elapsed deadline')
    error = ProjectInfrastructureError('compile boundary interrupted')
    error.__cause__ = cause
    outcome = interrupted_outcome(arm, error, target, started_epoch=100, elapsed_seconds=10)
    assert not outcome.host_verified and outcome.source_sha256 == source
    assert outcome.read_artifact(source) == b'kernel with repair in progress'
    obs = outcome.observations
    assert obs['interruption_reason'] == 'deadline-exhausted'
    assert obs['remote_operations'] == [dict(handle='remote:fixture:job:one', stage='compile',
        state='cancelling', expired=True)]
    ref = next(v for k,v in outcome.artifacts.items() if k.startswith('operation:'))
    assert json.loads(outcome.read_artifact(ref))['terminal_result']['stderr'].endswith('unknown type')
    assert obs['timing'] == dict(wall_seconds=10, by_stage_seconds={'provider':3,'compile':5},
        overlap_seconds=1, attributed_seconds=7, unattributed_seconds=3, provider_recorded_seconds=3)


def test_last_source_before_missing_observation_survives_actor_exception(tmp_path):
    arm = tmp_path/'arm'
    source = evidence(arm)
    store = ContentAddressedArtifacts(arm/'project/artifacts')
    latest = store.put_bytes(b'newest source not yet observed')
    error = TimeoutError('deadline')
    error.partial_evidence = dict(source_sha256=latest, actions=['write', 'compile'],
        admitted_artifact_sha256=[source, latest], latest_evidence={'classification':'host-mismatch'})
    outcome = interrupted_outcome(arm, error, ContentAddressedArtifacts(tmp_path/'destination'),
        started_epoch=100, elapsed_seconds=10)
    assert outcome.source_sha256 == latest
    assert outcome.read_artifact(latest) == b'newest source not yet observed'
    assert outcome.observations['evidence']['latest_evidence']['classification'] == 'host-mismatch'


def test_structural_operator_stop_is_preserved_through_wrapping():
    cause = TimeoutError('same boundary message')
    cause.interruption_reason = 'operator-stop'
    error = ProjectInfrastructureError('wrapped boundary')
    error.__cause__ = cause
    assert interruption_reason(error) == 'operator-stop'
    assert interruption_reason(TimeoutError('operator-stop words are not authoritative')) == 'deadline-exhausted'
    assert interruption_reason(ValueError('implementation bug')) == 'harness-error'


def test_legacy_provider_seconds_are_not_invented_as_timestamped_spans():
    result = timing_summary([dict(stage='provider', seconds=4)], 100, 10, 4)
    assert result['provider_recorded_seconds'] == 4
    assert result['unattributed_seconds'] == 10 and result['attributed_seconds'] == 0


@pytest.mark.parametrize('stage', ['compile', 'execute'])
def test_interrupted_arm_holds_peer_and_caches_portable_partial_result(tmp_path, stage):
    h = Harness()
    calls = []
    def interrupted(task, memory, directory, **kwargs):
        calls.append(stage)
        evidence(directory)
        raise TimeoutError('allocation exhausted during '+stage)
    h.verifier.execute_arm = interrupted
    first = evaluate(h, tmp_path)
    assert first['validity'] == {'control': False} and first['decision'] == 'inconclusive'
    assert first['arms']['control']['observations']['interruption_reason'] == 'deadline-exhausted'
    assert first['arms']['control']['source_sha256']
    assert not (tmp_path/'treatment').exists() and not h.requests
    assert evaluate(h, tmp_path) == first and calls == [stage]
    store = ContentAddressedArtifacts(tmp_path/'artifacts')
    assert all(store.read_bytes(h) for h in first['artifacts'].values())


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('kind', ['pi-provider', 'provider-rejection', 'cancelled'])
def test_actual_pi_and_provider_exception_types_survive_cause_wrapping(kind, wrapped):
    from coding.pi.rpc import PiProviderError, PiCancelled
    from benchmarks.a5kernels.phase1_provider import ProviderRejectionError
    error = {'pi-provider':lambda:PiProviderError('provider rejected'),
        'provider-rejection':lambda:ProviderRejectionError(code='rate_limit', status_code=429, request_id='fixture'),
        'cancelled':lambda:PiCancelled('generic aborted provider turn')}[kind]()
    expected = 'cancelled' if kind=='cancelled' else 'provider-failure'
    if wrapped:
        wrapper = ProjectInfrastructureError('wrapped boundary')
        wrapper.__cause__ = error
        error = wrapper
    assert interruption_reason(error) == expected


def test_actual_environment_report_contributes_timestamped_span(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from benchmarks.ascend import wave_environment
    arm = tmp_path/'arm'
    binding = arm/'runtime/binding'
    now = [0.]
    monkeypatch.setattr(wave_environment.time, 'monotonic', lambda:now[0])
    monkeypatch.setattr(wave_environment.time, 'time', lambda:100+now[0])
    class Transport:
        runtime_identity = {'physical_device':0}
        def compile(self, plan, root):
            now[0] += 2
            return SimpleNamespace(plan=plan, library_sha256='b'*64, stdout='', stderr='')
        def execute_receipt(self, compilation):
            now[0] += 3
            return SimpleNamespace(exit_code=0, job_handle='remote:fixture:job:probe',
                output=tuple(sum(xs) for xs in zip(*(t.values for t in compilation.plan.inputs))))
    check = wave_environment.NativeWaveEnvironmentCheck(SimpleNamespace(root=binding,
        target='fixture', arch='fixture', execution_profile='gz-a3'), [Transport()])
    check(1, lambda:10-now[0])
    outcome = interrupted_outcome(arm, TimeoutError('deadline'),
        ContentAddressedArtifacts(tmp_path/'destination'), started_epoch=100, elapsed_seconds=10)
    timing = outcome.observations['timing']
    assert timing['by_stage_seconds']['environment'] == 5
    assert timing['attributed_seconds'] == 5 and timing['unattributed_seconds'] == 5
    report_ref = next(v for k,v in outcome.artifacts.items() if k.startswith('environment:'))
    report = json.loads(outcome.read_artifact(report_ref))
    assert report['status']=='passed' and report['started_epoch']==100 and report['ended_epoch']==105


@pytest.mark.parametrize('kind', ['provider', 'cancelled', 'operator-stop'])
def test_actual_pi_error_arm_is_invalid_and_never_starts_peer(tmp_path, kind):
    from coding.pi.rpc import PiProviderError, PiCancelled
    h = Harness()
    calls = []
    def execute(task, memory, directory, **kwargs):
        calls.append(directory.name)
        error = PiProviderError('provider failed') if kind=='provider' else PiCancelled('aborted')
        if kind=='operator-stop':error.interruption_reason='operator-stop'
        raise error
    h.verifier.execute_arm=execute
    result = evaluate(h, tmp_path)
    assert result['decision']=='inconclusive' and result['validity']=={'control':False}
    expected = 'provider-failure' if kind=='provider' else kind
    assert result['arms']['control']['observations']['interruption_reason']==expected
    assert evaluate(h, tmp_path)==result and calls==['control']
    assert not h.requests and not (tmp_path/'treatment').exists()
