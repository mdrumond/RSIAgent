"""Focused repair replay and one KDB transfer, using normal admission publication.

Callbacks return host-owned outcomes; a started callback is never redispatched.
Timing evidence is optional, but performance claims cannot pass without it.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

from optimizer.jobs import atomic, canonical
from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from .arm_evidence import arm_evidence, interrupted_outcome
from .memory_verifier import MemoryVerifier, _hash, _seconds, _inconclusive
from .memory_status import evaluation_assessment
from .subject_execution import SubjectOutcome
from .time_budget import AttemptTimeBudget
from .environment_admission import native_reservation


def paired_measurements(report, runtime, source_request, before, after):
    """Accept only matched, repeated device timings with explicit correctness."""
    if not isinstance(report, dict) or report.get('runtime') != runtime:
        raise ValueError('measurement runtime differs from focused replay')
    if report.get('contract_sha256') != _hash(canonical(source_request)):
        raise ValueError('measurement contract differs from focused replay')
    if report.get('sources') != {'control': before, 'treatment': after}:
        raise ValueError('measurement sources differ from focused replay')
    pairs = report.get('pairs')
    if not isinstance(pairs, list) or len(pairs) < 3 or report.get('host_verified') is not True:
        raise ValueError('performance needs at least three correctness-verified pairs')
    for pair in pairs:
        if (not isinstance(pair, dict) or set(pair) != {'control_us', 'treatment_us'}
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in pair.values())):
            raise ValueError('paired device durations must be finite positive microseconds')
    return report


def workload_identity(spec):
    """Compare actual host semantics with input seeds independent of teaching metadata."""
    from benchmarks.a5kernels.phase1_project import materialize_inputs, evaluate_reference
    from benchmarks.a5kernels.phase1_project_executor import _tensor_payload, _expected_payload
    from .memory_usefulness import actor_project
    normalized = actor_project(spec)
    value = normalized.as_dict()
    semantics = {key: value[key] for key in ('dtype', 'block_count', 'padding_multiple', 'shape', 'graph')}
    semantics['padded_tail'] = 'padded-tail' in value['capabilities']
    # Ordinary materialization hashes the entire spec, including hypothesis and
    # failure tags. Those incidental seed changes are not a new transfer task.
    inputs = materialize_inputs(SimpleNamespace(as_dict=lambda: semantics,
        graph=normalized.graph, capabilities=normalized.capabilities,
        dtype=normalized.dtype, padding_multiple=normalized.padding_multiple))
    return _hash(canonical(dict(semantics=semantics,
        inputs=[_tensor_payload(item) for item in inputs],
        oracle=_expected_payload(evaluate_reference(normalized, inputs)))))


class FocusedMemoryVerifier(MemoryVerifier):
    mode = 'focused'
    protocol = 'focused-adaptation-v1'

    def __init__(self, complete, select_trial, replay_source, execute_transfer, read_artifact, *,
                 model_identity, runtime_identity, max_seconds=1800, arm_seconds=600,
                 measure_pair=None, witness_cache=None, replay_binding=None, clock=time.monotonic, wall_clock=time.time):
        super().__init__(complete, select_trial, execute_transfer, model_identity=model_identity,
            runtime_identity=runtime_identity, max_seconds=max_seconds, arm_seconds=arm_seconds,
            clock=clock, wall_clock=wall_clock)
        self.replay_source, self.read_artifact, self.measure_pair = replay_source, read_artifact, measure_pair
        self.witness_cache = Path(witness_cache) if witness_cache is not None else None
        self.replay_binding = json.loads(canonical(replay_binding or {}))

    def evaluate(self, proposal, baseline_bytes, treatment_bytes, directory, *, remaining_seconds,
                 source_request, task=None, sequence=0, check_remaining=None, **unused):
        parent = self.clock() + _seconds(remaining_seconds)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        identity = json.loads(canonical(dict(mode=self.mode, protocol=self.protocol, replay_binding=self.replay_binding, proposal=proposal,
            baseline_sha256=_hash(baseline_bytes), treatment_sha256=_hash(treatment_bytes),
            source_request=source_request, model=self.model, runtime=self.runtime,
            max_seconds=self.seconds, arm_seconds=self.arm_seconds, sequence=sequence,
            measurements=self.measure_pair is not None)))
        path = directory / 'evaluation.json'
        with self._lock(directory):
            state = json.loads(path.read_bytes()) if path.exists() else dict(identity=identity,
                allocation=min(self.seconds, remaining_seconds), task=None, stages={}, trace=[])
            if state['identity'] != identity or task is not None and state['task'] not in (None, task):
                raise ValueError('focused admission identity changed')
            if state.get('decision') is not None:
                self.last_trace = state['trace']
                return state['decision']
            atomic(path, canonical(state))
            store = ContentAddressedArtifacts(directory / 'artifacts')
            refs = {'proposal': store.put_json(proposal), 'source-request': store.put_json(source_request)}
            budget = AttemptTimeBudget(directory / 'elapsed.json', state['allocation'],
                clock=self.clock, wall_clock=self.wall_clock)

            def save():
                self.last_trace = state['trace']
                atomic(path, canonical(state))

            def remaining():
                cap = min(parent, budget.deadline) - self.clock()
                if check_remaining is not None:
                    available = check_remaining()
                    if available is not None:
                        cap = min(cap, available)
                if cap <= 0:
                    raise TimeoutError('focused admission elapsed allocation exhausted')
                return cap

            def execute_stage(name, callback):
                if name in state['stages']:
                    return state['stages'][name]
                started, epoch = self.clock(), self.wall_clock()
                stage = dict(status='started', outcome=None, started_epoch=epoch)
                state['stages'][name] = stage
                save()  # A lost callback may still own remote work: never dispatch twice.
                stage_dir = directory / name
                stage_dir.mkdir(exist_ok=True)
                allowance = min(self.arm_seconds, remaining())
                try:
                    outcome = callback(stage_dir, allowance)
                    if not isinstance(outcome, SubjectOutcome) or type(outcome.host_verified) is not bool:
                        raise TypeError('focused callbacks must return a host-owned SubjectOutcome')
                    for sha in outcome.artifacts.values():
                        store.put_bytes(outcome.read_artifact(sha))
                    retained, _, _, _, _, _ = arm_evidence(stage_dir, store)
                    outcome = SubjectOutcome(outcome.host_verified, outcome.source_sha256,
                        {**outcome.artifacts, **retained}, outcome.observations, store.root)
                    expected_source = {'control': proposal['before']['sha256'],
                                       'treatment': proposal['after']['sha256']}.get(name)
                    if expected_source is not None and outcome.source_sha256 != expected_source:
                        raise ValueError('replay outcome source differs from the frozen source')
                    classification = outcome.observations.get('classification')
                    valid = (self.clock() <= started + allowance
                        and classification in {'compile-failure', 'host-mismatch', 'host-verified'}
                        and outcome.host_verified == (classification == 'host-verified'))
                    stage.update(status='completed' if valid else 'invalid', outcome=outcome.as_dict())
                except Exception as error:
                    outcome = interrupted_outcome(stage_dir, error, store, started_epoch=epoch,
                        elapsed_seconds=max(0, self.clock()-started))
                    stage.update(status='interrupted', outcome=outcome.as_dict(), error_type=type(error).__name__)
                stage.update(elapsed_seconds=max(0, self.clock()-started), ended_epoch=self.wall_clock())
                state['trace'].append(dict(kind='focused-stage', stage=name, **stage))
                save()
                remaining()
                return stage

            def execute(name, callback):
                if name not in {'control', 'treatment'} or self.witness_cache is None:
                    return execute_stage(name, callback)
                source = proposal['before' if name == 'control' else 'after']['sha256']
                witness_identity = dict(protocol=self.protocol, source=source,
                    request=source_request, runtime=self.runtime, binding=self.replay_binding)
                cache = self.witness_cache / _hash(canonical(witness_identity))
                cache.mkdir(parents=True, exist_ok=True)
                with native_reservation(cache, remaining):
                    witness = cache / 'witness.json'
                    if name not in state['stages'] and witness.exists():
                        retained = json.loads(witness.read_bytes())
                        if retained['identity'] != witness_identity:
                            raise ValueError('replay witness identity changed')
                        row = retained['stage']
                        cached_store = ContentAddressedArtifacts(cache / 'artifacts')
                        for digest in row['outcome']['artifacts'].values():
                            if store.put_bytes(cached_store.read_bytes(digest)) != digest:
                                raise ValueError('replay witness bytes changed')
                        stage = dict(row, reused=True, elapsed_seconds=0,
                                     started_epoch=self.wall_clock(), ended_epoch=self.wall_clock())
                        state['stages'][name] = stage
                        state['trace'].append(dict(kind='replay-witness-reused', stage=name))
                        save()
                        return stage
                    stage = execute_stage(name, callback)
                    if stage['status'] == 'completed':
                        cached_store = ContentAddressedArtifacts(cache / 'artifacts')
                        for digest in stage['outcome']['artifacts'].values():
                            cached_store.put_bytes(store.read_bytes(digest))
                        atomic(witness, canonical(dict(identity=witness_identity, stage=stage)))
                    return stage

            try:
                remaining()
                before, after = proposal.get('before'), proposal.get('after')
                if before is None:
                    decision = _inconclusive('No retained before-source: keep as a source-reviewed reference, not a validated repair lesson.')
                elif not after or after['sha256'] != source_request['source_sha256']:
                    raise ValueError('after-source differs from the source-reviewed project')
                elif before['sha256'] == after['sha256']:
                    decision = _inconclusive('Before and after source are identical; a repair has not been demonstrated.')
                else:
                    for role, snippet in (('before-source', before), ('after-source', after)):
                        data = self.read_artifact(snippet['sha256'])
                        if store.put_bytes(data) != snippet['sha256']:
                            raise ValueError('retained source identity changed')
                        refs[role] = snippet['sha256']
                    if state['task'] is None:
                        selected = task if task is not None else self.select_trial(proposal, remaining_seconds=remaining())
                        if not isinstance(selected, dict) or not selected:
                            raise ValueError('freeze a nonempty related transfer task')
                        from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
                        contracts = source_request['artifacts']
                        contract_sha = contracts.get('contract', contracts.get('evidence:project-contract'))
                        contract = json.loads(self.read_artifact(contract_sha))
                        original = ProjectSpec.from_mapping(contract.get('spec', contract))
                        related = ProjectSpec.from_mapping(selected['spec'])
                        if workload_identity(original) == workload_identity(related):
                            raise ValueError('transfer must change the source workload contract')
                        # Original broken source is task scaffolding, not a delivered lesson.
                        # The repaired source and advice remain available only through KDB.
                        selected = {key:value for key,value in selected.items() if key != 'recovery'}
                        selected['recovery'] = dict(reason=(
                            'Adapt this retained broken implementation to the frozen related workload; '
                            'use the original and related host contracts for correctness.'),
                            artifacts={'before-source': before['sha256']})
                        state['task'] = json.loads(canonical(selected))
                        save()
                    callbacks = [
                        ('control', lambda d, s: self.replay_source(source_request, before['sha256'], d, remaining_seconds=s)),
                        ('treatment', lambda d, s: self.replay_source(source_request, after['sha256'], d, remaining_seconds=s)),
                        ('transfer', lambda d, s: self.execute_arm(state['task'], treatment_bytes, d, remaining_seconds=s))]
                    for name, callback in callbacks:
                        if execute(name, callback)['status'] != 'completed':
                            break
                        if name == 'treatment' and not state['stages'][name]['outcome']['host_verified']:
                            break
                    stages = state['stages']
                    required_stages = {'control', 'treatment', 'transfer'}
                    if not required_stages <= stages.keys() or any(stages[name]['status'] != 'completed' for name in required_stages):
                        decision = _inconclusive('Focused replay or transfer is incomplete, invalid, or failed; inspect retained stage evidence.')
                    else:
                        arms = {name: SubjectOutcome(**stages[name]['outcome'], artifact_root=store.root)
                                for name in ('control', 'treatment', 'transfer')}
                        if self.measure_pair is not None:
                            measured = execute('measurement', lambda d, s: self._measure(
                                source_request, before['sha256'], after['sha256'], d, s))
                            if measured['status'] == 'completed':
                                arms['measurement'] = SubjectOutcome(**measured['outcome'], artifact_root=store.root)
                        try:
                            decision = self.judge.review(proposal, arms, lambda sha, **kw: store.read_bytes(sha),
                                max_seconds=remaining(), check_remaining=remaining, focused=True)
                        finally:
                            state['trace'].extend(self.judge.last_trace)
                remaining()
            except Exception as error:
                decision = dict(_inconclusive('Focused evidence collection did not complete; observe retained work before retrying.'),
                    error_type=type(error).__name__, error=str(error))
            finally:
                budget.close()
            decision.update(mode=self.mode, protocol=self.protocol, evaluation_id=_hash(canonical(identity)), trial_task=state['task'],
                evidence_stages={name: dict(status=row['status'], host_verified=(row.get('outcome') or {}).get('host_verified'),
                                           elapsed_seconds=row.get('elapsed_seconds'))
                                 for name, row in state['stages'].items()},
                validity={name: row['status'] == 'completed' for name, row in state['stages'].items()},
                arms={name: row.get('outcome') for name, row in state['stages'].items()}, artifacts=refs)
            for name, row in state['stages'].items():
                decision['artifacts'].update({name+':'+role: sha for role, sha in (row.get('outcome') or {}).get('artifacts', {}).items()})
            decision['assessment'] = evaluation_assessment(decision)
            state['decision'] = decision
            save()
            return decision

    def _measure(self, request, before, after, directory, seconds):
        report = paired_measurements(self.measure_pair(request, before, after, directory,
            remaining_seconds=seconds), self.runtime, request, before, after)
        store = ContentAddressedArtifacts(directory / 'artifacts')
        sha = store.put_json(report)
        return SubjectOutcome(True, after, {'result:measurements': sha},
            {'classification': 'host-verified'}, store.root)
