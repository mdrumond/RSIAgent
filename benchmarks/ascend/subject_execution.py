"""Subject execution over existing host-verified graph and optimizer contracts.

Adapters return factual evidence only. They neither admit reusable memory nor
require a reviewed Phase 1 handoff. The caller owns curriculum and verification.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.phase1_wave_runtime import SemanticProjectFailure
from benchmarks.a5kernels.phase1_project_bz import ProjectInfrastructureError
from benchmarks.a5kernels.phase1_provider import ProviderRejectionError
from coding.pi.rpc import PiError
from optimizer.jobs import atomic, canonical, sha
from optimizer.memory import verify_snapshot


def optimizer_snapshot(memory_entries=(), *, files=None, source_identity=None):
    """Freeze accepted subject memory in the existing optimizer wire format."""
    body = dict(schema='optimizer-memory-snapshot-v1', phase='subject-curriculum',
                source_identity=dict(source_identity or {}), terminal=None,
                memory_entries=list(memory_entries), files={name: {
                    'sha256': sha(data), 'base64': base64.b64encode(data).decode(),
                } for name, data in (files or {}).items()})
    snapshot = {**body, 'snapshot_id': sha(canonical(body))}
    verify_snapshot(snapshot)
    return snapshot


@dataclass(frozen=True)
class SubjectOutcome:
    host_verified: bool
    source_sha256: str | None
    artifacts: dict[str, str]
    observations: dict
    artifact_root: Path

    def as_dict(self):
        """Portable references; artifact_root is a local reader dependency."""
        return dict(host_verified=self.host_verified, source_sha256=self.source_sha256,
                    artifacts=dict(self.artifacts), observations=self.observations)

    def read_artifact(self, digest):
        if digest not in self.artifacts.values():
            raise ValueError('artifact is not retained by this subject outcome')
        return ContentAddressedArtifacts(self.artifact_root).read_bytes(digest)


def _logs(root, artifacts, admitted):
    """Index project admissions and exact provider evidence, not older memory."""
    result = {}
    digests = set(admitted)
    paths = [root / 'evidence.jsonl', *sorted((root / 'sft-traces').glob('*.jsonl'))]
    for path in paths:
        if path.is_file():
            data = path.read_bytes()
            result['log:' + path.relative_to(root).as_posix()] = artifacts.put_bytes(data)
            if path.name == 'evidence.jsonl':
                for line in data.splitlines():
                    digest = json.loads(line)['payload'].get('sha256')
                    if digest is not None:
                        digests.add(digest)
            else:
                # Trace-chain and candidate IDs are not necessarily CAS objects.
                # Only validation's explicit evidence reference admits bytes.
                for line in data.splitlines():
                    row = json.loads(line)
                    if row['kind'] == 'validation':
                        digests.add(row['payload']['evidence_sha256'])
    for digest in sorted(digests):
        artifacts.read_bytes(digest)
        result['retained:' + digest] = digest
    return result


class SubjectExecutor:
    def __init__(self, graph_actor=None, optimizer_factory=None, *, clock=time.monotonic):
        self.graph_actor, self.optimizer_factory, self.clock = graph_actor, optimizer_factory, clock

    def graph(self, request, *, target=None, artifacts=None, runtime=None, policy=None, allow_device_migration=False, binding_root=None):
        """Execute AdaptiveProjectRequest without Actor-owned distillation.

        When target is supplied, artifacts contains its retained qualification;
        runtime must match the composed executor, never just caller assertions.
        artifacts is also the shared destination for portable verifier reads.
        """
        if self.graph_actor is None:
            raise ValueError('graph actor is not configured')
        composed = getattr(getattr(self.graph_actor, 'executor', None), 'runtime_identity', None)
        if policy is not None:
            policy.admit(request.spec)
        if target is not None:
            if target.spec != request.spec:
                raise ValueError('request differs from qualified subject target')
            if composed is None or (runtime is not None and dict(runtime) != dict(composed)):
                raise ValueError('target requires observed composed runtime identity')
            if artifacts is None:
                raise ValueError('qualified target requires its retained artifact store')
            target.verify(artifacts, composed, policy)
        started, epoch = self.clock(), time.time()
        request = replace(request, publish_learning=False)
        root = Path(request.branch_root)
        store = ContentAddressedArtifacts(root / 'artifacts')
        if target is not None:
            reference = artifacts.read_json(target.reference_sha256)
            qualified = [target.reference_sha256, reference['source_sha256'],
                         *reference['artifacts'].values(),
                         *(row['evidence_sha256'] for row in reference['timings'])]
            for digest in qualified:
                store.put_bytes(artifacts.read_bytes(digest))
        identity = dict(spec=request.spec.as_dict(), ordinal=request.authored_ordinal,
            memory_sha256=hashlib.sha256(request.pre_wave_memory).hexdigest(),
            target_id=target.target_id if target else None,
            runtime=dict(composed) if composed is not None else None)
        if request.actor_memory_delivery != 'indexed':
            identity['actor_memory_delivery'] = request.actor_memory_delivery
        destination = artifacts or store
        def portable(value):
            for digest in value['artifacts'].values():
                destination.put_bytes(store.read_bytes(digest))
            return SubjectOutcome(**value, artifact_root=destination.root)
        checkpoint = root / 'subject-execution.json'
        if checkpoint.exists():
            recorded = json.loads(checkpoint.read_bytes())
            previous = recorded['identity']
            same = previous == identity
            if not same and allow_device_migration and previous.get('runtime') and identity.get('runtime'):
                # The campaign has already authenticated unchanged frozen settings.
                # Only reuse a completed outcome; never alter its original identity.
                old_runtime = {key: value for key, value in previous['runtime'].items() if key != 'device'}
                new_runtime = {key: value for key, value in identity['runtime'].items() if key != 'device'}
                same = old_runtime == new_runtime and {key: value for key, value in previous.items() if key != 'runtime'} == {key: value for key, value in identity.items() if key != 'runtime'}
            if not same:
                raise ValueError('subject execution request changed; use its original frozen inputs')
            # Cached results must preserve the Actor's host-input/oracle gate.
            # Preparing is host-only: no compilation or device operation occurs.
            executor = getattr(self.graph_actor, 'executor', None)
            if executor is not None:
                prepared = executor.prepare(request.spec)
                inputs, oracle = prepared.inputs, prepared.expected
            else:
                from benchmarks.a5kernels.phase1_project import materialize_inputs, evaluate_reference
                inputs = materialize_inputs(request.spec)
                oracle = evaluate_reference(request.spec, inputs)
            if request.inputs != inputs or request.oracle != oracle:
                raise ValueError('request inputs or oracle differ from rematerialized project')
            value = recorded['outcome']
            if store.read_json(value['artifacts']['result:execution']) != value['observations']:
                raise ValueError('completed subject execution evidence changed')
            return portable(value)
        try:
            result = self.graph_actor(request)
            payload = json.loads(result.evidence_payload)
            observations = dict(classification=result.evidence.classification.value,
                summary=result.summary, host_facts=[asdict(f) for f in result.memory.host_facts],
                actions=list(result.memory.actions), evidence=payload)
            admitted = result.memory.admitted_artifact_sha256
            source, verified = result.memory.source_revision, result.evidence.verified
        except SemanticProjectFailure as error:
            payload = error.evidence
            observations = dict(classification='actor-unverified', summary=error.summary,
                                evidence=payload, host_facts=payload.get('host_facts', []))
            admitted = payload.get('admitted_artifact_sha256', [])
            source, verified = payload.get('source_sha256'), False
        except (ProjectInfrastructureError, TimeoutError, PiError, ProviderRejectionError) as error:
            from .arm_evidence import interrupted_outcome
            outcome = interrupted_outcome(root.parent, error, store, started_epoch=epoch,
                elapsed_seconds=max(0, self.clock()-started), project_root=root,
                binding_root=binding_root, project_id=request.spec.project_id if binding_root is not None else None)
            root.mkdir(parents=True, exist_ok=True)
            atomic(checkpoint, canonical(dict(identity=identity, outcome=outcome.as_dict())))
            return portable(outcome.as_dict())
        refs = {'result:execution': store.put_json(observations)}
        if source is not None:
            store.read_bytes(source)
            refs['source'] = source
        refs.update(_logs(root, store, [*admitted, *(qualified if target else [])]))
        outcome = SubjectOutcome(verified, source, refs, observations, store.root)
        root.mkdir(parents=True, exist_ok=True)
        atomic(checkpoint, canonical(dict(identity=identity, outcome=outcome.as_dict())))
        return portable(outcome.as_dict())

    def optimizer(self, directory, task, memory_snapshot, *, attempt_id,
                  remaining_seconds, feedback=None, artifacts=None):
        """Reuse a durable OptimizerTaskExecutor and its frozen reference suite."""
        if self.optimizer_factory is None:
            raise ValueError('optimizer execution factory is not configured')
        if (isinstance(remaining_seconds, bool) or not isinstance(remaining_seconds, (int, float))
                or not math.isfinite(remaining_seconds) or remaining_seconds <= 0):
            raise ValueError('subject remaining seconds must be finite and positive')
        verify_snapshot(memory_snapshot)
        started = self.clock()
        directory = Path(directory)
        initial_path = directory / 'subject-optimizer-initial-memory.json'
        initial = json.loads(initial_path.read_bytes()) if initial_path.exists() else memory_snapshot
        verify_snapshot(initial)
        # JobStore creation is immutable, while each attempt independently pins
        # the currently accepted learning. Reconstruct with the creation snapshot
        # so task/config/reference compatibility checks still run in the factory.
        execution = self.optimizer_factory(directory, task, initial)
        if not initial_path.exists():
            atomic(initial_path, canonical(initial))
        def bind():
            remaining = remaining_seconds - (self.clock() - started)
            if remaining <= 0:
                raise TimeoutError('subject optimizer elapsed allocation exhausted')
            execution.set_remaining_seconds(remaining)
        bind()
        baseline = execution.prepare()
        bind()
        result = execution.attempt(attempt_id, memory_snapshot=memory_snapshot, feedback=feedback)
        store = artifacts or ContentAddressedArtifacts(Path(directory) / 'artifacts')
        refs = {}
        for index, ref in enumerate(execution.store.artifacts(execution.job_id)):
            content = execution.store.read_artifact(execution.job_id, ref['sha256'])
            refs[f"optimizer:{index}:{ref['name']}"] = store.put_bytes(content)
        source = result.get('source_sha256')
        if source is not None:
            candidate = json.loads(execution.store.read_artifact(execution.job_id,
                                                               result['candidate']['sha256']))
            if not isinstance(candidate, str) or sha(candidate.encode()) != source:
                raise ValueError('optimizer candidate source does not match host evidence')
            refs['source'] = store.put_bytes(candidate.encode())
        refs['log:optimizer-events'] = store.put_json(execution.store.events(execution.job_id))
        observations = dict(classification=result['classification'], summary=result['summary'],
                            evidence=result, baseline=baseline, speedup=result.get('speedup'))
        refs['result:execution'] = store.put_json(observations)
        return SubjectOutcome(result['verified'], source, refs, observations, store.root)
