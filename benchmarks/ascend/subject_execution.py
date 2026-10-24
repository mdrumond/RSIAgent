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
    for digest in sorted(digests):
        artifacts.read_bytes(digest)
        result['retained:' + digest] = digest
    return result


class SubjectExecutor:
    def __init__(self, graph_actor=None, optimizer_factory=None, *, clock=time.monotonic):
        self.graph_actor, self.optimizer_factory, self.clock = graph_actor, optimizer_factory, clock

    def graph(self, request, *, target=None, artifacts=None, runtime=None, policy=None):
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
        destination = artifacts or store
        def portable(value):
            for digest in value['artifacts'].values():
                destination.put_bytes(store.read_bytes(digest))
            return SubjectOutcome(**value, artifact_root=destination.root)
        checkpoint = root / 'subject-execution.json'
        if checkpoint.exists():
            recorded = json.loads(checkpoint.read_bytes())
            if recorded['identity'] != identity:
                raise ValueError('subject execution request changed; use its original frozen inputs')
            value = recorded['outcome']
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
        execution = self.optimizer_factory(Path(directory), task, memory_snapshot)
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
