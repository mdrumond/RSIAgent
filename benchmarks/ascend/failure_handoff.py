"""Authenticated failure context, kept separate from reusable verified memory."""
from __future__ import annotations

import json

from .feedback import FeedbackStore


def phase1_contract(spec, inputs=None, expected=None):
    """Identify computation and exact host case independently of its exercise name."""
    from benchmarks.a5kernels.evidence import canonical_digest
    from benchmarks.a5kernels.phase1_project import materialize_inputs, evaluate_reference
    from benchmarks.a5kernels.phase1_project_executor import _tensor_payload, _expected_payload
    inputs = materialize_inputs(spec) if inputs is None else inputs
    expected = evaluate_reference(spec, inputs) if expected is None else expected
    return dict(graph=spec.graph.as_dict(), dtype=spec.dtype.value,
                inputs_sha256=canonical_digest({'tensors': [_tensor_payload(x) for x in inputs]}),
                expected_sha256=canonical_digest(_expected_payload(expected)),
                policy='phase1-classify-execution-v1')


def original_retry_inputs(store, feedback_id, spec):
    """Recover the selected observation's host case without campaign lineage."""
    from benchmarks.a5kernels.phase1_project import HostTensor, materialize_inputs, evaluate_reference
    from benchmarks.a5kernels.phase1_project_executor import _expected_payload
    from benchmarks.a5kernels.evidence import canonical_bytes
    cases = store.failure_cases(feedback_id)
    retained = [case for case in cases if 'inputs' in case.get('artifacts', {})]
    if not retained:
        return None  # Historical observations without a retained host case.
    layout = [(item.shape, item.physical_shape) for item in materialize_inputs(spec)]
    for case in retained:
        refs = case['artifacts']
        payload = json.loads(store.failure_artifact_bytes(feedback_id, refs['inputs']['sha256']))
        inputs = tuple(HostTensor(tuple(row['shape']), tuple(row['physical_shape']), tuple(row['storage']))
                       for row in payload['tensors'])
        if [(item.shape, item.physical_shape) for item in inputs] != layout:
            continue
        expected = evaluate_reference(spec, inputs)
        contract = phase1_contract(spec, inputs, expected)
        if (case.get('project_contract') == contract
                and refs.get('expected', {}).get('sha256') == contract['expected_sha256']
                and store.failure_artifact_bytes(feedback_id, refs['expected']['sha256']) ==
                    canonical_bytes(_expected_payload(expected))):
            return inputs
    raise ValueError('Original retry has no authenticated compatible retained host case')


class FailureContext:
    """Resolve only cases linked by the host to this remedial project."""
    def __init__(self, context, spec, inputs=None, expected=None):
        self.store = FeedbackStore(context['root'])
        self.feedback_ids = tuple(context['feedback_ids'])
        self.cases = [dict(case, feedback_id=key) for key in self.feedback_ids
                      for case in self.store.failure_cases(key)]
        from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
        from benchmarks.a5kernels.phase1_project import HostTensor, materialize_inputs, evaluate_reference
        from .failure_cases import phase1_policy
        mode = {'compile': 'compile-failure', 'host-mismatch': 'host-mismatch'}.get(spec.failure_mode.value)
        self.compatible = []
        self.inputs = self.expected = None
        target_inputs = materialize_inputs(spec)
        target_layout = [(x.shape, x.physical_shape) for x in target_inputs]
        for case in self.cases:
            if (case.get('classification') != mode or
                    case.get('availability', {}).get('complete') is not True or
                    not case.get('project_spec')):
                continue
            original = ProjectSpec.from_mapping(case['project_spec'])
            refs = case.get('artifacts', {})
            try:
                payload = json.loads(self.store.failure_artifact_bytes(case['feedback_id'], refs['inputs']['sha256']))
                original_inputs = tuple(HostTensor(tuple(row['shape']), tuple(row['physical_shape']),
                                                   tuple(row['storage'])) for row in payload['tensors'])
                original_layout = [(x.shape, x.physical_shape) for x in materialize_inputs(original)]
                if [(x.shape, x.physical_shape) for x in original_inputs] != original_layout:
                    continue
                original_expected = evaluate_reference(original, original_inputs)
            except (KeyError, TypeError, ValueError, OSError):
                continue
            if (original.graph != spec.graph or original.dtype != spec.dtype or
                    [(x.shape, x.physical_shape) for x in original_inputs] != target_layout or
                    case.get('project_contract') != phase1_contract(original, original_inputs, original_expected) or
                    case.get('policy') != phase1_policy(original_expected.dtype)):
                continue
            # Retained exact inputs and recomputed host oracle must match authenticated hashes.
            contract = case['project_contract']
            if any(case.get('artifacts', {}).get(role, {}).get('sha256') != contract[role + '_sha256']
                   for role in ('inputs', 'expected')):
                continue
            self.compatible.append(case)
            if self.inputs is None:
                self.inputs, self.expected = original_inputs, original_expected

    def presentation(self):
        ordered = self.compatible + [case for case in self.cases if case not in self.compatible]
        return dict(total_cases=len(self.cases), cases=[dict(
            feedback_id=case['feedback_id'], classification=case['classification'],
            description=str(case.get('description', ''))[:1024],
            compatible_negative_example=case in self.compatible,
            availability=case['availability'],
            artifacts={role: dict(sha256=ref['sha256'])
                       for role, ref in case.get('artifacts', {}).items()})
            for case in ordered[:8]])

    def require_repair_evidence(self, spec):
        if spec.failure_mode.value != 'none' and not self.compatible:
            details = '; '.join(str(case.get('availability', {}).get('reason', ''))
                                for case in self.cases[:3]) or 'No failure cases were linked.'
            raise ValueError(
                f"Repair project {spec.project_id!r} requires an authenticated {spec.failure_mode.value} "
                "example for the same computation, exact inputs, expected output and comparison policy. "
                "No compatible complete reproducer is attached. Select a matching repair task, "
                "or use failure_mode='none' and omit failure-repair for a different exploratory probe; "
                "the actor must not invent a failure. Retained evidence: " + details)

    def read_bytes(self, digest):
        for key in self.feedback_ids:
            try:
                return self.store.failure_artifact_bytes(key, digest)
            except (ValueError, FileNotFoundError):
                continue
        raise FileNotFoundError('Artifact is not available in this project\'s linked failure evidence')


class FailureArtifactReader:
    def __init__(self, current, failure):
        self.current, self.failure = current, failure

    def put_json(self, value):
        return self.current.put_json(value)

    def read_bytes(self, digest):
        try:
            return self.current.read_bytes(digest)
        except FileNotFoundError:
            return self.failure.read_bytes(digest)
