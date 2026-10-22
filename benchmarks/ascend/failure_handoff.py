"""Authenticated failure context, kept separate from reusable verified memory."""
from __future__ import annotations

from .phase2_feedback import FeedbackStore


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


class FailureContext:
    """Resolve only cases linked by the host to this remedial project."""
    def __init__(self, context, spec, inputs=None, expected=None):
        self.store = FeedbackStore(context['root'])
        self.feedback_ids = tuple(context['feedback_ids'])
        self.cases = [dict(case, feedback_id=key) for key in self.feedback_ids
                      for case in self.store.failure_cases(key)]
        contract = phase1_contract(spec, inputs, expected)
        mode = {'compile': 'compile-failure', 'host-mismatch': 'host-mismatch'}.get(spec.failure_mode.value)
        self.compatible = [case for case in self.cases if
                           case.get('classification') == mode and
                           case.get('availability', {}).get('complete') is True and
                           case.get('project_contract') == contract]

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

    def read_bytes(self, digest):
        try:
            return self.current.read_bytes(digest)
        except FileNotFoundError:
            return self.failure.read_bytes(digest)
