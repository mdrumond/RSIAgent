"""Host-verified measurements of exact adaptive Actor candidates."""

from dataclasses import replace
import math
import hashlib
from statistics import median, pstdev

from benchmarks.a5kernels.evidence import canonical_digest
from benchmarks.a5kernels.phase1_curriculum import Capability
from benchmarks.a5kernels.phase1_project import (
    EvidenceClassification, ProjectGraph, classify_execution,
    evaluate_reference, materialize_inputs,
)
from benchmarks.a5kernels.phase1_project_executor import ProjectExecutionRequest
from benchmarks.a5kernels.profiling import (
    CampaignKind, CaptureCommand, ProfileMetric, ProfileRequest, TimingCommand,
)


def specialize_timing(spec):
    """Rebuild the graph for each requested length without changing the candidate."""
    if len(spec.shape) != 1:
        raise ValueError("timing sweeps support one-dimensional projects only")
    if not spec.timing_lengths:
        raise ValueError("project has no timing lengths")
    return tuple(replace(
        spec, shape=(length,),
        capabilities=tuple(item for item in spec.capabilities if
                           item is not Capability.PADDED_TAIL or
                           length % spec.padding_multiple != 0),
        graph=ProjectGraph(tuple(replace(item, shape=(length,))
                                 for item in spec.graph.inputs),
                           spec.graph.nodes, spec.graph.output),
    ) for length in spec.timing_lengths)


def timing_summary(samples):
    if len(samples) != 3 or any(
        isinstance(x, bool) or not isinstance(x, (int, float))
        or not math.isfinite(x) or x <= 0 for x in samples
    ):
        raise ValueError("timing requires three positive finite samples")
    return {"measurement": "launch-plus-synchronize", "samples_us": list(samples),
            "median_us": median(samples), "dispersion_us": pstdev(samples),
            "dispersion": "population-standard-deviation",
            "warmups": 5, "launches_per_sample": 20}


class ProjectMeasurementBackend:
    """Reuse checked project execution and profiling; never trust Actor timing."""

    def __init__(self, project_backend, profile_backend):
        self._project = project_backend
        self._profile = profile_backend

    def _verify(self, request, artifacts):
        compilation = self._project.compile(request)
        runtime = self._project.run(request, compilation) if compilation.passed else None
        for observation in (compilation, runtime):
            if observation is not None:
                digest = observation.evidence_sha256
                artifacts[digest] = self._project.observation_payload(digest)
        verdict = classify_execution(
            request.expected, compile_succeeded=compilation.passed,
            runtime_error=(None if runtime is None or runtime.passed else runtime.diagnostic),
            actual=None if runtime is None else runtime.output,
        )
        if verdict.classification is not EvidenceClassification.HOST_VERIFIED:
            detail = compilation.diagnostic if runtime is None else verdict.detail
            raise ValueError(f"measurement requires host correctness: {detail}")

    def _request(self, request, kernel="*", *, timing=False):
        return ProfileRequest(self._project.measurement_plan(request), "catlass-dsl",
                              kernel, self._project.device,
                              warm_up=5 if timing else 0,
                              launch_count=20 if timing else 1)

    @staticmethod
    def _result(request, summary, artifacts):
        payload = {"project_id": request.spec.project_id,
                   "candidate_sha256": hashlib.sha256(request.candidate_source.encode()).hexdigest(),
                   "project_sha256": canonical_digest(request.spec.as_dict()),
                   "summary": summary, "supporting_evidence": list(artifacts)}
        digest = canonical_digest(payload)
        artifacts[digest] = payload
        return {"summary": summary, "artifact_payloads": artifacts,
                "evidence_sha256": tuple(artifacts), "measurement_sha256": digest}

    def timing(self, request: ProjectExecutionRequest):
        artifacts, results = {}, []
        for spec in specialize_timing(request.spec):
            inputs = materialize_inputs(spec)
            expected = evaluate_reference(spec, inputs)
            payload = {"tensors": [
                {"shape": list(x.shape), "physical_shape": list(x.physical_shape),
                 "storage": list(x.storage)} for x in inputs]}
            variant = replace(request, spec=spec, inputs=inputs, expected=expected,
                              inputs_sha256=canonical_digest(payload))
            self._verify(variant, artifacts)
            profile_request = self._request(variant, timing=True)
            values = []
            for index in range(3):
                replay = f"{profile_request.attempt_id}-timing-{index}"
                sample = self._profile.time_sample(TimingCommand(
                    CampaignKind.INTERMEDIATE, profile_request, replay))
                # Preserve the raw checked-wrapper stdout, not an agent estimate.
                raw = self._profile.timing_evidence_payload(replay)
                artifacts[canonical_digest(raw)] = raw
                values.append(sample.duration_us)
            results.append({"shape": list(spec.shape), **timing_summary(values)})
        return self._result(request, {"timing": results}, artifacts)

    def profile(self, request: ProjectExecutionRequest, metric: str):
        metrics = {"pipe-utilization": ProfileMetric.PIPE_UTILIZATION,
                   "memory-throughput": ProfileMetric.MEMORY,
                   "timeline": ProfileMetric.PIPE_TIMELINE}
        if metric not in metrics:
            raise ValueError("profile metric must be pipe-utilization, memory-throughput, or timeline")
        artifacts = {}
        self._verify(request, artifacts)
        profile_request = self._request(request)
        basic = self._profile.capture(CaptureCommand(
            CampaignKind.INTERMEDIATE, profile_request, ProfileMetric.BASIC_INFO,
            f"{profile_request.attempt_id}-basic"))
        if len(basic.exported_kernels) != 1:
            raise ValueError("BasicInfo must identify exactly one exported candidate kernel")
        kernel = basic.exported_kernels[0]
        capture = self._profile.capture(CaptureCommand(
            CampaignKind.INTERMEDIATE, replace(profile_request, expected_kernel=kernel),
            metrics[metric], f"{profile_request.attempt_id}-metric", kernel))
        for item in (basic, capture):
            raw = self._profile.capture_evidence_payload(item)
            artifacts[canonical_digest(raw)] = raw
        summary = {"kernel_name": kernel, "metric": metrics[metric].value,
                   "basic_info": list(basic.summary), "metrics": list(capture.summary),
                   "duration_semantics": "instrumented diagnostic kernel duration",
                   "capture_order": [basic.replay_id, capture.replay_id]}
        return self._result(request, summary, artifacts)
