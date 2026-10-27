"""Local native execution using the same host-owned candidate drivers as A3."""

import hashlib
import json
from pathlib import Path
from statistics import median
import signal
import sys

from benchmarks.a3kernels.candidate import CandidateCompilation, _parse_output
from benchmarks.a3kernels.native_candidate import NativeCandidatePlan
from benchmarks.a3kernels.native_measurements import NativeMeasurementBackend
from benchmarks.a3kernels.phase1_protocol import ExecutionPlan, ExecutionReceipt, FailedEvidence, VerifiedResult, attest
from benchmarks.a3kernels.profiling import parse_timing_output
from benchmarks.a5kernels.evidence import canonical_bytes, canonical_digest
from benchmarks.a5kernels.phase1_project_bz import ProjectInfrastructureError
from benchmarks.ascend.execution import LocalProcessRunner, local_environment, local_identity


def _failed_process_detail(result):
    detail = f"Native process failed with exit code {result.returncode}"
    if result.returncode < 0:
        try:
            detail += f" (terminated by {signal.Signals(-result.returncode).name})"
        except ValueError:
            detail += f" (terminated by signal {-result.returncode})"
    return detail + f".\nstdout:\n{result.stdout or '(empty)'}\nstderr:\n{result.stderr or '(empty)'}"


class LocalNativeCandidateBackend:
    """Stage immutable host payloads and invoke the activated Python directly."""

    profile = "local"

    def __init__(self, config, *, runner=None):
        self.config = config
        self._device = config["physical_device"]
        self._workspace = Path(config["workspace"])
        self.runner = runner or LocalProcessRunner(config)
        self.runtime_identity = local_identity(config)

    def remote_candidate_directory(self, plan):
        # The native measurement seam historically names this directory remote.
        return self._workspace / "native-candidates" / plan.execution_id

    def _stage(self, plan):
        if not isinstance(plan, (NativeCandidatePlan, ExecutionPlan)) or plan.execution_profile != "local":
            raise ValueError("local native execution requires a local NativeCandidatePlan")
        directory = self.remote_candidate_directory(plan)
        directory.mkdir(parents=True, exist_ok=True)
        payloads = {f.relative_path: f.content.encode() for f in plan.files}
        payloads["input.json"] = canonical_bytes(plan.input_payload if isinstance(plan, NativeCandidatePlan) else dict(
            input_a=plan.input_a, input_b=plan.input_b, logical_length=plan.logical_length,
            padded_length=plan.padded_length, block_count=plan.block_count))
        payloads["manifest.json"] = canonical_bytes(dict(
            execution_id=plan.execution_id, source_fingerprint=plan.source_fingerprint,
            files={name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}))
        for name, data in payloads.items():
            path = directory / name
            if path.exists() and path.read_bytes() != data:
                raise ProjectInfrastructureError("retained local candidate payload changed")
            if not path.exists():
                path.write_bytes(data)
        return directory

    def compile(self, plan, local_directory):
        directory = self._stage(plan)
        result = self.runner.run((sys.executable, str(directory / "host_driver.py"), "--compile-only"),
            Path(local_directory) / "compile", cwd=directory, env=local_environment(self.config))
        if result.returncode:
            kind = ("NativeCompileError" if "A3CANDIDATE_COMPILE_FAILED=1" in result.stdout.splitlines()
                    else "NativeHostError")
            return FailedEvidence.create(plan, stage="compile", error_type=kind,
                                         detail=_failed_process_detail(result))
        library = directory / "a3_candidate.so"
        digest = hashlib.sha256(library.read_bytes()).hexdigest()
        if [line for line in result.stdout.splitlines() if line.startswith("A3CANDIDATE_COMPILED=")] != ["A3CANDIDATE_COMPILED=" + digest]:
            raise ProjectInfrastructureError("local native compiler returned an invalid library identity")
        body = dict(plan=plan, library_sha256=digest, stdout=result.stdout, stderr=result.stderr)
        return CandidateCompilation(**body, attestation_sha256=attest(body))

    def execute_receipt(self, compilation):
        plan = compilation.plan
        directory = self._stage(plan)
        if hashlib.sha256((directory / "a3_candidate.so").read_bytes()).hexdigest() != compilation.library_sha256:
            raise ProjectInfrastructureError("retained native library changed")
        result = self.runner.run((sys.executable, str(directory / "host_driver.py"), "input.json"),
            directory / "execution", cwd=directory, env=local_environment(self.config))
        if result.returncode:
            return FailedEvidence.create(plan, stage="execute", error_type="NativeRuntimeError",
                detail=_failed_process_detail(result))
        return ExecutionReceipt(exit_code=0, output=_parse_output(result.stdout, plan.padded_length),
            stdout=result.stdout, stderr=result.stderr, job_handle="local:" + plan.execution_id,
            metadata=(("library_sha256", compilation.library_sha256), ("candidate_directory", str(directory))))


    def execute(self, compilation):
        plan = compilation.plan
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("generic native execution requires an external host oracle")
        receipt = self.execute_receipt(compilation)
        if isinstance(receipt, FailedEvidence):
            return receipt
        maximum = max(abs(actual - a - b) for actual, a, b in
                      zip(receipt.output[:plan.logical_length], plan.input_a, plan.input_b))
        return VerifiedResult.from_receipt(plan, receipt, max_abs_error=maximum)


class LocalNativeMeasurements(NativeMeasurementBackend):
    """Preserve native correctness, profiler report and timing evidence locally."""

    def __init__(self, *, compilation, candidate_backend, expected_output,
                 environment_identity, atol=1e-5, rtol=1e-5,
                 supported_metrics=("Basic", "PipeUtilization"), **options):
        self._compilation, self._candidate_backend = compilation, candidate_backend
        self._expected = tuple(expected_output)
        self._environment = dict(environment_identity)
        self._atol, self._rtol = atol, rtol
        self._supported_metrics = frozenset(supported_metrics)
        self.root = Path(options["evidence_directory"])
        self._verify_values(self._expected)
        if len(self._expected) != compilation.plan.output.numel:
            raise ValueError("host reference shape differs from candidate")

    def _measure(self, mode, warm_up, launch_count, replay_id, metric):
        if type(warm_up) is not int or warm_up < 0 or type(launch_count) is not int or launch_count < 1:
            raise ValueError("invalid native replay dimensions")
        plan = self._compilation.plan
        directory = self._candidate_backend.remote_candidate_directory(plan)
        verification = self._verify()
        identity = dict(execution_id=plan.execution_id, library_sha256=self._compilation.library_sha256,
            mode=mode, warm_up=warm_up, launch_count=launch_count, metric=metric,
            host_reference_sha256=canonical_digest(self._expected), environment=self._environment,
            execution=self._candidate_backend.runtime_identity,
            output=dict(shape=list(plan.output.shape), logical_shape=list(plan.output.logical_shape)))
        digest = canonical_digest(identity)
        argv = [sys.executable, str(directory / "native_measurement_driver.py"), mode,
            "--candidate-dir", str(directory), "--kernel", plan.kernel_name,
            "--execution-id", plan.execution_id, "--source-fingerprint", plan.source_fingerprint,
            "--library-sha256", self._compilation.library_sha256,
            "--warm-up", str(warm_up), "--launch-count", str(launch_count)]
        if metric:
            argv += ["--metric", "ArithmeticUtilization" if metric == "Basic" else metric]
        result = self._candidate_backend.runner.run(argv, self.root / (replay_id or digest),
            cwd=directory, env=local_environment(self._candidate_backend.config))
        if result.returncode:
            raise ProjectInfrastructureError(result.stderr or result.stdout or "native measurement failed")
        self._verify_values(_parse_output(result.stdout, len(self._expected)))
        meta = self._one_json_marker(result.stdout, "A3PROFILE_META=", "native metadata")
        if any(meta.get(k) != v for k, v in dict(mode=mode, kernel_name=plan.kernel_name,
                target=plan.target, logical_device=0).items()):
            raise ProjectInfrastructureError("native measurement returned a foreign identity")
        if mode == "timing":
            samples = parse_timing_output(result.stdout)
            if len(samples) != launch_count:
                raise ProjectInfrastructureError("native timing sample count differs")
            summary = dict(measurement="launch-plus-synchronize", samples_us=list(samples),
                           median_us=median(samples), instrumented=False)
        else:
            summary = self._one_json_marker(result.stdout, "A3PROFILE_COMPACT=", "native profile")
            if summary.get("exported_kernels") != [plan.kernel_name] or not summary.get("selected_columns") or not summary.get("report_sha256"):
                raise ProjectInfrastructureError("native profile lacks exact-kernel report evidence")
            summary.update(instrumented=True, metric=metric)
        raw = dict(stdout=result.stdout, stderr=result.stderr, verification=verification, metadata=meta)
        raw_hash = canonical_digest(raw)
        measured = dict(identity=identity, summary=summary, evidence_sha256=raw_hash,
                        artifact_payloads={raw_hash: raw})
        record = dict(request_id=digest, mode=mode, result=measured,
            evidence=dict(profile="local", handle="local:" + digest, status="completed",
                          physical_device=self._candidate_backend._device, logical_device=0,
                          remote_report=meta.get("remote_report")))
        destination = self.root / (replay_id or digest) / "measurement.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(canonical_bytes(record))
        return measured


class LocalLegacyProfiler:
    """Replay retained vector-add drivers for historical fixed A3 treatments."""

    def __init__(self, *, candidate, paths, worker_config):
        self.candidate, self.paths, self.config = candidate, paths, worker_config

    def _run(self, binding, dimensions, metric=None, request=None):
        from benchmarks.a3kernels.live_composition import _RecoveryCandidate, RemoteCandidateBundle
        from benchmarks.a3kernels.profiling import TimingResult, CompactProfileResult
        remote = self.candidate.backend
        if isinstance(remote, _RecoveryCandidate):
            remote = remote.backend
        if not isinstance(remote, RemoteCandidateBundle):
            raise TypeError("local replay requires a managed candidate bundle")
        verified = self.candidate.verified.get(binding.execution_id)
        if verified is None or verified.source_fingerprint != binding.source_fingerprint:
            raise ValueError("local replay requires this host-verified candidate")
        compiled = remote.compilations[binding.execution_id]
        directory = remote.backend.remote_candidate_directory(compiled.plan)
        mode = "profile" if metric else "timing"
        argv = [sys.executable, str(directory / "a3_profile_driver.py"), mode,
            "--candidate-dir", str(directory), "--logical-device", "0",
            "--length", str(dimensions.length), "--block-count", str(dimensions.block_count),
            "--warm-up", str(dimensions.warm_up), "--launch-count", str(dimensions.launch_count)]
        if metric:
            argv += ["--metric", "ArithmeticUtilization" if metric.value == "Basic" else metric.value,
                     "--kernel", "vector_add"]
        identity = canonical_digest(dict(binding=binding.as_dict(), dimensions=dimensions.as_dict(),
                                         mode=mode, metric=metric.value if metric else None))
        root = self.paths.root / "profiles" / identity
        result = remote.backend.runner.run(argv, root, cwd=directory, env=local_environment(self.config))
        if result.returncode:
            raise ProjectInfrastructureError(result.stderr or result.stdout or "local replay failed")
        meta = NativeMeasurementBackend._one_json_marker(result.stdout, "A3PROFILE_META=", "local replay")
        if metric:
            summary = NativeMeasurementBackend._one_json_marker(result.stdout, "A3PROFILE_COMPACT=", "local profile")
            observed = CompactProfileResult.create(request, exported_kernels=tuple(summary["exported_kernels"]),
                metric_values=tuple(tuple(pair) for pair in summary["metric_values"]),
                timeline=tuple(tuple(pair) for pair in summary["timeline"]), report_sha256=summary["report_sha256"])
        else:
            samples = parse_timing_output(result.stdout)
            if len(samples) != dimensions.launch_count:
                raise ProjectInfrastructureError("local timing sample count differs")
            observed = TimingResult.from_samples(binding, dimensions, samples, execution_profile="local")
        from dataclasses import asdict
        root.mkdir(parents=True, exist_ok=True)
        (root / "measurement.json").write_bytes(canonical_bytes(dict(result=asdict(observed),
            evidence=dict(profile="local", status="completed", remote_report=meta.get("remote_report")),
            stdout=result.stdout, stderr=result.stderr)))
        return observed

    def time(self, binding, dimensions):
        return self._run(binding, dimensions)

    def profile(self, request):
        from benchmarks.a3kernels.profiling import ProfilingTreatment
        if request.treatment is ProfilingTreatment.OFF:
            return None
        return self._run(request.binding, request.dimensions, request.metric, request)


def local_legacy_dependencies(config, worker_config, environ):
    """Compose fixed A3 Pi treatments without loading a remote-access client."""
    from benchmarks.a3kernels.live_composition import (
        LiveDependencies, RemoteCandidateBundle, model_actor_factory, local_knowledge_factory)
    from benchmarks.ascend.execution import preflight_local
    identity = preflight_local(worker_config)
    # This historical driver's registered ABI and profiling labels are 910B4.
    # The generic native composition supports other discovered A3 products.
    product = identity.get("product", "")
    if "910B4" not in product:
        raise ValueError("historical fixed A3 treatments require Ascend910B4; use the native A3 CLI for other products")
    root = Path(config.state_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "local-execution-identity.json"
    data = canonical_bytes(identity)
    if path.exists() and path.read_bytes() != data:
        raise ValueError("fixed A3 local execution identity changed")
    if not path.exists() and any(root.glob("cells/*/terminal.json")):
        raise ValueError("cannot resume remote fixed A3 state with local execution")
    path.write_bytes(data)
    return LiveDependencies(
        actor_factory=model_actor_factory(environ),
        candidate_factory=lambda *_: RemoteCandidateBundle(LocalNativeCandidateBackend(worker_config)),
        knowledge_factory=local_knowledge_factory(config),
        profiler_factory=lambda _cell, _proposal, paths, candidate: LocalLegacyProfiler(
            candidate=candidate, paths=paths, worker_config=worker_config),
        execution_profile="local")
