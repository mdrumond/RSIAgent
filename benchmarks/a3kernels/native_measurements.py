"""Bounded native measurements composed with the existing BZ retained transport."""

from dataclasses import asdict
import json
import math
import re
from pathlib import PurePosixPath
from statistics import median

from .native_candidate import NativeCandidatePlan
from .candidate import _parse_output
from .phase1_evidence import canonical_digest
from .phase1_protocol import ExecutionReceipt
from .profiling import parse_timing_output
from .profiling_bz import BZA3ProfilingBackend


class NativeMeasurementBackend(BZA3ProfilingBackend):
    """Reference values come from a host oracle, never from agent source/output."""

    def __init__(self, *, compilation, candidate_backend, expected_output,
                 environment_identity,
                 atol=1e-5, rtol=1e-5,
                 supported_metrics=("Basic", "PipeUtilization"), **options):
        if not isinstance(compilation.plan, NativeCandidatePlan):
            raise TypeError("native measurements require a generic native compilation")
        self._compilation = compilation
        if any(not environment_identity.get(key) for key in ("cann_version", "profiler_version")):
            raise ValueError("measurements require discovered CANN/profiler versions")
        self._environment = dict(environment_identity)
        self._candidate_backend = candidate_backend
        self._expected = tuple(expected_output)
        if len(self._expected) != compilation.plan.output.numel or any(not math.isfinite(v) for v in self._expected):
            raise ValueError("host reference must match the finite output shape")
        if any(not math.isfinite(v) or v < 0 for v in (atol, rtol)):
            raise ValueError("host tolerances must be finite and non-negative")
        self._atol, self._rtol = atol, rtol
        self._supported_metrics = frozenset(supported_metrics)
        super().__init__(verified_results={}, **options)
        if compilation.plan.execution_profile != self._profile:
            raise ValueError("native measurement profile differs from candidate")

    def _verify(self):
        receipt = self._candidate_backend.execute_receipt(self._compilation)
        if not isinstance(receipt, ExecutionReceipt) or receipt.exit_code != 0:
            raise ValueError("native measurement requires successful host verification")
        self._verify_values(receipt.output)
        return json.loads(json.dumps(asdict(receipt)))

    def _verify_values(self, output):
        if len(output) != len(self._expected) or any(
            not math.isfinite(actual) or abs(actual - expected) > self._atol + self._rtol * abs(expected)
            for actual, expected in zip(output, self._expected)
        ):
            raise ValueError("native measurement requires passing host verification")

    def timing(self, *, warm_up=5, launch_count=20, replay_id=None):
        return self._measure("timing", warm_up, launch_count, replay_id, None)

    def capture(self, metric, *, warm_up=0, launch_count=1, replay_id=None):
        if metric not in self._supported_metrics or metric not in ("Basic", "PipeUtilization"):
            raise ValueError(f"A3 metric {metric} unavailable in this measurement binding")
        return self._measure("profile", warm_up, launch_count, replay_id, metric)

    def _measure(self, mode, warm_up, launch_count, replay_id, metric):
        if type(warm_up) is not int or type(launch_count) is not int or warm_up < 0 or launch_count < 1:
            raise ValueError("warm-up must be non-negative and launch-count positive")
        plan = self._compilation.plan
        identity = dict(execution_id=plan.execution_id, source_fingerprint=plan.source_fingerprint,
            library_sha256=self._compilation.library_sha256, kernel_name=plan.kernel_name,
            target=plan.target, arch=plan.arch, execution_profile=self._profile,
            physical_device=self._device, logical_device=0, block_count=plan.block_count,
            environment=self._environment,
            inputs=[dict(name=t.name, shape=list(t.shape), dtype=t.dtype) for t in plan.inputs],
            output=dict(shape=list(plan.output.shape), dtype=plan.output.dtype),
            warm_up=warm_up, launch_count=launch_count, mode=mode, metric=metric,
            host_reference_sha256=canonical_digest(self._expected), atol=self._atol, rtol=self._rtol)
        request_id = canonical_digest(identity)
        replay = replay_id or f"a3-native-{request_id[:20]}"
        retained = self._load_replay(replay, request_id, mode)
        if retained is not None:
            return retained["result"]
        verification = self._verify()
        command = ("python", f"{self._remote}/native_measurement_driver.py", mode,
            "--candidate-dir", self._remote, "--kernel", plan.kernel_name,
            "--execution-id", plan.execution_id, "--source-fingerprint", plan.source_fingerprint,
            "--library-sha256", self._compilation.library_sha256,
            "--warm-up", str(warm_up), "--launch-count", str(launch_count))
        if metric is not None:
            command += ("--metric", "ArithmeticUtilization" if metric == "Basic" else metric)
        completed = self._run_or_observe(replay, request_id, mode, self._wrapper_argv(replay, command))
        meta, handle, status = self._validated_output(completed, mode=mode)
        self._verify_values(_parse_output(completed.stdout, len(self._expected)))
        if mode == "timing":
            samples = parse_timing_output(completed.stdout)
            if len(samples) != launch_count:
                raise RuntimeError("native timing returned an incorrect sample count")
            summary = dict(measurement="launch-plus-synchronize", samples_us=list(samples),
                           median_us=median(samples), instrumented=False)
        else:
            summary = self._one_json_marker(completed.stdout, "A3PROFILE_COMPACT=", "native profile")
            if summary.get("exported_kernels") != [plan.kernel_name] or not summary.get("selected_columns"):
                raise RuntimeError("native profile lacks exact-kernel evidence")
            if not re.fullmatch(r"[0-9a-f]{64}", str(summary.get("report_sha256"))) or not (summary.get("metric_values") or summary.get("timeline")):
                raise RuntimeError("native profile metrics unavailable or missing report identity")
            summary.update(instrumented=True, metric=metric,
                           duration_semantics="instrumented diagnostic; not canonical latency")
        raw = dict(stdout=completed.stdout, stderr=completed.stderr,
                   verification=verification, metadata=meta)
        raw_hash = canonical_digest(raw)
        result = dict(identity=identity, summary=summary, evidence_sha256=raw_hash,
                      artifact_payloads={raw_hash: raw})
        evidence = self._evidence(replay, request_id, mode, handle, status, meta, completed)
        self._publish(replay, request_id, mode, result, evidence)
        return result

    def _validated_output(self, completed, *, mode):
        # Reuse neutral-wrapper marker validation, with this manifest's target.
        values = {name: self._one_marker(completed.stdout, prefix, name) for name, prefix in {
            "profile": "CATLASS_VALIDATION_PROFILE=", "state": "CATLASS_VALIDATION_STATE=",
            "handle": "CATLASS_VALIDATION_HANDLE=", "exit": "CATLASS_VALIDATION_EXIT="}.items()}
        if values["profile"] != self._profile or values["state"] != "completed" or values["exit"] != "0" or not values["handle"].startswith(self._profile + ":"):
            raise RuntimeError("native measurement has invalid wrapper provenance")
        meta = self._one_json_marker(completed.stdout, "A3PROFILE_META=", "native metadata")
        expected = dict(mode=mode, target=self._compilation.plan.target,
                        kernel_name=self._compilation.plan.kernel_name,
                        language="ascend-c", runtime="native-ascend-c", logical_device=0)
        if any(meta.get(k) != v for k, v in expected.items()):
            raise RuntimeError("native measurement has foreign candidate metadata")
        report = meta.get("remote_report")
        if (mode == "timing" and report is not None) or (mode == "profile" and (not isinstance(report, str) or not PurePosixPath(report).is_absolute())):
            raise RuntimeError("native measurement report location is invalid")
        return meta, values["handle"], values["state"]
