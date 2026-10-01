"""Concrete timing and self-profiling execution through the neutral GZ-A3 route."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Callable, Mapping

from .phase1_protocol import VerifiedResult, attest
from .profiling import (
    CandidateBinding,
    CompactProfileResult,
    ProfileMetric,
    ProfileRequest,
    ProfilingTreatment,
    StudyDimensions,
    TimingRequest,
    TimingResult,
    parse_timing_output,
)


_SAFE_REPLAY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_MARKER_COMPACT = "A3PROFILE_COMPACT="
_MARKER_META = "A3PROFILE_META="
_STATE_MARKERS = {
    "profile": "CATLASS_VALIDATION_PROFILE=",
    "state": "CATLASS_VALIDATION_STATE=",
    "handle": "CATLASS_VALIDATION_HANDLE=",
    "exit": "CATLASS_VALIDATION_EXIT=",
}


@dataclass(frozen=True)
class GZA3RunEvidence:
    replay_id: str
    request_id: str
    mode: str
    handle: str
    status: str
    physical_device: int
    logical_device: int
    remote_report: str | None
    stdout_sha256: str
    stderr_sha256: str


class GZA3ProfilingBackend:
    """Host-owned neutral-wrapper backend for verified A3 candidates.

    The remote candidate directory must contain the staged candidate artifacts
    and the fixed ``a3_profile_driver.py`` supplied by the execution layer.  That
    driver owns timing and msprof extraction and emits only compact raw metrics;
    this class never applies A5 PMU formulas or interpretations.
    """

    def __init__(
        self,
        *,
        validation_wrapper: str,
        remote_candidate_directory: str,
        evidence_directory: Path,
        physical_device: int,
        verified_results: Mapping[str, VerifiedResult],
        process_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        timeout: int = 600,
    ) -> None:
        if Path(validation_wrapper).name != "catlass-validation.sh":
            raise ValueError("validation_wrapper must identify catlass-validation.sh")
        remote = PurePosixPath(remote_candidate_directory)
        if not remote.is_absolute() or ".." in remote.parts:
            raise ValueError("remote candidate directory must be an absolute normalized path")
        if type(physical_device) is not int or physical_device < 0:
            raise ValueError("physical_device must be a non-negative integer")
        if type(timeout) is not int or timeout < 1:
            raise ValueError("timeout must be a positive integer")
        self._wrapper = validation_wrapper
        self._remote = remote.as_posix()
        self._evidence_root = evidence_directory.resolve()
        self._device = physical_device
        self._verified = dict(verified_results)
        self._run = process_runner
        self._timeout = timeout

    def time(
        self,
        binding: CandidateBinding,
        dimensions: StudyDimensions,
        *,
        replay_id: str | None = None,
    ) -> TimingResult:
        self._require_verified(binding)
        request = TimingRequest(binding, dimensions, execution_profile="gz-a3")
        request_id = request.request_id
        replay = replay_id or f"a3-timing-{request_id[:16]}"
        self._validate_replay(replay)
        retained = self._load_replay(replay, request_id, "timing")
        if retained is not None:
            return self._timing_from_dict(retained["result"], request)
        argv = self._wrapper_argv(replay, self._driver_argv("timing", dimensions))
        completed = self._call(argv)
        meta, handle, status = self._validated_output(completed, mode="timing")
        result = TimingResult.from_samples(
            request, parse_timing_output(completed.stdout)
        )
        evidence = self._evidence(
            replay, request_id, "timing", handle, status, meta, completed
        )
        self._publish(replay, request_id, "timing", asdict(result), evidence)
        return result

    def profile(
        self,
        request: ProfileRequest,
        *,
        replay_id: str | None = None,
    ) -> CompactProfileResult | None:
        if request.treatment is ProfilingTreatment.OFF:
            return None
        self._require_verified(request.binding)
        if request.execution_profile != "gz-a3":
            raise ValueError(
                "profiling request requires the gz-a3 execution profile"
            )
        replay = replay_id or request.default_replay_id
        self._validate_replay(replay)
        retained = self._load_replay(replay, request.request_id, "profile")
        if retained is not None:
            return self._profile_from_dict(retained["result"], request)
        extra = (
            "--metric", request.metric.value,
            "--kernel", request.expected_kernel,
        )
        argv = self._wrapper_argv(
            replay, self._driver_argv("profile", request.dimensions, extra)
        )
        completed = self._call(argv)
        meta, handle, status = self._validated_output(completed, mode="profile")
        compact = self._one_json_marker(completed.stdout, _MARKER_COMPACT, "compact profile")
        try:
            result = CompactProfileResult.create(
                request,
                exported_kernels=tuple(compact["exported_kernels"]),
                metric_values=tuple(tuple(item) for item in compact["metric_values"]),
                timeline=tuple(tuple(item) for item in compact["timeline"]),
                report_sha256=compact["report_sha256"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("compact profile output is invalid") from exc
        evidence = self._evidence(
            replay, request.request_id, "profile", handle, status, meta, completed
        )
        self._publish(replay, request.request_id, "profile", asdict(result), evidence)
        return result

    def evidence(self, replay_id: str) -> GZA3RunEvidence:
        path = self._record_path(replay_id)
        if not path.is_file():
            raise KeyError(replay_id)
        try:
            return GZA3RunEvidence(**json.loads(path.read_text())["evidence"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("retained GZ-A3 evidence is corrupt") from exc

    def _require_verified(self, binding: CandidateBinding) -> None:
        result = self._verified.get(binding.execution_id)
        if (
            not isinstance(result, VerifiedResult)
            or not result.passed
            or result.execution_id != binding.execution_id
            or result.source_fingerprint != binding.source_fingerprint
            or result.attestation_sha256 != attest(result.attestation_payload())
        ):
            raise ValueError("profiling requires a host-verified passing candidate")

    def _driver_argv(
        self,
        mode: str,
        dimensions: StudyDimensions,
        extra: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        return (
            "python", f"{self._remote}/a3_profile_driver.py", mode,
            "--candidate-dir", self._remote,
            "--logical-device", "0",
            "--length", str(dimensions.length),
            "--block-count", str(dimensions.block_count),
            "--warm-up", str(dimensions.warm_up),
            "--launch-count", str(dimensions.launch_count),
            *extra,
        )

    def _wrapper_argv(self, replay: str, command: tuple[str, ...]) -> tuple[str, ...]:
        return (
            self._wrapper, "--profile", "gz-a3", "--operation", replay,
            "run", "--native", "--runtime", "py311-torch",
            "--device", str(self._device), "--timeout", str(self._timeout),
            "--", *command,
        )

    def _call(self, argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        # Do not smuggle device selection or A5/BZ settings around the neutral wrapper.
        environment = {"PATH": os.environ.get("PATH", "")}
        try:
            completed = self._run(
                argv, text=True, capture_output=True, check=False,
                timeout=self._timeout, env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("GZ-A3 profiling wrapper timed out") from exc
        if completed.returncode != 0:
            raise RuntimeError(
                "GZ-A3 profiling failed: "
                + (completed.stderr or completed.stdout or f"exit {completed.returncode}").strip()
            )
        return completed

    def _validated_output(
        self, completed: subprocess.CompletedProcess[str], *, mode: str
    ) -> tuple[dict[str, object], str, str]:
        values = {
            name: self._one_marker(completed.stdout, prefix, name)
            for name, prefix in _STATE_MARKERS.items()
        }
        if values["profile"] != "gz-a3":
            raise RuntimeError("GZ-A3 profiling provenance is foreign")
        if values["state"] != "completed" or values["exit"] != "0":
            raise RuntimeError("GZ-A3 profiling returned a failed terminal state")
        if not values["handle"].startswith("gz-a3:"):
            raise RuntimeError("GZ-A3 profiling returned an invalid handle")
        meta = self._one_json_marker(completed.stdout, _MARKER_META, "profile metadata")
        expected = {
            "language": "ascend-c", "logical_device": 0, "mode": mode,
            "runtime": "native-ascend-c", "target": "Ascend910B4",
        }
        if any(meta.get(key) != value for key, value in expected.items()):
            raise RuntimeError("GZ-A3 profiling metadata has foreign provenance")
        remote = meta.get("remote_report")
        if mode == "timing" and remote is not None:
            raise RuntimeError("timing evidence cannot claim an msprof report")
        if mode == "profile" and (
            type(remote) is not str or not PurePosixPath(remote).is_absolute()
        ):
            raise RuntimeError("profile evidence requires an absolute retained report")
        return meta, values["handle"], values["state"]

    @staticmethod
    def _one_marker(stdout: str, prefix: str, label: str) -> str:
        values = [line.removeprefix(prefix) for line in stdout.splitlines() if line.startswith(prefix)]
        if len(values) != 1:
            raise RuntimeError(f"GZ-A3 output requires one {label} marker")
        return values[0]

    @classmethod
    def _one_json_marker(cls, stdout: str, prefix: str, label: str) -> dict[str, object]:
        value = cls._one_marker(stdout, prefix, label)
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{label} is not valid JSON") from exc
        if type(decoded) is not dict:
            raise RuntimeError(f"{label} must be a JSON object")
        return decoded

    def _evidence(
        self, replay: str, request_id: str, mode: str, handle: str, status: str,
        meta: dict[str, object], completed: subprocess.CompletedProcess[str],
    ) -> GZA3RunEvidence:
        return GZA3RunEvidence(
            replay, request_id, mode, handle, status, self._device, 0,
            meta.get("remote_report") if isinstance(meta.get("remote_report"), str) else None,
            hashlib.sha256(completed.stdout.encode()).hexdigest(),
            hashlib.sha256(completed.stderr.encode()).hexdigest(),
        )

    def _record_path(self, replay: str) -> Path:
        self._validate_replay(replay)
        return self._evidence_root / replay / "record.json"

    def _load_replay(self, replay: str, request_id: str, mode: str) -> dict | None:
        path = self._record_path(replay)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("retained GZ-A3 replay is corrupt") from exc
        if (
            type(value) is not dict
            or set(value) != {"request_id", "mode", "result", "evidence"}
            or type(value["result"]) is not dict
            or type(value["evidence"]) is not dict
        ):
            raise RuntimeError("retained GZ-A3 replay is corrupt")
        if value.get("request_id") != request_id or value.get("mode") != mode:
            raise ValueError("conflicting replay request")
        try:
            evidence = GZA3RunEvidence(**value["evidence"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("retained GZ-A3 replay evidence is corrupt") from exc
        digest_ok = all(
            type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest)
            for digest in (evidence.stdout_sha256, evidence.stderr_sha256)
        )
        if (
            evidence.replay_id != replay
            or evidence.request_id != request_id
            or evidence.mode != mode
            or evidence.status != "completed"
            or type(evidence.handle) is not str
            or not evidence.handle.startswith("gz-a3:")
            or type(evidence.physical_device) is not int
            or evidence.physical_device != self._device
            or type(evidence.logical_device) is not int
            or evidence.logical_device != 0
            or not digest_ok
        ):
            raise RuntimeError("retained GZ-A3 replay evidence identity is invalid")
        return value

    def _publish(
        self, replay: str, request_id: str, mode: str,
        result: dict, evidence: GZA3RunEvidence,
    ) -> None:
        destination = self._record_path(replay)
        destination.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "request_id": request_id, "mode": mode, "result": result,
            "evidence": asdict(evidence),
        }
        data = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        with tempfile.NamedTemporaryFile(
            "w", dir=destination.parent, prefix=".record-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_text() != data:
                raise RuntimeError("replay conflicts with retained evidence") from None
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _validate_replay(value: str) -> None:
        if type(value) is not str or _SAFE_REPLAY.fullmatch(value) is None:
            raise ValueError("unsafe GZ-A3 replay id")

    @staticmethod
    def _timing_from_dict(
        value: dict, request: TimingRequest
    ) -> TimingResult:
        try:
            retained = dict(value)
            retained["dimensions"] = StudyDimensions(**retained["dimensions"])
            retained["samples_us"] = tuple(retained["samples_us"])
            result = TimingResult(**retained)
            expected = TimingResult.from_samples(request, result.samples_us)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("retained timing result is corrupt") from exc
        if result != expected:
            raise RuntimeError("retained timing result does not match its request or digest")
        return result

    @staticmethod
    def _profile_from_dict(
        value: dict, request: ProfileRequest
    ) -> CompactProfileResult:
        try:
            retained = dict(value)
            retained["metric"] = ProfileMetric(retained["metric"])
            for key in ("exported_kernels", "metric_values", "timeline"):
                retained[key] = tuple(
                    tuple(item) if isinstance(item, list) else item
                    for item in retained[key]
                )
            result = CompactProfileResult(**retained)
            expected = CompactProfileResult.create(
                request,
                exported_kernels=result.exported_kernels,
                metric_values=result.metric_values,
                timeline=result.timeline,
                report_sha256=result.report_sha256,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("retained profile result is corrupt") from exc
        if result != expected:
            raise RuntimeError("retained profile result does not match its request or digest")
        return result


__all__ = ["GZA3ProfilingBackend", "GZA3RunEvidence"]
