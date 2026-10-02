"""Timing and compact raw-msprof evidence through neutral BZ-A3 profiles."""

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
    CandidateBinding, CompactProfileResult, ProfileMetric, ProfileRequest,
    ProfilingTreatment, StudyDimensions, TimingRequest, TimingResult,
    parse_timing_output,
)

_PROFILES = frozenset(("bz-a3-1", "bz-a3-2"))
_WRAPPER_TIMEOUT_GRACE_SECONDS = 30
_SAFE_REPLAY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_STATE = {
    "profile": "CATLASS_VALIDATION_PROFILE=",
    "state": "CATLASS_VALIDATION_STATE=",
    "handle": "CATLASS_VALIDATION_HANDLE=",
    "exit": "CATLASS_VALIDATION_EXIT=",
}


@dataclass(frozen=True)
class BZA3RunEvidence:
    replay_id: str
    request_id: str
    mode: str
    profile: str
    handle: str
    status: str
    physical_device: int
    logical_device: int
    remote_report: str | None
    stdout_sha256: str
    stderr_sha256: str


class BZA3ProfilingBackend:
    """Run verified candidates while preserving raw evidence and no A5 formulas."""

    def __init__(
        self, *, validation_wrapper: str, cpl_remote: str, profile: str,
        remote_candidate_directory: str, evidence_directory: Path,
        physical_device: int, verified_results: Mapping[str, VerifiedResult],
        process_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        timeout: int = 600,
    ) -> None:
        remote = PurePosixPath(remote_candidate_directory)
        if Path(validation_wrapper).name != "catlass-validation.sh":
            raise ValueError("validation_wrapper must identify catlass-validation.sh")
        if Path(cpl_remote).name != "cpl-remote":
            raise ValueError("cpl_remote must identify the user-wide cpl-remote executable")
        if profile not in _PROFILES:
            raise ValueError("profile must be bz-a3-1 or bz-a3-2")
        if not remote.is_absolute() or ".." in remote.parts:
            raise ValueError("remote candidate directory must be absolute and normalized")
        if type(physical_device) is not int or physical_device < 0:
            raise ValueError("physical_device must be a non-negative integer")
        if type(timeout) is not int or timeout < 1:
            raise ValueError("timeout must be a positive integer")
        self._wrapper, self._cpl_remote, self._profile = validation_wrapper, cpl_remote, profile
        self._remote, self._evidence_root = remote.as_posix(), evidence_directory.resolve()
        self._device, self._verified = physical_device, dict(verified_results)
        self._run, self._timeout = process_runner, timeout

    def time(
        self, binding: CandidateBinding, dimensions: StudyDimensions, *,
        replay_id: str | None = None,
    ) -> TimingResult:
        request = TimingRequest(binding, dimensions, execution_profile=self._profile)
        self._require_verified(binding)
        replay = replay_id or f"a3-timing-{request.request_id[:16]}"
        retained = self._load(replay, request.request_id, "timing")
        if retained is not None:
            try:
                value = retained["result"]
                result = TimingResult(**{
                    **value, "dimensions": StudyDimensions(**value["dimensions"]),
                    "samples_us": tuple(value["samples_us"]),
                })
                expected = TimingResult.from_samples(request, result.samples_us)
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("retained timing result is corrupt") from exc
            if result != expected:
                raise RuntimeError("retained timing result does not match its request")
            return result
        completed, evidence = self._execute(
            replay, request.request_id, "timing",
            self._command(replay, "timing", binding, dimensions),
        )
        result = TimingResult.from_samples(request, parse_timing_output(completed.stdout))
        self._publish(replay, request.request_id, "timing", asdict(result), evidence)
        return result

    def profile(
        self, request: ProfileRequest, *, replay_id: str | None = None,
    ) -> CompactProfileResult | None:
        if request.treatment is ProfilingTreatment.OFF:
            return None
        self._require_verified(request.binding)
        if request.execution_profile != self._profile:
            raise ValueError("profiling request requires matching BZ-A3 execution profile")
        replay = replay_id or request.default_replay_id
        retained = self._load(replay, request.request_id, "profile")
        if retained is not None:
            try:
                value = retained["result"]
                result = CompactProfileResult(**{
                    **value, "metric": ProfileMetric(value["metric"]),
                    "exported_kernels": tuple(value["exported_kernels"]),
                    "metric_values": tuple(map(tuple, value["metric_values"])),
                    "timeline": tuple(map(tuple, value["timeline"])),
                })
                expected = CompactProfileResult.create(
                    request, exported_kernels=result.exported_kernels,
                    metric_values=result.metric_values, timeline=result.timeline,
                    report_sha256=result.report_sha256,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("retained profile result is corrupt") from exc
            if result != expected:
                raise RuntimeError("retained profile result does not match its request")
            return result
        command = self._command(
            replay, "profile", request.binding, request.dimensions,
            ("--metric", self._remote_metric(request.metric),
             "--kernel", request.expected_kernel),
        )
        completed, evidence = self._execute(replay, request.request_id, "profile", command)
        compact = self._json_marker(completed.stdout, "A3PROFILE_COMPACT=")
        try:
            result = CompactProfileResult.create(
                request, exported_kernels=tuple(compact["exported_kernels"]),
                metric_values=tuple(map(tuple, compact["metric_values"])),
                timeline=tuple(map(tuple, compact["timeline"])),
                report_sha256=compact["report_sha256"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("compact profile output is invalid") from exc
        self._publish(replay, request.request_id, "profile", asdict(result), evidence)
        return result

    def evidence(self, replay_id: str) -> BZA3RunEvidence:
        path = self._path(replay_id, "record.json")
        if not path.is_file():
            raise KeyError(replay_id)
        try:
            evidence = BZA3RunEvidence(**json.loads(path.read_text())["evidence"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("retained BZ-A3 evidence is corrupt") from exc
        self._validate_evidence(evidence, replay_id)
        return evidence

    @staticmethod
    def _remote_metric(metric: ProfileMetric) -> str:
        return "ArithmeticUtilization" if metric is ProfileMetric.BASIC else metric.value

    def _require_verified(self, binding: CandidateBinding) -> None:
        result = self._verified.get(binding.execution_id)
        if (
            not isinstance(result, VerifiedResult) or not result.passed
            or result.execution_id != binding.execution_id
            or result.source_fingerprint != binding.source_fingerprint
            or not isinstance(result.job_handle, str)
            or not result.job_handle.startswith(self._profile + ":")
            or result.attestation_sha256 != attest(result.attestation_payload())
        ):
            raise ValueError(
                "profiling requires a host-verified passing candidate from the "
                "selected BZ-A3 profile"
            )

    def _command(
        self, replay: str, mode: str, binding: CandidateBinding,
        dimensions: StudyDimensions,
        extra: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        driver = (
            "python", f"{self._remote}/a3_profile_driver.py", mode,
            "--candidate-dir", self._remote, "--logical-device", "0",
            "--execution-id", binding.execution_id,
            "--source-fingerprint", binding.source_fingerprint,
            "--length", str(dimensions.length), "--block-count", str(dimensions.block_count),
            "--warm-up", str(dimensions.warm_up), "--launch-count", str(dimensions.launch_count),
            *extra,
        )
        return (
            self._wrapper, "--profile", self._profile, "--operation", replay,
            "run", "--native", "--runtime", "py311-torch", "--device", str(self._device),
            "--timeout", str(self._timeout), "--", *driver,
        )

    def _execute(
        self, replay: str, request_id: str, mode: str, submit: tuple[str, ...],
    ) -> tuple[subprocess.CompletedProcess[str], BZA3RunEvidence]:
        pending = self._load_pending(replay, request_id, mode)
        argv = submit if pending is None else (
            self._wrapper, "--profile", self._profile, "--operation", replay,
            "observe", "--handle", pending["handle"],
        )
        try:
            completed = self._run(
                argv, text=True, capture_output=True, check=False,
                timeout=self._timeout + _WRAPPER_TIMEOUT_GRACE_SECONDS,
                env={"PATH": os.environ.get("PATH", ""), "CPL_REMOTE": self._cpl_remote},
            )
        except subprocess.TimeoutExpired as exc:
            if pending is not None:
                raise RuntimeError(
                    "BZ-A3 observation timed out; retry retained handle "
                    + pending["handle"]
                ) from exc
            self._write_once(
                self._path(replay, "unknown.json"),
                {"request_id": request_id, "mode": mode,
                 "status": "submission-timeout"},
            )
            raise RuntimeError(
                "BZ-A3 wrapper deadline expired; submission state is unknown and "
                "this replay is locked"
            ) from exc
        markers = self._markers(completed.stdout, required=completed.returncode == 0)
        if completed.returncode:
            if (
                markers.get("profile") == self._profile
                and markers.get("state") == "observation-unavailable"
                and markers.get("handle", "").startswith(self._profile + ":")
            ):
                if pending is not None and markers["handle"] != pending["handle"]:
                    raise RuntimeError("BZ-A3 observation returned a foreign handle")
                self._write_once(
                    self._path(replay, "pending.json"),
                    {"request_id": request_id, "mode": mode, "handle": markers["handle"]},
                )
                raise RuntimeError(
                    f"BZ-A3 observation unavailable; retry retained handle {markers['handle']}"
                )
            detail = (completed.stderr or completed.stdout or f"exit {completed.returncode}").strip()
            if (
                markers.get("profile") == self._profile
                and markers.get("state") == "failed"
                and markers.get("handle", "").startswith(self._profile + ":")
            ):
                self._write_once(
                    self._path(replay, "failed.json"),
                    {"request_id": request_id, "mode": mode, "status": "failed",
                     "profile": self._profile, "handle": markers["handle"],
                     "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                     "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
                     "detail": detail},
                )
                self._path(replay, "pending.json").unlink(missing_ok=True)
            raise RuntimeError(f"BZ-A3 profiling failed: {detail}")
        if pending is not None and markers["handle"] != pending["handle"]:
            raise RuntimeError("BZ-A3 observation returned a foreign handle")
        if (
            markers["profile"] != self._profile or markers["state"] != "completed"
            or markers["exit"] != "0"
            or not markers["handle"].startswith(self._profile + ":")
        ):
            raise RuntimeError("BZ-A3 profiling returned foreign or failed provenance")
        metadata = self._json_marker(completed.stdout, "A3PROFILE_META=")
        expected = {
            "language": "ascend-c", "logical_device": 0, "mode": mode,
            "runtime": "native-ascend-c", "target": "Ascend910B4",
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise RuntimeError("BZ-A3 profiling metadata has foreign provenance")
        report = metadata.get("remote_report")
        if mode == "timing" and report is not None:
            raise RuntimeError("timing evidence cannot claim an msprof report")
        if mode == "profile" and (
            type(report) is not str or not PurePosixPath(report).is_absolute()
        ):
            raise RuntimeError("profile evidence requires an absolute retained report")
        evidence = BZA3RunEvidence(
            replay, request_id, mode, self._profile, markers["handle"], markers["state"],
            self._device, 0, report if isinstance(report, str) else None,
            hashlib.sha256(completed.stdout.encode()).hexdigest(),
            hashlib.sha256(completed.stderr.encode()).hexdigest(),
        )
        return completed, evidence

    def _load_pending(self, replay: str, request_id: str, mode: str) -> dict[str, str] | None:
        path = self._path(replay, "pending.json")
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("retained BZ-A3 pending replay is corrupt") from exc
        if (
            type(value) is not dict or set(value) != {"request_id", "mode", "handle"}
            or not isinstance(value["handle"], str)
            or not value["handle"].startswith(self._profile + ":")
        ):
            raise RuntimeError("retained BZ-A3 pending replay is corrupt")
        if value["request_id"] != request_id or value["mode"] != mode:
            raise ValueError("conflicting pending replay request")
        return value

    def _load(self, replay: str, request_id: str, mode: str) -> dict[str, object] | None:
        self._raise_if_blocked(replay, request_id, mode)
        path = self._path(replay, "record.json")
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text())
            evidence = BZA3RunEvidence(**value["evidence"])
        except (KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("retained BZ-A3 replay is corrupt") from exc
        if value.get("request_id") != request_id or value.get("mode") != mode:
            raise ValueError("conflicting replay request")
        self._validate_evidence(evidence, replay, request_id, mode)
        return value

    def _raise_if_blocked(self, replay: str, request_id: str, mode: str) -> None:
        for name, label in (("unknown.json", "locked after an unknown submission"),
                            ("failed.json", "previously failed")):
            path = self._path(replay, name)
            if not path.exists():
                continue
            try:
                value = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("retained BZ-A3 replay state is corrupt") from exc
            if (
                type(value) is not dict
                or value.get("request_id") != request_id
                or value.get("mode") != mode
            ):
                raise ValueError("conflicting replay request")
            detail = value.get("detail")
            suffix = f": {detail}" if isinstance(detail, str) and detail else ""
            raise RuntimeError(f"BZ-A3 replay {label}; use a new replay id{suffix}")

    def _validate_evidence(
        self, evidence: BZA3RunEvidence, replay: str,
        request_id: str | None = None, mode: str | None = None,
    ) -> None:
        digests = (evidence.stdout_sha256, evidence.stderr_sha256)
        if (
            evidence.replay_id != replay
            or (request_id is not None and evidence.request_id != request_id)
            or (mode is not None and evidence.mode != mode)
            or evidence.profile != self._profile
            or not evidence.handle.startswith(self._profile + ":")
            or evidence.status != "completed" or type(evidence.physical_device) is not int
            or evidence.physical_device != self._device or type(evidence.logical_device) is not int
            or evidence.logical_device != 0
            or any(type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in digests)
        ):
            raise RuntimeError("retained BZ-A3 evidence identity is invalid")

    def _publish(
        self, replay: str, request_id: str, mode: str,
        result: dict[str, object], evidence: BZA3RunEvidence,
    ) -> None:
        self._write_once(
            self._path(replay, "record.json"),
            {"request_id": request_id, "mode": mode, "result": result,
             "evidence": asdict(evidence)},
        )
        self._path(replay, "pending.json").unlink(missing_ok=True)

    @staticmethod
    def _write_once(path: Path, value: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        if path.exists():
            if path.read_text() != data:
                raise RuntimeError("retained replay conflicts with new evidence")
            return
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=".a3-bz-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text() != data:
                raise RuntimeError("retained replay conflicts with new evidence") from None
        finally:
            temporary.unlink(missing_ok=True)

    def _path(self, replay: str, name: str) -> Path:
        if type(replay) is not str or _SAFE_REPLAY.fullmatch(replay) is None:
            raise ValueError("unsafe A3 replay id")
        return self._evidence_root / replay / name

    @staticmethod
    def _markers(stdout: str, *, required: bool) -> dict[str, str]:
        result = {}
        for name, prefix in _STATE.items():
            values = [line.removeprefix(prefix) for line in stdout.splitlines()
                      if line.startswith(prefix)]
            if len(values) == 1:
                result[name] = values[0]
            elif required:
                raise RuntimeError(f"A3 output requires one {name} marker")
        return result

    @staticmethod
    def _json_marker(stdout: str, prefix: str) -> dict[str, object]:
        values = [line.removeprefix(prefix) for line in stdout.splitlines()
                  if line.startswith(prefix)]
        if len(values) != 1:
            raise RuntimeError(f"A3 output requires one {prefix.rstrip('=')} marker")
        try:
            value = json.loads(values[0])
        except json.JSONDecodeError as exc:
            raise RuntimeError("A3 JSON marker is invalid") from exc
        if type(value) is not dict:
            raise RuntimeError("A3 JSON marker must contain an object")
        return value


__all__ = ["BZA3ProfilingBackend", "BZA3RunEvidence"]
