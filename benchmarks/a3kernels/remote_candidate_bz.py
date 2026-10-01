"""BZ-A3 transport for immutable agent-authored Ascend C candidates."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
from typing import Callable

from .phase1_protocol import ExecutionPlan
from .remote_candidate import (
    CandidateBundle,
    GZA3RemoteCandidateBackend,
    _COMPILE_SCRIPT,
    _EXECUTE_SCRIPT,
    _PendingObservation,
)


_PROFILES = frozenset(("bz-a3-1", "bz-a3-2"))
_WRAPPER_TIMEOUT_GRACE_SECONDS = 30


class BZA3RemoteCandidateBackend(GZA3RemoteCandidateBackend):
    """Run a candidate through one explicit native-only BZ-A3 profile.

    Bundle construction and host verification deliberately reuse the GZ
    implementation.  Transport and retained operation handling are replaced:
    BZ uploads use the user-wide ``cpl-remote`` interface, while compilation,
    execution, and observation use the neutral Catlass validation adapter.
    """

    def __init__(
        self, *, cpl_remote: str, validation_wrapper: str, profile: str,
        remote_workspace: str, physical_device: int,
        process_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        timeout: int = 1200, state_directory: Path | None = None,
    ) -> None:
        if Path(cpl_remote).name != "cpl-remote":
            raise ValueError("cpl_remote must identify the user-wide cpl-remote executable")
        if Path(validation_wrapper).name != "catlass-validation.sh":
            raise ValueError("validation_wrapper must identify catlass-validation.sh")
        if profile not in _PROFILES:
            raise ValueError("profile must be bz-a3-1 or bz-a3-2")
        workspace = PurePosixPath(remote_workspace)
        if not workspace.is_absolute() or ".." in workspace.parts:
            raise ValueError("remote_workspace must be an absolute normalized path")
        if type(physical_device) is not int or physical_device < 0:
            raise ValueError("physical_device must be a non-negative integer")
        if type(timeout) is not int or timeout <= 0:
            raise ValueError("timeout must be a positive integer")
        self._cpl_remote = cpl_remote
        self._wrapper = validation_wrapper
        self._profile = profile
        self._workspace = workspace
        self._device = physical_device
        self._run = process_runner
        self._timeout = timeout
        self._state_root = state_directory.resolve() if state_directory is not None else None

    @property
    def profile(self) -> str:
        return self._profile

    def _require_gz_plan(self, plan: ExecutionPlan) -> None:
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("plan must be an A3 ExecutionPlan")
        if plan.execution_profile != self._profile:
            raise ValueError(
                f"BZ-A3 backend requires a {self._profile} execution profile"
            )

    def _require_listener(self) -> None:
        """BZ transfer readiness is checked by cpl-remote itself."""

    def _upload(self, source: Path, destination: PurePosixPath) -> str:
        completed = self._call((
            self._cpl_remote, "upload", self._profile,
            str(source), destination.as_posix(),
        ))
        if completed.returncode != 0:
            raise RuntimeError(
                (completed.stderr or completed.stdout or "cpl-remote upload failed").strip()
            )
        expected = {
            "REMOTE_TARGET=": self._profile,
            "REMOTE_BACKEND=": "ssh",
            "REMOTE_OPERATION=": "upload",
            "REMOTE_STATE=": "completed",
            "REMOTE_EXIT=": "0",
        }
        if any(
            self._markers(completed.stdout, prefix) != [value]
            for prefix, value in expected.items()
        ):
            raise RuntimeError("cpl-remote upload returned invalid terminal markers")
        return "synchronous-bz-upload"

    def _poll_transfer(self, job_id: str) -> None:
        if job_id != "synchronous-bz-upload":
            raise RuntimeError("unexpected BZ transfer state")

    def _remote_archive_destination(self, plan: ExecutionPlan) -> PurePosixPath:
        return self._workspace / f".rsi-a3-candidate-{plan.execution_id}.tar"

    def _call(self, argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        wrapper_call = argv[0] == self._wrapper
        options = {
            "text": True,
            "capture_output": True,
            "check": False,
            "timeout": (
                self._timeout + _WRAPPER_TIMEOUT_GRACE_SECONDS
                if wrapper_call else self._timeout
            ),
        }
        if wrapper_call:
            options["env"] = {
                "PATH": os.environ.get("PATH", os.defpath),
                "CPL_REMOTE": self._cpl_remote,
            }
        try:
            return self._run(argv, **options)
        except subprocess.TimeoutExpired as exc:
            if wrapper_call:
                raise _PendingObservation(
                    "neutral wrapper exceeded its retained-operation timeout grace; "
                    "submission state is unknown"
                ) from exc
            raise RuntimeError("managed A3 operation timed out") from exc

    def _compile_argv(
        self, plan: ExecutionPlan, bundle: CandidateBundle,
        archive: PurePosixPath, directory: PurePosixPath,
    ) -> tuple[str, ...]:
        self._require_logical_device_zero(plan)
        return (
            self._wrapper, "--profile", self._profile, "--operation",
            f"a3-candidate-{plan.execution_id[:16]}", "run", "--native",
            "--runtime", "py311-torch", "--device", str(self._device),
            "--timeout", str(self._timeout), "--", "bash", "-c", _COMPILE_SCRIPT,
            "rsi-a3-candidate", archive.as_posix(), directory.as_posix(), bundle.sha256,
            plan.execution_id, plan.source_fingerprint, self._manifest_sha256(plan),
        )

    def _execute_argv(
        self, plan: ExecutionPlan, directory: PurePosixPath, library_sha256: str,
    ) -> tuple[str, ...]:
        self._require_logical_device_zero(plan)
        return (
            self._wrapper, "--profile", self._profile, "--operation",
            f"a3-execute-{plan.execution_id[:16]}", "run", "--native",
            "--runtime", "py311-torch", "--device", str(self._device),
            "--timeout", str(self._timeout), "--", "bash", "-c", _EXECUTE_SCRIPT,
            "rsi-a3-candidate", directory.as_posix(), plan.execution_id,
            plan.source_fingerprint, library_sha256, self._manifest_sha256(plan),
        )

    def _observe_argv(
        self, plan: ExecutionPlan, operation: str, handle: str,
    ) -> tuple[str, ...]:
        prefix = "a3-candidate" if operation == "compile" else "a3-execute"
        return (
            self._wrapper, "--profile", self._profile, "--operation",
            f"{prefix}-{plan.execution_id[:16]}", "observe", "--handle", handle,
        )

    def _load_pending(self, plan: ExecutionPlan, operation: str) -> dict[str, str] | None:
        path = self._pending_path(plan, operation)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("retained remote candidate operation is corrupt") from exc
        expected = {
            "execution_id": plan.execution_id,
            "source_fingerprint": plan.source_fingerprint,
            "operation": operation,
        }
        if (
            type(value) is not dict
            or set(value) != {*expected, "handle"}
            or any(value.get(key) != item for key, item in expected.items())
            or type(value.get("handle")) is not str
            or not value["handle"].startswith(self._profile + ":")
        ):
            raise RuntimeError("retained remote candidate operation is corrupt")
        return value

    def _publish_pending(
        self, plan: ExecutionPlan, operation: str, handle: str,
    ) -> None:
        if not handle.startswith(self._profile + ":"):
            raise RuntimeError("remote candidate observation returned a foreign handle")
        path = self._pending_path(plan, operation)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps({
            "execution_id": plan.execution_id,
            "source_fingerprint": plan.source_fingerprint,
            "operation": operation,
            "handle": handle,
        }, sort_keys=True, separators=(",", ":")) + "\n"
        if path.exists():
            if path.read_text() != data:
                raise RuntimeError("pending remote candidate operation conflicts")
            return
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=".pending-", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text() != data:
                raise RuntimeError("pending remote candidate operation conflicts") from None
        finally:
            temporary.unlink(missing_ok=True)

    def _run_operation(
        self, plan: ExecutionPlan, operation: str, argv: tuple[str, ...],
        pending: dict[str, str] | None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            completed = self._call(argv)
        except RuntimeError as exc:
            if pending is not None:
                raise _PendingObservation(
                    f"remote candidate observation incomplete; retry retained handle "
                    f"{pending['handle']}"
                ) from exc
            raise
        handles = self._markers(completed.stdout, "CATLASS_VALIDATION_HANDLE=")
        states = self._markers(completed.stdout, "CATLASS_VALIDATION_STATE=")
        profiles = self._markers(completed.stdout, "CATLASS_VALIDATION_PROFILE=")
        if (
            completed.returncode != 0
            and profiles == [self._profile]
            and states == ["observation-unavailable"]
            and len(handles) == 1
            and handles[0].startswith(self._profile + ":")
        ):
            if pending is not None and handles[0] != pending["handle"]:
                raise RuntimeError("remote candidate observation returned a foreign handle")
            self._publish_pending(plan, operation, handles[0])
            raise _PendingObservation(
                f"remote candidate observation unavailable; retry retained handle {handles[0]}"
            )
        if pending is not None:
            terminal = states in (["completed"], ["failed"])
            exact_identity = (
                profiles == [self._profile] and handles == [pending["handle"]]
            )
            coherent_exit = (
                (states == ["completed"] and completed.returncode == 0)
                or (states == ["failed"] and completed.returncode != 0)
            )
            if not terminal or not exact_identity or not coherent_exit:
                raise _PendingObservation(
                    f"remote candidate observation incomplete; retry retained handle "
                    f"{pending['handle']}"
                )
        if completed.returncode != 0 and states == ["failed"]:
            self._pending_path(plan, operation).unlink(missing_ok=True)
        return completed

    def _validate_wrapper(self, stdout: str) -> str:
        expected = {
            "CATLASS_VALIDATION_PROFILE=": self._profile,
            "CATLASS_VALIDATION_STATE=": "completed",
            "CATLASS_VALIDATION_EXIT=": "0",
        }
        if any(self._markers(stdout, prefix) != [value] for prefix, value in expected.items()):
            raise ValueError("neutral wrapper did not return a successful BZ-A3 terminal state")
        handles = self._markers(stdout, "CATLASS_VALIDATION_HANDLE=")
        if len(handles) != 1 or not handles[0].startswith(self._profile + ":"):
            raise ValueError("neutral wrapper returned an invalid BZ-A3 handle")
        return handles[0]

    @staticmethod
    def _require_logical_device_zero(plan: ExecutionPlan) -> None:
        if plan.logical_device != 0:
            raise ValueError("BZ-A3 staged execution requires logical device 0")


__all__ = ["BZA3RemoteCandidateBackend"]
