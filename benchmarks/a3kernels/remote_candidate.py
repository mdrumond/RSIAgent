"""Managed transfer and GZ-A3 execution for an immutable candidate bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile
import time
from typing import Callable, Mapping
import urllib.parse
import urllib.request

from .candidate import CandidateCompilation
from .phase1_protocol import ExecutionPlan, ExecutionReceipt, FailedEvidence, VerifiedResult, attest


@dataclass(frozen=True)
class CandidateBundle:
    path: Path
    sha256: str
    execution_id: str


class _PendingObservation(RuntimeError):
    pass


_COMPILE_SCRIPT = r'''set -eu
archive=$1
destination=$2
expected_archive=$3
execution_id=$4
source_fingerprint=$5
actual_archive=$(sha256sum "$archive" | cut -d ' ' -f 1)
test "$actual_archive" = "$expected_archive"
if test ! -d "$destination"; then
  temporary="${destination}.staging-${expected_archive}"
  mkdir -p "$(dirname "$destination")"
  mkdir "$temporary"
  tar -xf "$archive" -C "$temporary"
  mv "$temporary" "$destination"
fi
python -c 'import hashlib,json,os,sys; root=sys.argv[1]; p=json.load(open(os.path.join(root,"manifest.json"))); assert p["execution_id"] == sys.argv[2]; assert p["source_fingerprint"] == sys.argv[3]; assert all(hashlib.sha256(open(os.path.join(root,n),"rb").read()).hexdigest() == h for n,h in p["files"].items())' "$destination" "$execution_id" "$source_fingerprint"
cd "$destination"
printf 'A3REMOTE_STAGE=compile\n'
python host_driver.py --compile-only
'''

_EXECUTE_SCRIPT = r'''set -eu
destination=$1
execution_id=$2
source_fingerprint=$3
library_sha256=$4
python -c 'import hashlib,json,os,sys; root=sys.argv[1]; p=json.load(open(os.path.join(root,"manifest.json"))); assert p["execution_id"] == sys.argv[2]; assert p["source_fingerprint"] == sys.argv[3]; assert all(hashlib.sha256(open(os.path.join(root,n),"rb").read()).hexdigest() == h for n,h in p["files"].items()); assert hashlib.sha256(open(os.path.join(root,"a3_candidate.so"),"rb").read()).hexdigest() == sys.argv[4]' "$destination" "$execution_id" "$source_fingerprint" "$library_sha256"
cd "$destination"
printf 'A3REMOTE_STAGE=execute\n'
python host_driver.py input.json
'''


class GZA3RemoteCandidateBackend:
    def __init__(
        self, *, client: str, server: str, validation_wrapper: str,
        remote: str, remote_workspace: str, physical_device: int,
        health_reader: Callable[[], Mapping[str, object]] | None = None,
        process_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        sleeper: Callable[[float], None] = time.sleep, poll_interval: float = 1,
        timeout: int = 1200, max_polls: int = 1200,
        state_directory: Path | None = None,
    ) -> None:
        parsed = urllib.parse.urlparse(server)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
            raise ValueError("server must be an explicit localhost HTTP listener")
        if Path(client).name != "remote_agent_client.sh":
            raise ValueError("client must identify remote_agent_client.sh")
        if Path(validation_wrapper).name != "catlass-validation.sh":
            raise ValueError("validation_wrapper must identify catlass-validation.sh")
        workspace = PurePosixPath(remote_workspace)
        if not workspace.is_absolute() or ".." in workspace.parts:
            raise ValueError("remote_workspace must be an absolute normalized path")
        if type(physical_device) is not int or physical_device < 0:
            raise ValueError("physical_device must be non-negative")
        self._client = client
        self._server = server.rstrip("/")
        self._wrapper = validation_wrapper
        self._remote = remote
        self._workspace = workspace
        self._device = physical_device
        self._health = health_reader or self._read_health
        self._run = process_runner
        self._sleep = sleeper
        self._poll_interval = poll_interval
        self._timeout = timeout
        self._max_polls = max_polls
        self._state_root = state_directory.resolve() if state_directory is not None else None

    def build_bundle(self, plan: ExecutionPlan, destination: Path) -> CandidateBundle:
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("plan must be an A3 ExecutionPlan")
        input_value = {
            "input_a": plan.input_a, "input_b": plan.input_b,
            "logical_length": plan.logical_length, "padded_length": plan.padded_length,
            "block_count": plan.block_count,
        }
        members = {item.relative_path: item.content.encode() for item in plan.files}
        members["input.json"] = json.dumps(
            input_value, sort_keys=True, separators=(",", ":")
        ).encode()
        manifest = {
            "archive_schema": "rsi-a3-candidate-v1",
            "execution_id": plan.execution_id,
            "source_fingerprint": plan.source_fingerprint,
            "files": {name: hashlib.sha256(data).hexdigest()
                      for name, data in sorted(members.items())},
        }
        members["manifest.json"] = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream, tarfile.open(fileobj=stream, mode="w") as archive:
            for name, data in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = info.uid = info.gid = 0
                info.mode = 0o644
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(data))
        return CandidateBundle(
            destination, hashlib.sha256(destination.read_bytes()).hexdigest(), plan.execution_id
        )

    def compile(
        self, plan: ExecutionPlan, local_directory: Path
    ) -> CandidateCompilation | FailedEvidence:
        self._require_execution_profile(plan)
        self._bind_state_root(local_directory)
        pending = self._load_pending(plan, "compile")
        if pending is None:
            try:
                self._require_listener()
                bundle = self.build_bundle(plan, local_directory / f"{plan.execution_id}.tar")
                remote_archive = self._remote_archive_destination(plan)
                remote_directory = self.remote_candidate_directory(plan)
                transfer_id = self._upload(bundle.path, remote_archive)
                self._poll_transfer(transfer_id)
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                return FailedEvidence.create(
                    plan, stage="prepare", error_type=type(exc).__name__, detail=str(exc)
                )
            argv = self._compile_argv(plan, bundle, remote_archive, remote_directory)
        else:
            argv = self._observe_argv(plan, "compile", pending["handle"])
        try:
            completed = self._run_operation(plan, "compile", argv, pending)
        except _PendingObservation:
            raise
        except RuntimeError as exc:
            return FailedEvidence.create(
                plan, stage="compile", error_type="RuntimeError", detail=str(exc)
            )
        if completed.returncode != 0:
            return FailedEvidence.create(
                plan, stage="compile", error_type=self._operation_error_type(completed),
                detail=(completed.stderr or completed.stdout or "remote execution failed").strip(),
            )
        try:
            self._validate_wrapper(completed.stdout)
            libraries = self._markers(completed.stdout, "A3CANDIDATE_COMPILED=")
            if len(libraries) != 1 or re.fullmatch(r"[0-9a-f]{64}", libraries[0]) is None:
                raise ValueError("remote compile marker is invalid")
        except ValueError as exc:
            return FailedEvidence.create(
                plan, stage="compile", error_type="CompileOutputError", detail=str(exc)
            )
        body = {
            "plan": plan, "library_sha256": libraries[0],
            "stdout": completed.stdout, "stderr": completed.stderr,
        }
        result = CandidateCompilation(
            plan, libraries[0], completed.stdout, completed.stderr, attest(body)
        )
        self._pending_path(plan, "compile").unlink(missing_ok=True)
        return result

    def execute(
        self, compilation: CandidateCompilation
    ) -> VerifiedResult | FailedEvidence:
        if not isinstance(compilation, CandidateCompilation):
            raise TypeError("execute requires a CandidateCompilation")
        plan = compilation.plan
        self._require_execution_profile(plan)
        directory = self.remote_candidate_directory(plan)
        pending = self._load_pending(plan, "execute")
        argv = (
            self._execute_argv(plan, directory, compilation.library_sha256)
            if pending is None else self._observe_argv(plan, "execute", pending["handle"])
        )
        try:
            completed = self._run_operation(plan, "execute", argv, pending)
        except _PendingObservation:
            raise
        except RuntimeError as exc:
            return FailedEvidence.create(
                plan, stage="execute", error_type="RuntimeError", detail=str(exc)
            )
        if completed.returncode != 0:
            return FailedEvidence.create(
                plan, stage="execute", error_type=self._operation_error_type(completed),
                detail=(completed.stderr or completed.stdout or "remote execution failed").strip(),
            )
        try:
            handle = self._validate_wrapper(completed.stdout)
            output = self._parse_output(completed.stdout, plan.padded_length)
        except ValueError as exc:
            return FailedEvidence.create(
                plan, stage="verify", error_type="OutputError", detail=str(exc)
            )
        expected = tuple(a + b for a, b in zip(plan.input_a, plan.input_b))
        maximum = max(
            abs(output[index] - expected[index])
            for index in range(plan.logical_length)
        )
        receipt = ExecutionReceipt(
            exit_code=0, output=output, stdout=completed.stdout,
            stderr=completed.stderr, job_handle=handle,
            metadata=(
                ("library_sha256", compilation.library_sha256),
                ("remote_candidate_directory", directory.as_posix()),
            ),
        )
        result = VerifiedResult.from_receipt(plan, receipt, max_abs_error=maximum)
        self._pending_path(plan, "execute").unlink(missing_ok=True)
        return result

    def _operation_error_type(
        self, completed: subprocess.CompletedProcess[str],
    ) -> str:
        return "RemoteExecutionError"

    def run(self, plan: ExecutionPlan, local_directory: Path) -> VerifiedResult | FailedEvidence:
        compiled = self.compile(plan, local_directory)
        return compiled if isinstance(compiled, FailedEvidence) else self.execute(compiled)

    def remote_candidate_directory(self, plan: ExecutionPlan) -> PurePosixPath:
        """Return the retained directory consumed by the A3 profiling backend."""

        if not isinstance(plan, ExecutionPlan):
            raise TypeError("plan must be an A3 ExecutionPlan")
        return self._workspace / ".rsi-a3" / "candidates" / plan.execution_id

    def _remote_archive_destination(self, plan: ExecutionPlan) -> PurePosixPath:
        return self._workspace / ".rsi-a3" / "uploads" / f"{plan.execution_id}.tar"

    @staticmethod
    def _require_execution_profile(plan: ExecutionPlan) -> None:
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("plan must be an A3 ExecutionPlan")
        if plan.execution_profile != "gz-a3":
            raise ValueError("GZ-A3 backend requires a gz-a3 execution profile")

    def _bind_state_root(self, local_directory: Path) -> None:
        selected = (local_directory / ".a3-remote-state").resolve()
        if self._state_root is None:
            self._state_root = selected

    def _pending_path(self, plan: ExecutionPlan, operation: str) -> Path:
        if self._state_root is None:
            raise RuntimeError("remote candidate state directory is not configured")
        return self._state_root / plan.execution_id / f"{operation}.json"

    def _load_pending(self, plan: ExecutionPlan, operation: str) -> dict[str, str] | None:
        path = self._pending_path(plan, operation)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("retained remote candidate operation is corrupt") from exc
        expected = {
            "execution_id": plan.execution_id, "source_fingerprint": plan.source_fingerprint,
            "operation": operation,
        }
        if (
            type(value) is not dict or set(value) != {*expected, "handle"}
            or any(value.get(key) != item for key, item in expected.items())
            or type(value.get("handle")) is not str
            or not value["handle"].startswith("gz-a3:")
        ):
            raise RuntimeError("retained remote candidate operation is corrupt")
        return value

    def _publish_pending(
        self, plan: ExecutionPlan, operation: str, handle: str
    ) -> None:
        path = self._pending_path(plan, operation)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps({
            "execution_id": plan.execution_id,
            "source_fingerprint": plan.source_fingerprint,
            "operation": operation, "handle": handle,
        }, sort_keys=True, separators=(",", ":")) + "\n"
        if path.exists():
            if path.read_text() != data:
                raise RuntimeError("pending remote candidate operation conflicts")
            return
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=".pending-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
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
                    f"remote candidate observation incomplete; retry retained handle {pending['handle']}"
                ) from exc
            raise
        handles = self._markers(completed.stdout, "CATLASS_VALIDATION_HANDLE=")
        states = self._markers(completed.stdout, "CATLASS_VALIDATION_STATE=")
        profiles = self._markers(completed.stdout, "CATLASS_VALIDATION_PROFILE=")
        if (
            completed.returncode != 0 and profiles == ["gz-a3"]
            and states == ["observation-unavailable"] and len(handles) == 1
            and handles[0].startswith("gz-a3:")
        ):
            if pending is not None and handles[0] != pending["handle"]:
                raise RuntimeError("remote candidate observation returned a foreign handle")
            self._publish_pending(plan, operation, handles[0])
            raise _PendingObservation(
                f"remote candidate observation unavailable; retry retained handle {handles[0]}"
            )
        if completed.returncode == 0 and pending is not None and handles != [pending["handle"]]:
            raise RuntimeError("remote candidate observation returned a foreign handle")
        if completed.returncode != 0 and pending is not None and states != ["failed"]:
            raise _PendingObservation(
                f"remote candidate observation incomplete; retry retained handle {pending['handle']}"
            )
        if completed.returncode != 0 and states == ["failed"]:
            self._pending_path(plan, operation).unlink(missing_ok=True)
        return completed

    def _read_health(self) -> Mapping[str, object]:
        with urllib.request.urlopen(self._server + "/api/v1/healthz", timeout=5) as response:
            return json.load(response)

    def _require_listener(self) -> None:
        value = self._health()
        metadata = value.get("session_metadata") if isinstance(value, Mapping) else None
        if not isinstance(metadata, Mapping) or metadata.get("execution_mode") != "listener":
            raise RuntimeError("remote-agent listener execution mode is required")
        if metadata.get("listener_ssh_transport") != "asyncssh":
            raise RuntimeError("remote-agent listener requires asyncssh transfer transport")

    def _client_call(self, *arguments: str) -> dict[str, object]:
        completed = self._call((self._client, "--server", self._server, *arguments))
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout or "remote-agent client failed").strip())
        value = json.loads(completed.stdout)
        if not isinstance(value, dict):
            raise RuntimeError("remote-agent client returned a non-object response")
        return value

    def _upload(self, source: Path, destination: PurePosixPath) -> str:
        value = self._client_call(
            "transfer", "upload", "--remote", self._remote,
            "--src", str(source), "--dst", destination.relative_to(self._workspace).as_posix(),
        )
        job_id = value.get("id")
        if type(job_id) is not str or not job_id:
            raise RuntimeError("transfer upload returned no job id")
        return job_id

    def _poll_transfer(self, job_id: str) -> None:
        for _ in range(self._max_polls):
            value = self._client_call("transfer", "status", "--job-id", job_id)
            if value.get("id") != job_id:
                raise RuntimeError("transfer status returned a foreign job")
            status = value.get("status")
            if status == "succeeded":
                return
            if status in {"failed", "cancelled"}:
                raise RuntimeError(f"transfer {job_id} ended with {status}")
            if status not in {"queued", "running"}:
                raise RuntimeError("transfer returned an unknown status")
            self._sleep(self._poll_interval)
        raise RuntimeError(f"transfer {job_id} did not reach a terminal state")

    def _compile_argv(
        self, plan: ExecutionPlan, bundle: CandidateBundle,
        archive: PurePosixPath, directory: PurePosixPath,
    ) -> tuple[str, ...]:
        return (
            self._wrapper, "--profile", "gz-a3", "--operation",
            f"a3-candidate-{plan.execution_id[:16]}", "run", "--native",
            "--runtime", "py311-torch", "--device", str(self._device),
            "--timeout", str(self._timeout), "--", "bash", "-c", _COMPILE_SCRIPT,
            "rsi-a3-candidate", archive.as_posix(), directory.as_posix(), bundle.sha256,
            plan.execution_id, plan.source_fingerprint,
        )

    def _execute_argv(
        self, plan: ExecutionPlan, directory: PurePosixPath, library_sha256: str,
    ) -> tuple[str, ...]:
        return (
            self._wrapper, "--profile", "gz-a3", "--operation",
            f"a3-execute-{plan.execution_id[:16]}", "run", "--native",
            "--runtime", "py311-torch", "--device", str(self._device),
            "--timeout", str(self._timeout), "--", "bash", "-c", _EXECUTE_SCRIPT,
            "rsi-a3-candidate", directory.as_posix(), plan.execution_id,
            plan.source_fingerprint, library_sha256,
        )

    def _observe_argv(
        self, plan: ExecutionPlan, operation: str, handle: str,
    ) -> tuple[str, ...]:
        prefix = "a3-candidate" if operation == "compile" else "a3-execute"
        return (
            self._wrapper, "--profile", "gz-a3", "--operation",
            f"{prefix}-{plan.execution_id[:16]}", "observe", "--handle", handle,
        )

    def _call(self, argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        try:
            return self._run(
                argv, text=True, capture_output=True, check=False, timeout=self._timeout
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("managed A3 operation timed out") from exc

    @staticmethod
    def _markers(stdout: str, prefix: str) -> list[str]:
        return [line.removeprefix(prefix) for line in stdout.splitlines() if line.startswith(prefix)]

    @classmethod
    def _validate_wrapper(cls, stdout: str) -> str:
        expected = {
            "CATLASS_VALIDATION_PROFILE=": "gz-a3",
            "CATLASS_VALIDATION_STATE=": "completed",
            "CATLASS_VALIDATION_EXIT=": "0",
        }
        if any(cls._markers(stdout, prefix) != [value] for prefix, value in expected.items()):
            raise ValueError("neutral wrapper did not return a successful GZ-A3 terminal state")
        handles = cls._markers(stdout, "CATLASS_VALIDATION_HANDLE=")
        if len(handles) != 1 or not handles[0].startswith("gz-a3:"):
            raise ValueError("neutral wrapper returned an invalid GZ-A3 handle")
        return handles[0]

    @classmethod
    def _parse_output(cls, stdout: str, length: int) -> tuple[float, ...]:
        records = cls._markers(stdout, "A3KERNEL_OUTPUT=")
        if len(records) != 1:
            raise ValueError("remote host must emit one A3KERNEL_OUTPUT marker")
        value = json.loads(records[0])
        if (
            not isinstance(value, list) or len(value) != length
            or any(isinstance(item, bool) or not isinstance(item, (int, float))
                   or not math.isfinite(item) for item in value)
        ):
            raise ValueError("remote output must be a finite padded numeric array")
        return tuple(float(item) for item in value)

__all__ = ["CandidateBundle", "GZA3RemoteCandidateBackend"]
