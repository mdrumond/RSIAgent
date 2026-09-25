"""Adapter for durable execution through the checked-in BZ-A5 wrapper.

The command executor owns staging the declared files on the remote host.  This
module deliberately produces an argv tuple rather than a shell string, keeping
transport details mockable and preventing generated kernel text from becoming
shell syntax.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
from typing import Callable, Protocol

from benchmarks.a5kernels.protocol import ExecutionPlan, ExecutionReceipt, SourceFile


OUTPUT_MARKER = "A5KERNEL_OUTPUT="


@dataclass(frozen=True)
class CommandInvocation:
    argv: tuple[str, ...]
    files: tuple[SourceFile, ...]
    stdin: str
    remote_directory: str


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str = ""
    session_handle: str | None = None


class BZCommandExecutor(Protocol):
    """Stages an invocation and calls only the configured profile wrapper."""

    def run(self, invocation: CommandInvocation) -> CommandResult: ...

    def inspect(self, argv: tuple[str, ...]) -> CommandResult: ...

    @property
    def runtime_provenance(self) -> tuple[tuple[str, str], ...]: ...


class RuntimeUnavailableError(RuntimeError):
    """Raised before dispatch when no real language driver is registered."""


class ProfileCommandExecutor:
    """Stage registry files with the profile uploader, then run one session."""

    def __init__(
        self,
        *,
        upload_wrapper: str,
        process_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if Path(upload_wrapper).name != "upload.sh":
            raise ValueError("upload_wrapper must identify the checked-in upload.sh")
        self._upload_wrapper = upload_wrapper
        self._run = process_runner

    @property
    def runtime_provenance(self) -> tuple[tuple[str, str], ...]:
        return ()

    def run(self, invocation: CommandInvocation) -> CommandResult:
        with tempfile.TemporaryDirectory(prefix="a5kernel-") as temporary:
            stage = Path(temporary)
            for source in invocation.files:
                relative = PurePosixPath(source.relative_path)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"unsafe fixture path: {source.relative_path!r}")
                destination = stage.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(source.content, encoding="utf-8")
            uploaded = self._call(
                (self._upload_wrapper, "--recursive", f"{stage}/", invocation.remote_directory)
            )
        if uploaded.returncode != 0:
            return CommandResult(uploaded.returncode, uploaded.stdout, uploaded.stderr)

        completed = self._dispatch(invocation)
        handle = f"bz-a5:{_session_name(invocation.argv)}"
        return CommandResult(
            completed.returncode, completed.stdout, completed.stderr, handle
        )

    def _dispatch(self, invocation: CommandInvocation):
        return self._call(invocation.argv, stdin=invocation.stdin)

    def inspect(self, argv: tuple[str, ...]) -> CommandResult:
        completed = self._call(argv)
        return CommandResult(
            completed.returncode,
            completed.stdout,
            completed.stderr,
            f"bz-a5:{_session_name(argv)}",
        )

    def _call(self, argv: tuple[str, ...], *, stdin: str | None = None):
        return self._run(
            argv,
            input=stdin,
            text=True,
            capture_output=True,
            check=False,
        )


class CatlassValidationExecutor(ProfileCommandExecutor):
    """Run staged Catlass fixtures against one explicit retained revision."""

    def __init__(
        self,
        *,
        upload_wrapper: str,
        validation_wrapper: str,
        catlass_source: str,
        catlass_revision: str,
        process_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        super().__init__(
            upload_wrapper=upload_wrapper, process_runner=process_runner
        )
        if Path(validation_wrapper).name != "catlass-validation.sh":
            raise ValueError(
                "validation_wrapper must identify the checked-in catlass-validation.sh"
            )
        if not PurePosixPath(catlass_source).is_absolute():
            raise ValueError("catlass_source must be an absolute retained BZ path")
        if not _is_revision(catlass_revision):
            raise ValueError("catlass_revision must be a lowercase 40-character SHA")
        self._validation_wrapper = validation_wrapper
        self._provenance_wrapper = str(
            Path(validation_wrapper).parent / "bz-a5" / "catlass-provenance.sh"
        )
        self._catlass_source = catlass_source
        self._catlass_revision = catlass_revision
        self._runtime_provenance: tuple[tuple[str, str], ...] | None = None

    @property
    def runtime_provenance(self) -> tuple[tuple[str, str], ...]:
        if self._runtime_provenance is not None:
            return self._runtime_provenance
        completed = self._call(
            (
                self._provenance_wrapper,
                "--catlass-src",
                self._catlass_source,
                "--revision",
                self._catlass_revision,
            )
        )
        if completed.returncode != 0:
            raise RuntimeUnavailableError(
                f"Catlass runtime provenance probe failed: {completed.stderr.strip()}"
            )
        records = [
            line.removeprefix("CATLASS_RUNTIME_PROVENANCE=")
            for line in completed.stdout.splitlines()
            if line.startswith("CATLASS_RUNTIME_PROVENANCE=")
        ]
        if len(records) != 1:
            raise RuntimeUnavailableError("Catlass runtime provenance probe returned no unique record")
        try:
            record = json.loads(records[0])
        except json.JSONDecodeError as exc:
            raise RuntimeUnavailableError("Catlass runtime provenance probe returned invalid JSON") from exc
        keys = {
            "manifest_sha256",
            "bridge_sha256",
            "ascendnpu_ir_gitlink",
            "ascendnpu_ir_install_commit",
            "cann_version",
        }
        if not isinstance(record, dict) or set(record) != keys:
            raise RuntimeUnavailableError("Catlass runtime provenance probe returned invalid fields")
        if any(not isinstance(record[key], str) or not record[key] for key in keys):
            raise RuntimeUnavailableError("Catlass runtime provenance probe returned empty fields")
        if record["ascendnpu_ir_gitlink"] != record["ascendnpu_ir_install_commit"]:
            raise RuntimeUnavailableError("Catlass dependency provenance probe mismatch")
        self._runtime_provenance = (
            ("catlass_revision", self._catlass_revision),
            ("catlass_source", self._catlass_source),
            ("execution_profile", "bz-a5"),
            *(tuple((key, record[key]) for key in sorted(keys))),
        )
        return self._runtime_provenance

    def _dispatch(self, invocation: CommandInvocation):
        operation = _session_name(invocation.argv)
        timeout = _option_value(invocation.argv, "--observe-timeout")
        try:
            command_index = invocation.argv.index("--") + 1
        except ValueError as exc:
            raise ValueError("session invocation is missing command separator") from exc
        provenance = dict(self.runtime_provenance)
        command = (
            *invocation.argv[command_index:],
            self._catlass_revision,
            provenance["manifest_sha256"],
            provenance["bridge_sha256"],
            provenance["ascendnpu_ir_gitlink"],
            provenance["ascendnpu_ir_install_commit"],
            provenance["cann_version"],
        )
        return self._call(
            (
                self._validation_wrapper,
                "--profile",
                "bz-a5",
                "--operation",
                operation,
                "run",
                "--catlass-src",
                self._catlass_source,
                "--timeout",
                timeout,
                "--",
                *command,
            ),
            stdin=invocation.stdin,
        )

class BZSessionAdapter:
    """Translate execution plans into named durable BZ-A5 sessions."""

    def __init__(
        self,
        command_executor: BZCommandExecutor,
        *,
        session_wrapper: str,
        observe_timeout: int = 600,
    ) -> None:
        wrapper = Path(session_wrapper)
        if wrapper.name != "session.sh":
            raise ValueError("session_wrapper must identify the checked-in session.sh")
        self._executor = command_executor
        self._wrapper = str(wrapper)
        self._observe_timeout = observe_timeout

    @property
    def runtime_provenance(self) -> tuple[tuple[str, str], ...]:
        return tuple(getattr(self._executor, "runtime_provenance", ()))

    def execute(self, plan: ExecutionPlan) -> ExecutionReceipt:
        if plan.argv is None:
            raise RuntimeUnavailableError(
                f"no executable runtime is registered for {plan.language}"
            )
        if plan.runtime_provenance != self.runtime_provenance:
            raise RuntimeUnavailableError(
                "execution plan runtime provenance does not match the configured backend"
            )
        session_name = f"codex-a5hello-{plan.execution_id}-{plan.attempt_id}"
        remote_directory = f".a5kernels/{plan.execution_id}/{plan.attempt_id}"
        payload = json.dumps(
            {"input_a": plan.input_a, "input_b": plan.input_b},
            separators=(",", ":"),
        )
        staged_files = (*plan.files, SourceFile("input.json", payload))
        staged_paths = {item.relative_path for item in staged_files}
        remote_argv = tuple(
            f"{remote_directory}/{arg}" if arg in staged_paths else arg
            for arg in plan.argv
        )
        invocation = CommandInvocation(
            argv=(
                self._wrapper,
                "--name",
                session_name,
                "run",
                "--wait",
                "--observe-timeout",
                str(self._observe_timeout),
                "--",
                *remote_argv,
            ),
            files=staged_files,
            stdin="",
            remote_directory=remote_directory,
        )
        dispatch = self._executor.run(invocation)
        if dispatch.exit_code != 0 and not _proven_observation_uncertain(
            dispatch, session_name
        ):
            return ExecutionReceipt(
                exit_code=dispatch.exit_code,
                output=(),
                stdout=dispatch.stdout,
                stderr=dispatch.stderr,
                session_handle=dispatch.session_handle,
            )
        logs = self._executor.inspect((self._wrapper, "--name", session_name, "logs"))
        result = self._executor.inspect(
            (self._wrapper, "--name", session_name, "result")
        )
        output = _parse_output(logs.stdout) if result.exit_code == 0 else ()
        return ExecutionReceipt(
            exit_code=result.exit_code,
            output=output,
            stdout=_join_output(dispatch.stdout, logs.stdout),
            stderr="\n".join(
                part for part in (dispatch.stderr, logs.stderr, result.stderr) if part
            ),
            session_handle=(
                result.session_handle
                or logs.session_handle
                or dispatch.session_handle
                or f"bz-a5:{session_name}"
            ),
        )


def _parse_output(stdout: str) -> tuple[float, ...]:
    marked = [line[len(OUTPUT_MARKER) :] for line in stdout.splitlines() if line.startswith(OUTPUT_MARKER)]
    if len(marked) != 1:
        raise ValueError("remote output must contain exactly one A5KERNEL_OUTPUT record")
    decoded = json.loads(marked[0])
    if not isinstance(decoded, list) or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in decoded
    ):
        raise ValueError("A5KERNEL_OUTPUT must be a JSON array of numbers")
    return tuple(float(value) for value in decoded)


def _session_name(argv: tuple[str, ...]) -> str:
    try:
        return argv[argv.index("--name") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("session invocation is missing --name") from exc


def _option_value(argv: tuple[str, ...], option: str) -> str:
    try:
        return argv[argv.index(option) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"invocation is missing {option}") from exc


def _is_revision(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _proven_observation_uncertain(result: CommandResult, session_name: str) -> bool:
    return (
        result.exit_code == 75
        and "CATLASS_VALIDATION_STATE=observation-unavailable" in result.stdout
        and f"CATLASS_VALIDATION_HANDLE=bz-a5:{session_name}" in result.stdout
    )


def _join_output(*parts: str) -> str:
    return "\n".join(part.rstrip("\n") for part in parts if part)
