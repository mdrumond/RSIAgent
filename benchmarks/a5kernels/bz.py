"""Adapter for durable execution through the checked-in BZ-A5 wrapper.

The command executor owns staging the declared files on the remote host.  This
module deliberately produces an argv tuple rather than a shell string, keeping
transport details mockable and preventing generated kernel text from becoming
shell syntax.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

from benchmarks.a5kernels.protocol import ExecutionPlan, ExecutionReceipt, SourceFile


OUTPUT_MARKER = "A5KERNEL_OUTPUT="


@dataclass(frozen=True)
class CommandInvocation:
    argv: tuple[str, ...]
    files: tuple[SourceFile, ...]
    stdin: str


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


class RuntimeUnavailableError(RuntimeError):
    """Raised before dispatch when no real language driver is registered."""


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

    def execute(self, plan: ExecutionPlan) -> ExecutionReceipt:
        if plan.argv is None:
            raise RuntimeUnavailableError(
                f"no executable runtime is registered for {plan.language}"
            )
        session_name = f"codex-a5hello-{plan.request_id[:8]}-{plan.attempt_id}"
        payload = json.dumps(
            {"input_a": plan.input_a, "input_b": plan.input_b},
            separators=(",", ":"),
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
                *plan.argv,
            ),
            files=plan.files,
            stdin=payload,
        )
        dispatch = self._executor.run(invocation)
        logs = self._executor.inspect((self._wrapper, "--name", session_name, "logs"))
        result = self._executor.inspect(
            (self._wrapper, "--name", session_name, "result")
        )
        output = _parse_output(logs.stdout) if result.exit_code == 0 else ()
        return ExecutionReceipt(
            exit_code=result.exit_code,
            output=output,
            stdout=logs.stdout,
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
