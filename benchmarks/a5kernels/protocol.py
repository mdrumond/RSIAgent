"""Immutable messages crossing the A5 execution boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Any, Mapping


def canonical_hash(value: Mapping[str, Any]) -> str:
    """Hash JSON with stable ordering and no presentation whitespace."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RunRequest:
    """Host-owned description of one deterministic vector-add run."""

    language: str
    length: int = 32
    seed: int = 0
    dtype: str = "float32"
    # Optional host-owned verification extent; extra inputs are exactly zero.
    padded_length: int | None = None

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ValueError("length must be positive")
        if self.dtype != "float32":
            raise ValueError("the hello fixture supports only float32")
        if self.padded_length is not None and (
            type(self.padded_length) is not int or self.padded_length < self.length
        ):
            raise ValueError("padded_length must be an integer at least length")

    def to_dict(self) -> dict[str, Any]:
        fields = asdict(self)
        if self.padded_length is None:
            fields.pop("padded_length")
        return fields

    @property
    def request_id(self) -> str:
        return canonical_hash(self.to_dict())


@dataclass(frozen=True)
class SourceFile:
    relative_path: str
    content: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class ExecutionEnvironment:
    """Canonical environment policy rendered only at an execution boundary."""

    bindings: tuple[tuple[str, str], ...] = ()
    unset: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        bindings = tuple(self.bindings)
        unset = tuple(self.unset)
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(type(value) is str for value in item)
            for item in bindings
        ):
            raise TypeError("environment bindings must be string pairs")
        names = [item[0] for item in bindings]
        if any(type(name) is not str for name in unset):
            raise TypeError("environment unset names must be strings")
        if any(_ENVIRONMENT_NAME.fullmatch(name) is None for name in (*names, *unset)):
            raise ValueError("environment variable names must be valid identifiers")
        if len(names) != len(set(names)) or len(unset) != len(set(unset)):
            raise ValueError("environment variable declarations must be unique")
        if set(names).intersection(unset):
            raise ValueError("environment variables cannot be both bound and unset")
        object.__setattr__(self, "bindings", tuple(sorted(bindings)))
        object.__setattr__(self, "unset", tuple(sorted(unset)))

    def with_binding(self, name: str, value: str) -> ExecutionEnvironment:
        remaining = tuple(item for item in self.bindings if item[0] != name)
        return ExecutionEnvironment((*remaining, (name, value)), tuple(
            item for item in self.unset if item != name
        ))

    def with_unset(self, name: str) -> ExecutionEnvironment:
        remaining = tuple(item for item in self.bindings if item[0] != name)
        return ExecutionEnvironment(remaining, (*(
            item for item in self.unset if item != name
        ), name))

    def render(self, argv: tuple[str, ...]) -> tuple[str, ...]:
        if not self.bindings and not self.unset:
            return argv
        options = tuple(part for name in self.unset for part in ("-u", name))
        assignments = tuple(f"{name}={value}" for name, value in self.bindings)
        return ("env", *options, *assignments, *argv)


@dataclass(frozen=True)
class ExecutionPlan:
    """Complete, registry-generated workload submitted to the BZ adapter."""

    request_id: str
    attempt_id: str
    language: str
    files: tuple[SourceFile, ...]
    argv: tuple[str, ...] | None
    input_a: tuple[float, ...]
    input_b: tuple[float, ...]
    runtime_provenance: tuple[tuple[str, str], ...] = ()
    environment: ExecutionEnvironment = ExecutionEnvironment()

    def __post_init__(self) -> None:
        if not isinstance(self.environment, ExecutionEnvironment):
            raise TypeError("environment must be an ExecutionEnvironment")
        if self.argv and PurePosixPath(self.argv[0]).name == "env":
            raise ValueError("execution argv must not contain an env wrapper")

    @property
    def execution_id(self) -> str:
        return canonical_hash(
            {
                "argv": self.argv,
                "environment": asdict(self.environment),
                "input_a": self.input_a,
                "input_b": self.input_b,
                "request_id": self.request_id,
                "runtime_provenance": self.runtime_provenance,
                "source_fingerprint": self.source_fingerprint,
            }
        )

    @property
    def source_fingerprint(self) -> str:
        return canonical_hash(
            {item.relative_path: item.sha256 for item in self.files}
        )


@dataclass(frozen=True)
class ExecutionReceipt:
    """Untrusted raw data returned by remote execution.

    ``metadata`` may contain agent-authored notes or claimed scores. It is
    retained for auditability but is never read by the correctness oracle.
    """

    exit_code: int
    output: tuple[float, ...]
    stdout: str = ""
    stderr: str = ""
    session_handle: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class VerifiedResult:
    """Authoritative host-generated result packet."""

    request_id: str
    execution_id: str
    attempt_id: str
    language: str
    runtime_provenance: tuple[tuple[str, str], ...]
    passed: bool
    max_abs_error: float
    exit_code: int
    output_sha256: str
    source_fingerprint: str
    evidence_sha256: str
    session_handle: str | None
    attestation_sha256: str
