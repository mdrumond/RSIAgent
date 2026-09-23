"""Immutable messages crossing the A5 execution boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
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

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ValueError("length must be positive")
        if self.dtype != "float32":
            raise ValueError("the hello fixture supports only float32")

    @property
    def request_id(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class SourceFile:
    relative_path: str
    content: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


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

    @property
    def execution_id(self) -> str:
        return canonical_hash(
            {
                "request_id": self.request_id,
                "runtime_provenance": self.runtime_provenance,
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
