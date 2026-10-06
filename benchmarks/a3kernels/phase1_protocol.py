"""A3-owned immutable contracts for Phase 1 Ascend C experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import PurePosixPath
import hashlib
import math
from typing import Any, Mapping

from benchmarks.a3kernels.phase1_evidence import (
    A3_EXECUTION_PROFILES,
    canonical_digest,
)


A3_TARGET = "Ascend910B4"
A3_LANGUAGE = "ascend-c"
# Compatibility default for aggregate candidate helpers. ExecutionPlan keeps
# execution_profile required so every persisted identity names its route.
A3_EXECUTION_PROFILE = "gz-a3"
A3_RUNTIME = "native-ascend-c"


def attest(payload: Mapping[str, Any]) -> str:
    """Return a host-reproducible attestation over authoritative fields."""

    return canonical_digest(payload)


@dataclass(frozen=True, order=True)
class SourceFile:
    relative_path: str
    content: str

    def __post_init__(self) -> None:
        if type(self.relative_path) is not str or not self.relative_path:
            raise ValueError("source path must be a non-empty relative path")
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() == "."
        ):
            raise ValueError("source path must be a non-empty relative path")
        if type(self.content) is not str:
            raise TypeError("source content must be a string")
        object.__setattr__(self, "relative_path", path.as_posix())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExecutionPlan:
    """One host-owned A3 workload; agents may supply only staged sources."""

    request_id: str
    attempt_id: str
    project_id: str
    files: tuple[SourceFile, ...]
    argv: tuple[str, ...]
    input_a: tuple[float, ...]
    input_b: tuple[float, ...]
    execution_profile: str
    target: str = A3_TARGET
    language: str = A3_LANGUAGE
    runtime: str = A3_RUNTIME
    logical_device: int = 0
    logical_length: int | None = None
    padded_length: int | None = None
    block_count: int = 1
    verify_padding: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "input_a", tuple(self.input_a))
        object.__setattr__(self, "input_b", tuple(self.input_b))
        for name in ("request_id", "attempt_id", "project_id"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if self.target != A3_TARGET:
            raise ValueError(f"A3 target must be {A3_TARGET}")
        if self.language != A3_LANGUAGE:
            raise ValueError(f"A3 language must be {A3_LANGUAGE}; Catlass is unsupported")
        if self.execution_profile not in A3_EXECUTION_PROFILES:
            raise ValueError(
                "A3 execution profile must be bz-a3-1, bz-a3-2, or "
                "the gz-a3 compatibility profile"
            )
        if self.runtime != A3_RUNTIME:
            raise ValueError(f"A3 runtime must be {A3_RUNTIME}")
        if type(self.logical_device) is not int or self.logical_device != 0:
            raise ValueError("A3 logical device must be 0")
        if not self.files or any(not isinstance(item, SourceFile) for item in self.files):
            raise ValueError("files must contain at least one SourceFile")
        paths = [item.relative_path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("source paths must be unique")
        if not self.argv or any(type(item) is not str for item in self.argv):
            raise ValueError("argv must contain strings")
        if len(self.input_a) != len(self.input_b):
            raise ValueError("input vectors must have equal lengths")
        if self.logical_length is None:
            object.__setattr__(self, "logical_length", len(self.input_a))
        if self.padded_length is None:
            object.__setattr__(self, "padded_length", len(self.input_a))
        if (
            type(self.logical_length) is not int
            or type(self.padded_length) is not int
            or not 1 <= self.logical_length <= self.padded_length <= 4096
            or len(self.input_a) != self.padded_length
        ):
            raise ValueError(
                "logical_length and padded_length must describe the staged inputs"
            )
        if type(self.block_count) is not int or not 1 <= self.block_count <= 32:
            raise ValueError("block_count must be an integer in [1, 32]")
        if type(self.verify_padding) is not bool:
            raise ValueError("verify_padding must be a boolean")
        object.__setattr__(self, "files", tuple(sorted(self.files)))

    @property
    def source_fingerprint(self) -> str:
        return canonical_digest(
            {item.relative_path: item.sha256 for item in self.files}
        )

    @property
    def execution_id(self) -> str:
        value = {
                "argv": self.argv,
                "attempt_id": self.attempt_id,
                "execution_profile": self.execution_profile,
                "input_a": self.input_a,
                "input_b": self.input_b,
                "language": self.language,
                "logical_length": self.logical_length,
                "logical_device": self.logical_device,
                "padded_length": self.padded_length,
                "block_count": self.block_count,
                "project_id": self.project_id,
                "request_id": self.request_id,
                "runtime": self.runtime,
                "source_fingerprint": self.source_fingerprint,
                "target": self.target,
            }
        # Preserve the identity of retained Phase 1 logical-only evidence.
        if self.verify_padding:
            value["verify_padding"] = True
        return canonical_digest(value)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_fingerprint"] = self.source_fingerprint
        value["execution_id"] = self.execution_id
        return value


@dataclass(frozen=True)
class ExecutionReceipt:
    """Untrusted process output retained for audit, never used as a claim."""

    exit_code: int
    output: tuple[float, ...] = ()
    stdout: str = ""
    stderr: str = ""
    job_handle: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.exit_code) is not int:
            raise TypeError("exit_code must be an integer")
        object.__setattr__(self, "output", tuple(self.output))
        object.__setattr__(self, "metadata", tuple(tuple(item) for item in self.metadata))


@dataclass(frozen=True)
class VerificationMismatch:
    logical_index: int
    input_a: float
    input_b: float
    actual: float
    expected: float
    absolute_error: float

    def __post_init__(self) -> None:
        if type(self.logical_index) is not int or self.logical_index < 0:
            raise ValueError("logical_index must be a non-negative integer")
        for name in (
            "input_a", "input_b", "actual", "expected", "absolute_error",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, float(value))
        if self.expected != self.input_a + self.input_b:
            raise ValueError("expected must equal input_a plus input_b")
        if (
            self.absolute_error < 0
            or self.absolute_error != abs(self.actual - self.expected)
        ):
            raise ValueError(
                "absolute_error must equal the actual-to-expected difference"
            )


@dataclass(frozen=True)
class VerifiedResult:
    request_id: str
    execution_id: str
    attempt_id: str
    project_id: str
    passed: bool
    max_abs_error: float | None
    tolerance: float
    exit_code: int
    output_sha256: str
    source_fingerprint: str
    evidence_sha256: str
    job_handle: str | None
    attestation_sha256: str
    mismatch: VerificationMismatch | None = None
    padding_max_abs_error: float | None = None
    padding_mismatch_index: int | None = None

    def __post_init__(self) -> None:
        if self.mismatch is not None and not isinstance(
            self.mismatch, VerificationMismatch
        ):
            raise TypeError("mismatch must be a VerificationMismatch")
        if self.mismatch is not None and (self.passed or self.exit_code != 0):
            raise ValueError("only numerical verification failure has a mismatch")
        if self.mismatch is not None and (
            self.max_abs_error is None
            or self.mismatch.absolute_error != self.max_abs_error
        ):
            raise ValueError("mismatch absolute_error must equal max_abs_error")
        if (
            self.exit_code == 0
            and self.max_abs_error is not None
            and self.max_abs_error > self.tolerance
            and self.mismatch is None
        ):
            raise ValueError("failed numerical verification requires a mismatch")
        if self.padding_max_abs_error is not None and (
            type(self.padding_max_abs_error) not in (int, float)
            or not math.isfinite(self.padding_max_abs_error)
            or self.padding_max_abs_error < 0
        ):
            raise ValueError("padding_max_abs_error must be finite and non-negative")
        if self.padding_mismatch_index is not None and (
            type(self.padding_mismatch_index) is not int
            or self.padding_mismatch_index < 0
            or self.padding_max_abs_error is None
            or self.padding_max_abs_error <= self.tolerance
        ):
            raise ValueError("padding mismatch must identify a failed padded element")
        if (
            self.padding_max_abs_error is not None
            and self.padding_max_abs_error > self.tolerance
            and self.padding_mismatch_index is None
        ):
            raise ValueError("failed padding verification requires a mismatch index")
        if self.passed and (
            self.padding_max_abs_error is not None
            and self.padding_max_abs_error > self.tolerance
        ):
            raise ValueError("passed verification cannot contain padding corruption")

    @classmethod
    def from_receipt(
        cls,
        plan: ExecutionPlan,
        receipt: ExecutionReceipt,
        *,
        max_abs_error: float | None,
        tolerance: float = 1e-5,
    ) -> "VerifiedResult":
        if (
            type(tolerance) not in (int, float)
            or not math.isfinite(tolerance)
            or tolerance < 0
        ):
            raise ValueError("tolerance must be a finite non-negative number")
        if max_abs_error is not None and (
            type(max_abs_error) not in (int, float)
            or not math.isfinite(max_abs_error)
            or max_abs_error < 0
        ):
            raise ValueError("max_abs_error must be a finite non-negative number")
        if receipt.exit_code != 0 and max_abs_error is not None:
            raise ValueError("a failed exit code cannot have a verification metric")
        mismatch = None
        if max_abs_error is not None:
            if len(receipt.output) != plan.padded_length:
                raise ValueError(
                    "verification metric requires the registered padded output"
                )
            samples = tuple(
                VerificationMismatch(
                    logical_index=index,
                    input_a=plan.input_a[index],
                    input_b=plan.input_b[index],
                    actual=receipt.output[index],
                    expected=plan.input_a[index] + plan.input_b[index],
                    absolute_error=abs(
                        receipt.output[index]
                        - (plan.input_a[index] + plan.input_b[index])
                    ),
                )
                for index in range(plan.logical_length)
            )
            selected = max(samples, key=lambda item: item.absolute_error)
            if max_abs_error != selected.absolute_error:
                raise ValueError(
                    "max_abs_error must match the logical output maximum"
                )
        metric_ok = (
            max_abs_error is not None
            and max_abs_error <= tolerance
        )
        padding_samples = tuple(
            (index, abs(receipt.output[index]))
            for index in range(plan.logical_length, plan.padded_length)
        ) if plan.verify_padding else ()
        padding_max_abs_error = (
            max((error for _, error in padding_samples), default=0.0)
            if max_abs_error is not None and plan.verify_padding else None
        )
        padding_mismatch_index = None
        if (
            padding_max_abs_error is not None
            and padding_max_abs_error > tolerance
        ):
            padding_mismatch_index = max(
                padding_samples, key=lambda item: item[1]
            )[0]
        padding_ok = (
            not plan.verify_padding
            or (
                padding_max_abs_error is not None
                and padding_max_abs_error <= tolerance
            )
        )
        if not metric_ok and max_abs_error is not None:
            mismatch = selected
        body = {
            "request_id": plan.request_id,
            "execution_id": plan.execution_id,
            "attempt_id": plan.attempt_id,
            "project_id": plan.project_id,
            "passed": receipt.exit_code == 0 and metric_ok and padding_ok,
            "max_abs_error": max_abs_error,
            "tolerance": float(tolerance),
            "exit_code": receipt.exit_code,
            "output_sha256": canonical_digest(receipt.output),
            "source_fingerprint": plan.source_fingerprint,
            "evidence_sha256": canonical_digest(
                {
                    "exit_code": receipt.exit_code,
                    "job_handle": receipt.job_handle,
                    "metadata": receipt.metadata,
                    "output": receipt.output,
                    "stderr": receipt.stderr,
                    "stdout": receipt.stdout,
                }
            ),
            "job_handle": receipt.job_handle,
            "mismatch": mismatch,
        }
        if padding_max_abs_error is not None:
            body["padding_max_abs_error"] = padding_max_abs_error
            body["padding_mismatch_index"] = padding_mismatch_index
        return cls(**body, attestation_sha256=attest(body))

    def attestation_payload(self) -> dict[str, Any]:
        value = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "attestation_sha256"
        }
        # Older retained A3 evidence predates explicit padding verification.
        # Omitting absent fields preserves its original attestation identity.
        if self.padding_max_abs_error is None:
            value.pop("padding_max_abs_error")
            value.pop("padding_mismatch_index")
        return value


@dataclass(frozen=True)
class FailedEvidence:
    request_id: str
    execution_id: str
    attempt_id: str
    project_id: str
    source_fingerprint: str
    status: str
    stage: str
    error_type: str
    detail: str
    attestation_sha256: str

    @classmethod
    def create(
        cls,
        plan: ExecutionPlan,
        *,
        stage: str,
        error_type: str,
        detail: str,
    ) -> "FailedEvidence":
        if stage not in {"prepare", "compile", "execute", "verify"}:
            raise ValueError("failure stage is unsupported")
        body = {
            "request_id": plan.request_id,
            "execution_id": plan.execution_id,
            "attempt_id": plan.attempt_id,
            "project_id": plan.project_id,
            "source_fingerprint": plan.source_fingerprint,
            "status": "failed",
            "stage": stage,
            "error_type": error_type,
            "detail": detail,
        }
        return cls(**body, attestation_sha256=attest(body))

    def attestation_payload(self) -> dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "attestation_sha256"
        }
