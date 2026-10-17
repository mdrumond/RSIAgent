"""Host-authoritative compiler and runtime observations shared by Ascend targets."""

from dataclasses import dataclass
import re


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _require_digest(value: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError("evidence_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class CompileObservation:
    """An actual compiler observation supplied by the execution backend."""

    passed: bool
    evidence_sha256: str
    diagnostic: str
    artifact_id: str | None

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise ValueError("compile passed must be boolean")
        _require_digest(self.evidence_sha256)
        if not isinstance(self.diagnostic, str) or not self.diagnostic.strip():
            raise ValueError("compile diagnostic must be non-empty")
        if self.passed and (
            not isinstance(self.artifact_id, str) or not self.artifact_id.strip()
        ):
            raise ValueError("artifact_id is required after successful compilation")
        if not self.passed and self.artifact_id is not None:
            raise ValueError("artifact_id is invalid after failed compilation")


@dataclass(frozen=True)
class RuntimeObservation:
    """A synchronized runtime observation, including the complete host output."""

    passed: bool
    evidence_sha256: str
    diagnostic: str
    output: tuple[float, ...] | None

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise ValueError("runtime passed must be boolean")
        _require_digest(self.evidence_sha256)
        if not isinstance(self.diagnostic, str) or not self.diagnostic.strip():
            raise ValueError("runtime diagnostic must be non-empty")
        if self.passed != (self.output is not None):
            raise ValueError("output is required exactly when runtime passed")
        if self.output is not None and (
            not isinstance(self.output, tuple)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                for value in self.output
            )
        ):
            raise ValueError("runtime output must be a numeric tuple")
