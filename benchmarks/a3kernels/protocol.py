"""Immutable host-owned A3 execution identities and evidence packets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RunRequest:
    length: int = 32
    seed: int = 0
    dtype: str = "float32"
    target: str = "Ascend910B4"

    def __post_init__(self) -> None:
        if type(self.length) is not int or not 1 <= self.length <= 4096:
            raise ValueError("length must be an integer in [1, 4096]")
        if self.dtype != "float32":
            raise ValueError("the A3 hello kernel supports only float32")
        if self.target != "Ascend910B4":
            raise ValueError("the A3 hello kernel target must be Ascend910B4")

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
    request_id: str
    files: tuple[SourceFile, ...]
    argv: tuple[str, ...]
    input_a: tuple[float, ...]
    input_b: tuple[float, ...]
    runtime: str = "gz-a3-native-py311-torch"
    logical_device: int = 0

    @property
    def source_fingerprint(self) -> str:
        return canonical_hash({item.relative_path: item.sha256 for item in self.files})

    @property
    def execution_id(self) -> str:
        return canonical_hash(
            {
                "argv": self.argv,
                "input_a": self.input_a,
                "input_b": self.input_b,
                "logical_device": self.logical_device,
                "request_id": self.request_id,
                "runtime": self.runtime,
                "source_fingerprint": self.source_fingerprint,
            }
        )


@dataclass(frozen=True)
class VerifiedResult:
    request_id: str
    execution_id: str
    passed: bool
    max_abs_error: float | None
    exit_code: int
    source_fingerprint: str
    output_sha256: str
    a3_evidence_sha256: str
    attestation_sha256: str
