"""Shared host-owned dense attention cases, inputs, oracle and case grading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import struct
from typing import Protocol


MAX_DIAGNOSTIC_BYTES = 4096

def _digest(value: object) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


DTYPE_TOLERANCES = (
    ("float16", (("atol", 0.02), ("rtol", 0.02))),
    ("bfloat16", (("atol", 0.05), ("rtol", 0.05))),
)
HOST_TARGET_GRADING_CONTRACT = (
    ("reference", "dense QK^T, stable softmax, and V reduction with FP32 intermediates"),
    ("output", "complete finite flat [B,H,S,D] tensor"),
    ("comparison", "abs(actual-expected) <= atol + rtol*abs(expected)"),
    ("dtype_tolerances", DTYPE_TOLERANCES),
    ("suite_verdict", "pass only when every registered case passes"),
)
HOST_TARGET_GRADING_SHA256 = _digest(HOST_TARGET_GRADING_CONTRACT)


@dataclass(frozen=True)
class AttentionCase:
    """One host-owned `[B,H,S,D]` dense self-attention case."""

    case_id: str
    batch: int
    heads: int
    sequence: int
    head_dim: int
    dtype: str
    causal: bool
    seed: int

    def __post_init__(self) -> None:
        if not self.case_id or type(self.case_id) is not str:
            raise ValueError("case_id must be a non-empty string")
        if any(type(value) is not int or value < 1 for value in (
            self.batch, self.heads, self.sequence, self.head_dim,
        )):
            raise ValueError("attention dimensions must be positive integers")
        if self.dtype not in {"float16", "bfloat16"}:
            raise ValueError("attention dtype must be float16 or bfloat16")
        if type(self.causal) is not bool or type(self.seed) is not int:
            raise TypeError("causal must be bool and seed must be int")

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.batch, self.heads, self.sequence, self.head_dim

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)

    @property
    def output_elements(self) -> int:
        return self.batch * self.heads * self.sequence * self.head_dim

    @property
    def rtol(self) -> float:
        return dict(dict(DTYPE_TOLERANCES)[self.dtype])["rtol"]

    @property
    def atol(self) -> float:
        return dict(dict(DTYPE_TOLERANCES)[self.dtype])["atol"]


# These cases are large enough to require genuine attention tiling while remaining
# bounded for the pure-Python host oracle.  Every dtype/mask combination is present.
ATTENTION_CASES = (
    AttentionCase("fp16-b1-h2-s33-d64-full", 1, 2, 33, 64, "float16", False, 5133),
    AttentionCase("fp16-b1-h1-s65-d128-causal", 1, 1, 65, 128, "float16", True, 5265),
    AttentionCase("bf16-b2-h1-s33-d128-full", 2, 1, 33, 128, "bfloat16", False, 5333),
    AttentionCase("bf16-b1-h2-s65-d64-causal", 1, 2, 65, 64, "bfloat16", True, 5465),
)
SUITE_SHA256 = _digest([asdict(case) for case in ATTENTION_CASES])


@dataclass(frozen=True)
class AttentionExecution:
    """Complete immutable input passed to the untrusted runtime adapter."""

    case: AttentionCase
    q: tuple[float, ...]
    k: tuple[float, ...]
    v: tuple[float, ...]
    scale: float

    def __post_init__(self) -> None:
        if not isinstance(self.case, AttentionCase):
            raise TypeError("case must be an AttentionCase")
        expected = self.case.output_elements
        if any(len(item) != expected for item in (self.q, self.k, self.v)):
            raise ValueError("Q, K, and V must each have shape [B,H,S,D]")
        if self.scale != self.case.scale:
            raise ValueError("attention scale must be exactly 1/sqrt(D)")


@dataclass(frozen=True)
class AttentionReceipt:
    """Untrusted candidate output; metadata is retained but never grades it."""

    exit_code: int
    output: tuple[float, ...]
    stdout: str = ""
    stderr: str = ""
    session_handle: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()


class AttentionBackend(Protocol):
    def execute(
        self, candidate_source: str, execution: AttentionExecution, workdir: Path,
    ) -> AttentionReceipt: ...


class AttentionInfrastructureError(RuntimeError):
    """The target could not be observed; no candidate verdict may be emitted.

    Runtime adapters must use this for transport loss, unavailable devices, and
    indeterminate retained operations.  Candidate compile and runtime failures use
    ordinary exceptions and are retained as semantic failure evidence instead.
    """


@dataclass(frozen=True)
class AttentionCaseEvidence:
    case_id: str
    case_sha256: str
    passed: bool
    exit_code: int
    output_sha256: str
    expected_sha256: str
    max_abs_error: float | None
    max_rel_error: float | None
    mismatch_index: int | None
    mismatch_expected: float | None
    mismatch_actual: float | None
    failure: str | None
    diagnostic: str | None
    session_handle: str | None
    evidence_sha256: str

    def __post_init__(self) -> None:
        body = {
            field.name: getattr(self, field.name)
            for field in fields(self) if field.name != "evidence_sha256"
        }
        if self.evidence_sha256 != _digest(body):
            raise ValueError("case evidence digest is invalid")
        if self.passed != (self.failure is None):
            raise ValueError("case verdict and failure must agree")
        if self.passed and self.diagnostic is not None:
            raise ValueError("passed case evidence must not carry a diagnostic")
        if self.diagnostic is not None and (
            not self.diagnostic
            or len(self.diagnostic.encode("utf-8")) > MAX_DIAGNOSTIC_BYTES
            or "\r" in self.diagnostic
        ):
            raise ValueError("case diagnostic must be normalized and bounded")
        if self.passed and (
            self.exit_code != 0 or self.mismatch_index is not None
            or self.max_abs_error is None or self.max_rel_error is None
        ):
            raise ValueError("passed case evidence is incomplete")

    @classmethod
    def create(cls, **values: object) -> "AttentionCaseEvidence":
        return cls(**values, evidence_sha256=_digest(values))


def build_execution(case: AttentionCase) -> AttentionExecution:
    rng = random.Random(case.seed)
    count = case.output_elements
    quantize = _f16 if case.dtype == "float16" else _bf16
    tensors = tuple(
        tuple(quantize(rng.uniform(-0.75, 0.75)) for _ in range(count))
        for _ in range(3)
    )
    return AttentionExecution(case, *tensors, scale=case.scale)


@lru_cache(maxsize=32)
def attention_reference(execution: AttentionExecution) -> tuple[float, ...]:
    """Stable dense QK^T/softmax/V reference with FP32 intermediates."""

    case, q, k, v = execution.case, execution.q, execution.k, execution.v
    output = [0.0] * case.output_elements

    def offset(batch: int, head: int, token: int, dim: int) -> int:
        return (((batch * case.heads + head) * case.sequence + token)
                * case.head_dim + dim)

    for batch in range(case.batch):
        for head in range(case.heads):
            for query in range(case.sequence):
                key_count = query + 1 if case.causal else case.sequence
                scores = []
                for key in range(key_count):
                    total = 0.0
                    for dim in range(case.head_dim):
                        product = _f32(q[offset(batch, head, query, dim)]
                                       * k[offset(batch, head, key, dim)])
                        total = _f32(total + product)
                    scores.append(_f32(total * execution.scale))
                maximum = max(scores)
                weights = [_f32(math.exp(_f32(score - maximum))) for score in scores]
                denominator = 0.0
                for weight in weights:
                    denominator = _f32(denominator + weight)
                weights = [_f32(weight / denominator) for weight in weights]
                for dim in range(case.head_dim):
                    total = 0.0
                    for key, weight in enumerate(weights):
                        product = _f32(weight * v[offset(batch, head, key, dim)])
                        total = _f32(total + product)
                    output[offset(batch, head, query, dim)] = total
    return tuple(output)


def _grade(
    case: AttentionCase, receipt: AttentionReceipt, expected: tuple[float, ...],
) -> AttentionCaseEvidence:
    actual = tuple(receipt.output)
    failure = None
    mismatch = None
    max_abs = max_rel = 0.0
    if receipt.exit_code != 0:
        failure = "runtime-exit"
    elif len(actual) != len(expected):
        failure = "output-shape"
    else:
        for index, (got, want) in enumerate(zip(actual, expected)):
            if not math.isfinite(got):
                failure, mismatch = "nonfinite-output", (index, want, got)
                break
            absolute = abs(got - want)
            relative = absolute / max(abs(want), 1e-30)
            max_abs, max_rel = max(max_abs, absolute), max(max_rel, relative)
            if mismatch is None and absolute > case.atol + case.rtol * abs(want):
                mismatch = index, want, got
        if failure is None and mismatch is not None:
            failure = "numerical-mismatch"
    values = {
        "case_id": case.case_id,
        "case_sha256": _digest(asdict(case)),
        "passed": failure is None,
        "exit_code": receipt.exit_code,
        "output_sha256": _digest(actual),
        "expected_sha256": _digest(expected),
        "max_abs_error": (max_abs if len(actual) == len(expected)
                          and failure != "nonfinite-output" else None),
        "max_rel_error": (max_rel if len(actual) == len(expected)
                          and failure != "nonfinite-output" else None),
        "mismatch_index": None if mismatch is None else mismatch[0],
        "mismatch_expected": None if mismatch is None else mismatch[1],
        "mismatch_actual": (None if mismatch is None or
                            not math.isfinite(mismatch[2]) else mismatch[2]),
        "failure": failure,
        "diagnostic": (
            _receipt_diagnostic(receipt) if receipt.exit_code != 0 else None
        ),
        "session_handle": receipt.session_handle,
    }
    return AttentionCaseEvidence.create(**values)


def _execution_failure(
    case: AttentionCase, expected: tuple[float, ...], error: Exception,
) -> AttentionCaseEvidence:
    message = "\n".join(
        line.rstrip()
        for line in str(error).replace("\r\n", "\n").replace("\r", "\n").split(
            "\n"
        )
    ).strip()
    diagnostic = type(error).__name__
    if message:
        diagnostic += f": {message}"
    return AttentionCaseEvidence.create(
        case_id=case.case_id,
        case_sha256=_digest(asdict(case)),
        passed=False,
        exit_code=-1,
        output_sha256=_digest(()),
        expected_sha256=_digest(expected),
        max_abs_error=None,
        max_rel_error=None,
        mismatch_index=None,
        mismatch_expected=None,
        mismatch_actual=None,
        failure=f"execution-error:{type(error).__name__}",
        diagnostic=_truncate_utf8(diagnostic, MAX_DIAGNOSTIC_BYTES),
        session_handle=None,
    )


def _receipt_diagnostic(receipt: AttentionReceipt) -> str | None:
    """Return bounded host text for a failed receipt, never agent-authored prose."""

    streams = []
    for label, value in (("stdout", receipt.stdout), ("stderr", receipt.stderr)):
        normalized = "\n".join(
            line.rstrip()
            for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        ).strip()
        if normalized:
            streams.append((label, normalized))
    if not streams:
        return None

    overhead = sum(len(f"{label}:\n".encode("utf-8")) for label, _ in streams)
    overhead += len(streams) - 1  # Newlines between labeled streams.
    content_budget = MAX_DIAGNOSTIC_BYTES - overhead
    base, remainder = divmod(content_budget, len(streams))
    rendered = []
    for index, (label, value) in enumerate(streams):
        budget = base + (1 if index < remainder else 0)
        rendered.append(f"{label}:\n{_truncate_utf8(value, budget)}")
    return "\n".join(rendered)


def _truncate_utf8(value: str, budget: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= budget:
        return value
    if budget <= 0:
        return ""
    marker = b"\n...[truncated]...\n"
    if budget <= len(marker):
        return encoded[:budget].decode("utf-8", errors="ignore")
    content_budget = budget - len(marker)
    head_budget = (content_budget + 1) // 2
    tail_budget = content_budget - head_budget
    head = encoded[:head_budget].decode("utf-8", errors="ignore")
    tail = (
        encoded[-tail_budget:].decode("utf-8", errors="ignore")
        if tail_budget else ""
    )
    return head + marker.decode("ascii") + tail


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _f16(value: float) -> float:
    return struct.unpack("e", struct.pack("e", value))[0]


def _bf16(value: float) -> float:
    bits = struct.unpack("I", struct.pack("f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("f", struct.pack("I", bits))[0]


def verify_cases(backend: AttentionBackend, candidate_source: str, workdir: Path):
    """Run the same host-owned suite; infrastructure failures never become verdicts."""
    results = []
    for case in ATTENTION_CASES:
        execution = build_execution(case)
        expected = attention_reference(execution)
        case_dir = workdir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            receipt = backend.execute(candidate_source, execution, case_dir)
            if not isinstance(receipt, AttentionReceipt):
                raise TypeError("backend must return AttentionReceipt")
            results.append(_grade(case, receipt, expected))
        except AttentionInfrastructureError:
            raise
        except Exception as exc:
            results.append(_execution_failure(case, expected, exc))
    return tuple(results)
