"""Host-owned dense forward-prefill attention target for A5 task.

The execution backend is deliberately untrusted: it receives immutable inputs and
returns only raw output.  This module owns the cases, FP32 reference, tolerances,
and verdict so an agent-authored candidate cannot claim correctness.

The logical contract is Q/K/V in ``[B,H,S,D]`` float16 or bfloat16, causal or
unmasked, scaled by ``1/sqrt(D)``, with the attention output as its only result.
The candidate ABI presents each tensor as flat contiguous row-major storage and
provides the logical dimensions as compile-time constants.  Backward, dropout,
KV caching, GQA, paging, RoPE, ALiBi, and packed variable-length inputs are
deliberately outside this target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path

from benchmarks.ascend.attention import (
    ATTENTION_CASES, SUITE_SHA256, MAX_DIAGNOSTIC_BYTES,
    DTYPE_TOLERANCES, HOST_TARGET_GRADING_CONTRACT, HOST_TARGET_GRADING_SHA256,
    AttentionCase, AttentionExecution, AttentionReceipt, AttentionBackend,
    AttentionInfrastructureError, AttentionCaseEvidence,
    build_execution, attention_reference, _digest, _jsonable, _grade,
    _execution_failure, _receipt_diagnostic, _truncate_utf8, _f32, _f16, _bf16,
    verify_cases,
)

TARGET_ID = "a5-dense-forward-attention-v1"
SCHEMA = "a5-attention-evidence-v2"

CANDIDATE_ABI = (
    ("module_import", "import catlass.tla as tla"),
    ("module_members", (
        "one import, numeric or Catlass-dtype assignments, undecorated "
        "synchronous helpers, and one synchronous kernel definition"
    )),
    ("kernel_decorator", "@tla.kernel"),
    ("kernel_export", "dense_attention(gm_q, gm_k, gm_v, gm_output)"),
    ("signature_policy", "four positional arguments; no defaults, varargs, or kwargs"),
    ("host_bound_constants", (
        "BATCH", "HEADS", "SEQUENCE", "HEAD_DIM", "SCALE", "CAUSAL", "DTYPE",
    )),
    ("host_bound_dtype", (
        "DTYPE is the actual tla.Float16 or tla.BFloat16 type matching the case"
    )),
    ("candidate_assignment_policy", (
        "numeric literals and Catlass numeric dtype attributes; host-bound "
        "constant names are reserved"
    )),
    ("arguments", ("gm_q", "gm_k", "gm_v", "gm_output")),
    ("physical_layout", "flat-contiguous-row-major"),
    ("logical_shape", "[B,H,S,D]"),
    ("logical_dimensions", ("BATCH", "HEADS", "SEQUENCE", "HEAD_DIM")),
    ("launch_block_num", 1),
    ("launch_coverage", "block 0 must produce the complete output tensor"),
)
CANDIDATE_ABI_SHA256 = _digest(CANDIDATE_ABI)

@dataclass(frozen=True)
class AttentionTargetEvidence:
    schema: str
    target_id: str
    suite_sha256: str
    request_id: str
    attempt_id: str
    execution_profile: str
    candidate_sha256: str
    passed: bool
    cases: tuple[AttentionCaseEvidence, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))
        if (self.schema, self.target_id, self.suite_sha256) != (
            SCHEMA, TARGET_ID, SUITE_SHA256,
        ):
            raise ValueError("attention target identity is invalid")
        if tuple(item.case_id for item in self.cases) != tuple(
            case.case_id for case in ATTENTION_CASES
        ):
            raise ValueError("evidence must contain the complete ordered suite")
        if any(item.case_sha256 != _digest(asdict(case))
               for item, case in zip(self.cases, ATTENTION_CASES)):
            raise ValueError("case evidence changed a host-owned case")
        if self.passed != all(item.passed for item in self.cases):
            raise ValueError("aggregate verdict must be the conjunction of case verdicts")
        body = {
            field.name: getattr(self, field.name)
            for field in fields(self) if field.name != "evidence_sha256"
        }
        if self.evidence_sha256 != _digest(body):
            raise ValueError("aggregate evidence digest is invalid")

    @classmethod
    def create(cls, **values: object) -> "AttentionTargetEvidence":
        body = {"schema": SCHEMA, "target_id": TARGET_ID,
                "suite_sha256": SUITE_SHA256, **values}
        return cls(**body, evidence_sha256=_digest(body))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "AttentionTargetEvidence":
        value = json.loads(path.read_text(encoding="utf-8"))
        value["cases"] = tuple(AttentionCaseEvidence(**item) for item in value["cases"])
        return cls(**value)


class AttentionTargetVerifier:
    """Execute and grade the complete suite without trusting candidate claims."""

    def __init__(self, backend: AttentionBackend):
        self._backend = backend

    def verify(
        self, candidate_source: str, *, request_id: str, attempt_id: str,
        execution_profile: str, workdir: Path,
    ) -> AttentionTargetEvidence:
        if not candidate_source.strip():
            raise ValueError("candidate source must be non-empty")
        for name, value in (("request_id", request_id), ("attempt_id", attempt_id),
                            ("execution_profile", execution_profile)):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if execution_profile not in {"bz-a5", "local"}:
            raise ValueError("attention target requires bz-a5 or local execution")

        candidate_sha = hashlib.sha256(candidate_source.encode("utf-8")).hexdigest()
        results = verify_cases(self._backend, candidate_source, workdir)
        return AttentionTargetEvidence.create(
            request_id=request_id,
            attempt_id=attempt_id,
            execution_profile=execution_profile,
            candidate_sha256=candidate_sha,
            passed=all(item.passed for item in results),
            cases=tuple(results),
        )
