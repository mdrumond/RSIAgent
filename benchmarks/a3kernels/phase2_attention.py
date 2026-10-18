"""Native A3 attention binding for the shared host-owned Phase 2 suite.

This is a new target; the historical A3 tiled-vector-add identity is unchanged.
Candidates supply device source only. The host binds inputs, case constants,
one-block launch, and numerical grading. Architecture comes from the profile.
"""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path

from benchmarks.ascend.attention import (
    ATTENTION_CASES, SUITE_SHA256, HOST_TARGET_GRADING_CONTRACT,
    HOST_TARGET_GRADING_SHA256, AttentionCase, AttentionExecution, AttentionCaseEvidence,
    AttentionInfrastructureError, AttentionReceipt, attention_reference,
    build_execution, _digest, verify_cases,
)
from .native_candidate import NativeCandidatePlan, NativeTensor
from .phase1_evidence import A3_EXECUTION_PROFILES
from .phase1_protocol import FailedEvidence
from .remote_candidate import _PendingObservation
from .native_candidate import native_infrastructure_failure


TARGET_ID = "a3-phase2-dense-forward-attention-v1"
SCHEMA = "a3-phase2-attention-evidence-v1"
CANDIDATE_ABI = (
    ("kernel_export", 'extern "C" __global__ __aicore__ void dense_attention(GM_ADDR gm_q, GM_ADDR gm_k, GM_ADDR gm_v, GM_ADDR gm_output)'),
    ("physical_layout", "flat-contiguous-row-major [B,H,S,D]"),
    ("host_bound_constants", ("BATCH", "HEADS", "SEQUENCE", "HEAD_DIM", "SCALE", "CAUSAL", "DTYPE")),
    ("launch", "one block produces the complete output tensor"),
)
CANDIDATE_ABI_SHA256 = _digest(CANDIDATE_ABI)


def build_plan(source, execution, *, request_id, attempt_id, execution_profile, target, arch):
    """Bind immutable shared inputs and constants to the operation-neutral ABI."""
    case = execution.case
    dtype = {"float16": "half", "bfloat16": "bfloat16_t"}[case.dtype]
    constants = dict(BATCH=case.batch, HEADS=case.heads, SEQUENCE=case.sequence,
                     HEAD_DIM=case.head_dim, SCALE=f"{case.scale!r}f",
                     CAUSAL=int(case.causal), DTYPE=dtype)
    prefix = "\n".join(f"#define {key} {value}" for key, value in constants.items())
    inputs = tuple(NativeTensor(name, case.shape, case.dtype, values) for name, values in (
        ("gm_q", execution.q), ("gm_k", execution.k), ("gm_v", execution.v)))
    return NativeCandidatePlan(
        prefix + "\n" + source, "dense_attention", inputs,
        NativeTensor("gm_output", case.shape, case.dtype), request_id, attempt_id,
        case.case_id, execution_profile, target, arch,
    )


class NativeAttentionBackend:
    """Adapt the existing retained A3 compile/execute transport, not verification."""

    def __init__(self, runtime, *, request_id, attempt_id, execution_profile, target, arch):
        if execution_profile not in A3_EXECUTION_PROFILES:
            raise ValueError("native attention requires an A3 profile")
        self.runtime = runtime
        self.binding = dict(request_id=request_id, attempt_id=attempt_id,
                            execution_profile=execution_profile, target=target, arch=arch)

    def execute(self, source, execution, workdir):
        plan = build_plan(source, execution, **self.binding)
        try:
            result = self.runtime.compile(plan, workdir)
            if not isinstance(result, FailedEvidence):
                result = self.runtime.execute_receipt(result)
        except (_PendingObservation, OSError) as exc:
            raise AttentionInfrastructureError(str(exc)) from exc
        if isinstance(result, FailedEvidence):
            if native_infrastructure_failure(result):
                raise AttentionInfrastructureError(result.detail)
            return AttentionReceipt(1, (), stderr=f"{result.stage}: {result.detail}")
        return AttentionReceipt(result.exit_code, result.output, result.stdout, result.stderr,
                                result.job_handle, result.metadata)


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

    def __post_init__(self):
        object.__setattr__(self, "cases", tuple(self.cases))
        if (self.schema, self.target_id, self.suite_sha256) != (SCHEMA, TARGET_ID, SUITE_SHA256):
            raise ValueError("A3 attention target identity is invalid")
        if self.execution_profile not in A3_EXECUTION_PROFILES:
            raise ValueError("A3 evidence requires an A3 profile")
        if tuple(c.case_id for c in self.cases) != tuple(c.case_id for c in ATTENTION_CASES):
            raise ValueError("evidence must contain the complete shared suite")
        if any(e.case_sha256 != _digest(asdict(c)) for e, c in zip(self.cases, ATTENTION_CASES)):
            raise ValueError("case evidence changed a host-owned case")
        if self.passed != all(c.passed for c in self.cases):
            raise ValueError("suite verdict disagrees with host grading")
        body = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "evidence_sha256"}
        if self.evidence_sha256 != _digest(body):
            raise ValueError("attention evidence digest is invalid")

    def write(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path):
        body = json.loads(path.read_text(encoding="utf-8"))
        body["cases"] = tuple(AttentionCaseEvidence(**c) for c in body["cases"])
        return cls(**body)


class AttentionTargetVerifier:
    def __init__(self, backend: NativeAttentionBackend):
        self.backend = backend

    def verify(self, candidate_source, *, request_id, attempt_id, execution_profile, workdir):
        binding = self.backend.binding
        if (request_id, attempt_id, execution_profile) != tuple(
            binding[k] for k in ("request_id", "attempt_id", "execution_profile")
        ):
            raise ValueError("verification identity must match native backend binding")
        if not candidate_source.strip():
            raise ValueError("candidate source must be non-empty")
        cases = verify_cases(self.backend, candidate_source, workdir)
        body = dict(schema=SCHEMA, target_id=TARGET_ID, suite_sha256=SUITE_SHA256,
                    request_id=request_id, attempt_id=attempt_id, execution_profile=execution_profile,
                    candidate_sha256=hashlib.sha256(candidate_source.encode()).hexdigest(),
                    passed=all(c.passed for c in cases), cases=cases)
        return AttentionTargetEvidence(**body, evidence_sha256=_digest(body))
