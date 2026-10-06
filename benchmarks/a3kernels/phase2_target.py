"""Host-owned Phase 2 target suite for tiled A3 ``vector_add`` candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Callable, Protocol

from .candidate import CandidateCompilation
from .phase1_evidence import canonical_digest
from .phase1_protocol import (
    ExecutionPlan,
    FailedEvidence,
    VerificationMismatch,
    VerifiedResult,
    attest,
)


PHASE2_TARGET_ID = "a3-phase2-tiled-vector-add-v1"
_PADDING_CANARY_A = 17.0
_PADDING_CANARY_B = -5.0


@dataclass(frozen=True)
class Phase2TargetCase:
    case_id: str
    logical_length: int
    padded_length: int
    block_count: int
    seed: int

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not self.case_id:
            raise ValueError("case_id must be a non-empty string")
        if (
            type(self.logical_length) is not int
            or type(self.padded_length) is not int
            or not 1 <= self.logical_length <= self.padded_length <= 4096
        ):
            raise ValueError("case lengths must describe an A3 candidate allocation")
        if type(self.block_count) is not int or not 1 <= self.block_count <= 32:
            raise ValueError("block_count must be an integer in [1, 32]")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")


PHASE2_TARGET_CASES = (
    Phase2TargetCase("n1-p64-b1", 1, 64, 1, 2101),
    Phase2TargetCase("n31-p64-b1", 31, 64, 1, 2131),
    Phase2TargetCase("n33-p64-b1", 33, 64, 1, 2133),
    Phase2TargetCase("n65-p128-b2", 65, 128, 2, 2165),
    Phase2TargetCase("n255-p256-b4", 255, 256, 4, 2255),
    Phase2TargetCase("n400-p448-b4", 400, 448, 4, 2400),
    Phase2TargetCase("n4096-p4096-b8", 4096, 4096, 8, 6096),
)
PHASE2_TARGET_SUITE_SHA256 = canonical_digest(
    tuple(asdict(case) for case in PHASE2_TARGET_CASES)
)


@dataclass(frozen=True)
class Phase2CaseEvidence:
    case: Phase2TargetCase
    request_id: str
    attempt_id: str
    project_id: str
    execution_id: str
    source_fingerprint: str
    compile_attestation_sha256: str | None
    library_sha256: str | None
    result_kind: str
    authority: VerifiedResult | FailedEvidence
    passed: bool
    result_attestation_sha256: str
    job_handle: str | None
    max_abs_error: float | None = None
    output_sha256: str | None = None
    failure_stage: str | None = None
    error_type: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.case, Phase2TargetCase):
            raise TypeError("case must be a Phase2TargetCase")
        for name in ("request_id", "attempt_id", "project_id", "execution_id",
                     "source_fingerprint", "result_attestation_sha256"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.result_kind not in {"verified", "failed"}:
            raise ValueError("result_kind must be verified or failed")
        if not isinstance(self.authority, (VerifiedResult, FailedEvidence)):
            raise TypeError("authority must be authoritative A3 result evidence")
        if self.authority.attestation_sha256 != attest(
            self.authority.attestation_payload()
        ):
            raise ValueError("authority attestation is invalid")
        if (
            self.authority.request_id != self.request_id
            or self.authority.attempt_id != self.attempt_id
            or self.authority.project_id != self.project_id
            or self.authority.execution_id != self.execution_id
            or self.authority.source_fingerprint != self.source_fingerprint
            or self.authority.attestation_sha256 != self.result_attestation_sha256
        ):
            raise ValueError("case fields do not match authoritative result")
        if self.passed:
            if (
                self.result_kind != "verified"
                or not isinstance(self.authority, VerifiedResult)
                or not self.authority.passed
                or self.compile_attestation_sha256 is None
                or self.library_sha256 is None
                or self.job_handle is None
                or self.failure_stage is not None
                or self.max_abs_error is None
                or self.output_sha256 is None
            ):
                raise ValueError("passed case evidence is incomplete")
        elif self.failure_stage not in {"compile", "execute", "verify", "prepare"}:
            raise ValueError("failed case evidence requires a failure stage")
        if self.result_kind == "failed" and not isinstance(
            self.authority, FailedEvidence
        ):
            raise ValueError("failed case must retain FailedEvidence authority")
        if self.result_kind == "failed" and (
            self.failure_stage != self.authority.stage
            or self.error_type != self.authority.error_type
            or self.detail != self.authority.detail
        ):
            raise ValueError("case failure does not match FailedEvidence authority")
        if self.result_kind == "verified" and (
            not isinstance(self.authority, VerifiedResult)
            or self.authority.padding_max_abs_error is None
            or self.passed != self.authority.passed
            or self.job_handle != self.authority.job_handle
            or self.max_abs_error != self.authority.max_abs_error
            or self.output_sha256 != self.authority.output_sha256
        ):
            raise ValueError("case verdict does not match VerifiedResult authority")


@dataclass(frozen=True)
class Phase2TargetEvidence:
    target_id: str
    suite_sha256: str
    request_id: str
    attempt_id: str
    execution_profile: str
    candidate_sha256: str
    source_fingerprint: str
    passed: bool
    cases: tuple[Phase2CaseEvidence, ...]
    attestation_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))
        if self.target_id != PHASE2_TARGET_ID:
            raise ValueError("target identity does not match the Phase 2 target")
        if self.suite_sha256 != PHASE2_TARGET_SUITE_SHA256:
            raise ValueError("suite identity does not match the Phase 2 cases")
        if tuple(item.case for item in self.cases) != PHASE2_TARGET_CASES:
            raise ValueError("aggregate evidence must contain every target case in order")
        if any(item.source_fingerprint != self.source_fingerprint for item in self.cases):
            raise ValueError("every case must bind the aggregate source fingerprint")
        if any(
            item.request_id != self.request_id or item.attempt_id != self.attempt_id
            for item in self.cases
        ):
            raise ValueError("every case must bind the aggregate request and attempt")
        if self.passed != all(item.passed for item in self.cases):
            raise ValueError("aggregate pass must equal the conjunction of case verdicts")
        if self.attestation_sha256 != canonical_digest(self.attestation_payload()):
            raise ValueError("aggregate evidence attestation is invalid")

    @classmethod
    def create(
        cls, *, request_id: str, attempt_id: str, execution_profile: str,
        candidate_sha256: str, source_fingerprint: str,
        cases: tuple[Phase2CaseEvidence, ...],
    ) -> "Phase2TargetEvidence":
        body = {
            "target_id": PHASE2_TARGET_ID,
            "suite_sha256": PHASE2_TARGET_SUITE_SHA256,
            "request_id": request_id,
            "attempt_id": attempt_id,
            "execution_profile": execution_profile,
            "candidate_sha256": candidate_sha256,
            "source_fingerprint": source_fingerprint,
            "passed": all(item.passed for item in cases),
            "cases": cases,
        }
        return cls(**body, attestation_sha256=canonical_digest(body))

    def attestation_payload(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "attestation_sha256"
        }

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def read(cls, path: Path) -> "Phase2TargetEvidence":
        value = json.loads(path.read_text(encoding="utf-8"))
        if type(value) is not dict or type(value.get("cases")) is not list:
            raise ValueError("Phase 2 target evidence must be a JSON object")
        cases = []
        for item in value.pop("cases"):
            authority = item["authority"]
            if item["result_kind"] == "verified":
                mismatch = authority.get("mismatch")
                authority = VerifiedResult(
                    **{
                        **authority,
                        "mismatch": (
                            VerificationMismatch(**mismatch)
                            if mismatch is not None else None
                        ),
                    }
                )
            else:
                authority = FailedEvidence(**authority)
            cases.append(Phase2CaseEvidence(
                **{
                    **item,
                    "case": Phase2TargetCase(**item["case"]),
                    "authority": authority,
                }
            ))
        return cls(**value, cases=cases)

    def write(self, path: Path) -> None:
        data = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() == data:
                return
            raise RuntimeError("persisted Phase 2 target evidence conflicts")
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=".phase2-target-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise RuntimeError("persisted Phase 2 target evidence conflicts") from None
        finally:
            temporary.unlink(missing_ok=True)


class Phase2CandidateBackend(Protocol):
    def compile(
        self, plan: ExecutionPlan, local_directory: Path
    ) -> CandidateCompilation | FailedEvidence: ...

    def execute(
        self, compilation: CandidateCompilation
    ) -> VerifiedResult | FailedEvidence: ...


PlanBuilder = Callable[..., ExecutionPlan]


class Phase2TargetVerifier:
    """Run one candidate unchanged through every host-owned target case."""

    def __init__(self, backend: Phase2CandidateBackend, *, plan_builder: PlanBuilder):
        self._backend = backend
        self._plan = plan_builder

    def verify(
        self, source: str, *, request_id: str, attempt_id: str,
        execution_profile: str, workdir: Path,
    ) -> Phase2TargetEvidence:
        outcomes: list[Phase2CaseEvidence] = []
        fingerprint: str | None = None
        for case in PHASE2_TARGET_CASES:
            project_id = f"{PHASE2_TARGET_ID}/{case.case_id}"
            plan = self._plan(
                source,
                request_id=request_id,
                attempt_id=attempt_id,
                project_id=project_id,
                length=case.logical_length,
                padded_length=case.padded_length,
                block_count=case.block_count,
                seed=case.seed,
                execution_profile=execution_profile,
            )
            _require_plan_binding(
                plan, source=source, request_id=request_id,
                attempt_id=attempt_id, project_id=project_id,
                execution_profile=execution_profile, case=case,
            )
            plan = _with_padding_canary(plan)
            if fingerprint is None:
                fingerprint = plan.source_fingerprint
            elif fingerprint != plan.source_fingerprint:
                raise RuntimeError("plan builder changed the candidate source identity")
            compilation = self._backend.compile(plan, workdir / case.case_id)
            if isinstance(compilation, FailedEvidence):
                outcomes.append(_failed_case(case, plan, compilation, None, None))
                continue
            if not isinstance(compilation, CandidateCompilation) or compilation.plan != plan:
                raise RuntimeError("candidate backend returned a foreign compilation")
            result = self._backend.execute(compilation)
            if isinstance(result, FailedEvidence):
                outcomes.append(_failed_case(
                    case, plan, result, compilation.attestation_sha256,
                    compilation.library_sha256,
                ))
            elif isinstance(result, VerifiedResult):
                _require_result_binding(plan, result)
                outcomes.append(Phase2CaseEvidence(
                    case=case,
                    request_id=plan.request_id,
                    attempt_id=plan.attempt_id,
                    project_id=plan.project_id,
                    execution_id=plan.execution_id,
                    source_fingerprint=plan.source_fingerprint,
                    compile_attestation_sha256=compilation.attestation_sha256,
                    library_sha256=compilation.library_sha256,
                    result_kind="verified",
                    authority=result,
                    passed=result.passed,
                    result_attestation_sha256=result.attestation_sha256,
                    job_handle=result.job_handle,
                    max_abs_error=result.max_abs_error,
                    output_sha256=result.output_sha256,
                    failure_stage=None if result.passed else "verify",
                ))
            else:
                raise RuntimeError("candidate backend returned an unsupported result")
        assert fingerprint is not None
        return Phase2TargetEvidence.create(
            request_id=request_id,
            attempt_id=attempt_id,
            execution_profile=execution_profile,
            candidate_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            source_fingerprint=fingerprint,
            cases=tuple(outcomes),
        )


def _require_result_binding(plan: ExecutionPlan, result: VerifiedResult) -> None:
    if (
        result.execution_id != plan.execution_id
        or result.source_fingerprint != plan.source_fingerprint
        or result.request_id != plan.request_id
        or result.attempt_id != plan.attempt_id
        or result.project_id != plan.project_id
    ):
        raise RuntimeError("candidate backend returned foreign verification evidence")


def _failed_case(
    case: Phase2TargetCase, plan: ExecutionPlan, result: FailedEvidence,
    compile_attestation: str | None, library_sha256: str | None,
) -> Phase2CaseEvidence:
    if (
        result.request_id != plan.request_id
        or result.attempt_id != plan.attempt_id
        or result.execution_id != plan.execution_id
        or result.source_fingerprint != plan.source_fingerprint
        or result.project_id != plan.project_id
    ):
        raise RuntimeError("candidate backend returned foreign failure evidence")
    return Phase2CaseEvidence(
        case=case,
        request_id=plan.request_id,
        attempt_id=plan.attempt_id,
        project_id=plan.project_id,
        execution_id=plan.execution_id,
        source_fingerprint=plan.source_fingerprint,
        compile_attestation_sha256=compile_attestation,
        library_sha256=library_sha256,
        result_kind="failed",
        authority=result,
        passed=False,
        result_attestation_sha256=result.attestation_sha256,
        job_handle=None,
        failure_stage=result.stage,
        error_type=result.error_type,
        detail=result.detail,
    )


def _require_plan_binding(
    plan: ExecutionPlan, *, source: str, request_id: str, attempt_id: str,
    project_id: str, execution_profile: str, case: Phase2TargetCase,
) -> None:
    if not isinstance(plan, ExecutionPlan):
        raise RuntimeError("plan builder must return an A3 ExecutionPlan")
    expected = (
        request_id, attempt_id, project_id, execution_profile,
        case.logical_length, case.padded_length, case.block_count, False,
    )
    actual = (
        plan.request_id, plan.attempt_id, plan.project_id, plan.execution_profile,
        plan.logical_length, plan.padded_length, plan.block_count,
        plan.verify_padding,
    )
    candidates = tuple(
        item.content for item in plan.files if item.relative_path == "candidate.cpp"
    )
    rng = random.Random(case.seed)
    expected_a = tuple(rng.uniform(-1.0, 1.0) for _ in range(case.logical_length))
    expected_b = tuple(rng.uniform(-1.0, 1.0) for _ in range(case.logical_length))
    padding = (0.0,) * (case.padded_length - case.logical_length)
    if (
        actual != expected
        or candidates != (source,)
        or plan.input_a != expected_a + padding
        or plan.input_b != expected_b + padding
    ):
        raise RuntimeError("plan builder returned a plan outside the requested target case")


def _with_padding_canary(plan: ExecutionPlan) -> ExecutionPlan:
    tail = plan.padded_length - plan.logical_length
    if tail == 0:
        return replace(plan, verify_padding=True)
    return replace(
        plan,
        input_a=plan.input_a[:plan.logical_length] + (_PADDING_CANARY_A,) * tail,
        input_b=plan.input_b[:plan.logical_length] + (_PADDING_CANARY_B,) * tail,
        verify_padding=True,
    )


__all__ = [
    "PHASE2_TARGET_CASES",
    "PHASE2_TARGET_ID",
    "PHASE2_TARGET_SUITE_SHA256",
    "Phase2CaseEvidence",
    "Phase2TargetCase",
    "Phase2TargetEvidence",
    "Phase2TargetVerifier",
]
