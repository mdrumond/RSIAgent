"""Host-attested execution and verification for A5 hello kernels."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import uuid
from typing import Protocol

from benchmarks.a5kernels.evidence import EvidenceKind, EvidenceLedger, canonical_digest
from benchmarks.a5kernels.fixtures import fixture_for
from benchmarks.a5kernels.protocol import (
    ExecutionPlan,
    ExecutionReceipt,
    RunRequest,
    VerifiedResult,
    canonical_hash,
)


class ExecutionBackend(Protocol):
    """Boundary implemented by the checked-in BZ-A5 wrapper adapter."""

    def execute(self, plan: ExecutionPlan) -> ExecutionReceipt: ...


class A5KernelRunner:
    def __init__(
        self,
        backend: ExecutionBackend,
        *,
        atol: float = 1e-5,
        evidence_ledger: EvidenceLedger | None = None,
    ) -> None:
        self._backend = backend
        self._atol = atol
        self._ledger = evidence_ledger

    def prepare(
        self, request: RunRequest, *, attempt_id: str | None = None
    ) -> ExecutionPlan:
        fixture = fixture_for(request.language)
        attempt_id = uuid.uuid4().hex if attempt_id is None else attempt_id
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", attempt_id):
            raise ValueError("attempt_id must be a safe 1-64 character identifier")
        if fixture.max_length is not None and request.length > fixture.max_length:
            raise ValueError(
                f"{fixture.language.value} hello length cannot exceed {fixture.max_length}"
            )
        rng = random.Random(request.seed)
        input_a = tuple(rng.uniform(-1.0, 1.0) for _ in range(request.length))
        input_b = tuple(rng.uniform(-1.0, 1.0) for _ in range(request.length))
        return ExecutionPlan(
            request_id=request.request_id,
            attempt_id=attempt_id,
            language=fixture.language.value,
            files=fixture.files,
            argv=fixture.argv,
            input_a=input_a,
            input_b=input_b,
            runtime_provenance=getattr(self._backend, "runtime_provenance", ()),
        )

    def run(
        self, request: RunRequest, *, attempt_id: str | None = None
    ) -> VerifiedResult:
        plan = self.prepare(request, attempt_id=attempt_id)
        self._record_plan(request, plan)
        try:
            receipt = self._backend.execute(plan)
        except Exception as exc:
            if self._ledger is not None:
                self._ledger.append(
                    EvidenceKind.RESULT,
                    {
                        "request_id": plan.request_id,
                        "execution_id": plan.execution_id,
                        "attempt_id": plan.attempt_id,
                        "status": "execution_error",
                        "error_type": type(exc).__name__,
                    },
                )
            raise
        expected = tuple(a + b for a, b in zip(plan.input_a, plan.input_b))
        correct_length = len(receipt.output) == len(expected)
        finite = all(math.isfinite(value) for value in receipt.output)
        max_error = (
            max((abs(got - want) for got, want in zip(receipt.output, expected)), default=0.0)
            if correct_length and finite
            else math.inf
        )
        passed = receipt.exit_code == 0 and correct_length and finite and max_error <= self._atol

        output_json = json.dumps(receipt.output, separators=(",", ":"))
        output_sha = hashlib.sha256(output_json.encode("utf-8")).hexdigest()
        evidence_sha = canonical_hash(
            {
                "exit_code": receipt.exit_code,
                "stdout": receipt.stdout,
                "stderr": receipt.stderr,
                "session_handle": receipt.session_handle,
                "output_sha256": output_sha,
                "execution_id": plan.execution_id,
                "runtime_provenance": plan.runtime_provenance,
            }
        )
        fields = {
            "request_id": plan.request_id,
            "execution_id": plan.execution_id,
            "attempt_id": plan.attempt_id,
            "language": plan.language,
            "runtime_provenance": plan.runtime_provenance,
            "passed": passed,
            "max_abs_error": max_error,
            "exit_code": receipt.exit_code,
            "output_sha256": output_sha,
            "source_fingerprint": plan.source_fingerprint,
            "evidence_sha256": evidence_sha,
            "session_handle": receipt.session_handle,
        }
        result = VerifiedResult(**fields, attestation_sha256=canonical_hash(fields))
        if self._ledger is not None:
            result_payload = dict(result.__dict__)
            if not math.isfinite(result.max_abs_error):
                result_payload["max_abs_error"] = None
                result_payload["max_abs_error_status"] = "non-finite"
            result_payload["status"] = "verified"
            result_payload["execution_evidence"] = {
                "retained_logs": receipt.stdout,
                "diagnostics": receipt.stderr,
                "result_exit_code": receipt.exit_code,
                "output_sha256": result.output_sha256,
            }
            self._ledger.append(EvidenceKind.RESULT, result_payload)
        return result

    def _record_plan(self, request: RunRequest, plan: ExecutionPlan) -> None:
        if self._ledger is None:
            return
        identity = {
            "request_id": plan.request_id,
            "execution_id": plan.execution_id,
            "attempt_id": plan.attempt_id,
        }
        self._ledger.append(
            EvidenceKind.REQUEST,
            {**identity, "request": request.__dict__},
        )
        self._ledger.append(
            EvidenceKind.ACTION,
            {
                **identity,
                "language": plan.language,
                "argv": plan.argv,
                "inputs_sha256": canonical_digest(
                    {"input_a": plan.input_a, "input_b": plan.input_b}
                ),
            },
        )
        self._ledger.append(
            EvidenceKind.ARTIFACT,
            {
                **identity,
                "source_fingerprint": plan.source_fingerprint,
                "source_sha256": {
                    source.relative_path: source.sha256 for source in plan.files
                },
                "artifact_sha256": {
                    "source_bundle": plan.source_fingerprint,
                },
            },
        )
