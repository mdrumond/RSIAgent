"""Host-attested execution and verification for A5 hello kernels."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import uuid
from typing import Protocol

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
    def __init__(self, backend: ExecutionBackend, *, atol: float = 1e-5) -> None:
        self._backend = backend
        self._atol = atol

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
        )

    def run(
        self, request: RunRequest, *, attempt_id: str | None = None
    ) -> VerifiedResult:
        plan = self.prepare(request, attempt_id=attempt_id)
        receipt = self._backend.execute(plan)
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
            }
        )
        fields = {
            "request_id": plan.request_id,
            "attempt_id": plan.attempt_id,
            "language": plan.language,
            "passed": passed,
            "max_abs_error": max_error,
            "exit_code": receipt.exit_code,
            "output_sha256": output_sha,
            "source_fingerprint": plan.source_fingerprint,
            "evidence_sha256": evidence_sha,
            "session_handle": receipt.session_handle,
        }
        return VerifiedResult(**fields, attestation_sha256=canonical_hash(fields))
