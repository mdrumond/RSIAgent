from __future__ import annotations

from dataclasses import replace
import math

import pytest

from benchmarks.a5kernels.attention import (
    ATTENTION_CASES,
    CANDIDATE_ABI,
    CANDIDATE_ABI_SHA256,
    HOST_TARGET_GRADING_CONTRACT,
    HOST_TARGET_GRADING_SHA256,
    SUITE_SHA256,
    AttentionCase,
    AttentionInfrastructureError,
    AttentionReceipt,
    AttentionTargetEvidence,
    AttentionTargetVerifier,
    MAX_DIAGNOSTIC_BYTES,
    _truncate_utf8,
    attention_reference,
    build_execution,
)


SOURCE = """\
import catlass.tla as tla

@tla.kernel
def dense_attention(q, k, v, output):
    pass
"""


class FakeBackend:
    def __init__(
        self, *, mutate=None, fail_case=None, infrastructure_error=None,
        claimed_pass=False,
    ):
        self.mutate = mutate
        self.fail_case = fail_case
        self.infrastructure_error = infrastructure_error
        self.claimed_pass = claimed_pass
        self.calls = []

    def execute(self, source, execution, workdir):
        self.calls.append((source, execution, workdir))
        if execution.case.case_id == self.fail_case:
            raise RuntimeError("synthetic runtime failure")
        if self.infrastructure_error is not None:
            raise self.infrastructure_error
        output = attention_reference(execution)
        if self.mutate is not None:
            output = self.mutate(execution.case, output)
        return AttentionReceipt(
            exit_code=0,
            output=tuple(output),
            session_handle=f"bz-a5:{execution.case.case_id}",
            metadata=(("candidate_claimed_pass", str(self.claimed_pass)),),
        )


def _verify(tmp_path, backend):
    return AttentionTargetVerifier(backend).verify(
        SOURCE,
        request_id="phase2-cell-1",
        attempt_id="target-1",
        execution_profile="bz-a5",
        workdir=tmp_path / "runs",
    )


def test_suite_covers_fixed_dense_forward_contract():
    assert len(ATTENTION_CASES) == 4
    assert len({case.case_id for case in ATTENTION_CASES}) == 4
    assert len(SUITE_SHA256) == 64
    assert {(case.dtype, case.causal) for case in ATTENTION_CASES} == {
        ("float16", False), ("float16", True),
        ("bfloat16", False), ("bfloat16", True),
    }
    assert any(case.batch > 1 for case in ATTENTION_CASES)
    assert any(case.heads > 1 for case in ATTENTION_CASES)
    assert {case.sequence for case in ATTENTION_CASES} == {33, 65}
    assert {case.head_dim for case in ATTENTION_CASES} == {64, 128}
    assert all(case.scale == 1 / math.sqrt(case.head_dim) for case in ATTENTION_CASES)
    assert {(case.dtype, case.atol, case.rtol) for case in ATTENTION_CASES} == {
        ("float16", 0.02, 0.02),
        ("bfloat16", 0.05, 0.05),
    }


def test_candidate_abi_identity_is_stable_and_explicit():
    assert CANDIDATE_ABI == (
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
    assert len(CANDIDATE_ABI_SHA256) == 64


def test_host_grading_identity_is_stable_and_explicit():
    contract = dict(HOST_TARGET_GRADING_CONTRACT)
    assert contract["comparison"] == (
        "abs(actual-expected) <= atol + rtol*abs(expected)"
    )
    assert dict(contract["dtype_tolerances"]) == {
        "float16": (("atol", 0.02), ("rtol", 0.02)),
        "bfloat16": (("atol", 0.05), ("rtol", 0.05)),
    }
    assert contract["suite_verdict"] == (
        "pass only when every registered case passes"
    )
    assert len(HOST_TARGET_GRADING_SHA256) == 64


def test_execution_inputs_are_deterministic_quantized_bhsd_tensors():
    for case in ATTENTION_CASES:
        first = build_execution(case)
        second = build_execution(case)
        assert first == second
        assert len(first.q) == len(first.k) == len(first.v) == case.output_elements
        assert first.scale == case.scale
        assert first.q != first.k != first.v


def test_host_reference_obeys_causal_mask_and_full_attention():
    case = AttentionCase("mask-check", 1, 1, 2, 1, "float16", True, 1)
    execution = build_execution(case)
    execution = replace(execution, q=(1.0, 1.0), k=(1.0, 1.0), v=(2.0, 8.0))
    causal = attention_reference(execution)
    full = attention_reference(replace(execution, case=replace(case, causal=False)))

    assert causal == (2.0, 5.0)
    assert full == (5.0, 5.0)


def test_complete_fake_backend_passes_and_round_trips_evidence(tmp_path):
    backend = FakeBackend(claimed_pass=False)
    evidence = _verify(tmp_path, backend)

    assert evidence.passed is True
    assert len(evidence.cases) == len(ATTENTION_CASES)
    assert all(item.passed and item.failure is None for item in evidence.cases)
    assert [call[1].case for call in backend.calls] == list(ATTENTION_CASES)
    assert all(call[2].parent == tmp_path / "runs" for call in backend.calls)

    path = tmp_path / "evidence.json"
    evidence.write(path)
    assert AttentionTargetEvidence.read(path) == evidence


def test_passing_receipt_does_not_publish_host_diagnostics(tmp_path):
    class NoisyPassingBackend(FakeBackend):
        def execute(self, source, execution, workdir):
            receipt = super().execute(source, execution, workdir)
            return replace(
                receipt,
                stdout="compiler warning from checked host",
                stderr="runtime warning from checked host",
            )

    evidence = _verify(tmp_path, NoisyPassingBackend())

    assert evidence.passed is True
    assert all(item.diagnostic is None for item in evidence.cases)


def test_nonzero_receipt_publishes_bounded_digest_bound_host_diagnostics(tmp_path):
    class FailingBackend(FakeBackend):
        def execute(self, source, execution, workdir):
            return AttentionReceipt(
                exit_code=17,
                output=(),
                stdout=(
                    "Traceback (most recent call last):\r\ncompile.py:1\r\n"
                    + "x" * 5000 + "\nValueError: tile overflow"
                ),
                stderr="runtime failed\rwith ordinary input",
                session_handle=f"bz-a5:{execution.case.case_id}",
            )

    evidence = _verify(tmp_path, FailingBackend())

    assert evidence.passed is False
    for item in evidence.cases:
        assert item.failure == "runtime-exit"
        assert item.diagnostic is not None
        assert item.diagnostic.startswith(
            "stdout:\nTraceback (most recent call last):\ncompile.py:1\n"
        )
        assert "\n...[truncated]...\n" in item.diagnostic
        assert "ValueError: tile overflow\nstderr:\nruntime failed" in item.diagnostic
        assert "\r" not in item.diagnostic
        assert len(item.diagnostic.encode("utf-8")) <= 4096
        with pytest.raises(ValueError, match="digest"):
            replace(item, diagnostic="agent-authored explanation")

    path = tmp_path / "failed-evidence.json"
    evidence.write(path)
    assert AttentionTargetEvidence.read(path) == evidence


@pytest.mark.parametrize(
    ("mutate", "failure"),
    [
        (lambda case, output: (*output[:-1], output[-1] + 1.0), "numerical-mismatch"),
        (lambda case, output: (*output[:-1], float("nan")), "nonfinite-output"),
        (lambda case, output: output[:-1], "output-shape"),
    ],
)
def test_host_rejects_wrong_nonfinite_and_wrong_shape_outputs(tmp_path, mutate, failure):
    evidence = _verify(tmp_path, FakeBackend(mutate=mutate, claimed_pass=True))

    assert evidence.passed is False
    assert all(item.failure == failure for item in evidence.cases)


def test_numerical_mismatch_fields_are_directly_consumable(tmp_path):
    class NoisyMismatchBackend(FakeBackend):
        def execute(self, source, execution, workdir):
            receipt = super().execute(source, execution, workdir)
            return replace(
                receipt,
                stdout="A5ATTENTION_OUTPUT=[large host tensor omitted by evidence]",
                stderr="host runner completed",
            )

    evidence = _verify(
        tmp_path,
        NoisyMismatchBackend(
            mutate=lambda case, output: (output[0] + 1.0, *output[1:])
        ),
    )

    for item in evidence.cases:
        assert item.failure == "numerical-mismatch"
        assert item.mismatch_index == 0
        assert item.mismatch_expected is not None
        assert item.mismatch_actual == pytest.approx(item.mismatch_expected + 1.0)
        assert item.max_abs_error == pytest.approx(1.0)
        assert item.max_rel_error is not None
        assert item.diagnostic is None


def test_backend_exception_is_case_local_and_complete_suite_is_retained(tmp_path):
    failed = ATTENTION_CASES[2].case_id
    evidence = _verify(tmp_path, FakeBackend(fail_case=failed))

    assert evidence.passed is False
    assert len(evidence.cases) == len(ATTENTION_CASES)
    item = next(item for item in evidence.cases if item.case_id == failed)
    assert item.failure == "execution-error:RuntimeError"
    assert item.diagnostic == "RuntimeError: synthetic runtime failure"
    assert len(evidence.cases) == len(ATTENTION_CASES)


def test_local_value_error_message_is_normalized_bounded_and_retained(tmp_path):
    class RejectingBackend(FakeBackend):
        def execute(self, source, execution, workdir):
            raise ValueError(
                "candidate validation failed\r\n" + "x" * 5000
                + "\ninvalid dense_attention signature"
            )

    evidence = _verify(tmp_path, RejectingBackend())

    for item in evidence.cases:
        assert item.failure == "execution-error:ValueError"
        assert item.diagnostic is not None
        assert item.diagnostic.startswith("ValueError: candidate validation failed\n")
        assert "\n...[truncated]...\n" in item.diagnostic
        assert item.diagnostic.endswith("invalid dense_attention signature")
        assert "\r" not in item.diagnostic
        assert len(item.diagnostic.encode("utf-8")) <= MAX_DIAGNOSTIC_BYTES


def test_utf8_truncation_preserves_short_text_and_long_head_and_tail():
    assert _truncate_utf8("alpha 🙂 omega", 100) == "alpha 🙂 omega"

    rendered = _truncate_utf8("head:" + "🙂" * 100 + ":tail", 64)

    assert rendered.startswith("head:")
    assert rendered.endswith(":tail")
    assert "\n...[truncated]...\n" in rendered
    assert len(rendered.encode("utf-8")) <= 64


def test_infrastructure_failure_aborts_without_semantic_evidence(tmp_path):
    failure = AttentionInfrastructureError("retained operation is unavailable")
    backend = FakeBackend(infrastructure_error=failure)

    with pytest.raises(AttentionInfrastructureError) as raised:
        _verify(tmp_path, backend)

    assert raised.value is failure
    assert len(backend.calls) == 1


def test_tolerance_is_dtype_specific_and_candidate_metadata_cannot_decide(tmp_path):
    def perturb(case, output):
        delta = 0.03
        return tuple(value + delta for value in output)

    evidence = _verify(tmp_path, FakeBackend(mutate=perturb, claimed_pass=True))
    by_dtype = {
        case.dtype: evidence.cases[index].passed
        for index, case in enumerate(ATTENTION_CASES)
    }
    assert by_dtype == {"float16": False, "bfloat16": True}


def test_identity_profile_and_evidence_tampering_are_rejected(tmp_path):
    verifier = AttentionTargetVerifier(FakeBackend())
    with pytest.raises(ValueError, match="bz-a5"):
        verifier.verify(
            SOURCE, request_id="r", attempt_id="a", execution_profile="a3",
            workdir=tmp_path,
        )
    evidence = _verify(tmp_path, FakeBackend())
    with pytest.raises(ValueError, match="aggregate verdict"):
        replace(evidence, passed=False, evidence_sha256=evidence.evidence_sha256)
    with pytest.raises(ValueError, match="digest"):
        replace(evidence.cases[0], output_sha256="0" * 64)
