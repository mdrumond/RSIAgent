from dataclasses import FrozenInstanceError, replace

import pytest

from benchmarks.a3kernels.phase1_protocol import (
    ExecutionPlan,
    ExecutionReceipt,
    FailedEvidence,
    SourceFile,
    VerifiedResult,
    attest,
)


def plan(**changes):
    values = {
        "request_id": "request-7",
        "attempt_id": "attempt-1",
        "project_id": "vector-add",
        "files": (SourceFile("kernel.cpp", "extern \"C\" {}\n"),),
        "argv": ("python", "host_driver.py", "input.json"),
        "input_a": (1.0, -2.0),
        "input_b": (3.0, 5.0),
    }
    values.update(changes)
    return ExecutionPlan(**values)


def test_plan_has_deterministic_a3_source_and_execution_identity():
    files = [SourceFile("z.cpp", "z"), SourceFile("a.cpp", "a")]
    first = plan(files=files, argv=["python", "host_driver.py", "input.json"])
    reordered = plan(files=tuple(reversed(files)))

    assert first == reordered
    assert first.source_fingerprint == reordered.source_fingerprint
    assert first.execution_id == reordered.execution_id
    assert first.target == "Ascend910B4"
    assert first.language == "ascend-c"
    assert first.execution_profile == "gz-a3"
    assert first.logical_device == 0
    assert first.files == tuple(reversed(files))
    assert isinstance(first.argv, tuple)
    assert len(first.execution_id) == 64
    with pytest.raises(FrozenInstanceError):
        first.project_id = "changed"


def test_execution_dimensions_are_validated_and_identity_bound():
    execution = plan(logical_length=2, padded_length=2, block_count=1)
    assert execution.execution_id != plan(
        logical_length=1, padded_length=2, block_count=1
    ).execution_id
    assert execution.execution_id != plan(
        logical_length=2, padded_length=2, block_count=2
    ).execution_id
    with pytest.raises(ValueError, match="padded_length"):
        plan(logical_length=2, padded_length=3)


@pytest.mark.parametrize("profile", ["bz-a3-1", "bz-a3-2"])
def test_bz_execution_profiles_are_explicit_identity_fields(profile):
    execution = plan(execution_profile=profile)
    assert execution.execution_profile == profile
    assert execution.execution_id != plan().execution_id


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"target": "Ascend950"}, "target"),
        ({"language": "catlass-dsl"}, "language"),
        ({"execution_profile": "bz-a5"}, "profile"),
        ({"runtime": "a5-catlass"}, "runtime"),
        ({"logical_device": 1}, "logical device"),
    ],
)
def test_plan_rejects_non_a3_or_catlass_identity(changes, message):
    with pytest.raises(ValueError, match=message):
        plan(**changes)


@pytest.mark.parametrize("path", ["/kernel.cpp", "../kernel.cpp", "a/../../kernel.cpp"])
def test_source_paths_are_relative_and_cannot_escape(path):
    with pytest.raises(ValueError, match="relative"):
        SourceFile(path, "source")


def test_plan_rejects_duplicate_sources_and_identity_changes_with_content():
    source = SourceFile("kernel.cpp", "one")
    with pytest.raises(ValueError, match="unique"):
        plan(files=(source, source))
    assert plan(files=(source,)).execution_id != plan(
        files=(replace(source, content="two"),)
    ).execution_id


def test_receipt_is_untrusted_and_attestation_is_host_derived():
    execution = plan()
    receipt = ExecutionReceipt(
        exit_code=0,
        output=[4.0, 3.0],
        stdout="A3_OUTPUT=[4,3]",
        metadata=(("agent_claim", "perfect"),),
    )
    result = VerifiedResult.from_receipt(execution, receipt, max_abs_error=0.0)

    assert result.passed is True
    assert isinstance(receipt.output, tuple)
    assert result.output_sha256
    assert result.attestation_sha256 == attest(result.attestation_payload())
    assert "agent_claim" not in str(result.attestation_payload())


def test_failed_evidence_is_structured_and_attested():
    execution = plan()
    failure = FailedEvidence.create(
        execution, stage="compile", error_type="CompileError", detail="bad source"
    )

    assert failure.status == "failed"
    assert failure.stage == "compile"
    assert failure.request_id == execution.request_id
    assert failure.source_fingerprint == execution.source_fingerprint
    assert failure.attestation_sha256 == attest(failure.attestation_payload())


def test_verified_result_cannot_claim_pass_on_failed_process():
    with pytest.raises(ValueError, match="exit code"):
        VerifiedResult.from_receipt(
            plan(), ExecutionReceipt(exit_code=2), max_abs_error=0.0
        )


@pytest.mark.parametrize("metric", [-1.0, -0.000001, True])
def test_verified_result_rejects_negative_or_boolean_error(metric):
    with pytest.raises(ValueError, match="max_abs_error"):
        VerifiedResult.from_receipt(
            plan(), ExecutionReceipt(exit_code=0, output=(4.0, 3.0)),
            max_abs_error=metric,
        )


@pytest.mark.parametrize("tolerance", [-1.0, float("inf"), True])
def test_verified_result_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        VerifiedResult.from_receipt(
            plan(), ExecutionReceipt(exit_code=0, output=(4.0, 3.0)),
            max_abs_error=0.0, tolerance=tolerance,
        )
