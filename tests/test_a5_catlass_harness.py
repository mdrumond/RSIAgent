from __future__ import annotations

from dataclasses import replace
import json

import pytest

from benchmarks.a5kernels.catlass_harness import (
    CatlassContractHarness,
    HarnessContract,
    SCHEMA,
    SPECS,
    _case,
    report_results,
    validate_contract_source,
    write_result,
)
from benchmarks.a5kernels.protocol import ExecutionReceipt
import run_catlass_dsl_harness as cli


def source_for(contract: HarnessContract) -> str:
    spec = SPECS[contract]
    arguments = ", ".join(f"{name}: tla.Tensor" for name in spec.arguments)
    body = {
        HarnessContract.PADDED_SIMD: '''\
    with tla.vector():
        tla.copy(gm_out, gm_a)
        with tla.vec.func(mode="simd"):
            gm_out.store(gm_a.load())''',
        HarnessContract.MULTIBLOCK_SIMT: '''\
    block = tla.arch.block_idx()
    blocks = tla.arch.block_num()
    with tla.vector():
        with tla.vec.func(mode="simt"):
            thread = tla.arch.thread_idx()
            index = block + thread[0] * blocks
            gm_out[index] = gm_a[index]''',
        HarnessContract.CUBE_MATMUL: '''\
    with tla.cube():
        tla.copy(gm_out, gm_a)
        tla.mmad(gm_out, gm_a, gm_b)''',
    }[contract]
    return f'''\
import catlass.tla as tla

SIZE = 16

@tla.kernel
def {spec.kernel_name}({arguments}) -> None:
{body}
'''


class FakeBackend:
    runtime_provenance = (
        ("catlass_revision", "9a6ac627b5f4078060287844189730cf0d184800"),
        ("execution_profile", "bz-a5"),
    )

    def __init__(self, *, corrupt=False, exit_code=0):
        self.corrupt = corrupt
        self.exit_code = exit_code
        self.plans = []

    def prepare_execution(self, plan):
        return replace(plan, runtime_provenance=plan.runtime_provenance)

    def execute(self, plan):
        self.plans.append(plan)
        contract = next(
            item for item, spec in SPECS.items()
            if f"def {spec.kernel_name}(" in plan.files[0].content
        )
        output = list(_case(contract)[2])
        if self.corrupt:
            output[-1] += 1.0
        if self.exit_code:
            output = []
        return ExecutionReceipt(
            self.exit_code,
            tuple(output),
            stdout=f"A5KERNEL_NAME={SPECS[contract].kernel_name}__kernel0\nlogs",
            stderr="compiler failed" if self.exit_code else "",
            session_handle="bz-a5:test-session",
        )


@pytest.mark.parametrize("contract", list(HarnessContract))
def test_contract_source_accepts_exact_agent_kernel(contract):
    validate_contract_source(source_for(contract), contract)


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda text: "", "must not be empty"),
        (lambda text: text.replace("import catlass.tla as tla", "import os"), "import only"),
        (lambda text: text + "\ndef helper(): pass\n", "exactly one"),
        (lambda text: text.replace("@tla.kernel", ""), "exactly @tla.kernel"),
        (lambda text: text.replace("gm_out", "output"), "must use"),
    ],
)
def test_contract_source_rejects_non_contract_modules(edit, message):
    source = source_for(HarnessContract.PADDED_SIMD)
    with pytest.raises(ValueError, match=message):
        validate_contract_source(edit(source), HarnessContract.PADDED_SIMD)


def test_multiblock_source_must_use_declared_block_count():
    source = source_for(HarnessContract.MULTIBLOCK_SIMT)
    without_block_count = source.replace(
        "    blocks = tla.arch.block_num()\n", "    blocks = 4\n"
    )

    with pytest.raises(ValueError, match="simt vector execution"):
        validate_contract_source(without_block_count, HarnessContract.MULTIBLOCK_SIMT)


def test_preflight_binds_exact_source_revision_profile_and_device():
    source = source_for(HarnessContract.PADDED_SIMD)
    payload = CatlassContractHarness(FakeBackend(), device=3).preflight(
        source, HarnessContract.PADDED_SIMD
    )

    assert payload == {
        "ready": True,
        "contract": "padded-simd",
        "source_sha256": payload["source_sha256"],
        "catlass_revision": "9a6ac627b5f4078060287844189730cf0d184800",
        "execution_profile": "bz-a5",
        "device": 3,
    }
    assert len(payload["source_sha256"]) == 64


def test_preflight_rejects_unpinned_runtime():
    backend = FakeBackend()
    backend.runtime_provenance = (
        ("catlass_revision", "a" * 40),
        ("execution_profile", "bz-a5"),
    )
    with pytest.raises(ValueError, match="requires Catlass revision"):
        CatlassContractHarness(backend).preflight(
            source_for(HarnessContract.PADDED_SIMD), HarnessContract.PADDED_SIMD
        )


@pytest.mark.parametrize("contract", list(HarnessContract))
def test_run_uses_host_case_runtime_oracle_and_evidence(contract):
    backend = FakeBackend()
    source = source_for(contract)
    result = CatlassContractHarness(backend, device=2).run(source, contract, "agent-1")

    assert result.schema == SCHEMA
    assert result.contract == contract.value
    assert result.host_verdict == "PASS"
    assert result.max_abs_error == 0.0
    assert result.mismatch_index is None
    assert result.session_handle == "bz-a5:test-session"
    assert len(result.record_sha256) == len(result.retained_evidence_sha256) == 64
    plan = backend.plans[0]
    assert dict(plan.environment.bindings) == {
        "A5KERNEL_BLOCK_NUM": str(SPECS[contract].block_count),
        "BZ_A5_PROFILE_PHYSICAL_DEVICE": "2",
    }
    assert len(plan.input_a) == len(plan.input_b) == SPECS[contract].physical_length
    kernel = next(item.content for item in plan.files if item.relative_path == "kernel.py")
    assert source.rstrip() in kernel
    assert "--npu-arch 3510" in kernel
    assert "torch.npu.synchronize()" in kernel
    assert 'physical_device = int(os.environ["BZ_A5_PROFILE_PHYSICAL_DEVICE"])' in kernel
    assert "torch.npu.set_device(0)" in kernel
    assert "torch.npu.set_device(physical_device)" not in kernel
    assert "def run(input_a, input_b):" in kernel


def test_multiblock_contract_is_indexed_and_host_partitioned():
    contract = HarnessContract.MULTIBLOCK_SIMT
    backend = FakeBackend()
    CatlassContractHarness(backend).run(source_for(contract), contract, "multiblock")
    plan = backend.plans[0]

    assert dict(plan.environment.bindings)["A5KERNEL_BLOCK_NUM"] == "4"
    assert plan.input_b[:4] == (0.0, 1 / 257.0, 2 / 257.0, 3 / 257.0)
    assert plan.input_a[257:] == plan.input_b[257:] == (0.0,) * 63


def test_cube_contract_uses_host_owned_fp16_inputs_and_fp32_output():
    backend = FakeBackend()
    contract = HarnessContract.CUBE_MATMUL
    CatlassContractHarness(backend).run(source_for(contract), contract, "cube")
    kernel = backend.plans[0].files[0].content

    assert "dtype=torch.float16" in kernel
    assert ".reshape(16, 16)" in kernel
    assert "dtype=torch.float32" in kernel
    assert "torch.full((16, 16)" in kernel


def test_wrong_result_and_compile_failure_are_host_failures():
    source = source_for(HarnessContract.PADDED_SIMD)
    wrong = CatlassContractHarness(FakeBackend(corrupt=True)).run(
        source, HarnessContract.PADDED_SIMD, "wrong"
    )
    failed = CatlassContractHarness(FakeBackend(exit_code=7)).run(
        source, HarnessContract.PADDED_SIMD, "compile"
    )

    assert wrong.host_verdict == "FAIL"
    assert wrong.mismatch_index == 127
    assert wrong.mismatch_expected == 0.0
    assert wrong.mismatch_actual == 1.0
    assert wrong.max_abs_error == 1.0
    assert failed.host_verdict == "FAIL"
    assert failed.exit_code == 7
    assert failed.max_abs_error is None
    assert failed.mismatch_index == 0


def test_result_round_trip_and_report_rejects_tampering(tmp_path):
    harness = CatlassContractHarness(FakeBackend())
    for contract in HarnessContract:
        write_result(
            harness.run(source_for(contract), contract, contract.value),
            tmp_path / f"{contract.value}.json",
        )
    assert report_results(tmp_path) == {
        "schema": SCHEMA,
        "total": 3,
        "passed": 3,
        "failed": 0,
        "contracts": {item.value: 1 for item in HarnessContract},
    }

    path = tmp_path / "padded-simd.json"
    record = json.loads(path.read_text())
    record["host_verdict"] = "FAIL"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="digest"):
        report_results(tmp_path)


def test_cli_preflight_run_and_report(monkeypatch, tmp_path, capsys):
    source_path = tmp_path / "kernel.py"
    source_path.write_text(source_for(HarnessContract.PADDED_SIMD))
    fake = CatlassContractHarness(FakeBackend(), device=0)
    monkeypatch.setattr(cli, "_harness", lambda args: fake)
    common = [
        "--source", str(source_path), "--contract", "padded-simd",
        "--tla-root", str(tmp_path), "--catlass-src", "/remote/catlass",
        "--catlass-revision", "9a6ac627b5f4078060287844189730cf0d184800",
        "--device", "0",
    ]
    assert cli.main(["preflight", *common]) == 0
    assert json.loads(capsys.readouterr().out)["ready"] is True

    output = tmp_path / "results/result.json"
    assert cli.main(["run", *common, "--attempt-id", "cli-1", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["host_verdict"] == "PASS"
    assert cli.main(["report", str(output.parent)]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] == 1
