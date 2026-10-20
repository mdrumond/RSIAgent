import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmarks.ascend.phase2 import admit_native_snapshot, build_native_phase2, source_cell_id, REVIEW_SCHEMA
from benchmarks.ascend.attention import ATTENTION_CASES, attention_reference, build_execution
from benchmarks.a5kernels.evidence import canonical_bytes
from benchmarks.a5kernels.phase1_provider import DirectCompletion
from tests.test_a5_phase2_adaptive_memory import _source
from tests.test_a5_phase2_source_provider import GUIDE, ENVIRONMENT


SOURCE = 'extern "C" __global__ __aicore__ void dense_attention(GM_ADDR gm_q, GM_ADDR gm_k, GM_ADDR gm_v, GM_ADDR gm_output) {}'


def native_source(tmp_path, *, profiling=False):
    root, source, review_path, memory = _source(tmp_path)
    identity = dict(schema="ascend-native-adaptive-identity-v1", language="ascend-c",
                    target="Ascend910_9382", arch="dav-c220", execution_profile="bz-a3-1",
                    guide_sha256=hashlib.sha256(GUIDE.encode()).hexdigest(), cpl_skills_revision="1" * 40,
                    model=source.model.as_dict(), dtypes=["float16", "bfloat16"],
                    environment_identity=dict(cann_version="test-cann", profiler_version="test-profiler"),
                    profiling_guidance="Use the native profiler" if profiling else None,
                    profiling_mode="new" if profiling else "without-profiling-guidance",
                    measurement_actions={"profiling": profiling, "timing": False},
                    kdb_collection=None)
    (root / "campaign-identity.json").write_bytes(canonical_bytes(identity))
    review = json.loads(review_path.read_bytes())
    review.update(schema=REVIEW_SCHEMA, cell_id=source_cell_id(identity))
    review_path.write_bytes(canonical_bytes(review))
    return root, review_path, memory


def test_native_handoff_is_immutable_and_rejects_a5(tmp_path):
    root, review, memory = native_source(tmp_path)
    snapshot = admit_native_snapshot(root, b"", review, tmp_path / "phase2")
    assert snapshot.memory_entries() == (memory,)
    assert snapshot.destination.as_dict()["language"] == "ascend-c"
    assert snapshot.destination.as_dict()["target"] == "a3"
    (root / "state.json").unlink()
    snapshot.verify()
    snapshot.memory_path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        snapshot.verify()


def test_a5_identity_is_not_admitted_as_native(tmp_path):
    root, source, review, _ = _source(tmp_path)
    with pytest.raises(ValueError):
        admit_native_snapshot(root, b"", review, tmp_path / "phase2")


def test_native_new_profiling_label_and_exact_content_are_preserved(tmp_path):
    from benchmarks.ascend.phase2 import NativeSourceProvider
    from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider
    from benchmarks.a5kernels.phase2_live import Phase2AttemptRequest
    root, review, memory = native_source(tmp_path, profiling=True)
    snapshot = admit_native_snapshot(root, b"", review, tmp_path / "phase2")
    assert snapshot.destination.profiling.value == "new"
    provider = NativeSourceProvider(DirectCompletionProvider(ENVIRONMENT, transport=lambda **_: None),
                                   evidence_root=tmp_path / "provider", guide_text=GUIDE,
                                   profiling_guidance="Use the native profiler")
    request = Phase2AttemptRequest(snapshot.destination, "1" * 64, 1, "req", "try", (memory,), ())
    provider._validate_request(request)
    provider.profiling_guidance = "wrong product guide"
    with pytest.raises(ValueError, match="guidance changed"):
        provider._validate_request(request)


@pytest.mark.parametrize("repair", [False, True])
def test_reviewed_handoff_to_shared_model_loop_and_numerical_verifier(tmp_path, repair):
    root, review, memory = native_source(tmp_path)
    snapshot = admit_native_snapshot(root, b"", review, tmp_path / "phase2")
    calls = []

    def provider_transport(**kwargs):
        calls.append(kwargs)
        return DirectCompletion(json.dumps({"candidate_source": {"source": SOURCE}}), 1)

    class Runtime:
        def compile(self, plan, directory):
            return plan

        def execute_receipt(self, plan):
            case = next(c for c in ATTENTION_CASES if c.case_id == plan.project_id)
            output = attention_reference(build_execution(case))
            if repair and plan.attempt_id.endswith("a01"):
                output = (42.0,) * len(output)
            return SimpleNamespace(exit_code=0, output=output,
                                   stdout="", stderr="", job_handle="bz-a3-1:fixture", metadata=())

    runner = build_native_phase2(snapshot, GUIDE.encode(), Runtime(), environment=ENVIRONMENT,
                                 provider_transport=provider_transport)
    result = runner.run()
    assert result["status"] == "passed"
    prompt = json.loads(calls[0]["request"]["messages"][1]["content"])
    assert prompt["phase1_memory"] == [memory]
    assert "Ascend C" in prompt["task"]
    assert prompt["target_contract"]["target_id"].startswith("a3-")
    assert "Catlass DSL" not in calls[0]["request"]["messages"][0]["content"]
    assert "catlass" not in json.dumps(prompt["target_contract"]).lower()
    assert runner.run() == result and len(calls) == (2 if repair else 1)
    if repair:
        second = json.loads(calls[1]["request"]["messages"][1]["content"])
        assert second["phase2_memory"][0]["passed"] is False
        assert second["phase2_memory"][0]["failed_cases"][0]["failure"] == "numerical-mismatch"


def test_cli_admit_and_injected_run(tmp_path, capsys):
    from benchmarks.ascend.phase2_cli import main
    root, review, _ = native_source(tmp_path)
    initial = tmp_path / "initial.bin"
    initial.write_bytes(b"")
    args = ["--target", "a3", "admit", "--config", str(tmp_path / "runtime.json"),
            "--root", str(tmp_path / "phase2"), "--phase1-root", str(root),
            "--initial-memory", str(initial), "--review", str(review)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["cell_id"].startswith("a3-phase2-cell-")
    args[2] = "run"
    assert main(args, runner_factory=lambda _: SimpleNamespace(run=lambda: {"status": "passed"})) == 0


def test_cli_native_builder_reuses_shared_runtime_config(tmp_path, monkeypatch):
    from benchmarks.ascend import phase1_cli, phase2_cli
    from benchmarks.a5kernels import phase1_production
    root, review, _ = native_source(tmp_path)
    identity = json.loads((root / "campaign-identity.json").read_bytes())
    guide = tmp_path / "native-guide.md"
    guide.write_text(GUIDE)
    initial = tmp_path / "initial.bin"
    initial.write_bytes(b"")
    config = {key: identity[key] for key in ("target", "arch", "execution_profile", "cpl_skills_revision", "environment_identity")}
    config.update(guide=str(guide), cpl_remote="/tools/cpl-remote",
                  validation_wrapper="/tools/catlass-validation.sh", remote_workspace="/owned", physical_device=1)
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(config))
    args = SimpleNamespace(config=path, phase1_root=root, initial_memory=initial,
                           review=review, root=tmp_path / "phase2", env_file=None)
    monkeypatch.setattr(phase1_cli, "native_transport", lambda config, root: object())
    monkeypatch.setattr(phase1_production, "load_direct_environment", lambda *_, **__: ENVIRONMENT)
    runner = phase2_cli.native_runner(args)
    assert runner.execution_profile == "bz-a3-1"
    assert not hasattr(runner.dependencies.cell.guide, "catlass_revision")
    config["environment_identity"]["cann_version"] = "changed"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="environment differs"):
        phase2_cli.native_runner(args)


pytestmark = pytest.mark.usefixtures("scripted_pi")
