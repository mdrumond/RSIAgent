import json
from pathlib import PurePosixPath
from dataclasses import replace
import pytest

from benchmarks.ascend.phase1_cli import main
from benchmarks.a3kernels.candidate import CandidateCompilation
from benchmarks.a3kernels.phase1_protocol import ExecutionReceipt
from benchmarks.a5kernels.phase1_provider import DirectCompletion
from benchmarks.a5kernels.evidence import canonical_digest


def test_a5_selection_delegates_exact_legacy_arguments():
    calls = []
    main(["--target", "a5", "report", "--root", "existing"], a5_main=lambda args: calls.append(args))
    assert calls == [["report", "--root", "existing"]]


@pytest.mark.parametrize("sync_performance", [False, True])
def test_native_launcher_runs_fake_provider_through_host_oracle_and_profile(tmp_path, capsys, sync_performance):
    guide = tmp_path / "guide.md"
    guide.write_text("Native guide; do not use Catlass DSL.")
    memory = tmp_path / "initial.md"
    memory.write_text("")
    config = dict(guide=str(guide), target="discovered-a3", arch="discovered-arch",
        execution_profile="bz-a3-1", cpl_skills_revision="a" * 40,
        cpl_remote="/tools/cpl-remote", validation_wrapper="/checked/catlass-validation.sh",
        remote_workspace="/owned", physical_device=1,
        environment_identity=dict(cann_version="discovered-cann", profiler_version="discovered-msprof"))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    root = tmp_path / "campaign"
    events = []

    class Transport:
        def __init__(self, **kwargs):
            assert kwargs["profile"] == "bz-a3-1"
        def compile(self, plan, directory):
            events.append("compile")
            self.plan = plan
            return CandidateCompilation(plan, "b" * 64, "compiled", "", "a" * 64)
        def execute_receipt(self, compiled):
            events.append("run")
            return ExecutionReceipt(0, compiled.plan.inputs[0].values, "synchronized")
        def remote_candidate_directory(self, plan):
            return PurePosixPath("/owned") / plan.execution_id

    class Measurements:
        def __init__(self, **kwargs):
            assert kwargs["expected_output"]
            assert kwargs["compilation"].plan.kernel_name == "project_kernel"
        def capture(self, metric):
            events.append("profile")
            raw = dict(actual_metric=metric, report_sha256="d" * 64)
            digest = canonical_digest(raw)
            return dict(identity=dict(kernel_name="project_kernel"), summary=raw,
                        artifact_payloads={digest: raw}, evidence_sha256=digest)

    source = 'extern "C" __global__ __aicore__ void project_kernel(GM_ADDR gm_input0, GM_ADDR gm_output) {}'
    actions = [json.dumps({"write": {"slot": "kernel", "content": source}}),
               '{"compile":{}}', '{"run":{}}', '{"request-profile":{}}']
    if sync_performance:
        actions += [json.dumps({"write": {"slot": "kernel", "content": source + " // variant"}}),
                    '{"run":{}}', '{"request-profile":{}}']
    actions = iter([*actions, '{"submit":{}}'])
    def provider(**kwargs):
        messages = kwargs["request"]["messages"]
        if "Curriculum Agent" in messages[0]["content"]:
            context = json.loads(messages[-1]["content"].rsplit("\n", 1)[-1])
            spec = dict(project_id="profile-copy", capabilities=["gm-ub-movement", "profiling"],
                shape=[4], dtype="float32", block_count=1, padding_multiple=1, failure_mode="none",
                timing_lengths=[], profile_metric="pipe-utilization", hypothesis="Study native copy pipeline.",
                graph=dict(inputs=[dict(name="x", shape=[4])], nodes=[dict(name="out", op="copy", operands=["x"], axis=None)], output="out"))
            if sync_performance:
                spec["learning_focus"] = ["sync-performance"]
            text = "DECISION: WAVE\n" + json.dumps(dict(wave_id=context["next_wave_id"],
                pre_wave_memory_sha256=context["pre_wave_memory_sha256"], pre_wave_coverage=context["pre_wave_coverage"],
                projects=[dict(spec=spec, requires=[])]))
        elif "Distill only" in messages[0]["content"] or "Distill only" in messages[-1]["content"]:
            text = '{"distillation":{"no_new_knowledge":true,"lessons":[]}}'
        else:
            text = next(actions)
        return DirectCompletion(text, 1)

    main(["--target", "a3", "run", "--root", str(root), "--objective", "Learn native copy profiling",
          "--initial-memory", str(memory), "--config", str(config_path), "--model", "openai/gpt-5.6-sol",
          "--knowledge", "without-kdb", "--profiling", "without-profiling-guidance",
          "--project-target", "1", "--max-projects", "1"],
         transport_factory=Transport, provider_transport=provider, measurement_factory=Measurements,
         environment_loader=lambda **kwargs: dict(OPENAI_API_KEY="fake", DEEPSEEK_API_KEY="fake"))
    assert "profile" in events
    assert events.index("run") < events.index("profile")
    if sync_performance:
        assert events.count("profile") == 2
        from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
        import hashlib
        store = ContentAddressedArtifacts(root / "waves/wave-001/branches/profile-copy/artifacts")
        payloads = [json.loads(path.read_bytes()) for path in store.objects.glob("*/*")
                    if path.read_bytes().startswith(b'{')]
        assert {hashlib.sha256(value.encode()).hexdigest() for value in (source, source + " // variant")} <= {
            row.get("candidate_sha256") for row in payloads}
    assert json.loads((root / "campaign-identity.json").read_text())["language"] == "ascend-c"
    assert "discovered-a3" in (root / "campaign-identity.json").read_text()
    assert "CATLASS" not in capsys.readouterr().out


def test_timing_kernel_contract_has_host_shape_macros():
    from benchmarks.a3kernels.native_smoke import smoke_plan
    plan = smoke_plan(target="discovered-a3", arch="discovered-arch")
    wrapper = next(f.content for f in plan.files if f.relative_path == "host_wrapper.inc")
    assert "#define RSI_NATIVE_INPUT0_NUMEL 32" in wrapper
    assert "#define RSI_NATIVE_OUTPUT_DIM0 32" in wrapper


def test_native_timing_action_recompiles_each_shape_with_host_inputs(tmp_path):
    from test_ascend_target_production import spec
    from benchmarks.ascend.production import NativeProjectBackend, NativeProjectBinding
    from benchmarks.ascend.phase1_cli import NativeProjectMeasurements
    from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
    from benchmarks.a5kernels.phase1_curriculum import Capability
    from benchmarks.a5kernels.evidence import EvidenceLedger
    shapes = []
    logical_shapes = []
    class Transport:
        def compile(self, plan, directory):
            shapes.append(plan.output.shape)
            logical_shapes.append(plan.output.logical_shape)
            wrapper = next(f.content for f in plan.files if f.relative_path == "host_wrapper.inc")
            assert f"#define RSI_NATIVE_INPUT0_LOGICAL_NUMEL {plan.inputs[0].logical_shape[0]}" in wrapper
            assert "#define RSI_NATIVE_INPUT0_NUMEL 16" in wrapper
            return CandidateCompilation(plan, "a" * 64, "compiled", "", "b" * 64)
        def remote_candidate_directory(self, plan):
            return PurePosixPath("/owned") / plan.execution_id
    class Measurement:
        def __init__(self, **kwargs):
            self.plan = kwargs["compilation"].plan
        def timing(self, **kwargs):
            payload = dict(shape=list(self.plan.output.shape), logical_shape=list(self.plan.output.logical_shape))
            digest = canonical_digest(payload)
            return dict(identity=dict(output=payload), summary=dict(median_us=1.0), artifact_payloads={digest: payload})
    backend = NativeProjectBackend(Transport(), tmp_path, target="actual-a3", arch="native-arch", profile="bz-a3-1")
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "ledger"), binding=NativeProjectBinding())
    study = replace(spec(), capabilities=(Capability.GM_UB_MOVEMENT, Capability.PADDED_TAIL, Capability.TIMING_SWEEP),
                    padding_multiple=16, timing_lengths=(3, 5))
    prepared = executor.prepare(study)
    request = executor.request(prepared, prepared.abi.signature + " {}")
    config = dict(environment_identity=dict(cann_version="cann", profiler_version="msprof"),
        cpl_remote="/tools/cpl-remote", validation_wrapper="/checked/catlass-validation.sh",
        execution_profile="bz-a3-1", physical_device=1)
    result = NativeProjectMeasurements(backend, config, tmp_path, factory=Measurement).timing(request)
    assert shapes == [(16,), (16,)]
    assert logical_shapes == [(3,), (5,)]
    assert [item["shape"] for item in result["summary"]["timing"]] == [[3], [5]]
    import hashlib
    source_sha256 = hashlib.sha256(request.candidate_source.encode()).hexdigest()
    assert len([payload for payload in result["artifact_payloads"].values()
                if payload.get("candidate_sha256") == source_sha256]) == 2


def test_native_runtime_config_fails_early_for_missing_version_identity(tmp_path):
    from benchmarks.ascend.phase1_cli import load_native_config
    path = tmp_path / "config.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="missing fields"):
        load_native_config(path)
