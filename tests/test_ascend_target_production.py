from types import SimpleNamespace
from dataclasses import replace
import hashlib
import json

import pytest

from benchmarks.ascend.production import NativeProjectBinding, NativeProjectBackend
from benchmarks.ascend.production import NativeProductionConfig, build_native_adaptive_phase1
from benchmarks.a5kernels.evidence import EvidenceLedger
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
from benchmarks.a3kernels.phase1_protocol import ExecutionReceipt


def spec():
    return ProjectSpec.from_mapping({
        "project_id": "native-copy", "capabilities": ["gm-ub-movement"],
        "shape": [4], "dtype": "float32", "block_count": 1,
        "padding_multiple": 1, "failure_mode": "none", "timing_lengths": [],
        "profile_metric": None, "hypothesis": "Learn a native copy.",
        "graph": {"inputs": [{"name": "x", "shape": [4]}], "nodes": [{
            "name": "out", "op": "copy", "operands": ["x"], "axis": None}],
            "output": "out"},
    })


class NativeTransport:
    def compile(self, plan, directory):
        self.plan = plan
        return SimpleNamespace(library_sha256="b" * 64, stdout="compiled", stderr="")

    def execute_receipt(self, compilation):
        return ExecutionReceipt(0, self.plan.inputs[0].values, "synchronized")


def test_native_backend_uses_shared_host_oracle(tmp_path):
    backend = NativeProjectBackend(
        NativeTransport(), tmp_path / "bundles", target="Ascend910B4",
        arch="dav-c220", profile="bz-a3-1",
    )
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "ledger.jsonl"),
                               binding=NativeProjectBinding())
    prepared = executor.prepare(spec())
    assert "tla" not in prepared.instruction
    source = prepared.abi.signature + " {}"
    assert executor.compile(spec(), source).observation.passed
    assert executor.execute(spec(), source).evidence.verified


def test_native_backend_rejects_source_error_without_device_launch(tmp_path):
    transport = NativeTransport()
    backend = NativeProjectBackend(transport, tmp_path, target="Ascend910B4",
                                   arch="dav-c220", profile="bz-a3-1")
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "ledger.jsonl"),
                               binding=NativeProjectBinding())
    result = executor.compile(spec(), "wrong signature")
    assert not result.observation.passed
    assert not hasattr(transport, "plan")


@pytest.mark.parametrize("phase2", [False, True])
@pytest.mark.parametrize("selector", [None, "native", ["native", "triton"]])
def test_native_composition_uses_target_policy_and_pinned_guide(tmp_path, selector, phase2):
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.phase1_provider import direct_profile
    guide = b"Native Ascend C programming guide"
    config = NativeProductionConfig(
        tmp_path, "Ascend910B4", "dav-c220", "bz-a3-1",
        hashlib.sha256(guide).hexdigest(), "a" * 40,
    )
    embeddings = None
    if selector:
        from test_expanded_kdb import Embeddings
        from benchmarks.a5kernels.knowledge import KnowledgeDB
        embeddings = Embeddings()
        source = tmp_path / "guide.cpp"
        source.write_text("vector copy kernel")
        kdb = tmp_path / "native.sqlite3"
        with KnowledgeDB(kdb, embeddings) as database:
            database.index(tmp_path, [source], collection="native", language="en",
                           kernel_languages=("ascend-c",))
            database.index(tmp_path, [source], collection="triton", language="mixed",
                           kernel_languages=("triton",))
        config = replace(config, kdb=kdb, collection=selector)
    kwargs = dict(config=config, native_transport=NativeTransport(),
                  profile=direct_profile(MODEL_IDENTITIES[0]), guide_bytes=guide,
                  objective="Learn native kernels", initial_memory=b"",
                  environment={"OPENAI_API_KEY": "test", "DEEPSEEK_API_KEY": "test"},
                  embeddings=embeddings)
    if phase2:
        from test_ascend_phase2_curriculum import snapshot
        kwargs["reviewed_snapshot"] = snapshot()
    controller = build_native_adaptive_phase1(**kwargs)
    assert controller.curriculum.target_policy.target == "a3"
    assert controller.project_target == 64
    assert "A3 Ascend C" in controller.execute_project.actor.system_prompt
    assert controller.execute_project.retain_semantic_failure is phase2
    identity = json.loads((tmp_path / "campaign-identity.json").read_bytes())
    if phase2:
        assert identity["phase"] == "adaptive-phase2"
        assert identity["snapshot_id"] == "reviewed-seed"
    assert build_native_adaptive_phase1(**kwargs).project_target == 64
    from benchmarks.ascend.training_traces import recorder_factory
    factory = recorder_factory(phase="phase1", round_id="native-builder-round",
                               model="fixture", source_manifest_sha256="b" * 64)
    traced = build_native_adaptive_phase1(**kwargs, trace_recorder_factory=factory)
    assert traced.execute_project.trace_recorder_factory is factory
    assert controller.execute_project.trace_recorder_factory is None
    with pytest.raises(ValueError, match="different native target"):
        build_native_adaptive_phase1(**dict(kwargs, config=replace(config, arch="other")))
    with pytest.raises(ValueError, match="guide identity"):
        build_native_adaptive_phase1(**dict(kwargs, guide_bytes=b"other guide"))


def test_native_binding_does_not_label_vector_reduction_as_simt():
    value = spec().as_dict()
    value["capabilities"] = ["vector-reduction"]
    value["graph"]["nodes"][0].update(op="reduce-sum", axis=0)
    assert NativeProjectBinding().family(ProjectSpec.from_mapping(value)).value == "vector-reduction-mask"
