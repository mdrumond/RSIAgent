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


@pytest.mark.parametrize("reviewed", [False, True])
@pytest.mark.parametrize("selector", [None, "native", ["native", "triton"]])
def test_native_composition_uses_target_policy_and_pinned_guide(tmp_path, selector, reviewed):
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
    if reviewed:
        from reviewed_fixtures import snapshot
        kwargs["reviewed_snapshot"] = snapshot()
    controller = build_native_adaptive_phase1(**kwargs)
    assert controller.curriculum.target_policy.target == "a3"
    assert controller.project_target == 64
    assert controller.execute_project.retain_semantic_failure is True
    identity = json.loads((tmp_path / "campaign-identity.json").read_bytes())
    if reviewed:
        assert identity["reviewed_snapshot_id"] == "reviewed-seed"
        assert "phase" not in identity
    assert build_native_adaptive_phase1(**kwargs).project_target == 64
    # New goal qualification may observe a compiler without changing reviewed seed pins.
    observed = build_native_adaptive_phase1(**dict(kwargs, config=replace(config, compiler_version="observed-compiler")))
    assert json.loads((tmp_path / "campaign-identity.json").read_bytes()) == identity
    assert observed.snapshot.snapshot_id == "reviewed-seed" if reviewed else True
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


def test_parallel_native_composition_preserves_each_workers_runtime_on_resume(tmp_path):
    from benchmarks.ascend.production import NativeProjectWorker, native_target_runtime
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.phase1_provider import direct_profile

    guide = b"Native Ascend C guide"
    config = NativeProductionConfig(tmp_path, "observed-product", "observed-arch", "bz-a3-1",
        hashlib.sha256(guide).hexdigest(), "a" * 40,
        environment_identity={"cann_version": "observed-cann"}, compiler_version="observed-compiler")
    workers = tuple(NativeProjectWorker(tmp_path / f"worker-{device}",
        SimpleNamespace(_device=device, profile="bz-a3-1")) for device in (2, 5))
    for _ in range(2):
        controller = build_native_adaptive_phase1(config=config, native_transport=workers[0].transport,
            profile=direct_profile(MODEL_IDENTITIES[0]), guide_bytes=guide,
            objective="Learn native kernels", initial_memory=b"",
            environment={"OPENAI_API_KEY": "test", "DEEPSEEK_API_KEY": "test"},
            project_workers=workers)
        assert controller.max_parallel == 2
        actors = tuple(controller.execute_project.actors.queue)
        assert [actor.executor.runtime_identity for actor in actors] == [
            native_target_runtime(config, worker.transport) for worker in workers]


@pytest.mark.parametrize('lock_wait', [1, 6])
def test_native_curriculum_bounds_shared_lock_and_forwards_remaining(tmp_path, monkeypatch, lock_wait):
    import benchmarks.ascend.production as production
    import benchmarks.a5kernels.phase1_adaptive_production as adaptive
    from benchmarks.a5kernels.knowledge import KnowledgeDB
    from benchmarks.a5kernels.knowledge_agent import KnowledgeQuery
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.phase1_provider import direct_profile
    from test_expanded_kdb import Embeddings
    clock, acquired, released, queries = SimpleNamespace(value=1000), [], [], []
    class WaitingLock:
        def acquire(self, *, timeout):
            acquired.append(timeout)
            clock.value += lock_wait
            return True
        def release(self):
            released.append(True)
    monkeypatch.setattr(production, 'Lock', WaitingLock)
    def query(action, *, check_remaining):
        queries.append((action.query, check_remaining()))
        return ()
    monkeypatch.setattr(adaptive, 'curriculum_knowledge', lambda *args: (query, {'enabled': True, 'manifests': {}}))
    guide, embeddings = b'Native guide', Embeddings()
    source = tmp_path/'guide.cpp'
    source.write_text('vector copy kernel')
    database_path = tmp_path/'native.sqlite3'
    with KnowledgeDB(database_path, embeddings) as database:
        database.index(tmp_path, [source], collection='native', language='en', kernel_languages=('ascend-c',))
    config = NativeProductionConfig(tmp_path/'campaign', 'Ascend910B4', 'dav-c220', 'bz-a3-1',
        hashlib.sha256(guide).hexdigest(), 'a'*40, kdb=database_path, collection='native', curriculum_kdb=True)
    controller = build_native_adaptive_phase1(config=config, native_transport=NativeTransport(),
        profile=direct_profile(MODEL_IDENTITIES[0]), guide_bytes=guide, objective='Learn native kernels',
        initial_memory=b'', environment={'OPENAI_API_KEY': 'test', 'DEEPSEEK_API_KEY': 'test'}, embeddings=embeddings)
    def remaining():
        seconds = 1005-clock.value
        if seconds <= 0:
            raise TimeoutError('selection expired while awaiting model lock')
        return seconds
    if lock_wait < 5:
        assert controller.curriculum.query_knowledge(KnowledgeQuery('copy'), check_remaining=remaining) == ()
        assert queries == [('copy', 4)]
    else:
        with pytest.raises(TimeoutError):
            controller.curriculum.query_knowledge(KnowledgeQuery('copy'), check_remaining=remaining)
        assert queries == []
    assert acquired == [5]
    assert released == [True]
