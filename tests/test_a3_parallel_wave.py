"""Independent native device slots inside the existing publication barrier."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import PurePosixPath
from threading import Barrier, Lock, local
import time

import pytest

from benchmarks.ascend.phase1_cli import main, native_devices
from benchmarks.ascend.production import NativeProjectPool
from benchmarks.a3kernels.candidate import CandidateCompilation
from benchmarks.a3kernels.phase1_protocol import ExecutionReceipt
from benchmarks.a5kernels.evidence import canonical_digest
from benchmarks.a5kernels.phase1_provider import DirectCompletion


@pytest.mark.parametrize("value", [[], [1, 1], [-1], [True], "1,2", None])
def test_pool_rejects_invalid_devices(value):
    with pytest.raises(ValueError, match="physical_devices"):
        native_devices({"physical_devices": value})


def test_legacy_device_and_explicit_pool():
    assert native_devices({"physical_device": 7}) == [7]
    assert native_devices({"physical_device": 7, "physical_devices": [2, 3]}) == [2, 3]


@pytest.mark.parametrize("explicit_device, expected", [(None, 2), (7, 7)])
def test_phase1_pool_config_hands_off_to_one_serial_phase2_device(tmp_path, monkeypatch, explicit_device, expected):
    from types import SimpleNamespace
    from benchmarks.ascend import phase1_cli, phase2_cli
    guide, initial = tmp_path / "guide.md", tmp_path / "initial.md"
    guide.write_text("Native guide")
    initial.write_bytes(b"")
    config = dict(guide=str(guide), target="a3", arch="arch", execution_profile="bz-a3-1",
        cpl_skills_revision="a" * 40, cpl_remote="/tools/cpl-remote",
        validation_wrapper="/checked/catlass-validation.sh", remote_workspace="/owned",
        physical_devices=[2, 5], environment_identity=dict(cann_version="cann", profiler_version="msprof"))
    if explicit_device is not None:
        config["physical_device"] = explicit_device
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    snapshot = SimpleNamespace(destination=SimpleNamespace(source_identity=config))
    monkeypatch.setattr(phase2_cli, "admit_native_snapshot", lambda *args: snapshot)
    monkeypatch.setattr("benchmarks.a5kernels.phase1_production.load_direct_environment", lambda *args, **kwargs: {})
    transports = []
    class Transport:
        def __init__(self, **kwargs):
            transports.append(kwargs)
    native_transport = phase1_cli.native_transport
    monkeypatch.setattr(phase1_cli, "native_transport",
        lambda config, root: native_transport(config, root, factory=Transport))
    runner = object()
    def build(admitted, guide_bytes, transport, **kwargs):
        assert admitted is snapshot and guide_bytes == b"Native guide"
        assert isinstance(transport, Transport)
        return runner
    monkeypatch.setattr(phase2_cli, "build_native_phase2", build)
    args = SimpleNamespace(config=path, phase1_root=tmp_path / "phase1", initial_memory=initial,
        review=tmp_path / "review.json", root=tmp_path / "phase2", env_file=None)
    assert phase2_cli.native_runner(args) is runner
    assert len(transports) == 1
    assert transports[0]["physical_device"] == expected
    assert transports[0]["state_directory"] == args.root / "transport-state"


def test_failed_project_unblocks_slots_without_dispatch_until_recovered():
    seen = []
    def execute(request):
        seen.append(request)
        if request == "fail":
            raise RuntimeError("provider interrupted")
        return request
    pool = NativeProjectPool([execute])
    with pytest.raises(RuntimeError, match="interrupted"):
        pool("fail")
    with ThreadPoolExecutor() as threads:
        future = threads.submit(pool, "next")
        with pytest.raises(RuntimeError, match="settle retained operations"):
            future.result(timeout=2)
    assert seen == ["fail"]
    # Explicit recovery starts a fresh pool, never reuses uncertain device state.
    assert NativeProjectPool([execute])("next") == "next"


@pytest.mark.parametrize("explicit_device, expected", [(None, 2), (3, 3)])
@pytest.mark.parametrize("original_retry", [False, True])
def test_public_campaign_build_controller_keeps_reviewed_retries_serial(tmp_path, monkeypatch, explicit_device, expected, original_retry):
    import hashlib
    from types import SimpleNamespace
    from benchmarks.ascend import campaign_cli, phase1_cli
    from benchmarks.ascend.phase2_curriculum import AdaptivePhase2Coordinator
    from test_ascend_phase2_curriculum import snapshot as reviewed_snapshot
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    guide = tmp_path / "guide.md"
    guide.write_text("Native guide")
    runtime = dict(guide=str(guide), target="a3", arch="arch", execution_profile="bz-a3-1",
        cpl_skills_revision="a" * 40, cpl_remote="/tools/cpl-remote",
        validation_wrapper="/checked/catlass-validation.sh", remote_workspace="/owned",
        physical_devices=[2, 3], environment_identity=dict(cann_version="cann", profiler_version="msprof"))
    if explicit_device is not None:
        runtime["physical_device"] = explicit_device
    runtime_path = tmp_path / "native.json"
    runtime_path.write_text(json.dumps(runtime))
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps(dict(phase1_arguments=["--config", str(runtime_path),
        "--model", "openai/gpt-5.6-sol", "--knowledge", "without-kdb",
        "--profiling", "without-profiling-guidance"])))
    measurement_bindings = []
    class Measurements:
        def __init__(self, backend, config, root, **kwargs):
            measurement_bindings.append((backend.transport._device, config["physical_device"], root))
        def profile(self, *args):
            pass
        def timing(self, *args):
            pass
    monkeypatch.setattr(phase1_cli, "NativeProjectMeasurements", Measurements)
    job = dict(target="a3", phase=1, config=str(settings),
        config_sha256=hashlib.sha256(settings.read_bytes()).hexdigest(),
        model="openai/gpt-5.6-sol", round_id="test-round", source_manifest_sha256="c" * 64,
        root=str(tmp_path / "phase1"), objective="Learn native kernels", project_target=2)
    phase1_root = tmp_path / "phase1"
    phase1_root.mkdir()
    (phase1_root / "initial-memory.bin").write_bytes(b"")
    phase1 = campaign_cli.build_controller(job)
    assert isinstance(phase1.execute_project, NativeProjectPool)
    assert [actor.executor._backend.transport._device for actor in phase1.execute_project.actors.queue] == [2, 3]
    source = json.loads((phase1_root / "campaign-identity.json").read_bytes())
    snapshot = reviewed_snapshot()
    snapshot.destination = SimpleNamespace(source_identity=source)
    monkeypatch.setattr(campaign_cli, "_snapshot", lambda *args: snapshot)
    phase2_root = tmp_path / "phase2"
    phase2_root.mkdir()
    (phase2_root / "initial-memory.bin").write_bytes(b"")
    measurement_bindings.clear()
    reviewed_job = {**job, "phase": 1 if original_retry else 2, "root": str(phase2_root)}
    if original_retry:
        from benchmarks.ascend.feedback import FeedbackStore
        from benchmarks.ascend.manual_retry import ManualPhase1Retry
        from test_generic_failure_feedback import phase1 as failed_campaign
        from test_a5_phase1_adaptive import _project, _result
        original_spec = _project('original', 'gm-ub-movement')
        failed_campaign(tmp_path/'failed', spec=original_spec)
        feedback = FeedbackStore(tmp_path/'feedback')
        feedback.import_phase1(tmp_path/'failed', target='a3', language='ascend-c')
        original, = feedback.records()
        selection = json.dumps(dict(selected=[original])).encode()
        (phase2_root/'phase2-feedback.json').write_bytes(selection)
        reviewed_job.update(phase1_original_retry=True, phase2_max_seconds=30,
            feedback_root=str(feedback.root), feedback_selection_sha256=hashlib.sha256(selection).hexdigest())
    phase2 = campaign_cli.build_controller(reviewed_job)
    if original_retry:
        assert isinstance(phase2, ManualPhase1Retry)
        retry = phase2
        phase2 = retry.runner
        assert retry.row['outcome']['original_spec'] == original_spec
    assert isinstance(phase2, AdaptivePhase2Coordinator)
    assert phase2.snapshot is snapshot
    assert phase2.max_parallel == 1
    transport = phase2.execute_project.executor._backend.transport
    assert transport._device == expected
    assert transport._state_root == phase2_root / "transport-state"
    assert measurement_bindings == [(expected, expected, phase2_root / "measurements")]

    if original_retry:
        requests = []
        def execute(request):
            requests.append(request)
            return _result(request.spec, True, 'host acceptance')
        phase2.execute_project = execute
        assert retry.resume()['status'] == 'FIXED'
        request, = requests
        assert request.spec.as_dict() == original_spec
        envelope = json.loads(request.pre_wave_memory)
        import base64
        reviewed = json.loads(base64.b64decode(envelope['initial_memory_base64']))
        assert reviewed['snapshot_id'] == snapshot.snapshot_id
        assert reviewed['phase1_memory'] == list(snapshot.memory_entries())
        assert envelope['project_memory_base64'] == []


def test_semantic_project_failure_does_not_stop_the_pool():
    from benchmarks.a5kernels.phase1_wave_runtime import SemanticProjectFailure
    def execute(request):
        if request == "fail":
            raise SemanticProjectFailure("no valid candidate", {})
        return request
    pool = NativeProjectPool([execute])
    with pytest.raises(SemanticProjectFailure):
        pool("fail")
    assert pool("next") == "next"


@pytest.mark.parametrize("interrupt", [False, True])
def test_parallel_cli_isolates_devices_measurements_and_preserves_resume(tmp_path, capsys, interrupt):
    guide = tmp_path / "guide.md"
    guide.write_text("Native guide")
    initial = tmp_path / "initial.md"
    initial.write_text("")
    config = dict(guide=str(guide), target="discovered-a3", arch="discovered-arch",
        execution_profile="bz-a3-1", cpl_skills_revision="a" * 40,
        cpl_remote="/tools/cpl-remote", validation_wrapper="/checked/catlass-validation.sh",
        remote_workspace="/owned", physical_devices=[2, 5],
        environment_identity=dict(cann_version="cann", profiler_version="msprof"))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    root = tmp_path / "campaign"
    barrier, lock = Barrier(2), Lock()
    transports, measurements, openings = [], [], []
    first_run = True

    class Transport:
        def __init__(self, **kwargs):
            self.device = kwargs["physical_device"]
            self.state = kwargs["state_directory"]
            self.first = True
            transports.append(self)
        def compile(self, plan, directory):
            if plan.project_id == "sum-three":
                return CandidateCompilation(plan, "b" * 64, "compiled", "", "a" * 64)
            if self.first and first_run:
                self.first = False
                barrier.wait(timeout=5)  # Fails if the CLI silently remains serial.
            assert f"device-{self.device}" in str(directory)
            return CandidateCompilation(plan, "b" * 64, "compiled", "", "a" * 64)
        def execute_receipt(self, compiled):
            if compiled.plan.project_id == "sum-three":
                return ExecutionReceipt(0, tuple(sum(xs) for xs in zip(*(t.values for t in compiled.plan.inputs))))
            if interrupt and first_run and compiled.plan.project_id == "copy-1":
                raise OSError("temporary host failure")
            return ExecutionReceipt(0, compiled.plan.inputs[0].values, "synchronized")
        def remote_candidate_directory(self, plan):
            return PurePosixPath("/owned/.rsi-a3/candidates") / plan.execution_id

    class Measurement:
        def __init__(self, **kwargs):
            self.plan = kwargs["compilation"].plan
            self.device = kwargs["physical_device"]
            assert self.device == kwargs["candidate_backend"].device
            assert f"device-{self.device}" in str(kwargs["evidence_directory"])
        def result(self, kind):
            with lock:
                measurements.append((self.plan.project_id, self.device, kind))
            payload = dict(project=self.plan.project_id, device=self.device, kind=kind)
            digest = canonical_digest(payload)
            return dict(identity=dict(output=dict(shape=list(self.plan.output.shape),
                logical_shape=list(self.plan.output.logical_shape))), summary=dict(median_us=1.0, kind=kind),
                artifact_payloads={digest: payload})
        def timing(self, **kwargs):
            return self.result("timing")
        def capture(self, metric):
            return self.result("profile")

    state = local()
    source = 'extern "C" __global__ __aicore__ void project_kernel(GM_ADDR gm_input0, GM_ADDR gm_output) {}'
    actions = [json.dumps({"write": {"slot": "kernel", "content": source}}),
               '{"run":{}}', '{"request-timing":{}}', '{"request-profile":{}}', '{"submit":{}}']

    def provider(**kwargs):
        messages = kwargs["request"]["messages"]
        if "Curriculum Agent" in messages[0]["content"]:
            context = json.loads(messages[-1]["content"].rsplit("\n", 1)[-1])
            projects = []
            for i, size in enumerate((4, 8)):
                spec = dict(project_id=f"copy-{i}", capabilities=["gm-ub-movement", "timing-sweep", "profiling"],
                    shape=[size], dtype="float32", block_count=1, padding_multiple=1, failure_mode="none",
                    timing_lengths=[size, size + 1], profile_metric="pipe-utilization", hypothesis="Study native movement.",
                    graph=dict(inputs=[dict(name="x", shape=[size])], nodes=[
                        dict(name="out", op="copy", operands=["x"], axis=None)], output="out"))
                projects.append(dict(spec=spec, requires=[]))
            text = "DECISION: WAVE\n" + json.dumps(dict(wave_id=context["next_wave_id"],
                pre_wave_memory_sha256=context["pre_wave_memory_sha256"],
                pre_wave_coverage=context["pre_wave_coverage"], projects=projects))
        elif "Distill only" in messages[0]["content"] or "Distill only" in messages[-1]["content"]:
            text = '{"distillation":{"no_new_knowledge":true,"lessons":[]}}'
        else:
            index = getattr(state, "index", 0)
            if index == 0:
                with lock:
                    openings.append(messages[1]["content"])
            text = actions[index]
            state.index = index + 1
        return DirectCompletion(text, 1)

    args = ["--target", "a3", "run", "--root", str(root), "--objective", "Learn native measurement",
        "--initial-memory", str(initial), "--config", str(path), "--model", "openai/gpt-5.6-sol",
        "--knowledge", "without-kdb", "--profiling", "without-profiling-guidance",
        "--project-target", "2", "--max-projects", "2"]
    deps = dict(transport_factory=Transport, provider_transport=provider, measurement_factory=Measurement,
        environment_loader=lambda **kwargs: dict(OPENAI_API_KEY="fake", DEEPSEEK_API_KEY="fake"))
    main(args, **deps)
    if interrupt:
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == "WAVE_CLOSED"
        assert report["published_projects"] == ["copy-0"]
        assert report["wave_results"][-1]["status"] == "partial"
    environment_reports = list(root.glob("environment-checks/*/*/report.json"))
    assert environment_reports and all(json.loads(p.read_text())["status"] == "passed" for p in environment_reports)
    assert {transport.device for transport in transports} == {2, 5}
    assert len({transport.state for transport in transports}) == 2
    assert all("native-workers" in str(transport.state) for transport in transports)
    assert len(openings) == 2
    # Both projects see exactly the same production context and pre-wave memory.
    assert openings[0].split("\nPre-wave memory (immutable, readable index): ")[1] == openings[1].split(
        "\nPre-wave memory (immutable, readable index): ")[1]
    for project in (("copy-0",) if interrupt else ("copy-0", "copy-1")):
        rows = [row for row in measurements if row[0] == project]
        assert {row[2] for row in rows} == {"timing", "profile"}
        assert len({row[1] for row in rows}) == 1
    if not interrupt:
        assert len({row[1] for row in measurements}) == 2
    publication = root / "waves/wave-001/published.json"
    before = {p: p.read_bytes() for p in root.glob("waves/wave-001/branches/*/*") if p.is_file()}
    assert publication.exists()
    assert list(json.loads(publication.read_bytes())["commits"])[0]["project_id"] == "copy-0"
    main(["resume" if arg == "run" else arg for arg in args], **deps)
    assert before == {p: p.read_bytes() for p in before}
    assert len(openings) == 2


def test_phase2_rejects_parallel_workers_before_opening_state(tmp_path):
    from benchmarks.ascend.production import build_native_adaptive_phase1
    with pytest.raises(ValueError, match="Phase 2 remains serial"):
        build_native_adaptive_phase1(config=None, native_transport=None, profile=None,
            guide_bytes=b"guide", objective="phase2", initial_memory=b"", environment={},
            project_workers=(object(),), reviewed_snapshot=object())


def test_parallel_actors_serialize_shared_kdb_query(tmp_path, monkeypatch):
    import hashlib
    from benchmarks.ascend.production import (
        NativeProductionConfig, NativeProjectWorker, build_native_adaptive_phase1,
    )
    from benchmarks.a5kernels.phase1_provider import direct_profile
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.knowledge import KnowledgeDB
    from test_expanded_kdb import Embeddings
    embeddings = Embeddings()
    source = tmp_path / "source.cpp"
    source.write_text("Native vector example")
    kdb = tmp_path / "kdb.sqlite3"
    with KnowledgeDB(kdb, embeddings) as database:
        database.index(tmp_path, [source], collection="native", language="ascend-c")
    active, peak = 0, 0
    lock = Lock()
    def query(payload, context):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return payload
    monkeypatch.setattr("benchmarks.a5kernels.phase1_adaptive_production._knowledge_action",
                        lambda *args: query)
    guide = b"Native guide"
    config = NativeProductionConfig(tmp_path / "campaign", "a3", "arch", "bz-a3-1",
        hashlib.sha256(guide).hexdigest(), "a" * 40, kdb=kdb, collection="native")
    coordinator = build_native_adaptive_phase1(config=config, native_transport=None,
        profile=direct_profile(MODEL_IDENTITIES[0]), guide_bytes=guide, objective="Learn native kernels",
        initial_memory=b"", environment=dict(OPENAI_API_KEY="fake", DEEPSEEK_API_KEY="fake"),
        embeddings=embeddings, project_workers=tuple(
            NativeProjectWorker(tmp_path / f"worker-{i}", object()) for i in range(2)))
    actors = list(coordinator.execute_project.actors.queue)
    with ThreadPoolExecutor(max_workers=2) as threads:
        calls = [threads.submit(actor.query_knowledge, {"query": "copy"}, None) for actor in actors]
        assert all(call.result(timeout=2) == {"query": "copy"} for call in calls)
    assert peak == 1



pytestmark = pytest.mark.usefixtures("scripted_pi")
