import json
from dataclasses import dataclass
from pathlib import PurePosixPath

import pytest

from benchmarks.a3_experiments import KnowledgeMode, ProfilingGuidance
from benchmarks.a3_experiments import BackendModel
from benchmarks.a3_model_profiles import (
    A3Completion, A3TransportResult, DEEPSEEK_BASE_URL, load_a3_model_profile,
)
from benchmarks.a3kernels.candidate import A3CandidateBackend, CandidateCompilation
from benchmarks.a3kernels.live_composition import (
    AuthoritativeResultStore,
    LiveComposition,
    LiveDependencies,
    RemoteCandidateBundle,
    local_knowledge_factory,
    model_actor_factory,
)
from benchmarks.a3kernels.knowledge_agent import (
    KnowledgeAgent,
    KnowledgeQuery,
    QueryJournal,
)
from benchmarks.a3kernels.phase1_evidence import canonical_digest
from benchmarks.a3kernels.phase1_protocol import (
    ExecutionReceipt, FailedEvidence, VerifiedResult, attest,
)
from benchmarks.a3kernels.phase1_registry import DEFAULT_PROPOSALS
from benchmarks.a3kernels.project_execution import ProjectRuntimePolicy, RecoveryEvidence
from benchmarks.a3kernels.phase1_wave import (
    Phase1Config,
    Phase1Wave,
    foundation_cells,
)
from benchmarks.a3kernels.profiling import (
    CandidateBinding, CompactProfileResult, ProfileMetric, ProfileRequest,
    StudyDimensions,
    TimingResult,
)


SOURCE = '''extern "C" __global__ __aicore__ void vector_add(
    GM_ADDR input_a, GM_ADDR input_b, GM_ADDR output,
    uint32_t count, uint32_t buffer_bytes) {}
'''


def config(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    wrapper = tmp_path / "catlass-validation.sh"
    wrapper.write_text("#!/bin/sh\n"); wrapper.chmod(0o755)
    embedding = tmp_path / "embedding"; embedding.mkdir()
    corpus = tmp_path / "corpus"; corpus.mkdir()
    database = tmp_path / "kdb.sqlite"; database.write_bytes(b"db")
    manifest = tmp_path / "manifest.json"; manifest.write_text("{}")
    return Phase1Config(tmp_path / "state", wrapper, embedding, corpus, database, manifest)


class FakeCandidate:
    def __init__(self, proposal=None):
        self.base = A3CandidateBackend(lambda *a, **k: None)
        self.policy = ProjectRuntimePolicy.from_proposal(proposal) if proposal else None

    def compile(self, source, workdir, **options):
        plan = self.base.plan(source, **options)
        if (
            self.policy and self.policy.recovery
            and self.policy.recovery.required_evidence is RecoveryEvidence.COMPILE_FAILURE
            and source == self.policy.recovery.source
        ):
            return FailedEvidence.create(
                plan, stage="compile", error_type="CompileError", detail="registered fault"
            )
        body = {"plan": plan, "library_sha256": "1" * 64, "stdout": "", "stderr": ""}
        return CandidateCompilation(plan, "1" * 64, "", "", attest(body))

    def run(self, source, workdir, **options):
        plan = self.base.plan(source, **options)
        if (
            self.policy and self.policy.recovery
            and self.policy.recovery.required_evidence is RecoveryEvidence.HOST_VERIFICATION_FAILURE
            and source == self.policy.recovery.source
        ):
            return FailedEvidence.create(
                plan, stage="verify", error_type="Mismatch", detail="registered fault"
            )
        output = tuple(a + b for a, b in zip(plan.input_a, plan.input_b))
        return VerifiedResult.from_receipt(
            plan, ExecutionReceipt(0, output, job_handle=f"gz-a3:{plan.project_id[:12]}"),
            max_abs_error=0.0,
        )


class FakeKnowledge:
    def __init__(self, enabled, calls): self.enabled, self.calls = enabled, calls
    def query(self, query): self.calls.append(query.query); return ()


class FakeProfiler:
    def __init__(self, enabled, calls): self.enabled, self.calls = enabled, calls
    def profile(self, request):
        if not self.enabled: return None
        self.calls.append(request.request_id)
        return CompactProfileResult.create(
            request, exported_kernels=("vector_add",), metric_values=(("pipe", 1.0),),
            timeline=(("kernel", 1.0),), report_sha256="2" * 64,
        )
    def time(self, binding, dimensions):
        self.calls.append("timing")
        return TimingResult.from_samples(binding, dimensions, (1.0, 2.0))


def actor_factory(cell, proposal):
    actions = [
        {"action": "write_source", "source": SOURCE}, {"action": "compile"},
        {"action": "run"},
    ]
    if cell.knowledge is KnowledgeMode.WITH_KDB:
        actions.append({"action": "query", "query": "vector add", "limit": 1})
    if cell.profiling is ProfilingGuidance.WITH_GUIDANCE:
        actions.append({"action": "profile", "metric": "PipeUtilization"})
    actions.append({"action": "submit", "interpretation": "host facts only", "supports": []})
    iterator = iter(actions)
    def actor(profile, context):
        return A3Completion(
            json.dumps(next(iterator)), 1,
            {"profile_sha256": profile.fingerprint},
        )
    return actor


def dependencies(knowledge_calls, profile_calls):
    return LiveDependencies(
        actor_factory=actor_factory,
        candidate_factory=lambda cell, proposal, paths: FakeCandidate(proposal),
        knowledge_factory=lambda cell, paths: FakeKnowledge(
            cell.knowledge is KnowledgeMode.WITH_KDB, knowledge_calls
        ),
        profiler_factory=lambda cell, proposal, paths, verified: FakeProfiler(
            cell.profiling is ProfilingGuidance.WITH_GUIDANCE, profile_calls
        ),
    )


def test_complete_eight_cell_local_proof_is_isolated_and_resumable(tmp_path):
    cfg = config(tmp_path); knowledge_calls, profile_calls = [], []
    composition = LiveComposition(cfg, dependencies(knowledge_calls, profile_calls))
    wave = Phase1Wave(cfg, composition.execute)
    records = wave.run()
    assert len(records) == 8 and {row["status"] for row in records} == {"passed"}
    assert len(knowledge_calls) == 4 * len(DEFAULT_PROPOSALS)
    assert profile_calls.count("timing") == 8 * 3
    assert len([value for value in profile_calls if value != "timing"]) == 4 * len(DEFAULT_PROPOSALS)
    cells = __import__("benchmarks.a3kernels.phase1_wave", fromlist=["foundation_cells"]).foundation_cells()
    assert all(len(json.loads(wave.paths(cell).memory.read_text().splitlines()[0])["entry_sha256"]) == 64 for cell in cells)
    for cell in cells:
        memories = [json.loads(line)["memory"] for line in wave.paths(cell).memory.read_text().splitlines()]
        assert sum(
            fact["category"] == "profiling" and fact["statement"] == "host timing samples captured"
            for memory in memories for fact in memory["host_facts"]
        ) == 3
        failures = [
            json.loads(line)["payload"].get("failed_evidence", {})
            for line in wave.paths(cell).evidence.read_text().splitlines()
            if json.loads(line)["kind"] == "failure"
        ]
        assert {item.get("stage") for item in failures} >= {"compile", "verify"}
    assert Phase1Wave(cfg, composition.execute).resume() == records


def test_live_composition_recovers_torn_evidence_before_resume(tmp_path):
    cfg = config(tmp_path)
    composition = LiveComposition(cfg, dependencies([], []))
    cell = foundation_cells()[0]
    paths = Phase1Wave(cfg, composition.execute).paths(cell)
    first = composition.execute(cell, paths)
    committed = paths.evidence.read_bytes()
    with paths.evidence.open("ab") as stream:
        stream.write(b'{"sequence":999')

    resumed = composition.execute(cell, paths)

    assert resumed == first
    assert paths.evidence.read_bytes() == committed


def test_local_knowledge_factory_recovers_torn_query_before_continuation(tmp_path):
    cfg = config(tmp_path)
    cell = next(
        item for item in foundation_cells()
        if item.knowledge is KnowledgeMode.WITHOUT_KDB
    )
    paths = Phase1Wave(cfg, lambda *_: {}).paths(cell)
    journal_path = paths.root / "knowledge-queries.jsonl"
    reference_path = tmp_path / "reference-queries.jsonl"
    for target in (journal_path, reference_path):
        KnowledgeAgent(enabled=False, journal=QueryJournal(target)).query(
            KnowledgeQuery("first")
        )
    with journal_path.open("ab") as stream:
        stream.write(b'{"sequence":2')

    resumed = local_knowledge_factory(cfg)(cell, paths)
    resumed.query(KnowledgeQuery("second"))
    KnowledgeAgent(enabled=False, journal=QueryJournal(reference_path)).query(
        KnowledgeQuery("second")
    )

    assert journal_path.read_bytes() == reference_path.read_bytes()


def test_authority_store_is_durable_exact_and_conflict_rejecting(tmp_path):
    path = tmp_path / "authority.jsonl"; store = AuthoritativeResultStore(path)
    record = store.register("compile", "a" * 64, "b" * 64, "c" * 64, True)
    assert AuthoritativeResultStore(path).resolve("a" * 64) == record
    try:
        store.register("compile", "a" * 64, "b" * 64, "d" * 64, True)
    except ValueError as exc:
        assert "conflicting" in str(exc)
    else: raise AssertionError("conflicting evidence was accepted")


def test_authority_store_recovers_only_an_unterminated_final_record(tmp_path):
    path = tmp_path / "authority.jsonl"
    record = AuthoritativeResultStore(path).register(
        "compile", "a" * 64, "b" * 64, "c" * 64, True
    )
    committed = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b'{"kind":"host-verification"')

    recovered = AuthoritativeResultStore(path)

    assert recovered.resolve("a" * 64) == record
    assert path.read_bytes() == committed


def test_authority_store_rejects_malformed_committed_record(tmp_path):
    path = tmp_path / "authority.jsonl"
    path.write_bytes(b"not-json\n")

    with pytest.raises(ValueError, match="line 1"):
        AuthoritativeResultStore(path)


def test_completion_provenance_and_no_secret_in_terminal_output(tmp_path):
    cfg = config(tmp_path); secret = "never-print-this"
    deps = dependencies([], [])
    composition = LiveComposition(cfg, deps)
    record = Phase1Wave(cfg, composition.execute).run()[0]
    assert secret not in json.dumps(record)
    assert record["evidence_sha256"] == canonical_digest(composition.cell_digests[record["cell_id"]])


def test_deepseek_actor_uses_native_pinned_route_without_exposing_secret():
    seen = {}
    def transport(**kwargs):
        seen.update(kwargs)
        return A3TransportResult('{"action":"compile"}', 3)
    factory = model_actor_factory({"DEEPSEEK_API_KEY": "private-value"}, transport)
    cell = next(cell for cell in __import__("benchmarks.a3kernels.phase1_wave", fromlist=["foundation_cells"]).foundation_cells() if cell.backend_model is BackendModel.DEEPSEEK_FLASH)
    completion = factory(cell, DEFAULT_PROPOSALS[0])(
        load_a3_model_profile(cell.backend_model), "{}"
    )
    assert seen["base_url"] == DEEPSEEK_BASE_URL
    assert seen["api_key"] == "private-value"
    assert "private-value" not in json.dumps(completion.as_dict())


def test_remote_bundle_compile_then_run_reuses_one_managed_execution(tmp_path):
    class Managed:
        def __init__(self): self.compile_calls = self.execute_calls = 0
        def compile(self, plan, workdir):
            self.compile_calls += 1
            body = {"plan": plan, "library_sha256": "3" * 64, "stdout": "", "stderr": ""}
            return CandidateCompilation(plan, "3" * 64, "", "", attest(body))
        def execute(self, compilation):
            self.execute_calls += 1
            plan = compilation.plan
            output = tuple(a + b for a, b in zip(plan.input_a, plan.input_b))
            return VerifiedResult.from_receipt(
                plan, ExecutionReceipt(0, output, job_handle="gz-a3:managed"),
                max_abs_error=0.0,
            )
    managed = Managed(); bundle = RemoteCandidateBundle(managed)
    options = dict(request_id="cell", project_id="project", length=32,
                   padded_length=64, block_count=1, seed=0)
    compiled = bundle.compile(SOURCE, tmp_path, attempt_id="turn-2", **options)
    verified = bundle.run(SOURCE, tmp_path, attempt_id="turn-3", **options)
    assert isinstance(compiled, CandidateCompilation)
    assert isinstance(verified, VerifiedResult) and verified.passed
    assert verified.execution_id == compiled.plan.execution_id
    assert verified.attempt_id == "turn-2"
    assert (managed.compile_calls, managed.execute_calls) == (1, 1)


def test_profile_uses_exact_retained_compile_binding_and_directory(tmp_path, monkeypatch):
    import benchmarks.a3kernels.live_composition as live
    class Managed:
        def compile(self, plan, workdir):
            body = {"plan": plan, "library_sha256": "4" * 64, "stdout": "", "stderr": ""}
            return CandidateCompilation(plan, "4" * 64, "", "", attest(body))
        def execute(self, compilation):
            plan = compilation.plan
            output = tuple(a + b for a, b in zip(plan.input_a, plan.input_b))
            return VerifiedResult.from_receipt(
                plan, ExecutionReceipt(0, output, job_handle="gz-a3:exact"),
                max_abs_error=0.0,
            )
        def remote_candidate_directory(self, plan):
            return PurePosixPath("/remote/candidates") / plan.execution_id
    managed = Managed()
    store = AuthoritativeResultStore(tmp_path / "authority.jsonl")
    candidate = live._RecordingCandidate(RemoteCandidateBundle(managed), store)
    options = dict(request_id="cell", project_id="a" * 64, length=32,
                   padded_length=64, block_count=1, seed=0)
    candidate.compile(SOURCE, tmp_path, attempt_id="turn-2", **options)
    verified = candidate.run(SOURCE, tmp_path, attempt_id="turn-3", **options)
    request = ProfileRequest(
        CandidateBinding(verified.execution_id, verified.source_fingerprint),
        StudyDimensions(32, 1, 5, 20), ProfileMetric.PIPE_UTILIZATION,
    )
    seen = {}
    class ProfileBackend:
        def __init__(self, **kwargs): seen.update(kwargs)
        def profile(self, selected):
            return CompactProfileResult.create(
                selected, exported_kernels=("vector_add",),
                metric_values=(("pipe", 1.0),), timeline=(("kernel", 1.0),),
                report_sha256="5" * 64,
            )
    monkeypatch.setattr(live, "GZA3ProfilingBackend", ProfileBackend)
    cfg = config(tmp_path / "config")
    paths = Phase1Wave(cfg, lambda *_: {}).paths(
        __import__("benchmarks.a3kernels.phase1_wave", fromlist=["foundation_cells"]).foundation_cells()[0]
    )
    result = live._LazyGZProfiler(
        config=cfg, paths=paths, candidate=candidate, physical_device=0,
    ).profile(request)
    assert result.candidate_execution_id == verified.execution_id
    assert seen["remote_candidate_directory"].endswith(verified.execution_id)
    assert seen["verified_results"] == {verified.execution_id: verified}


def test_cli_run_and_resume_route_bz_and_compose_all_eight_cells(
    tmp_path, monkeypatch, capsys,
):
    import run_a3_phase1
    cfg = config(tmp_path)
    deps = dependencies([], [])
    routed = {}
    preflights = []
    monkeypatch.setattr(
        run_a3_phase1,
        "bz_live_dependencies",
        lambda *a, **k: routed.update(k) or deps,
    )
    monkeypatch.setattr(
        run_a3_phase1, "_bz_preflight",
        lambda *args: preflights.append(args) or {"state": "completed"},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    common = [
        "--state-root", str(cfg.state_root),
        "--validation-wrapper", str(cfg.validation_wrapper),
        "--embedding-cache", str(cfg.embedding_cache),
        "--corpus-artifacts", str(cfg.corpus_artifacts),
        "--knowledge-database", str(cfg.knowledge_database),
        "--knowledge-manifest", str(cfg.knowledge_manifest),
        "--profile", "bz-a3-2", "--cpl-remote", "/checked/cpl-remote",
        "--remote-workspace", "/data2/research", "--physical-device", "0",
    ]
    for command in ("run", "resume"):
        routed.clear()
        assert run_a3_phase1.main([command, *common]) == 0
        output = capsys.readouterr().out
        assert len(json.loads(output)) == 8
        assert "openrouter-secret" not in output and "deepseek-secret" not in output
        environment = routed.pop("environ")
        assert environment is __import__("os").environ
        assert routed == {
            "cpl_remote": "/checked/cpl-remote",
            "profile": "bz-a3-2",
            "remote_workspace": "/data2/research",
            "physical_device": 0,
        }
    assert preflights == [
        (cfg.validation_wrapper, "bz-a3-2", "/checked/cpl-remote"),
        (cfg.validation_wrapper, "bz-a3-2", "/checked/cpl-remote"),
    ]


def test_cli_selected_bz_transport_reaches_candidate_and_profiler(
    tmp_path, monkeypatch, capsys,
):
    import run_a3_phase1

    cfg = config(tmp_path)
    cpl_remote = tmp_path / "cpl-remote"
    cpl_remote.write_text("#!/bin/sh\n")
    cpl_remote.chmod(0o755)
    selected = {}

    class InspectingComposition:
        def __init__(self, _config, dependencies):
            self.dependencies = dependencies

        def execute(self, cell, paths):
            proposal = DEFAULT_PROPOSALS[0]
            candidate = self.dependencies.candidate_factory(cell, proposal, paths)
            profiler = self.dependencies.profiler_factory(
                cell, proposal, paths, object(),
            )
            selected.update(
                dependency_profile=self.dependencies.execution_profile,
                candidate_profile=candidate.backend.profile,
                candidate_cpl_remote=candidate.backend._cpl_remote,
                candidate_workspace=candidate.backend._workspace.as_posix(),
                candidate_device=candidate.backend._device,
                profiler_profile=profiler.profile_name,
                profiler_cpl_remote=profiler.cpl_remote,
                profiler_device=profiler.physical_device,
            )
            return {
                "status": "passed",
                "evidence_sha256": canonical_digest(selected),
            }

    monkeypatch.setattr(run_a3_phase1, "LiveComposition", InspectingComposition)
    monkeypatch.setattr(
        run_a3_phase1, "_bz_preflight", lambda *args: {"state": "completed"},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    cell = __import__(
        "benchmarks.a3kernels.phase1_wave", fromlist=["foundation_cells"]
    ).foundation_cells()[0]
    argv = [
        "run", "--state-root", str(cfg.state_root),
        "--validation-wrapper", str(cfg.validation_wrapper),
        "--embedding-cache", str(cfg.embedding_cache),
        "--corpus-artifacts", str(cfg.corpus_artifacts),
        "--knowledge-database", str(cfg.knowledge_database),
        "--knowledge-manifest", str(cfg.knowledge_manifest),
        "--profile", "bz-a3-2", "--cpl-remote", str(cpl_remote),
        "--remote-workspace", "/remote/rsi", "--physical-device", "4",
        "--cell-id", cell.cell_id,
    ]

    assert run_a3_phase1.main(argv) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1
    assert selected == {
        "dependency_profile": "bz-a3-2",
        "candidate_profile": "bz-a3-2",
        "candidate_cpl_remote": str(cpl_remote),
        "candidate_workspace": "/remote/rsi",
        "candidate_device": 4,
        "profiler_profile": "bz-a3-2",
        "profiler_cpl_remote": str(cpl_remote),
        "profiler_device": 4,
    }
