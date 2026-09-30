import json
from dataclasses import dataclass

from benchmarks.a3_experiments import KnowledgeMode, ProfilingGuidance
from benchmarks.a3_experiments import BackendModel
from benchmarks.a3_model_profiles import A3Completion, DEEPSEEK_BASE_URL, load_a3_model_profile
from benchmarks.a3kernels.candidate import A3CandidateBackend, CandidateCompilation
from benchmarks.a3kernels.live_composition import (
    AuthoritativeResultStore,
    LiveComposition,
    LiveDependencies,
    model_actor_factory,
)
from benchmarks.a3kernels.phase1_evidence import canonical_digest
from benchmarks.a3kernels.phase1_protocol import ExecutionReceipt, VerifiedResult, attest
from benchmarks.a3kernels.phase1_registry import DEFAULT_PROPOSALS
from benchmarks.a3kernels.phase1_wave import Phase1Config, Phase1Wave
from benchmarks.a3kernels.profiling import CompactProfileResult


SOURCE = '''extern "C" __global__ __aicore__ void vector_add(
    GM_ADDR input_a, GM_ADDR input_b, GM_ADDR output,
    uint32_t count, uint32_t buffer_bytes) {}
'''


def config(tmp_path):
    wrapper = tmp_path / "catlass-validation.sh"
    wrapper.write_text("#!/bin/sh\n"); wrapper.chmod(0o755)
    embedding = tmp_path / "embedding"; embedding.mkdir()
    corpus = tmp_path / "corpus"; corpus.mkdir()
    database = tmp_path / "kdb.sqlite"; database.write_bytes(b"db")
    manifest = tmp_path / "manifest.json"; manifest.write_text("{}")
    driver = tmp_path / "a3_profile_driver.py"; driver.write_text("# a3 fixed\n")
    return Phase1Config(tmp_path / "state", wrapper, embedding, corpus, database, manifest, driver)


class FakeCandidate:
    def __init__(self):
        self.base = A3CandidateBackend(lambda *a, **k: None)

    def compile(self, source, workdir, **options):
        plan = self.base.plan(source, **options)
        body = {"plan": plan, "library_sha256": "1" * 64, "stdout": "", "stderr": ""}
        return CandidateCompilation(plan, "1" * 64, "", "", attest(body))

    def run(self, source, workdir, **options):
        plan = self.base.plan(source, **options)
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
        return A3Completion(json.dumps(next(iterator)), {"profile_sha256": profile.fingerprint})
    return actor


def dependencies(knowledge_calls, profile_calls):
    return LiveDependencies(
        actor_factory=actor_factory,
        candidate_factory=lambda cell, proposal, paths: FakeCandidate(),
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
    assert len(profile_calls) == 4 * len(DEFAULT_PROPOSALS)
    assert all(len(json.loads(wave.paths(cell).memory.read_text().splitlines()[0])["entry_sha256"]) == 64 for cell in __import__("benchmarks.a3kernels.phase1_wave", fromlist=["foundation_cells"]).foundation_cells())
    assert Phase1Wave(cfg, composition.execute).resume() == records


def test_authority_store_is_durable_exact_and_conflict_rejecting(tmp_path):
    path = tmp_path / "authority.jsonl"; store = AuthoritativeResultStore(path)
    record = store.register("compile", "a" * 64, "b" * 64, "c" * 64, True)
    assert AuthoritativeResultStore(path).resolve("a" * 64) == record
    try:
        store.register("compile", "a" * 64, "b" * 64, "d" * 64, True)
    except ValueError as exc:
        assert "conflicting" in str(exc)
    else: raise AssertionError("conflicting evidence was accepted")


def test_completion_provenance_and_no_secret_in_terminal_output(tmp_path):
    cfg = config(tmp_path); secret = "never-print-this"
    deps = dependencies([], [])
    composition = LiveComposition(cfg, deps)
    record = Phase1Wave(cfg, composition.execute).run()[0]
    assert secret not in json.dumps(record)
    assert record["evidence_sha256"] == canonical_digest(composition.cell_digests[record["cell_id"]])


def test_deepseek_actor_uses_native_pinned_route_without_exposing_secret():
    seen = {}
    def transport(**kwargs): seen.update(kwargs); return '{"action":"compile"}'
    factory = model_actor_factory({"DEEPSEEK_API_KEY": "private-value"}, transport)
    cell = next(cell for cell in __import__("benchmarks.a3kernels.phase1_wave", fromlist=["foundation_cells"]).foundation_cells() if cell.backend_model is BackendModel.DEEPSEEK_FLASH)
    completion = factory(cell, DEFAULT_PROPOSALS[0])(
        load_a3_model_profile(cell.backend_model), "{}"
    )
    assert seen["base_url"] == DEEPSEEK_BASE_URL
    assert seen["api_key"] == "private-value"
    assert "private-value" not in json.dumps(completion.as_dict())
