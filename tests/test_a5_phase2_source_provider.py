from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json

import pytest

from benchmarks.a5kernels.knowledge_agent import (
    Citation,
    KnowledgeResult,
    ProgressiveMemoryJournal,
)
from benchmarks.a5kernels.phase1_experiments import (
    GUIDE_SCHEMA,
    MODEL_IDENTITIES,
    PINNED_CATLASS_REVISION,
    KnowledgeMode,
    ProfilingGuidance,
    ProgrammingGuideIdentity,
)
from benchmarks.a5kernels.phase1_live import InfrastructureFailure
from benchmarks.a5kernels.phase1_provider import DirectCompletion, DirectCompletionProvider
from benchmarks.a5kernels.phase2_live import Phase2AttemptRequest
from benchmarks.a5kernels.phase2_memory import phase2_cells
import benchmarks.a5kernels.phase2_source_provider as source_provider_module
from benchmarks.a5kernels.phase2_source_provider import Phase2SourceProvider


GUIDE = "Use host verification for every candidate."
GUIDE_ID = ProgrammingGuideIdentity(
    GUIDE_SCHEMA,
    hashlib.sha256(GUIDE.encode()).hexdigest(),
    "1" * 40,
    PINNED_CATLASS_REVISION,
)
QUALIFICATION = "2" * 64
ENVIRONMENT = {"OPENAI_API_KEY": "one", "DEEPSEEK_API_KEY": "two"}


class FakeKnowledge:
    def __init__(self, journal, *, enabled=True):
        self.enabled = enabled
        self.journal = journal
        self.queries = []
        self.closed = False

    def close(self):
        self.closed = True

    def query(self, action):
        self.queries.append(action)
        result = KnowledgeResult(
            Citation("catlass", "examples/add.py", 4, 9, "a" * 64),
            "reference text",
            0.9,
        )
        self.journal.append(
            enabled=True,
            collection="catlass",
            query=action.query,
            citations=(result.citation,),
            manifest=type("Manifest", (), {
                "fingerprint": "b" * 64,
                "embedding_model": "bge",
                "embedding_revision": "rev",
            })(),
        )
        return (result,)


def _cell(model=MODEL_IDENTITIES[0], knowledge=KnowledgeMode.WITHOUT_KDB,
          profiling=ProfilingGuidance.WITHOUT_GUIDANCE):
    return next(cell for cell in phase2_cells(GUIDE_ID, QUALIFICATION) if (
        cell.model == model and cell.knowledge is knowledge
        and cell.profiling is profiling
    ))


def _request(cell, *, attempt=1):
    return Phase2AttemptRequest(
        cell=cell,
        protocol_sha256="3" * 64,
        attempt=attempt,
        request_id=f"request-{attempt}",
        attempt_id=f"attempt-{attempt}",
        phase1_memory=tuple(
            {"project": ordinal, "host_facts": ["observed"]}
            for ordinal in range(1, 9)
        ),
        phase2_memory=({
            "attempt": 1,
            "candidate_sha256": "4" * 64,
            "evidence_sha256": "5" * 64,
            "passed": False,
            "failed_cases": [{"case_id": "odd", "failure": "mismatch"}],
            "entry_sha256": "6" * 64,
        },) if attempt > 1 else (),
    )


def _provider(tmp_path, replies, *, cell=None, knowledge_factory=None,
              guidance=None, max_turns=4, max_queries=None,
              max_completion_bytes=256 * 1024, compile_probe=None,
              max_compile_probes=0):
    calls = []
    replies = iter(replies)

    def transport(**kwargs):
        calls.append(kwargs)
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return DirectCompletion(reply, 3)

    provider = Phase2SourceProvider(
        DirectCompletionProvider(ENVIRONMENT, transport=transport),
        evidence_root=tmp_path,
        guide_text=GUIDE,
        knowledge_factory=knowledge_factory,
        profiling_guidance=guidance,
        max_turns=max_turns,
        max_queries=min(2, max_turns) if max_queries is None else max_queries,
        compile_probe=compile_probe,
        max_compile_probes=max_compile_probes,
        max_completion_bytes=max_completion_bytes,
    )
    return provider, calls


@pytest.mark.parametrize("model", MODEL_IDENTITIES)
def test_registered_models_receive_exact_host_context_and_emit_source(tmp_path, model):
    cell = _cell(model=model)
    provider, calls = _provider(
        tmp_path, ['{"candidate_source":{"source":"def kernel(): pass\\n"}}']
    )
    request = _request(cell, attempt=2)

    assert provider(request) == "def kernel(): pass\n"
    prompt = json.loads(calls[0]["request"]["messages"][1]["content"])
    assert prompt["cell"] == cell.as_dict()
    assert prompt["protocol_sha256"] == request.protocol_sha256
    assert prompt["target_suite_sha256"] == cell.suite_sha256
    contract = prompt["target_contract"]
    assert contract["target_id"] == source_provider_module.TARGET_ID
    assert contract["suite_sha256"] == cell.suite_sha256
    assert contract["candidate_abi_sha256"] == (
        source_provider_module.CANDIDATE_ABI_SHA256
    )
    assert contract["candidate_abi"] == json.loads(json.dumps(
        dict(source_provider_module.CANDIDATE_ABI)
    ))
    assert contract["host_target_grading_sha256"] == (
        source_provider_module.HOST_TARGET_GRADING_SHA256
    )
    assert contract["host_target_grading"] == json.loads(json.dumps(dict(
        source_provider_module.HOST_TARGET_GRADING_CONTRACT
    )))
    assert contract["candidate_module"] == {
        "only_import": "import catlass.tla as tla",
        "module_members": (
            "one import, numeric or Catlass-dtype assignments, undecorated "
            "synchronous helpers, and one synchronous kernel definition"
        ),
        "only_decorator": "@tla.kernel",
        "only_export": "dense_attention(gm_q, gm_k, gm_v, gm_output)",
        "signature_policy": (
            "four positional arguments; no defaults, varargs, or kwargs"
        ),
        "host_binds_compile_time_constants": [
            "BATCH", "HEADS", "SEQUENCE", "HEAD_DIM", "SCALE", "CAUSAL", "DTYPE",
        ],
        "host_bound_dtype": (
            "DTYPE is the actual tla.Float16 or tla.BFloat16 type matching the case"
        ),
        "candidate_assignment_policy": (
            "numeric literals and Catlass numeric dtype attributes; host-bound "
            "constant names are reserved"
        ),
        "launch_policy": {
            "block_num": 1,
            "required_coverage": "block 0 must produce the complete output tensor",
        },
    }
    assert contract["semantics"] == {
        "tensor_arguments": (
            "gm_q/gm_k/gm_v/gm_output are flat contiguous row-major tensors "
            "representing logical [B,H,S,D]"
        ),
        "logical_indexing": (
            "use BATCH/HEADS/SEQUENCE/HEAD_DIM compile-time constants"
        ),
        "evaluation": (
            "candidate source is evaluated unchanged; unsupported Catlass "
            "constructs surface as host compile/runtime evidence"
        ),
        "operation": "dense forward-prefill attention",
        "dtypes": ["float16", "bfloat16"],
        "masks": ["noncausal", "causal"],
        "scale": "1/sqrt(D)",
        "result": "attention output only",
    }
    expected_cases = [
        {
            **asdict(case), "shape": list(case.shape), "scale": case.scale,
            "rtol": case.rtol, "atol": case.atol,
        }
        for case in source_provider_module.ATTENTION_CASES
    ]
    assert contract["cases"] == expected_cases
    assert all(
        set(case) == {
            "case_id", "batch", "heads", "sequence", "head_dim", "dtype",
            "causal", "seed", "shape", "scale", "rtol", "atol",
        }
        for case in contract["cases"]
    )
    assert contract["excluded"] == [
        "backward", "dropout", "KV cache", "GQA", "paging", "RoPE", "ALiBi",
        "packed variable-length inputs",
    ]
    assert prompt["programming_guide"] == {
        "identity": cell.guide.as_dict(), "text": GUIDE,
    }
    assert prompt["phase1_memory"] == list(request.phase1_memory)
    assert prompt["phase2_memory"] == list(request.phase2_memory)
    manifest = json.loads((tmp_path / cell.cell_id / request.attempt_id / "provider.json").read_text())
    assert manifest["provider"]["profile_id"] == model.profile_id
    assert manifest["completion_tokens"] == 3
    assert manifest["candidate_sha256"] == hashlib.sha256(
        b"def kernel(): pass\n"
    ).hexdigest()
    assert manifest["target_contract_sha256"] == hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert "passed" not in manifest and "correct" not in manifest


def test_kdb_query_is_agent_chosen_and_retains_validated_citations(tmp_path):
    cell = _cell(knowledge=KnowledgeMode.WITH_KDB)
    agents = []

    def factory(path, received_cell):
        assert received_cell == cell
        agent = FakeKnowledge(ProgressiveMemoryJournal(path))
        agents.append(agent)
        return agent

    provider, calls = _provider(tmp_path, [
        '{"knowledge_query":{"query":"A5 vector movement","limit":2}}',
        '{"candidate_source":{"source":"source"}}',
    ], knowledge_factory=factory)

    assert provider(_request(cell)) == "source"
    assert agents[0].queries[0].query == "A5 vector movement"
    assert agents[0].closed is True
    observation = json.loads(calls[1]["request"]["messages"][-1]["content"])
    assert observation["knowledge_query"]["results"][0]["citation"]["path"] == "examples/add.py"
    manifest = json.loads(next(tmp_path.rglob("provider.json")).read_text())
    assert manifest["knowledge_queries"][0]["citations"][0]["path"] == "examples/add.py"


def test_without_kdb_never_constructs_agent_and_reports_unavailable(tmp_path):
    def forbidden(*_args):
        raise AssertionError("disabled treatment touched the KDB")

    provider, calls = _provider(tmp_path, [
        '{"knowledge_query":{"query":"try"}}',
        '{"candidate_source":{"source":"source"}}',
    ], knowledge_factory=forbidden)

    assert provider(_request(_cell())) == "source"
    assert json.loads(calls[1]["request"]["messages"][-1]["content"]) == {
        "knowledge_query": {"available": False, "results": []}
    }


@pytest.mark.parametrize(
    ("profiling", "guidance", "error"),
    [
        (ProfilingGuidance.WITH_GUIDANCE, None, "requires non-empty"),
        (ProfilingGuidance.WITHOUT_GUIDANCE, "unexpected", "must not receive"),
    ],
)
def test_guidance_treatment_is_enforced_before_dispatch(
    tmp_path, profiling, guidance, error,
):
    provider, calls = _provider(tmp_path, [], guidance=guidance)
    with pytest.raises(ValueError, match=error):
        provider(_request(_cell(profiling=profiling)))
    assert calls == []


def test_guidance_text_and_digest_are_bound_into_prompt_and_manifest(tmp_path):
    provider, calls = _provider(
        tmp_path, ['{"candidate_source":{"source":"source"}}'], guidance="Use msprof."
    )
    request = _request(_cell(profiling=ProfilingGuidance.WITH_GUIDANCE))
    provider(request)
    prompt = json.loads(calls[0]["request"]["messages"][1]["content"])
    assert prompt["profiling_guidance"] == "Use msprof."
    manifest = json.loads(next(tmp_path.rglob("provider.json")).read_text())
    assert manifest["profiling_guidance_sha256"] == hashlib.sha256(
        b"Use msprof."
    ).hexdigest()


@pytest.mark.parametrize("bad", [
    "prose before {\"candidate_source\":{\"source\":\"x\"}}",
    '{"candidate_source":{"source":"x"},"knowledge_query":{"query":"y"}}',
    '{"candidate_source":{"source":""}}',
])
def test_invalid_or_multiple_actions_exhaust_the_turn_budget(tmp_path, bad):
    provider, _calls = _provider(tmp_path, [bad], max_turns=1)
    with pytest.raises(RuntimeError, match="turn budget exhausted"):
        provider(_request(_cell()))
    assert not list(tmp_path.rglob("provider.json"))


def test_query_budget_is_bounded(tmp_path):
    cell = _cell(knowledge=KnowledgeMode.WITH_KDB)
    provider, _calls = _provider(
        tmp_path,
        ['{"knowledge_query":{"query":"one"}}', '{"knowledge_query":{"query":"two"}}'],
        knowledge_factory=lambda path, _cell: FakeKnowledge(ProgressiveMemoryJournal(path)),
        max_turns=2,
        max_queries=1,
    )
    with pytest.raises(RuntimeError, match="turn budget exhausted"):
        provider(_request(cell))


def test_compile_probe_is_opt_in_and_disabled_action_consumes_ordinary_turn(tmp_path):
    provider, calls = _provider(
        tmp_path, ['{"compile_probe":{"source":"source"}}'], max_turns=1,
    )
    with pytest.raises(RuntimeError, match="turn budget exhausted"):
        provider(_request(_cell()))
    assert "compile_probe" not in calls[0]["request"]["messages"][0]["content"]
    assert not list(tmp_path.rglob("compile-probes"))


def test_compile_probe_does_not_consume_ordinary_turn_and_retains_tokens(tmp_path):
    probes = []

    def compile_probe(source, workdir):
        probes.append((source, workdir))
        return {
            "status": "compile-failed", "case_id": "fp16-noncausal",
            "exit_code": 1, "diagnostic": "error: bad tile",
            "session_handle": "bz-a5:probe-1",
        }

    provider, calls = _provider(tmp_path, [
        '{"compile_probe":{"source":"broken"}}',
        '{"candidate_source":{"source":"fixed"}}',
    ], max_turns=1, compile_probe=compile_probe, max_compile_probes=1)
    request = _request(_cell())
    assert provider(request) == "fixed"
    observation = json.loads(calls[1]["request"]["messages"][-1]["content"])
    assert observation["compile_probe"]["status"] == "compile-failed"
    assert observation["compile_probe"]["correctness_established"] is False
    manifest = json.loads(next(tmp_path.rglob("provider.json")).read_text())
    assert manifest["ordinary_turns"] == 1
    assert manifest["compile_probes"] == 1
    assert manifest["completion_tokens"] == 6
    assert len(probes) == 1


def test_compile_probe_is_exempt_but_knowledge_query_consumes_last_turn(tmp_path):
    def compile_probe(_source, _workdir):
        return {
            "status": "compiled", "case_id": "case", "exit_code": 0,
            "diagnostic": "", "session_handle": "bz-a5:probe",
        }
    provider, calls = _provider(tmp_path, [
        '{"compile_probe":{"source":"candidate"}}',
        '{"knowledge_query":{"query":"Catlass copy"}}',
        '{"candidate_source":{"source":"must-not-run"}}',
    ], max_turns=1, max_queries=1, compile_probe=compile_probe,
       max_compile_probes=1)

    with pytest.raises(RuntimeError, match="turn budget exhausted"):
        provider(_request(_cell()))
    assert len(calls) == 2


def test_compile_probe_limit_exhaustion_is_bounded(tmp_path):
    def callback(_source, _workdir):
        return {
            "status": "compiled", "case_id": "case", "exit_code": 0,
            "diagnostic": "", "session_handle": "bz-a5:probe",
        }
    provider, calls = _provider(tmp_path, [
        '{"compile_probe":{"source":"one"}}',
        '{"compile_probe":{"source":"two"}}',
        '{"candidate_source":{"source":"done"}}',
    ], max_turns=2, compile_probe=callback, max_compile_probes=1)
    assert provider(_request(_cell())) == "done"
    observation = json.loads(calls[2]["request"]["messages"][-1]["content"])
    assert observation["compile_probe"]["status"] == "probe-budget-exhausted"


def test_compile_probe_rejects_unbounded_or_invalid_host_evidence(tmp_path):
    for result in (
        {"status": "compiled"},
        {
            "status": "compile-failed", "case_id": "case", "exit_code": 1,
            "diagnostic": "x" * 4097, "session_handle": None,
        },
    ):
        provider, _ = _provider(
            tmp_path / str(len(result)),
            ['{"compile_probe":{"source":"source"}}'],
            compile_probe=lambda _source, _workdir, result=result: result,
            max_compile_probes=1,
        )
        with pytest.raises(ValueError, match="compile probe returned"):
            provider(_request(_cell()))


def test_completed_probe_is_source_bound_and_reused_after_interruption(tmp_path):
    calls = []

    def probe(source, _workdir):
        calls.append(source)
        return {
            "status": "invalid-source", "case_id": "case", "exit_code": 2,
            "diagnostic": "candidate source is not valid Python",
            "session_handle": None,
        }

    provider, _ = _provider(tmp_path, [
        '{"compile_probe":{"source":"bad source"}}', TimeoutError("offline"),
    ], compile_probe=probe, max_compile_probes=1)
    request = _request(_cell())
    with pytest.raises(InfrastructureFailure):
        provider(request)

    resumed, resumed_calls = _provider(
        tmp_path, ['{"candidate_source":{"source":"fixed"}}'],
        compile_probe=probe, max_compile_probes=1,
    )
    assert resumed(request) == "fixed"
    assert calls == ["bad source"]
    assert json.loads(
        resumed_calls[0]["request"]["messages"][-1]["content"]
    )["compile_probe"]["status"] == "invalid-source"

    retained = next(tmp_path.rglob("compile-probes/probe-01/candidate.py"))
    retained.write_text("different", encoding="utf-8")
    replay, _ = _provider(
        tmp_path, ['{"compile_probe":{"source":"bad source"}}'],
        compile_probe=probe, max_compile_probes=1,
    )
    with pytest.raises(ValueError, match="compile probe"):
        replay(request)


def test_probe_infrastructure_failure_resumes_pending_source_without_model_recall(
    tmp_path,
):
    probe_calls = []

    def probe(source, _workdir):
        probe_calls.append(source)
        if len(probe_calls) == 1:
            raise InfrastructureFailure("BZ unavailable")
        return {
            "status": "compiled", "case_id": "case", "exit_code": 0,
            "diagnostic": "", "session_handle": "bz-a5:resumed",
        }

    request = _request(_cell())
    provider, _ = _provider(
        tmp_path, ['{"compile_probe":{"source":"pending"}}'],
        compile_probe=probe, max_compile_probes=1,
    )
    with pytest.raises(InfrastructureFailure, match="BZ unavailable"):
        provider(request)
    resumed, calls = _provider(
        tmp_path, ['{"candidate_source":{"source":"done"}}'],
        compile_probe=probe, max_compile_probes=1,
    )
    assert resumed(request) == "done"
    assert probe_calls == ["pending", "pending"]
    assert len(calls) == 1


def test_retryable_transport_propagates_without_completed_artifacts(tmp_path):
    provider, _calls = _provider(tmp_path, [TimeoutError("offline")])
    with pytest.raises(InfrastructureFailure, match="provider transport"):
        provider(_request(_cell()))
    assert not list(tmp_path.rglob("provider.json"))
    assert not list(tmp_path.rglob("candidate.py"))


def test_completion_and_candidate_bytes_are_bounded_and_agent_is_closed(tmp_path):
    cell = _cell(knowledge=KnowledgeMode.WITH_KDB)
    agents = []

    def factory(path, _cell):
        agents.append(FakeKnowledge(ProgressiveMemoryJournal(path)))
        return agents[-1]

    oversized = json.dumps({"candidate_source": {"source": "x" * 1100}})
    provider, _calls = _provider(
        tmp_path, [oversized], knowledge_factory=factory, max_turns=1,
        max_completion_bytes=1024,
    )
    with pytest.raises(RuntimeError, match="turn budget exhausted"):
        provider(_request(cell))
    assert agents[0].closed is True
    assert not list(tmp_path.rglob("candidate.py"))


def test_completed_attempt_resumes_without_provider_and_rejects_tampering(tmp_path):
    request = _request(_cell())
    provider, calls = _provider(
        tmp_path, ['{"candidate_source":{"source":"source"}}']
    )
    assert provider(request) == "source"
    assert provider(request) == "source"
    assert len(calls) == 1

    candidate = next(tmp_path.rglob("candidate.py"))
    candidate.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="completed provider evidence"):
        provider(request)


def test_interrupted_candidate_publication_replays_same_uncommitted_attempt(
    tmp_path, monkeypatch,
):
    request = _request(_cell())
    provider, calls = _provider(
        tmp_path,
        [
            '{"candidate_source":{"source":"unbound-source"}}',
            '{"candidate_source":{"source":"replayed-source"}}',
        ],
    )
    publish = source_provider_module._publish_exact
    interrupted = True

    def stop_before_manifest(path, value):
        nonlocal interrupted
        if path.name == "provider.json" and interrupted:
            interrupted = False
            raise RuntimeError("simulated publication interruption")
        return publish(path, value)

    monkeypatch.setattr(source_provider_module, "_publish_exact", stop_before_manifest)
    with pytest.raises(RuntimeError, match="publication interruption"):
        provider(request)
    attempt_root = tmp_path / request.cell.cell_id / request.attempt_id
    assert (attempt_root / "candidate.py").read_text() == "unbound-source"
    assert not (attempt_root / "provider.json").exists()

    assert provider(request) == "replayed-source"
    assert (attempt_root / "candidate.py").read_text() == "replayed-source"
    assert (attempt_root / "provider.json").is_file()
    assert len(calls) == 2
    assert calls[0]["request"]["messages"] == calls[1]["request"]["messages"]


def test_resume_rejects_changed_request_context(tmp_path):
    request = _request(_cell())
    provider, _calls = _provider(
        tmp_path, ['{"candidate_source":{"source":"source"}}']
    )
    provider(request)
    changed = replace(
        request, phase1_memory=tuple({"different": ordinal} for ordinal in range(8))
    )
    with pytest.raises(ValueError, match="completed provider evidence"):
        provider(changed)


def test_resume_rejects_changed_host_target_contract(tmp_path, monkeypatch):
    request = _request(_cell())
    provider, _calls = _provider(
        tmp_path, ['{"candidate_source":{"source":"source"}}']
    )
    provider(request)
    changed_case = replace(source_provider_module.ATTENTION_CASES[0], seed=999)
    monkeypatch.setattr(
        source_provider_module,
        "ATTENTION_CASES",
        (changed_case, *source_provider_module.ATTENTION_CASES[1:]),
    )
    with pytest.raises(ValueError, match="completed provider evidence"):
        provider(request)
