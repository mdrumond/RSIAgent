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
              max_completion_bytes=256 * 1024):
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
        "module_members": "exactly one import and one synchronous kernel definition",
        "only_decorator": "@tla.kernel",
        "only_export": "dense_attention(gm_q, gm_k, gm_v, gm_output)",
        "signature_policy": (
            "four positional arguments; no defaults, varargs, or kwargs"
        ),
        "host_prepends_compile_time_constants": [
            "BATCH", "HEADS", "SEQUENCE", "HEAD_DIM", "SCALE", "CAUSAL", "DTYPE",
        ],
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
