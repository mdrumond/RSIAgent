from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.evidence import EvidenceKind, EvidenceLedger
from benchmarks.a5kernels.knowledge_agent import KnowledgeAgent, ProgressiveMemoryJournal
from benchmarks.a5kernels.matrix import Workload
from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
from benchmarks.a5kernels.phase1_live import InfrastructureFailure
from benchmarks.a5kernels.phase1_provider import (
    DirectActorDriver,
    DirectCompletion,
    DirectCompletionProvider,
    direct_profile,
    openai_compatible_transport,
)
from benchmarks.a5kernels.phase1_registry import DEFAULT_PROPOSALS
from benchmarks.a5kernels.phase1_runtime import Phase1ProjectRuntime
from benchmarks.a5kernels.protocol import ExecutionPlan, SourceFile, VerifiedResult
from benchmarks.a5kernels.trial import CandidateRun, TrialOrchestrator


ENVIRONMENT = {
    "OPENAI_API_KEY": "openai-secret",
    "DEEPSEEK_API_KEY": "deepseek-secret",
}
SOURCE = "agent source"


def test_openai_transport_bounds_each_attempt_and_disables_sdk_retries(monkeypatch):
    constructor = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            constructor.append(kwargs)
            usage = SimpleNamespace(completion_tokens=3)
            message = SimpleNamespace(content="answer")
            response = SimpleNamespace(
                choices=[SimpleNamespace(message=message)], usage=usage,
            )
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_request: response),
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    result = openai_compatible_transport(
        base_url="https://provider.invalid",
        api_key="secret",
        request={"model": "model", "messages": []},
    )

    assert result == DirectCompletion("answer", 3)
    assert constructor == [{
        "api_key": "secret",
        "base_url": "https://provider.invalid",
        "timeout": 120,
        "max_retries": 0,
    }]


class FakeBackend:
    def __init__(self):
        self.runs = []

    def compile(self, workspace, language, attempt_id):
        return {"exit_code": 0, "attempt_id": attempt_id}

    def run(self, workspace, language, workload, attempt_id, ledger):
        source = (workspace / "kernel.py").read_text(encoding="utf-8")
        plan = ExecutionPlan(
            request_id="r" * 64,
            attempt_id=attempt_id,
            language=language,
            files=(SourceFile("kernel.py", source),),
            argv=("python", "driver.py"),
            input_a=(1.0,),
            input_b=(2.0,),
            runtime_provenance=(("execution_profile", "bz-a5"),),
        )
        verified = VerifiedResult(
            request_id=plan.request_id,
            execution_id=plan.execution_id,
            attempt_id=attempt_id,
            language=language,
            runtime_provenance=plan.runtime_provenance,
            passed=True,
            max_abs_error=0.0,
            exit_code=0,
            output_sha256="o" * 64,
            source_fingerprint=plan.source_fingerprint,
            evidence_sha256="e" * 64,
            session_handle="bz-a5:fake",
            attestation_sha256="a" * 64,
        )
        candidate = CandidateRun(
            plan,
            verified,
            "vector_add_generated",
            workload,
            hashlib.sha256(source.encode("utf-8")).hexdigest(),
        )
        self.runs.append(candidate)
        return candidate


class FakeProfile:
    def intermediate(self, run):
        return {"kernel": run.kernel_name}

    def final(self, run):
        return {"duration_us": 3.0, "kernel": run.kernel_name}


def _scripted_provider(replies, calls):
    replies = iter(replies)

    def transport(**kwargs):
        calls.append(kwargs)
        return DirectCompletion(next(replies), 2)

    return DirectCompletionProvider(ENVIRONMENT, transport=transport)


def test_direct_profiles_issue_exact_isolated_requests_without_fallbacks():
    calls = []
    provider = _scripted_provider(['{"submit":{}}'] * 2, calls)

    for identity in MODEL_IDENTITIES:
        provider.complete(direct_profile(identity), [{"role": "user", "content": "one"}])

    assert [call["base_url"] for call in calls] == [
        "https://api.openai.com/v1",
        "https://api.deepseek.com",
    ]
    assert [call["api_key"] for call in calls] == [
        "openai-secret",
        "deepseek-secret",
    ]
    openai_request, deepseek_request = [call["request"] for call in calls]
    assert openai_request == {
        "model": "gpt-5.6-sol",
        "messages": [{"role": "user", "content": "one"}],
        "reasoning_effort": "high",
        "max_completion_tokens": 32768,
    }
    assert deepseek_request == {
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "one"}],
        "reasoning_effort": "high",
        "max_tokens": 8192,
        "top_p": 1.0,
        "extra_body": {"thinking": {"type": "disabled"}},
    }


def test_provider_requires_both_cell_credentials_before_dispatch():
    calls = []

    def transport(**kwargs):
        calls.append(kwargs)
        return DirectCompletion('{"submit":{}}', 1)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        DirectCompletionProvider(
            {"OPENAI_API_KEY": "present"}, transport=transport
        )

    assert calls == []


@pytest.mark.parametrize("failure", [
    ConnectionError("offline"),
    TimeoutError("timed out"),
    type("RateLimit", (Exception,), {"status_code": 429})("limited"),
    type("ServerError", (Exception,), {"status_code": 503})("unavailable"),
])
def test_retryable_provider_failures_use_the_shared_infrastructure_signal(failure):
    def transport(**_kwargs):
        raise failure

    provider = DirectCompletionProvider(ENVIRONMENT, transport=transport)
    with pytest.raises(InfrastructureFailure, match="provider transport"):
        provider.complete(direct_profile(MODEL_IDENTITIES[0]), [])


@pytest.mark.parametrize("exception_name", ["APIConnectionError", "APITimeoutError"])
def test_openai_sdk_transport_failures_use_infrastructure_signal(
    monkeypatch, exception_name,
):
    class APIConnectionError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    sdk = SimpleNamespace(
        APIConnectionError=APIConnectionError,
        APITimeoutError=APITimeoutError,
    )
    monkeypatch.setitem(sys.modules, "openai", sdk)
    failure = getattr(sdk, exception_name)("provider unavailable")

    def transport(**_kwargs):
        raise failure

    provider = DirectCompletionProvider(ENVIRONMENT, transport=transport)
    with pytest.raises(InfrastructureFailure, match="provider transport"):
        provider.complete(direct_profile(MODEL_IDENTITIES[0]), [])


def test_direct_actor_reprompts_invalid_output_and_preserves_host_observations():
    calls = []
    provider = _scripted_provider([
        "not-json",
        json.dumps({"write": {"slot": "kernel", "content": SOURCE}}),
        '{"submit":{}}',
    ], calls)
    actor = DirectActorDriver(provider)
    seen = []

    def execute(action):
        seen.append(action.kind)
        from benchmarks.a5kernels.trial import ActionOutcome

        return ActionOutcome(
            json.dumps({"host": {"event": action.kind}}),
            terminal=action.kind == "submit",
        )

    from benchmarks.a5kernels.trial import parse_trial_action

    outcome = actor(
        "instruction",
        profile=direct_profile(MODEL_IDENTITIES[0]),
        workspace=None,
        context=None,
        turn_parser=parse_trial_action,
        action_executor=execute,
    )

    assert seen == ["write", "submit"]
    assert outcome.iterations == 3
    assert outcome.tokens == 6
    second_request = calls[1]["request"]["messages"]
    assert json.loads(second_request[-1]["content"])["host"]["event"] == "invalid-action"
    assert calls[2]["request"]["messages"][-1] == {
        "role": "user",
        "content": json.dumps({"host": {"event": "write"}}),
    }


@pytest.mark.parametrize("proposal", DEFAULT_PROPOSALS, ids=lambda item: item.family.value)
def test_every_registered_project_family_can_use_the_bound_trial_executor(
    tmp_path, proposal
):
    runtime = Phase1ProjectRuntime.from_proposal(proposal)
    calls = []
    provider = _scripted_provider([
        json.dumps({"write": {"slot": "kernel", "content": SOURCE}}),
        '{"run":{}}',
        '{"submit":{}}',
    ], calls)
    backend = FakeBackend()
    knowledge = KnowledgeAgent(
        enabled=False,
        journal=ProgressiveMemoryJournal(tmp_path / "knowledge.jsonl"),
    )
    ledger = EvidenceLedger(tmp_path / "evidence.jsonl")

    result = TrialOrchestrator.run_bound(
        trial_root=tmp_path / "trial",
        request_payload={
            "schema": "a5-phase1-provider-family-test-v1",
            "proposal": proposal.as_dict(),
            "runtime": runtime.as_dict(),
        },
        language="catlass-dsl",
        workload=Workload.SMOKE_VECTOR_ADD,
        instruction=f"execute registered {proposal.family.value}",
        profile=direct_profile(MODEL_IDENTITIES[0]),
        backend=backend,
        actor=DirectActorDriver(provider),
        knowledge=knowledge,
        profiling=FakeProfile(),
        profiling_guidance=False,
        ledger=ledger,
    )

    assert result.verified.passed
    assert result.outcome.tokens == 6
    assert len(backend.runs) == 2
    assert result.final_profile["kernel"] == "vector_add_generated"
    assert result.workspace.joinpath("kernel.py").read_text() == SOURCE
    request_entries = [
        entry for entry in ledger.entries if entry.kind == EvidenceKind.REQUEST.value
    ]
    assert request_entries[-1].payload["proposal"] == proposal.as_dict()
    assert request_entries[-1].payload["runtime"] == runtime.as_dict()


def test_bound_trial_requires_a_host_verified_submit(tmp_path):
    provider = _scripted_provider([
        json.dumps({"write": {"slot": "kernel", "content": SOURCE}})
    ], [])
    actor = DirectActorDriver(provider, max_turns=1)
    knowledge = KnowledgeAgent(
        enabled=False,
        journal=ProgressiveMemoryJournal(tmp_path / "knowledge.jsonl"),
    )

    with pytest.raises(RuntimeError, match="host-verified submit"):
        TrialOrchestrator.run_bound(
            trial_root=tmp_path / "trial",
            request_payload={"schema": "a5-phase1-provider-family-test-v1"},
            language="catlass-dsl",
            workload=Workload.SMOKE_VECTOR_ADD,
            instruction="instruction",
            profile=direct_profile(MODEL_IDENTITIES[0]),
            backend=FakeBackend(),
            actor=actor,
            knowledge=knowledge,
            profiling=FakeProfile(),
            profiling_guidance=False,
            ledger=EvidenceLedger(tmp_path / "evidence.jsonl"),
        )
