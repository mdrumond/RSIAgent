from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.evidence import EvidenceKind, EvidenceLedger
from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.knowledge_agent import KnowledgeAgent, ProgressiveMemoryJournal
from benchmarks.a5kernels.matrix import Workload
from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
from benchmarks.a5kernels.phase1_live import InfrastructureFailure
from benchmarks.a5kernels.phase1_provider import (
    DISTILLATION_PROMPT,
    DirectActorDriver,
    DirectCompletion,
    DirectCompletionProvider,
    direct_profile,
    openai_compatible_transport,
)
from benchmarks.a5kernels.phase1_registry import DEFAULT_PROPOSALS
from benchmarks.a5kernels.phase1_runtime import Phase1ProjectRuntime
from benchmarks.a5kernels.protocol import ExecutionPlan, SourceFile, VerifiedResult
from benchmarks.a5kernels.trial import CandidateRun, TrialOrchestrator, _SYSTEM_PROMPT


ENVIRONMENT = {
    "OPENAI_API_KEY": "openai-secret",
    "DEEPSEEK_API_KEY": "deepseek-secret",
}
SOURCE = "agent source"
ADMITTED = "a" * 64
NO_NEW_DISTILLATION = (
    '{"distillation":{"no_new_knowledge":true,"lessons":[]}}'
)


def _distillation(*, evidence=ADMITTED, category="correctness"):
    return json.dumps({"distillation": {
        "no_new_knowledge": False,
        "lessons": [{
            "category": category,
            "claim": "The submitted padded vector add passed host verification.",
            "applicability": "This exact registered padded vector-add contract.",
            "preconditions": "The host uses the retained source and dimensions.",
            "limitations": "No claim is made for other shapes.",
            "evidence_artifact_sha256": [evidence],
            "failure_signature": None,
            "repair": None,
            "confidence": "high",
        }],
    }}, sort_keys=True, separators=(",", ":"))


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


def test_direct_actor_reprompts_invalid_output_and_preserves_host_observations(tmp_path):
    calls = []
    provider = _scripted_provider([
        "not-json",
        json.dumps({"write": {"slot": "kernel", "content": SOURCE}}),
        '{"submit":{}}',
        NO_NEW_DISTILLATION,
    ], calls)
    trace_events = []
    actor = DirectActorDriver(
        provider, trace_event=lambda kind, payload: trace_events.append((kind, payload))
    )
    seen = []

    def execute(action):
        seen.append(action.kind)
        from benchmarks.a5kernels.trial import ActionOutcome

        return ActionOutcome(
            json.dumps({"host": {"event": action.kind}}),
            terminal=action.kind == "submit",
        )

    from benchmarks.a5kernels.trial import parse_trial_action

    workspace = tmp_path / "trial/workspace"
    workspace.mkdir(parents=True)
    outcome = actor(
        "instruction",
        profile=direct_profile(MODEL_IDENTITIES[0]),
        workspace=workspace,
        context=tmp_path / "trial/context",
        turn_parser=parse_trial_action,
        action_executor=execute,
    )

    assert seen == ["write", "submit"]
    assert outcome.iterations == 3
    assert outcome.tokens == 6
    assert trace_events[0] == (
        "message", {"role": "system", "content": _SYSTEM_PROMPT}
    )
    assert trace_events[1] == (
        "message", {"role": "user", "content": "instruction"}
    )
    assert trace_events[-1] == (
        "terminal", {"status": "done", "admitted_artifacts": {}}
    )
    second_request = calls[1]["request"]["messages"]
    assert json.loads(second_request[-1]["content"])["host"]["event"] == "invalid-action"
    assert calls[2]["request"]["messages"][-1] == {
        "role": "user",
        "content": json.dumps({"host": {"event": "write"}}),
    }
    outcome = actor.distill(outcome)
    assert outcome.tokens == 8
    ledger = EvidenceLedger(tmp_path / "trial/evidence.jsonl")
    artifacts = ContentAddressedArtifacts(tmp_path / "trial/artifacts")
    retained = [entry.payload for entry in ledger.entries
                if entry.kind == EvidenceKind.ARTIFACT.value]
    assert [entry["role"] for entry in retained].count("provider-request") == 4
    assert [entry["role"] for entry in retained].count("provider-response") == 4
    responses = [artifacts.read_bytes(entry["sha256"]).decode()
                 for entry in retained if entry["role"] == "provider-response"]
    assert responses[0] == "not-json"
    actor_observations = [entry for entry in retained
                          if entry["role"] == "host-observation"
                          and entry.get("origin") == "actor"]
    assert json.loads(artifacts.read_bytes(
        actor_observations[0]["sha256"]
    ))["host"]["event"] == "invalid-action"


def test_direct_actor_delivers_terminal_host_observation_before_returning(tmp_path):
    calls = []
    provider = _scripted_provider([
        '{"submit":{}}',
        _distillation(),
    ], calls)
    actor = DirectActorDriver(provider, max_turns=1)

    from benchmarks.a5kernels.trial import ActionOutcome, parse_trial_action

    terminal = json.dumps({"host": {"event": "submitted", "passed": True}})
    workspace = tmp_path / "trial/workspace"
    workspace.mkdir(parents=True)
    outcome = actor(
        "instruction",
        profile=direct_profile(MODEL_IDENTITIES[0]),
        workspace=workspace,
        context=tmp_path / "trial/context",
        turn_parser=parse_trial_action,
        action_executor=lambda _action: ActionOutcome(
            terminal, terminal=True,
            admitted_artifacts={ADMITTED: "host-verification"},
        ),
    )

    assert len(calls) == 1
    assert outcome.distillation is None
    outcome = actor.distill(outcome)
    assert len(calls) == 2
    assert calls[-1]["request"]["messages"][-2] == {
        "role": "user", "content": terminal,
    }
    prompt = calls[-1]["request"]["messages"][-1]
    assert prompt["role"] == "user"
    assert DISTILLATION_PROMPT in prompt["content"]
    assert ADMITTED in prompt["content"]
    assert outcome.terminal_response == _distillation()
    assert outcome.tokens == 4


def test_post_terminal_provider_failure_does_not_erase_host_success(tmp_path):
    calls = 0

    def transport(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return DirectCompletion('{"submit":{}}', 2)
        raise ConnectionError("provider unavailable after host completion")

    provider = DirectCompletionProvider(ENVIRONMENT, transport=transport)
    actor = DirectActorDriver(provider, max_turns=1)
    workspace = tmp_path / "trial/workspace"
    workspace.mkdir(parents=True)
    from benchmarks.a5kernels.trial import ActionOutcome, parse_trial_action

    outcome = actor(
        "instruction", profile=direct_profile(MODEL_IDENTITIES[0]),
        workspace=workspace, context=tmp_path / "trial/context",
        turn_parser=parse_trial_action,
        action_executor=lambda _action: ActionOutcome(
            '{"host":{"event":"submitted","passed":true}}', terminal=True,
        ),
    )
    outcome = actor.distill(outcome)

    assert outcome.status == "done"
    assert outcome.terminal_response is None
    ledger = EvidenceLedger(tmp_path / "trial/evidence.jsonl")
    retained = [entry.payload for entry in ledger.entries
                if entry.kind == EvidenceKind.ARTIFACT.value]
    assert [entry["role"] for entry in retained][-2:] == [
        "provider-request", "provider-response-failure",
    ]


def test_distillation_correction_is_bounded_and_does_not_rerun_actions(tmp_path):
    calls = []
    provider = _scripted_provider([
        '{"submit":{}}',
        "not-json",
        _distillation(),
    ], calls)
    actor = DirectActorDriver(provider, max_turns=1)
    from benchmarks.a5kernels.trial import ActionOutcome, parse_trial_action
    actions = []
    workspace = tmp_path / "trial/workspace"
    workspace.mkdir(parents=True)

    outcome = actor(
        "instruction", profile=direct_profile(MODEL_IDENTITIES[0]),
        workspace=workspace, context=tmp_path / "trial/context",
        turn_parser=parse_trial_action,
        action_executor=lambda action: (
            actions.append(action.kind)
            or ActionOutcome(
                '{"host":{"event":"submitted","passed":true}}',
                terminal=True, admitted_artifacts={ADMITTED: "host-verification"},
            )
        ),
    )
    learned = actor.distill(outcome)

    assert actions == ["submit"]
    assert learned.distillation is not None
    assert len(calls) == 3
    assert "invalid structured distillation response" in (
        calls[-1]["request"]["messages"][-1]["content"]
    )


def test_distillation_correction_stops_after_three_invalid_responses():
    provider = _scripted_provider([
        '{"submit":{}}', "bad-1", "bad-2", "bad-3",
    ], [])
    actor = DirectActorDriver(provider, max_turns=1)
    from benchmarks.a5kernels.trial import ActionOutcome, parse_trial_action
    outcome = actor(
        "instruction", profile=direct_profile(MODEL_IDENTITIES[0]),
        workspace=None, context=None, turn_parser=parse_trial_action,
        action_executor=lambda _action: ActionOutcome(
            '{"host":{"event":"submitted","passed":true}}', terminal=True,
            admitted_artifacts={ADMITTED: "host-verification"},
        ),
    )

    with pytest.raises(ValueError, match="invalid structured distillation"):
        actor.distill(outcome)


@pytest.mark.parametrize("response,match,role", [
    ('{"distillation":{"no_new_knowledge":true,"lessons":[],"extra":1}}',
     "distillation response", "compile"),
    (_distillation(evidence="b" * 64), "admitted artifact", "compile"),
    (_distillation(category="correctness"), "correctness", "compile"),
    (_distillation(category="performance"), "performance", "compile"),
])
def test_direct_actor_rejects_invalid_or_unsupported_distillation(response, match, role):
    # The last case deliberately admits only compile evidence, which cannot
    # establish correctness even when its digest is genuine.
    provider = _scripted_provider(['{"submit":{}}', response, response, response], [])
    actor = DirectActorDriver(provider, max_turns=1)
    from benchmarks.a5kernels.trial import ActionOutcome, parse_trial_action

    outcome = actor(
        "instruction", profile=direct_profile(MODEL_IDENTITIES[0]),
        workspace=None, context=None, turn_parser=parse_trial_action,
        action_executor=lambda _action: ActionOutcome(
            '{"host":{"event":"submitted","passed":true}}',
            terminal=True, admitted_artifacts={ADMITTED: role},
        ),
    )
    with pytest.raises(ValueError, match=match):
        actor.distill(outcome)


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
    assert result.outcome.distillation is None
    assert len(backend.runs) == 2
    assert result.final_profile["kernel"] == "vector_add_generated"
    assert result.workspace.joinpath("kernel.py").read_text() == SOURCE
    artifacts = ContentAddressedArtifacts(result.artifacts)
    retained = [entry.payload for entry in ledger.entries
                if entry.kind == EvidenceKind.ARTIFACT.value]
    by_role = {entry["role"]: entry["sha256"] for entry in retained}
    assert artifacts.read_bytes(by_role["actor-instruction"]).decode() == (
        f"execute registered {proposal.family.value}"
    )
    assert artifacts.read_bytes(by_role["actor-system-prompt"]).decode() == _SYSTEM_PROMPT
    assert artifacts.read_json(by_role["final-profile"]) == result.final_profile
    assert artifacts.read_bytes(by_role["candidate-source"]).decode() == SOURCE
    assert artifacts.read_bytes(by_role["final-source"]).decode() == SOURCE
    assert "agent-distillation" not in by_role
    assert artifacts.verify()
    retained_bytes = b"".join(path.read_bytes()
                              for path in artifacts.objects.glob("*/*"))
    assert ENVIRONMENT["OPENAI_API_KEY"].encode() not in retained_bytes
    assert ENVIRONMENT["DEEPSEEK_API_KEY"].encode() not in retained_bytes
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
