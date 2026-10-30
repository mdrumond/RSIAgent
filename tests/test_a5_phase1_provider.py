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
    DirectActorDriver,
    DirectCompletion,
    DirectCompletionProvider,
    ProviderRejectionError,
    direct_profile,
    openai_compatible_transport,
)
from benchmarks.a5kernels.phase1_registry import DEFAULT_PROPOSALS
from benchmarks.a5kernels.phase1_runtime import Phase1ProjectRuntime
from benchmarks.a5kernels.protocol import ExecutionPlan, SourceFile, VerifiedResult
from benchmarks.a5kernels.trial import (
    ActionOutcome,
    CandidateRun,
    TrialOrchestrator,
    parse_trial_action,
    _SYSTEM_PROMPT,
)


ENVIRONMENT = {
    "OPENAI_API_KEY": "openai-secret",
    "DEEPSEEK_API_KEY": "deepseek-secret",
}
SOURCE = "agent source"
ADMITTED = "a" * 64
NO_NEW_DISTILLATION = (
    '{"distillation":{"no_new_knowledge":true,"lessons":[]}}'
)


class PromptRejected(Exception):
    status_code = 400
    code = "invalid_prompt"
    request_id = "req_ordinary_123"
    body = {"error": {"code": "invalid_prompt", "message": "openai-secret deepseek-secret"}}


def test_invalid_prompt_retries_identical_request_without_replaying_actions(tmp_path):
    calls, delays, actions = [], [], []
    def transport(**kwargs):
        calls.append(kwargs["request"])
        if len(calls) == 1:
            raise PromptRejected("openai-secret")
        return DirectCompletion('{"submit":{}}', 2)
    provider = DirectCompletionProvider(ENVIRONMENT, transport=transport, sleeper=delays.append)
    # Provider retry/receipt semantics remain a provider contract; programming
    # stage control is covered by canonical domain tests, not the retired loop.
    from benchmarks.a5kernels.phase1_provider import _ProviderEvidenceRecorder
    recorder = _ProviderEvidenceRecorder(tmp_path / 'actor')
    completion = provider.complete(direct_profile(MODEL_IDENTITIES[0]),
        [{'role': 'user', 'content': 'write an ordinary kernel'}],
        attempt_observer=lambda kind, payload: recorder.attempt(kind, payload, stage='action', turn=1))
    recorder.response(completion.text, stage='action', turn=1)
    action = parse_trial_action(completion.text)
    actions.append(action.kind)
    assert calls[0] == calls[1]
    assert delays == [120]
    assert actions == ["submit"] and completion.completion_tokens == 2
    rows = [json.loads(line)["payload"] for line in (tmp_path / "evidence.jsonl").read_text().splitlines()]
    requests = [row for row in rows if row["role"] == "provider-request"]
    assert len(requests) == 2 and requests[0]["sha256"] == requests[1]["sha256"]
    assert [row["attempt"] for row in requests] == [1, 2]
    failures = [row for row in rows if row["role"] == "provider-attempt-rejection"]
    retained = ContentAddressedArtifacts(tmp_path / "artifacts").read_json(failures[0]["sha256"])
    assert retained["request_id"] == "req_ordinary_123" and retained["backoff_seconds"] == 120
    assert retained["request_sha256"] == requests[0]["sha256"]
    assert "secret" not in json.dumps(retained)


def test_invalid_prompt_exhaustion_is_noninfra_and_safe():
    calls, delays, events = [], [], []
    def transport(**kwargs):
        calls.append(kwargs["request"])
        raise PromptRejected("openai-secret")
    provider = DirectCompletionProvider(ENVIRONMENT, transport=transport, sleeper=delays.append, clock=lambda: len(delays)*120)
    with pytest.raises(ProviderRejectionError) as caught:
        provider.complete(direct_profile(MODEL_IDENTITIES[0]), [{"role": "user", "content": "kernel"}],
                          attempt_observer=lambda event, payload: events.append((event, payload)))
    assert len(calls) == 4 and all(call == calls[0] for call in calls)
    assert delays == [120, 120, 120]
    assert [payload["retry"] for event, payload in events if event == "rejection"] == [True, True, True, False]
    assert caught.value.code == "invalid_prompt" and caught.value.status_code == 400
    assert not isinstance(caught.value, InfrastructureFailure)
    assert "secret" not in str(caught.value) + json.dumps(caught.value.evidence())


@pytest.mark.parametrize("exhaust", [False, True])
def test_rejections_are_logged_safely_without_actor_observer(caplog, exhaust):
    calls = []
    def transport(**kwargs):
        calls.append(kwargs["request"])
        if exhaust or len(calls) == 1:
            raise PromptRejected("openai-secret deepseek-secret raw body")
        return DirectCompletion("ordinary response", 1)
    provider = DirectCompletionProvider(ENVIRONMENT, transport=transport, sleeper=lambda _: None, clock=lambda: max(0, len(calls)-1)*120)
    if exhaust:
        with pytest.raises(ProviderRejectionError):
            provider.complete(direct_profile(MODEL_IDENTITIES[0]), [])
    else:
        provider.complete(direct_profile(MODEL_IDENTITIES[0]), [])
    logs = [json.loads(record.getMessage().split(": ", 1)[1]) for record in caplog.records
            if record.name == "benchmarks.a5kernels.phase1_provider"]
    assert len(logs) == (4 if exhaust else 1)
    assert [entry["attempt"] for entry in logs] == list(range(1, len(logs) + 1))
    assert all(entry["code"] == "invalid_prompt" and entry["status_code"] == 400 for entry in logs)
    assert all(entry["request_id"] == "req_ordinary_123" for entry in logs)
    assert len({entry["request_sha256"] for entry in logs}) == 1
    assert logs[-1]["retry"] is not exhaust
    assert logs[-1]["backoff_seconds"] == (0 if exhaust else 120)
    assert "secret" not in caplog.text and "raw body" not in caplog.text


@pytest.mark.parametrize("code,status", [("invalid_schema", 400), ("invalid_prompt", 401),
                                        ("context_length_exceeded", 400)])
def test_other_provider_rejections_are_not_retried(code, status):
    calls = []
    error = PromptRejected("ordinary rejection")
    error.code, error.status_code, error.body = code, status, None
    def transport(**kwargs):
        calls.append(kwargs)
        raise error
    provider = DirectCompletionProvider(ENVIRONMENT, transport=transport,
                                       sleeper=lambda _: pytest.fail("unexpected backoff"))
    with pytest.raises(PromptRejected):
        provider.complete(direct_profile(MODEL_IDENTITIES[0]), [])
    assert len(calls) == 1


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
        "timeout": 600,
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


def test_distillation_correction_stops_at_deadline():
    calls = []
    provider = _scripted_provider([
        '{"submit":{}}', "bad-1", "bad-2", "bad-3",
    ], calls)
    actor = DirectActorDriver(provider, max_seconds=4, clock=lambda: len(calls))
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
    calls = []
    provider = _scripted_provider(['{"submit":{}}', response, response, response], calls)
    actor = DirectActorDriver(provider, max_seconds=4, clock=lambda: len(calls))
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


@pytest.mark.parametrize("terminal", [True, False])
def test_direct_actor_counts_all_turns_but_exempts_compile_budget(tmp_path, terminal):
    calls = []
    provider = _scripted_provider([
        '{"write":{"slot":"kernel","content":"source"}}',
        '{"compile":{}}',
        '{"compile":{}}',
        '{"submit":{}}',
        _distillation(),
    ], calls)
    actor = DirectActorDriver(provider, max_turns=2, max_seconds=5 if terminal else 4, clock=lambda: len(calls))
    workspace = tmp_path / "trial/workspace"
    workspace.mkdir(parents=True)

    outcome = actor(
        "instruction",
        profile=direct_profile(MODEL_IDENTITIES[0]),
        workspace=workspace,
        context=None,
        turn_parser=parse_trial_action,
        action_executor=lambda action: ActionOutcome(
            json.dumps({"host": {"event": action.kind}}),
            terminal=terminal and action.kind == "submit",
            budget_exempt=action.kind == "compile",
            admitted_artifacts={ADMITTED: "host-verification"},
        ),
    )

    assert outcome.status == ("done" if terminal else "time-budget-exhausted")
    assert outcome.iterations == 4
    assert outcome.tokens == 8
    assert len(calls) == 4
    if not terminal:
        return
    learned = actor.distill(outcome)
    assert learned.iterations == 4
    assert learned.tokens == 10
    retained = [entry.payload for entry in EvidenceLedger(
        tmp_path / "trial/evidence.jsonl"
    ).entries if entry.kind == EvidenceKind.ARTIFACT.value]
    assert [(entry["stage"], entry["turn"]) for entry in retained
            if entry["role"] == "provider-request"] == [
        ("action", 1), ("action", 2), ("action", 3), ("action", 4),
        ("distillation", 5),
    ]


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
    assert artifacts.read_bytes(by_role["actor-instruction"]).decode().startswith(
        f"execute registered {proposal.family.value}\n"
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
    assert request_entries[-1].payload["trial_budget"] == {
        "max_seconds": 3600,
        "policy": "elapsed-time-v1",
    }


def test_bound_trial_retains_and_explains_unmetered_compile_allowance(tmp_path):
    class RecordingCompileBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.compiles = []

        def compile(self, workspace, language, attempt_id):
            self.compiles.append(attempt_id)
            return super().compile(workspace, language, attempt_id)

    calls = []
    provider = _scripted_provider([
        json.dumps({"write": {"slot": "kernel", "content": SOURCE}}),
        '{"compile":{}}',
        '{"submit":{}}',
    ], calls)
    backend = RecordingCompileBackend()
    memory = tmp_path / "memory"
    memory.mkdir()
    knowledge = KnowledgeAgent(
        enabled=False,
        journal=ProgressiveMemoryJournal(memory / "knowledge.jsonl"),
    )
    ledger = EvidenceLedger(tmp_path / "evidence.jsonl")

    result = TrialOrchestrator.run_bound(
        trial_root=tmp_path / "trial",
        request_payload={"schema": "unmetered-compile-test-v1"},
        language="catlass-dsl",
        workload=Workload.SMOKE_VECTOR_ADD,
        instruction="develop the kernel",
        profile=direct_profile(MODEL_IDENTITIES[0]),
        backend=backend,
        actor=DirectActorDriver(provider, max_turns=2),
        knowledge=knowledge,
        profiling=FakeProfile(),
        profiling_guidance=False,
        ledger=ledger,
        unmetered_compile_probes=1,
    )

    assert result.verified.passed
    assert result.outcome.iterations == 3
    assert result.outcome.tokens == 6
    assert len(backend.compiles) == 1
    opening = calls[0]["request"]["messages"][1]["content"]
    assert "within the remaining time" in opening
    assert "does not launch the kernel or establish correctness" in opening
    artifacts = ContentAddressedArtifacts(result.artifacts)
    instruction = next(entry.payload for entry in ledger.entries
                       if entry.kind == EvidenceKind.ARTIFACT.value
                       and entry.payload["role"] == "actor-instruction")
    assert artifacts.read_bytes(instruction["sha256"]).decode() == opening
    actions = [entry.payload for entry in ledger.entries
               if entry.kind == EvidenceKind.ACTION.value]
    assert [entry["ordinal"] for entry in actions] == [1, 2, 3]
    assert actions[1]["budget"]["remaining_seconds"] > 0
    assert all("payload_artifact_sha256" in entry for entry in actions)
    request = next(
        entry for entry in ledger.entries
        if entry.kind == EvidenceKind.REQUEST.value
    )
    assert request.payload["trial_budget"] == {
        "max_seconds": 3600,
        "policy": "elapsed-time-v1",
    }


def test_bound_trial_requires_a_host_verified_submit(tmp_path):
    calls = []
    provider = _scripted_provider([
        json.dumps({"write": {"slot": "kernel", "content": SOURCE}})
    ], calls)
    actor = DirectActorDriver(provider, max_seconds=1, clock=lambda: len(calls))
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


def test_single_model_provider_does_not_require_unrelated_credentials():
    selected = MODEL_IDENTITIES[0]
    provider = DirectCompletionProvider({selected.credential_env: 'fixture-only'},
        models=(selected.backend_model,), transport=lambda **_: DirectCompletion('ok', 1))
    assert provider.coding_environment() == {selected.credential_env: 'fixture-only'}
    assert provider.complete(direct_profile(selected), [{'role':'user', 'content':'test'}]).text == 'ok'
    with pytest.raises(ValueError, match='not configured'):
        provider.complete(direct_profile(MODEL_IDENTITIES[1]), [{'role':'user', 'content':'test'}])


def test_bounded_provider_requests_shrink_http_time_and_do_not_dispatch_after_deadline():
    now, calls = [0.0], []
    def transport(**kwargs):
        calls.append(kwargs['request'])
        now[0] += 1
        if len(calls) == 1:
            raise PromptRejected()
        return DirectCompletion('ok', 1)
    def sleep(seconds):
        now[0] += seconds
    provider = DirectCompletionProvider(ENVIRONMENT, transport=transport, clock=lambda: now[0],
        sleeper=sleep, invalid_prompt_backoff_seconds=1)
    profile = direct_profile(MODEL_IDENTITIES[0])
    assert provider.complete(profile, [{'role': 'user', 'content': 'test'}], max_seconds=10).text == 'ok'
    assert [call['timeout'] for call in calls] == [10, 8]
    calls.clear()
    with pytest.raises(TimeoutError):
        provider.complete(profile, [], max_seconds=0)
    assert calls == []
    provider._sleep = lambda seconds: now.__setitem__(0, now[0]+20)
    with pytest.raises(TimeoutError):
        provider.complete(profile, [], max_seconds=10)
    assert len(calls) == 1


def test_bounded_transport_consumes_timeout_without_sending_api_payload_field(monkeypatch):
    constructors, payloads = [], []
    class OpenAI:
        def __init__(self, **kwargs):
            constructors.append(kwargs)
            def create(**payload):
                payloads.append(payload)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ok'))],
                    usage=SimpleNamespace(completion_tokens=1))
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=OpenAI))
    request = {'model': 'test', 'messages': [], 'timeout': 4}
    openai_compatible_transport(base_url='https://provider.invalid', api_key='secret', request=request)
    assert constructors[0]['timeout'] == 4
    assert payloads == [{'model': 'test', 'messages': []}]
    assert request['timeout'] == 4
