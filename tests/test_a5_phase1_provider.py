from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from canonical_kernel_fixtures import canonical_kernel

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
    CandidateRun,
    TrialOrchestrator,
    parse_trial_action,
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


def test_direct_actor_reprompts_invalid_output_and_preserves_host_observations(canonical_kernel):
    case = canonical_kernel
    case.invalid = True
    outcome = case.run()
    assert outcome.status == 'host-verified'
    invalid = [row for row in case.runner.events if row['type'] == 'actor-invalid-action']
    assert len(invalid) == 1
    assert any(context['observation'] == {
        key: value for key, value in invalid[0].items()
        if key in ('event', 'error', 'accepted', 'allowed')}
        for context in case.contexts)
    assert any(row['passed'] is False for row in case.runner.workflow.state['history'])
    assert case.actions.latest.evidence.verified


def _distill_case(case, transport):
    case.provider._transport = transport
    outcome = case.run()
    assert outcome.status == 'host-verified'
    assert case.actions.latest.evidence.verified
    evidence = next(ref for ref, role in outcome.admitted_artifacts.items() if role == 'host-verification')
    assert case.runner.store.read_bytes(evidence)
    return outcome, evidence


def test_direct_actor_delivers_terminal_host_observation_before_returning(canonical_kernel):
    case = canonical_kernel
    calls = []
    def transport(**options):
        calls.append(options['request'])
        return DirectCompletion(NO_NEW_DISTILLATION, 2)
    outcome, evidence = _distill_case(case, transport)
    assert not calls and outcome.distillation is None
    learned = case.driver.distill(outcome)
    assert learned.distillation is not None and len(calls) == 1
    context = json.loads(calls[0]['messages'][-1]['content'].split('\n')[-1])
    assert context['admitted_artifacts'][evidence] == 'host-verification'
    assert context['workflow_history'][-1]['passed'] is True
    assert context['artifact_payloads'][evidence] == case.runner.store.read_bytes(evidence).decode()
    assert learned.terminal_response == NO_NEW_DISTILLATION
    assert learned.tokens == 2


def test_post_terminal_provider_failure_does_not_erase_host_success(canonical_kernel):
    case = canonical_kernel
    def transport(**_):
        raise ConnectionError('provider unavailable after host completion')
    outcome, _ = _distill_case(case, transport)
    calls = list(case.calls)
    learned = case.driver.distill(outcome)
    assert learned.status == 'host-verified'
    assert learned.terminal_response is None
    assert learned.distillation_failure is not None
    failure = case.runner.store.read_json(learned.distillation_failure)
    assert failure['host_outcome'] == 'host-verified'
    assert failure['source_hash'] == case.runner.backend.project_hash()
    assert case.calls == calls
    assert case.actions.latest.evidence.verified


def test_distillation_correction_is_bounded_and_does_not_rerun_actions(canonical_kernel):
    case = canonical_kernel
    requests = []
    def transport(**options):
        requests.append(options['request'])
        if len(requests) == 1:
            return DirectCompletion('not-json', 2)
        context = json.loads(requests[0]['messages'][-1]['content'].split('\n')[-1])
        evidence = next(ref for ref, role in context['admitted_artifacts'].items() if role == 'host-verification')
        return DirectCompletion(_distillation(evidence=evidence), 2)
    outcome, _ = _distill_case(case, transport)
    calls = list(case.calls)
    learned = case.driver.distill(outcome)
    assert learned.distillation is not None and len(requests) == 2
    original = requests[0]['messages']
    corrected = requests[1]['messages']
    assert corrected[:-3] == original[:-1]
    assert corrected[-3] == dict(role='assistant', content='not-json')
    # The current terminal contract stays pinned after correction feedback.
    assert corrected[-1] == original[-1]
    assert len(corrected) == len(original) + 2
    assert corrected[-1]['role'] == 'user'
    assert case.calls == calls


def test_distillation_correction_stops_at_deadline(canonical_kernel):
    from rsi.runtime import WorkflowError
    case = canonical_kernel
    requests = []
    def transport(**options):
        requests.append(options['request'])
        case.now += 121
        return DirectCompletion('invalid-distillation', 1)
    outcome, _ = _distill_case(case, transport)
    calls = list(case.calls)
    with pytest.raises(WorkflowError):
        case.driver.distill(outcome)
    assert len(requests) == 1 and case.calls == calls
    assert case.actions.latest.evidence.verified


@pytest.mark.parametrize('response,match,role', [
    ('{"distillation":{"no_new_knowledge":true,"lessons":[],"extra":1}}',
     'distillation response', 'compile'),
    (_distillation(evidence='b' * 64), 'admitted artifact', 'compile'),
    (_distillation(category='correctness'), 'correctness', 'compile'),
    (_distillation(category='performance'), 'performance', 'compile'),
])
def test_direct_actor_rejects_invalid_or_unsupported_distillation(canonical_kernel, response, match, role):
    from benchmarks.a5kernels.phase1_memory import parse_agent_distillation
    case = canonical_kernel
    outcome = case.run()
    assert outcome.status == 'host-verified'
    # Use an actual retained compile artifact, never a fabricated host receipt.
    compile_ref = next(ref for ref, category in outcome.admitted_artifacts.items() if category == role)
    response = response.replace(ADMITTED, compile_ref)
    with pytest.raises(ValueError, match=match):
        parse_agent_distillation(response, {compile_ref: role})
    assert case.runner.store.read_bytes(compile_ref)


@pytest.mark.parametrize('terminal', [True, False])
def test_direct_actor_counts_all_turns_but_exempts_compile_budget(canonical_kernel, terminal):
    case = canonical_kernel
    from rsi.contracts import BudgetLimits
    if not terminal:
        case.after_host = lambda kind: setattr(case, 'now', case.now + 121) if kind == 'compile' else None
    actor = case.make_driver(max_turns=1,
        canonical_budgets=BudgetLimits(actor_seconds=120, wall_seconds=120))
    outcome = case.run(actor)
    if terminal:
        assert outcome.status == 'host-verified'
        assert len(case.contexts) > 1
        assert case.calls.count('compile') >= 1 and case.calls.count('run') >= 2
        # A compile attempt is an execution reservation, not a model-turn limit.
        history = case.runner.workflow.state['history']
        assert any(not row['passed'] for row in history)
        assert history[-1]['passed'] is True
    else:
        assert outcome.status == 'budget_exhausted'
        assert case.calls == ['compile']
        assert not case.actions.latest
        assert case.runner.workflow.accounting.seconds('compilation') >= 121
        assert case.runner.workflow.remaining()['actor_seconds'] == 0
        assert case.runner.workflow.remaining()['wall_seconds'] == 0


@pytest.mark.parametrize("proposal", DEFAULT_PROPOSALS, ids=lambda item: item.family.value)
def test_every_registered_project_family_can_use_the_bound_trial_executor(
    tmp_path, proposal, canonical_kernel
):
    runtime = Phase1ProjectRuntime.from_proposal(proposal)
    from test_a5_phase1_production import ProjectExecution
    from tests.test_a5_catlass_candidate import SOURCE as candidate_source
    from rsi.domain_bindings import kernel_adoption
    case = canonical_kernel
    case.session.initial = case.session.repaired = candidate_source
    provider = DirectCompletionProvider(ENVIRONMENT,
        transport=lambda **_: pytest.fail('coding must use the canonical session'))
    execution = ProjectExecution()
    backend = runtime.backend(execution)
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
        actor=DirectActorDriver(provider, canonical=kernel_adoption(baseline_query=case.baseline_query)),
        knowledge=knowledge,
        profiling=FakeProfile(),
        profiling_guidance=False,
        ledger=ledger,
    )

    assert result.verified.passed
    assert result.outcome.status == "host-verified"
    assert result.outcome.distillation is None
    # Mandatory baseline queries must not dispatch the disabled experience tool.
    assert any(row['type'] == 'project-tool' and row['name'] == 'query_kdb'
               for row in case.runner.events)
    assert not any(row['action'] == 'query-knowledge'
                   for row in case.runner.backend.host.observations)
    assert len(execution.plans) >= 3
    assert result.final_profile["kernel"] == result.run.kernel_name
    assert all(len(plan.input_a) == runtime.padded_length for plan in execution.plans)
    assert result.workspace.joinpath("kernel.py").read_text() == candidate_source
    artifacts = ContentAddressedArtifacts(result.artifacts)
    artifact_rows = [entry.payload for entry in ledger.entries
                     if entry.kind == EvidenceKind.ARTIFACT.value]
    by_role = {row['role']: row['sha256'] for row in artifact_rows if 'role' in row}
    required_roles = {'actor-instruction', 'actor-system-prompt', 'action-payload',
                      'host-observation', 'candidate-source', 'final-source', 'final-profile'}
    assert required_roles <= by_role.keys()
    for row in artifact_rows:
        if 'role' in row:
            artifacts.read_bytes(row['sha256'])  # Hash-check every referenced retained object.
    # Host launch provenance shares the ledger with role-tagged trial artifacts.
    plans = {plan.execution_id: plan for plan in execution.plans}
    launch_artifacts = [row for row in artifact_rows if 'execution_id' in row]
    assert launch_artifacts
    for row in launch_artifacts:
        plan = plans[row['execution_id']]
        assert row['source_fingerprint'] == plan.source_fingerprint
        assert row['source_sha256'] == {source.relative_path: source.sha256 for source in plan.files}
    presented = [context['task']['domain']['instruction'] for context in case.contexts]
    assert artifacts.read_bytes(by_role['actor-instruction']).decode() in presented
    assert artifacts.read_json(by_role["final-profile"]) == result.final_profile
    assert artifacts.read_bytes(by_role["candidate-source"]).decode() == candidate_source
    assert artifacts.read_bytes(by_role["final-source"]).decode() == candidate_source
    assert "agent-distillation" not in by_role
    assert artifacts.verify()
    retained_bytes = b"".join(path.read_bytes()
                              for path in artifacts.objects.glob("*/*"))
    assert ENVIRONMENT["OPENAI_API_KEY"].encode() not in retained_bytes
    assert ENVIRONMENT["DEEPSEEK_API_KEY"].encode() not in retained_bytes
    request_entries = [entry for entry in ledger.entries
                       if entry.kind == EvidenceKind.REQUEST.value
                       and entry.payload.get('schema') == 'a5-phase1-provider-family-test-v1']
    assert len(request_entries) == 1
    assert request_entries[0].payload["proposal"] == proposal.as_dict()
    assert request_entries[0].payload["runtime"] == runtime.as_dict()
    assert request_entries[0].payload["trial_budget"] == {
        "max_seconds": 3600,
        "policy": "elapsed-time-v1",
    }


def test_bound_trial_retains_and_explains_unmetered_compile_allowance(tmp_path, canonical_kernel):
    from test_a5_phase1_production import ProjectExecution
    from tests.test_a5_catlass_candidate import SOURCE as candidate_source
    from rsi.domain_bindings import kernel_adoption
    case = canonical_kernel
    case.session.initial = case.session.repaired = candidate_source
    execution = ProjectExecution()
    backend = Phase1ProjectRuntime.from_proposal(DEFAULT_PROPOSALS[0]).backend(execution)
    provider = DirectCompletionProvider(ENVIRONMENT,
        transport=lambda **_: pytest.fail('canonical coding must not use old provider loop'))
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
        actor=DirectActorDriver(provider, max_turns=2,
            canonical=kernel_adoption(baseline_query=case.baseline_query)),
        knowledge=knowledge,
        profiling=FakeProfile(),
        profiling_guidance=False,
        ledger=ledger,
        unmetered_compile_probes=1,
    )

    assert result.verified.passed
    assert result.outcome.status == 'host-verified'
    assert len(case.contexts) > 2  # max_turns is telemetry, not a stopping budget.
    assert execution.plans
    opening = case.contexts[0]['task']['domain']['instruction']
    artifacts = ContentAddressedArtifacts(result.artifacts)
    instruction = next(entry.payload for entry in ledger.entries
                       if entry.kind == EvidenceKind.ARTIFACT.value
                       and entry.payload["role"] == "actor-instruction")
    assert artifacts.read_bytes(instruction["sha256"]).decode() == opening
    action_rows = [entry.payload for entry in ledger.entries
                   if entry.kind == EvidenceKind.ACTION.value]
    actions = [row for row in action_rows if 'action' in row]
    launches = [row for row in action_rows if 'execution_id' in row]
    from collections import Counter
    from rsi.domain_checkpoint import decode
    compile_plans = [plan for plan in execution.plans
                     if dict(plan.environment.bindings).get('A5KERNEL_COMPILE_ONLY') == '1']
    run_plans = [plan for plan in execution.plans
                 if dict(plan.environment.bindings).get('A5KERNEL_COMPILE_ONLY') != '1']
    assert len(compile_plans) == 1 and len(run_plans) == 3
    assert Counter(row['execution_id'] for row in launches) == Counter(plan.execution_id for plan in run_plans)
    dispatches = {row['reference']: case.runner.store.read_json(row['reference'])
                  for row in case.runner.events if row['type'] == 'domain-dispatch'}
    compile_results = []
    for event in case.runner.events:
        if event['type'] != 'domain-result':
            continue
        receipt = case.runner.store.read_json(event['reference'])
        dispatch = dispatches[receipt['dispatch_ref']]
        if dispatch['name'] != 'compile':
            continue
        assert receipt['observed_terminal'] and receipt['status'] == 'complete'
        assert receipt['operation_id'] == dispatch['operation_id']
        assert receipt['activity_id'] == dispatch['activity_id']
        assert receipt['source_hash'] == dispatch['source_hash'] == hashlib.sha256(candidate_source.encode()).hexdigest()
        outcome, observed = decode(receipt['result'])
        result = observed['result']
        assert json.loads(outcome.observation)['host']['result'] == result
        assert result['passed'] and result['exit_code'] == 0
        assert len(bytes.fromhex(result['attestation_sha256'])) == 32
        compile_results.append(result)
    assert Counter(row['execution_id'] for row in compile_results) == Counter(plan.execution_id for plan in compile_plans)
    assert compile_results[0]['source_fingerprint'] == compile_plans[0].source_fingerprint
    assert [entry["ordinal"] for entry in actions] == list(range(1, len(actions) + 1))
    compiles = [entry for entry in actions if entry["action"] == "compile"]
    assert compiles and all(entry["budget"]["remaining_seconds"] > 0 for entry in compiles)
    assert all("payload_artifact_sha256" in entry for entry in actions)
    request = next(
        entry for entry in ledger.entries
        if entry.kind == EvidenceKind.REQUEST.value
    )
    assert request.payload["trial_budget"] == {
        "max_seconds": 3600,
        "policy": "elapsed-time-v1",
    }


def test_bound_trial_requires_a_host_verified_submit(tmp_path, canonical_kernel):
    from rsi.domain_bindings import kernel_adoption
    case = canonical_kernel
    case.after_select = lambda context, selected: setattr(case, 'now', case.now + 121) if selected[0] == 'perform' else None
    actor = case.make_driver(canonical=kernel_adoption(baseline_query=case.baseline_query))
    knowledge = KnowledgeAgent(
        enabled=False,
        journal=ProgressiveMemoryJournal(tmp_path / "knowledge.jsonl"),
    )

    backend = FakeBackend()
    with pytest.raises(RuntimeError, match="host-verified submit"):
        TrialOrchestrator.run_bound(
            trial_root=tmp_path / "trial",
            request_payload={"schema": "a5-phase1-provider-family-test-v1"},
            language="catlass-dsl",
            workload=Workload.SMOKE_VECTOR_ADD,
            instruction="instruction",
            profile=direct_profile(MODEL_IDENTITIES[0]),
            backend=backend,
            actor=actor,
            knowledge=knowledge,
            profiling=FakeProfile(),
            profiling_guidance=False,
            ledger=EvidenceLedger(tmp_path / "evidence.jsonl"),
        )
    assert backend.runs == []
    assert not any(row['type'] == 'domain-dispatch' for row in case.runner.events)


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
