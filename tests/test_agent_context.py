import json

import pytest

from benchmarks.ascend.agent_context import ContextBudget, ContextBudgetError, WorkingContext, request_bytes
from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
from benchmarks.a5kernels.phase1_provider import DirectActorDriver, DirectCompletion, DirectCompletionProvider, direct_profile
from benchmarks.a5kernels.trial import ActionOutcome, parse_trial_action


ENV = {"OPENAI_API_KEY": "test", "DEEPSEEK_API_KEY": "test"}
BUDGET = ContextBudget(5000, 3500, 7000)


@pytest.mark.parametrize("identity", MODEL_IDENTITIES)
def test_provider_rejects_oversize_without_transport_or_rewriting(identity):
    calls = []
    provider = DirectCompletionProvider(ENV, transport=lambda **kw: calls.append(kw),
        context_budgets={identity.backend_model.value: BUDGET})
    with pytest.raises(ContextBudgetError):
        provider.complete(direct_profile(identity), [{"role": "user", "content": "x" * 7000}])
    assert calls == []


def test_work_log_keeps_host_facts_outside_agent_summary_and_resets_source():
    profile = direct_profile(MODEL_IDENTITIES[0])
    working = WorkingContext([{"role": "system", "content": "system"},
                              {"role": "user", "content": "task"}])
    summaries = []
    for i in range(12):
        working.append({"role": "assistant", "content": "repair" * 120})
        working.append({"role": "user", "content": json.dumps({"host": {
            "event": "host-run", "source_sha256": "a" * 64, "passed": True}})})
    def summarize(messages):
        summaries.append(messages)
        assert request_bytes(profile.request(messages)) <= BUDGET.target
        return "Agent thinks all kernels pass, but must verify current source."
    messages = working.prepare(profile, BUDGET, summarize)
    assert summaries and request_bytes(profile.request(messages)) <= BUDGET.high
    working.append({"role": "user", "content": json.dumps({"host": {
        "event": "candidate-written", "source_sha256": "b" * 64}})})
    assert "host-run" not in working.host_state
    assert working.host_state["source_sha256"] == "b" * 64
    assert working.fixed[-1]["content"] == "task"


@pytest.mark.parametrize("summary", ["", "x" * 1000])
def test_invalid_work_log_stops_and_retains_unfolded_messages(summary):
    working = WorkingContext([{"role": "system", "content": "s"}, {"role": "user", "content": "task"}])
    working.recent = [{"role": "user", "content": "x" * 900} for _ in range(8)]
    old = list(working.recent)
    with pytest.raises(ContextBudgetError, match="work log"):
        working.prepare(direct_profile(MODEL_IDENTITIES[0]), BUDGET, lambda _: summary)
    assert working.recent == old


def test_impossible_fixed_task_never_calls_summarizer():
    working = WorkingContext([{"role": "system", "content": "s"}, {"role": "user", "content": "x" * 8000}])
    with pytest.raises(ContextBudgetError, match="fixed task"):
        working.prepare(direct_profile(MODEL_IDENTITIES[0]), BUDGET,
                        lambda _: pytest.fail("cannot summarize away task"))


def test_actor_compaction_usage_requests_raw_trace_and_distillation(tmp_path):
    calls, events = [], []
    action_count = 0
    def transport(**kw):
        nonlocal action_count
        messages = kw["request"]["messages"]
        calls.append(kw["request"])
        if messages[0]["content"].startswith("Update a compact working log"):
            return DirectCompletion("Remember repair and source hashes; verify with host.", 7)
        if "The host has now returned" in messages[-1]["content"]:
            return DirectCompletion('{"distillation":{"no_new_knowledge":true,"lessons":[]}}', 2)
        action_count += 1
        return DirectCompletion('{"submit":{}}' if action_count == 8 else '{"run":{}}', 2)
    provider = DirectCompletionProvider(ENV, transport=transport,
        context_budgets={MODEL_IDENTITIES[0].backend_model.value: BUDGET})
    actor = DirectActorDriver(provider, max_turns=8, system_prompt="system",
        trace_event=lambda kind, payload: events.append((kind, payload))).with_trace(
            lambda kind, payload: events.append((kind, payload)))
    workspace = tmp_path / "project/actor"
    workspace.mkdir(parents=True)
    def execute(action):
        return ActionOutcome(json.dumps({"host": {"event": "host-run", "passed": False,
            "source_sha256": "a" * 64, "detail": "diagnostic" * 100}}), terminal=action.kind == "submit")
    outcome = actor("task", profile=direct_profile(MODEL_IDENTITIES[0]), workspace=workspace,
                    context=None, turn_parser=parse_trial_action, action_executor=execute)
    assert outcome.iterations == 8
    assert outcome.tokens > 16
    result = actor.distill(outcome)
    assert result.iterations == 8 and result.tokens > outcome.tokens
    request_events = [p["request"] for kind, p in events if kind == "tool" and p["name"] == "provider-request"]
    assert request_events == calls
    assert all(request_bytes(call) <= BUDGET.hard for call in calls)
    assert sum(kind == "message" and p["role"] == "assistant" for kind, p in events) == 9
    assert any(kind == "tool" and p["name"] == "context-compaction" for kind, p in events)
    assert "compaction" in (tmp_path / "project/evidence.jsonl").read_text()


def test_compaction_transport_failure_is_explicit_noninfra(tmp_path):
    provider = DirectCompletionProvider(ENV, transport=lambda **_: (_ for _ in ()).throw(TimeoutError()),
        context_budgets={MODEL_IDENTITIES[0].backend_model.value: BUDGET})
    working = WorkingContext([{"role": "system", "content": "s"}, {"role": "user", "content": "task"}])
    working.recent = [{"role": "user", "content": "x" * 900} for _ in range(8)]
    from benchmarks.a5kernels.phase1_provider import _ProviderEvidenceRecorder
    with pytest.raises(ContextBudgetError, match="summarization failed"):
        DirectActorDriver(provider)._prepare_context(working, direct_profile(MODEL_IDENTITIES[0]),
                                                     _ProviderEvidenceRecorder(None), turn=3)


def test_distillation_large_host_data_and_correction_are_bounded(tmp_path):
    calls = []
    def transport(**kw):
        calls.append(kw["request"])
        if len(calls) == 1:
            return DirectCompletion('{"submit":{}}', 1)
        if len(calls) == 2:
            return DirectCompletion("invalid distillation", 1)
        return DirectCompletion('{"distillation":{"no_new_knowledge":true,"lessons":[]}}', 1)
    provider = DirectCompletionProvider(ENV, transport=transport)
    events = []
    actor = DirectActorDriver(provider, trace_event=lambda kind, payload: events.append((kind, payload)))
    workspace = tmp_path / "project/actor"
    workspace.mkdir(parents=True)
    outcome = actor("task", profile=direct_profile(MODEL_IDENTITIES[1]), workspace=workspace,
        context=None, turn_parser=parse_trial_action,
        action_executor=lambda _: ActionOutcome('{"host":{"event":"done"}}', terminal=True))
    result = actor.distill(outcome, host_summaries=[{"tensor": list(range(10000))}] * 16)
    assert result.tokens == 3
    assert all(request_bytes(call) <= 48 * 1024 for call in calls)
    assert any(kind == "message" and "previous distillation was invalid" in payload["content"]
               for kind, payload in events)


def test_request_size_counts_unicode_and_wire_metadata():
    assert request_bytes({"messages": [{"content": "é"}]}) > len("é")
    with pytest.raises(ValueError):
        ContextBudget(1024, 2000, 3000)
