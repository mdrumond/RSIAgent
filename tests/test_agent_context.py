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


@pytest.mark.parametrize("stage", ["compaction", "distillation"])
def test_provider_retry_preserves_compaction_and_distillation_requests(tmp_path, stage):
    from benchmarks.a5kernels.phase1_provider import _ProviderEvidenceRecorder
    calls, actions, delays = [], [], []
    class Rejected(Exception):
        status_code, code, request_id = 400, "invalid_prompt", "req_stage_retry"
    stage_calls = []
    def transport(**kw):
        request = kw["request"]
        calls.append(request)
        is_summary = request["messages"][0]["content"].startswith("Update a compact working log")
        is_distill = "The host has now returned" in request["messages"][-1]["content"]
        if (stage == "compaction" and is_summary) or (stage == "distillation" and is_distill):
            stage_calls.append(request)
            if len(stage_calls) == 1:
                raise Rejected("not retained")
        return DirectCompletion("Preserve current source and host facts." if is_summary else
            '{"distillation":{"no_new_knowledge":true,"lessons":[]}}' if is_distill else
            '{"submit":{}}', 2)
    profile = direct_profile(MODEL_IDENTITIES[0])
    provider = DirectCompletionProvider(ENV, transport=transport, sleeper=delays.append,
        context_budgets={profile.identity.backend_model.value: BUDGET})
    actor = DirectActorDriver(provider, system_prompt="kernel actions")
    workspace = tmp_path / "actor"
    if stage == "compaction":
        working = WorkingContext([{"role": "system", "content": "s"}, {"role": "user", "content": "task"}])
        working.recent = [{"role": "user", "content": "x" * 900} for _ in range(8)]
        actor._prepare_context(working, profile, _ProviderEvidenceRecorder(workspace), turn=3)
    else:
        def execute(action):
            actions.append(action.kind)
            return ActionOutcome('{"host":{"event":"done"}}', terminal=True)
        outcome = actor("task", profile=profile, workspace=workspace, context=None,
            turn_parser=parse_trial_action, action_executor=execute)
        actor.distill(outcome)
    assert delays == [120] and stage_calls[0] == stage_calls[1]
    assert actions == ([] if stage == "compaction" else ["submit"])
    rows = [json.loads(line)["payload"] for line in (tmp_path / "evidence.jsonl").read_text().splitlines()]
    stage_rows = [row for row in rows if row.get("stage") == stage]
    requests = [row for row in stage_rows if row["role"] == "provider-request"]
    assert [row["attempt"] for row in requests[:2]] == [1, 2]
    assert requests[0]["sha256"] == requests[1]["sha256"]
    assert sum(row["role"] == "provider-attempt-rejection" for row in stage_rows) == 1


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


def test_legacy_written_and_run_observations_pin_only_current_candidate():
    working = WorkingContext([])
    for observation in ({"event": "written", "slot": "kernel", "sha256": "a" * 64},
                        {"event": "run", "passed": True, "status": "passed"}):
        working.append({"role": "user", "content": json.dumps({"host": observation})})
    assert working.host_state["run"]["passed"] is True
    working.append({"role": "user", "content": json.dumps({"host": {
        "event": "written", "slot": "kernel", "sha256": "b" * 64}})})
    assert working.host_state["source_sha256"] == "b" * 64
    assert "run" not in working.host_state


def test_distillation_schema_and_citation_map_survive_correction_compaction(tmp_path):
    calls, distill_calls = [], []
    def transport(**kw):
        request = kw["request"]
        calls.append(request)
        if request["messages"][0]["content"].startswith("Update a compact working log"):
            return DirectCompletion("Retain host verification and citation identity.", 7)
        if len(calls) == 1:
            return DirectCompletion('{"submit":{}}', 1)
        distill_calls.append(request)
        return DirectCompletion("x" * 3500 if len(distill_calls) == 1 else
            '{"distillation":{"no_new_knowledge":true,"lessons":[]}}', 1)
    profile = direct_profile(MODEL_IDENTITIES[0])
    provider = DirectCompletionProvider(ENV, transport=transport,
        context_budgets={profile.identity.backend_model.value: BUDGET})
    actor = DirectActorDriver(provider, system_prompt="kernel actions only")
    workspace = tmp_path / "project/actor"
    workspace.mkdir(parents=True)
    outcome = actor("task", profile=profile, workspace=workspace, context=None,
        turn_parser=parse_trial_action, action_executor=lambda _: ActionOutcome(
            json.dumps({"host": {"event": "done", "detail": "terminal" * 10}}), terminal=True,
            admitted_artifacts={"a" * 64: "host-verification"}))
    outcome.continuation.working_context.recent[:0] = [
        {"role": "assistant", "content": "old repair" * 40},
        {"role": "user", "content": "old diagnostic" * 40},
    ]
    actor.distill(outcome)
    assert len(distill_calls) == 2
    assert any(call["messages"][0]["content"].startswith("Update a compact working log") for call in calls)
    for request in distill_calls:
        text = json.dumps(request)
        assert "no_new_knowledge" in text and "a" * 64 in text


def test_distillation_infrastructure_failure_retains_summary_usage(tmp_path):
    calls = []
    def transport(**kw):
        calls.append(kw["request"])
        if len(calls) == 1:
            return DirectCompletion('{"submit":{}}', 1)
        if kw["request"]["messages"][0]["content"].startswith("Update a compact working log"):
            return DirectCompletion("Host terminal result retained.", 7)
        raise TimeoutError()
    profile = direct_profile(MODEL_IDENTITIES[0])
    provider = DirectCompletionProvider(ENV, transport=transport,
        context_budgets={profile.identity.backend_model.value: ContextBudget(5000, 3500, 15000)})
    actor = DirectActorDriver(provider, system_prompt="system")
    workspace = tmp_path / "project/actor"
    workspace.mkdir(parents=True)
    outcome = actor("task", profile=profile, workspace=workspace, context=None,
        turn_parser=parse_trial_action, action_executor=lambda _: ActionOutcome(
            json.dumps({"host": {"event": "done", "detail": "x" * 2400}}), terminal=True))
    # Additional earlier exchange forces a successful work-log update before timeout.
    outcome.continuation.working_context.recent.insert(0, {"role": "user", "content": "old repair" * 30})
    result = actor.distill(outcome)
    assert result.tokens >= 8


def test_recorded_oversized_work_log_corrects_from_original_input():
    profile = direct_profile(MODEL_IDENTITIES[1])
    old_budget = ContextBudget(32 * 1024, 24 * 1024, 48 * 1024)
    working = WorkingContext([{"role": "system", "content": "s"}, {"role": "user", "content": "task"}])
    working.recent = [{"role": "user", "content": "x" * 5000} for _ in range(8)]
    working.work_log = "previous discoveries"
    original = list(working.recent)
    calls = []
    def summarize(messages):
        calls.append(messages)
        assert request_bytes(profile.request(messages)) <= old_budget.target
        # Nothing is folded before a replacement is accepted.
        assert working.recent == original
        return "x" * 6321 if len(calls) == 1 else "corrected log"
    working.prepare(profile, old_budget, summarize)
    assert len(calls) == 2
    assert calls[1][:2] == calls[0]
    assert "6321 UTF-8 bytes" in calls[1][-1]["content"]
    assert "6144 UTF-8 bytes" in calls[1][-1]["content"]
    assert working.work_log == "corrected log"
    assert working.recent == original[len(json.loads(calls[0][1]["content"])["exchanges"]):]


@pytest.mark.parametrize("invalid,measurement", [("", "0 UTF-8 bytes"),
    (" " * 4, "4 UTF-8 bytes"), ("é" * 438, "876 UTF-8 bytes"),
    (None, "non-text (UTF-8 size unavailable)")])
def test_summary_correction_exhaustion_keeps_original_input(invalid, measurement):
    working = WorkingContext([{"role": "system", "content": "s"}, {"role": "user", "content": "task"}])
    working.recent = [{"role": "user", "content": "x" * 900} for _ in range(8)]
    working.work_log = "old log"
    original = working.messages()
    calls = []
    def summarize(messages):
        calls.append(messages)
        return invalid
    with pytest.raises(ContextBudgetError, match="after 3 attempts") as error:
        working.prepare(direct_profile(MODEL_IDENTITIES[0]), BUDGET, summarize)
    assert measurement in str(error.value)
    assert "875 UTF-8 bytes" in str(error.value)
    assert len(calls) == 3
    assert calls[1][:2] == calls[2][:2] == calls[0]
    assert len(calls[1]) == len(calls[2]) == 3
    assert working.messages() == original
    assert all(request_bytes(direct_profile(MODEL_IDENTITIES[0]).request(call)) <= BUDGET.target for call in calls)


def test_summary_limit_accepts_utf8_boundary_without_truncation():
    working = WorkingContext([{"role": "system", "content": "s"}, {"role": "user", "content": "task"}])
    working.recent = [{"role": "user", "content": "x" * 900} for _ in range(8)]
    boundary = "é" * 437 + "x"
    assert len(boundary.encode("utf-8")) == BUDGET.target // 4
    working.prepare(direct_profile(MODEL_IDENTITIES[0]), BUDGET, lambda _: boundary)
    assert working.work_log == boundary


def test_deepseek_default_preserves_fixed_task_and_ordinary_turns():
    from benchmarks.ascend.agent_context import DEFAULT_CONTEXT_BUDGETS
    budget = DEFAULT_CONTEXT_BUDGETS["deepseek-flash"]
    assert budget == DEFAULT_CONTEXT_BUDGETS["openai/gpt-5.6-sol"] == ContextBudget(131072, 81920, 196608)
    profile = direct_profile(MODEL_IDENTITIES[1])
    working = WorkingContext([{"role": "system", "content": "s"}, {"role": "user", "content": "t" * 25863}])
    for _ in range(20):
        working.append({"role": "assistant", "content": "action" * 100})
        working.append({"role": "user", "content": "host result" * 100})
        working.prepare(profile, budget, lambda _: pytest.fail("ordinary turns must not compact repeatedly"))
    assert working.fixed[1]["content"] == "t" * 25863


@pytest.mark.parametrize("exhaust", [False, True])
def test_compaction_correction_records_every_request_response_and_token(tmp_path, exhaust):
    from benchmarks.a5kernels.phase1_provider import _ProviderEvidenceRecorder
    calls, events = [], []
    def transport(**kw):
        calls.append(kw["request"])
        return DirectCompletion("x" * 1000 if exhaust or len(calls) == 1 else "accepted log", 7)
    profile = direct_profile(MODEL_IDENTITIES[0])
    provider = DirectCompletionProvider(ENV, transport=transport,
        context_budgets={profile.identity.backend_model.value: BUDGET})
    actor = DirectActorDriver(provider, trace_event=lambda kind, payload: events.append((kind, payload)))
    working = WorkingContext([{"role": "system", "content": "s"}, {"role": "user", "content": "task"}])
    working.recent = [{"role": "user", "content": "x" * 900} for _ in range(8)]
    if exhaust:
        with pytest.raises(ContextBudgetError, match="after 3 attempts"):
            actor._prepare_context(working, profile, _ProviderEvidenceRecorder(tmp_path / "actor"), turn=3)
        assert len(calls) == 3
    else:
        _, tokens = actor._prepare_context(working, profile, _ProviderEvidenceRecorder(tmp_path / "actor"), turn=3)
        assert tokens == 7 * len(calls)
        assert len(calls) >= 2
    rows = [json.loads(line)["payload"] for line in (tmp_path / "evidence.jsonl").read_text().splitlines()]
    assert sum(row["role"] == "provider-request" for row in rows) == len(calls)
    assert sum(row["role"] == "provider-response" for row in rows) == len(calls)
    summaries = [payload for kind, payload in events if kind == "tool" and payload["name"] == "context-compaction"]
    assert len(summaries) == len(calls)
    assert sum(payload["completion_tokens"] for payload in summaries) == 7 * len(calls)


def test_compaction_transport_failure_does_not_use_summary_correction():
    working = WorkingContext([{"role": "system", "content": "s"}, {"role": "user", "content": "task"}])
    working.recent = [{"role": "user", "content": "x" * 900} for _ in range(8)]
    calls = []
    def summarize(messages):
        calls.append(messages)
        raise TimeoutError()
    with pytest.raises(TimeoutError):
        working.prepare(direct_profile(MODEL_IDENTITIES[0]), BUDGET, summarize)
    assert len(calls) == 1


@pytest.mark.parametrize("exhaust", [False, True])
def test_sdk_empty_compaction_is_corrected_and_retained(monkeypatch, tmp_path, exhaust):
    import sys
    from types import SimpleNamespace
    from benchmarks.a5kernels.phase1_provider import _ProviderEvidenceRecorder
    calls, events = [], []
    def create(**request):
        calls.append(request)
        text = "  " if exhaust or len(calls) == 1 else "accepted log"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                               usage=SimpleNamespace(completion_tokens=5))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **_: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))))
    profile = direct_profile(MODEL_IDENTITIES[0])
    provider = DirectCompletionProvider(ENV, context_budgets={profile.identity.backend_model.value: BUDGET})
    actor = DirectActorDriver(provider, trace_event=lambda kind, payload: events.append((kind, payload)))
    working = WorkingContext([{"role": "system", "content": "s"}, {"role": "user", "content": "task"}])
    working.recent = [{"role": "user", "content": "x" * 900} for _ in range(8)]
    original = working.messages()
    recorder = _ProviderEvidenceRecorder(tmp_path / "actor")
    if exhaust:
        with pytest.raises(ContextBudgetError, match="after 3 attempts.*2 UTF-8 bytes"):
            actor._prepare_context(working, profile, recorder, turn=3)
        assert len(calls) == 3 and working.messages() == original
    else:
        _, tokens = actor._prepare_context(working, profile, recorder, turn=3)
        assert tokens == 5 * len(calls)
        assert working.work_log == "accepted log"
    assert "2 UTF-8 bytes" in calls[1]["messages"][-1]["content"]
    rows = [json.loads(line)["payload"] for line in (tmp_path / "evidence.jsonl").read_text().splitlines()]
    assert sum(row["role"] == "provider-request" for row in rows) == len(calls)
    assert sum(row["role"] == "provider-response" for row in rows) == len(calls)
    traces = [p for kind, p in events if kind == "tool" and p["name"] == "context-compaction"]
    assert len(traces) == len(calls) and sum(p["completion_tokens"] for p in traces) == len(calls) * 5
    empty_digest = next(row["sha256"] for row in rows if row["role"] == "provider-response")
    assert recorder.artifacts.read_bytes(empty_digest) == b"  "


@pytest.mark.parametrize("content,usage", [("", 5), ("  ", 5), (None, 5), ([], 5), ("", 0)])
def test_sdk_empty_action_contract_still_rejects(monkeypatch, content, usage):
    import sys
    from types import SimpleNamespace
    calls = []
    def create(**request):
        calls.append(request)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                               usage=SimpleNamespace(completion_tokens=usage))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **_: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))))
    with pytest.raises(ValueError):
        DirectCompletionProvider(ENV).complete(direct_profile(MODEL_IDENTITIES[0]),
                                              [{"role": "user", "content": "act"}])
    assert len(calls) == 1


@pytest.mark.parametrize("content,usage", [(None, 5), ([], 5), ("", 0)])
def test_sdk_invalid_response_is_not_compaction_correction(monkeypatch, content, usage):
    import sys
    from types import SimpleNamespace
    from benchmarks.a5kernels.phase1_provider import _ProviderEvidenceRecorder
    calls = []
    def create(**request):
        calls.append(request)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                               usage=SimpleNamespace(completion_tokens=usage))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **_: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))))
    profile = direct_profile(MODEL_IDENTITIES[0])
    provider = DirectCompletionProvider(ENV, context_budgets={profile.identity.backend_model.value: BUDGET})
    working = WorkingContext([{"role": "system", "content": "s"}, {"role": "user", "content": "task"}])
    working.recent = [{"role": "user", "content": "x" * 900} for _ in range(8)]
    with pytest.raises(ContextBudgetError, match="summarization failed"):
        DirectActorDriver(provider)._prepare_context(working, profile, _ProviderEvidenceRecorder(None), turn=3)
    assert len(calls) == 1
