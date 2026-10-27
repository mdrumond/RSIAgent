from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import copy

import pytest

from benchmarks.a5kernels.evidence import EvidenceLedger, canonical_digest
from benchmarks.a5kernels.phase1_adaptive import AdaptiveProjectRequest
from benchmarks.a5kernels.knowledge_agent import Citation
from benchmarks.a5kernels.phase1_adaptive_actor import (
    AdaptiveActionResult,
    AdaptiveActorProjectExecutor,
)
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
from benchmarks.a5kernels.phase1_memory import HostFact, parse_agent_distillation
from benchmarks.a5kernels.phase1_project import EvidenceClassification, ProjectGraph
from benchmarks.a5kernels.phase1_project_executor import (
    CompileObservation,
    ProjectExecutor,
    RuntimeObservation,
)
from benchmarks.a5kernels.trial import ActorOutcome


def _spec() -> ProjectSpec:
    graph = ProjectGraph.from_mapping({
        "inputs": [{"name": "x", "shape": [5]}],
        "nodes": [{
            "name": "out", "op": "relu", "operands": ["x"], "axis": None,
        }],
        "output": "out",
    })
    return ProjectSpec.from_mapping({
        "project_id": "repair-relu",
        "capabilities": ["gm-ub-movement", "simd-vector", "padded-tail"],
        "shape": [5],
        "dtype": "float32",
        "block_count": 1,
        "padding_multiple": 16,
        "failure_mode": "none",
        "timing_lengths": [],
        "profile_metric": None,
        "hypothesis": "Repair from exact host evidence.",
        "graph": graph.as_dict(),
    })


class SourceBackend:
    def __init__(self):
        self.requests = []
        self.run_requests = []

    def compile(self, request):
        self.requests.append(request)
        passed = request.candidate_source != "compile failure"
        return CompileObservation(
            passed, canonical_digest({"compile": request.candidate_source}),
            "compiler accepted source" if passed else "frontend rejected source",
            "object" if passed else None,
        )

    def run(self, request, compilation):
        del compilation
        self.run_requests.append(request)
        if request.candidate_source == "runtime failure":
            return RuntimeObservation(
                False, canonical_digest({"runtime": request.candidate_source}),
                "device launch failed", None,
            )
        output = (
            request.expected.storage
            if request.candidate_source == "fixed source"
            else tuple(-1.0 for _ in request.expected.storage)
        )
        return RuntimeObservation(
            True, canonical_digest({"runtime": request.candidate_source}),
            "runtime synchronized", tuple(output),
        )


class ScriptedActor:
    def __init__(self, replies):
        self.replies = tuple(replies)
        self.instruction = None
        self.observations = []
        self.distill_calls = []

    def __call__(
        self, instruction, *, profile, workspace, context,
        turn_parser, action_executor,
    ):
        del profile, workspace
        self.instruction = instruction
        assert context.read_bytes() == b'{"prior":"memory"}'
        for ordinal, reply in enumerate(self.replies, 1):
            action = turn_parser(reply)
            assert action is not None
            outcome = action_executor(action)
            self.observations.append(json.loads(outcome.observation)["host"])
            if outcome.terminal:
                return ActorOutcome(
                    outcome.status, ordinal, 10,
                    admitted_artifacts=dict(outcome.admitted_artifacts),
                    continuation="awaiting-distillation",
                )
        return ActorOutcome("turn-budget-exhausted", len(self.replies), 10)

    def distill(self, outcome, *, admitted_artifacts=None, host_summaries=()):
        admitted = dict(admitted_artifacts or {})
        self.distill_calls.append((outcome, admitted, host_summaries))
        verified = next(
            digest for digest, role in admitted.items() if role == "host-verification"
        )
        text = json.dumps({"distillation": {
            "no_new_knowledge": False,
            "lessons": [{
                "category": "failure-repair",
                "claim": "The revised source repaired the observed mismatch.",
                "applicability": "This exact adaptive ReLU project.",
                "preconditions": "The retained host classification is verified.",
                "limitations": "Other shapes were not measured.",
                "evidence_artifact_sha256": [verified],
                "failure_signature": "host output differs from the oracle",
                "repair": "Revise the source and rerun host verification.",
                "confidence": "high",
            }],
        }}, sort_keys=True, separators=(",", ":"))
        return replace(
            outcome,
            terminal_response=text,
            distillation=parse_agent_distillation(text, admitted),
            admitted_artifacts=admitted,
            continuation=None,
        )


def _special_spec(*, failure="none", timing=False, profiling=False):
    value = _spec().as_dict()
    suffix = []
    if failure != "none":
        value["capabilities"].append("failure-repair")
        value["failure_mode"] = failure
        suffix.append(failure)
    if timing:
        value["capabilities"].append("timing-sweep")
        value["timing_lengths"] = [5, 17]
        suffix.append("timing")
    if profiling:
        value["capabilities"].append("profiling")
        value["profile_metric"] = "timeline"
        suffix.append("profile")
    value["project_id"] = "repair-" + "-".join(suffix or ["relu"])
    return ProjectSpec.from_mapping(value)


def _request(tmp_path, executor, spec=None):
    spec = _spec() if spec is None else spec
    prepared = executor.prepare(spec)
    return AdaptiveProjectRequest(
        spec, 1, b'{"prior":"memory"}', tmp_path / "branch",
        prepared.inputs, prepared.expected,
    )


class TracedScriptedActor(ScriptedActor):
    def with_trace(self, callback):
        result = copy.copy(self)
        result.trace_event = callback
        return result

    def __call__(self, instruction, **kwargs):
        self.trace_event("message", {"role": "user", "content": instruction})
        original = kwargs["action_executor"]
        replies = iter(self.replies)
        def execute(action):
            self.trace_event("message", {"role": "assistant", "content": next(replies)})
            return original(action)
        return super().__call__(instruction, **{**kwargs, "action_executor": execute})


def test_adaptive_trace_retains_repair_but_excludes_distillation(tmp_path):
    from benchmarks.ascend.training_traces import recorder_factory
    from benchmarks.a5kernels.sft_dataset import _read_trace, _example
    executor = ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path / "evidence.jsonl"))
    request = _request(tmp_path, executor)
    class DeadlineCheckingActor(TracedScriptedActor):
        def __call__(self, instruction, **kwargs):
            import time
            actions = kwargs['action_executor']
            assert 0 < actions.deadline-time.monotonic() <= 5
            assert actions.query_knowledge is not None
            return super().__call__(instruction, **kwargs)
    from dataclasses import replace
    request = replace(request, remaining_seconds=5)
    actor = DeadlineCheckingActor([
        '{"write":{"slot":"kernel","content":"broken source"}}',
        '{"run":{}}',
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}', '{"submit":{}}',
    ])
    driver = AdaptiveActorProjectExecutor(executor, actor, object(), trace_recorder_factory=recorder_factory(
        phase="phase1", round_id="round-1", model="fixture", source_manifest_sha256="a" * 64,
    ))
    driver(request)
    rows = _read_trace(next((request.branch_root / "sft-traces").glob("*.jsonl")))
    example, metadata = _example(rows)
    assert metadata["repair"]
    assistants = [m for m in example["messages"] if m["role"] == "assistant"]
    assert assistants[-1] == {"role": "assistant", "content": "fixed source", "weight": 1}
    assert all(m["weight"] == 0 for m in assistants[:-1])
    assert all("distillation" not in m["content"] for m in assistants)
    assert not hasattr(actor, "trace_event")


def test_adaptive_failed_trace_is_retained_without_positive_label(tmp_path):
    from benchmarks.ascend.training_traces import recorder_factory
    from benchmarks.a5kernels.sft_dataset import _read_trace, _example
    executor = ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path / "evidence.jsonl"))
    request = _request(tmp_path, executor)
    actor = TracedScriptedActor(['{"write":{"slot":"kernel","content":"broken source"}}', '{"run":{}}'])
    driver = AdaptiveActorProjectExecutor(executor, actor, object(), trace_recorder_factory=recorder_factory(
        phase="phase2", round_id="round-1", model="fixture", source_manifest_sha256="a" * 64,
    ))
    with pytest.raises(RuntimeError, match="without host-verified"):
        driver(request)
    rows = _read_trace(next((request.branch_root / "sft-traces").glob("*.jsonl")))
    assert _example(rows) is None
    assert any(row["kind"] == "validation" for row in rows)


def test_parallel_project_traces_are_isolated_and_abort_evidence_is_retained(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from benchmarks.ascend.training_traces import recorder_factory
    from benchmarks.a5kernels.sft_dataset import _read_trace, _example
    actor = TracedScriptedActor(['{"write":{"slot":"kernel","content":"fixed source"}}',
                                '{"run":{}}', '{"submit":{}}'])
    factory = recorder_factory(phase="phase2", round_id="round-2", model="fixture",
                               source_manifest_sha256="a" * 64)
    def run(index):
        executor = ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path / str(index) / "evidence.jsonl"))
        request = _request(tmp_path / str(index), executor)
        request = replace(request, authored_ordinal=index + 1)
        AdaptiveActorProjectExecutor(executor, actor, object(), trace_recorder_factory=factory)(request)
        return next((request.branch_root / "sft-traces").glob("*.jsonl"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(run, range(2)))
    assert paths[0].name != paths[1].name
    for path in paths:
        assert _example(_read_trace(path))[0]["messages"][-1]["content"] == "fixed source"
    assert not hasattr(actor, "trace_event")

    class BrokenBackend(SourceBackend):
        def compile(self, request):
            raise OSError("retained infrastructure failure")
    executor = ProjectExecutor(BrokenBackend(), EvidenceLedger(tmp_path / "abort-evidence.jsonl"))
    request = _request(tmp_path / "abort", executor)
    with pytest.raises(OSError):
        AdaptiveActorProjectExecutor(executor, actor, object(), trace_recorder_factory=factory)(request)
    rows = _read_trace(next((request.branch_root / "sft-traces").glob("*.jsonl")))
    assert _example(rows) is None
    validation = next(row["payload"] for row in rows if row["kind"] == "validation")
    assert validation["status"] == "infrastructure-failure"
    digest = validation["evidence_sha256"]
    assert (request.branch_root / "artifacts" / "objects" / digest[:2] / digest[2:]).is_file()


def test_context_budget_abort_is_not_recorded_as_infrastructure(tmp_path):
    from benchmarks.ascend.agent_context import ContextBudget, ContextBudgetError
    from benchmarks.ascend.training_traces import recorder_factory
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.phase1_provider import DirectActorDriver, DirectCompletionProvider, direct_profile
    from benchmarks.a5kernels.sft_dataset import _read_trace

    identity = MODEL_IDENTITIES[0]
    provider = DirectCompletionProvider(
        {"OPENAI_API_KEY": "test", "DEEPSEEK_API_KEY": "test"},
        transport=lambda **_: pytest.fail("oversized request must not reach transport"),
        context_budgets={identity.backend_model.value: ContextBudget(2000, 1024, 3000)},
    )
    executor = ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path / "evidence.jsonl"))
    request = _request(tmp_path, executor)
    driver = AdaptiveActorProjectExecutor(executor, DirectActorDriver(provider), direct_profile(identity),
        trace_recorder_factory=recorder_factory(phase="phase1", round_id="context-limit",
            model="fixture", source_manifest_sha256="a" * 64))
    with pytest.raises(ContextBudgetError):
        driver(request)
    trace = _read_trace(next((request.branch_root / "sft-traces").glob("*.jsonl")))
    validations = [row["payload"] for row in trace if row["kind"] == "validation"]
    assert validations[-1]["status"] == "failed"
    assert validations[-1]["details"]["abort_status"] == "context-budget-exhausted"
    assert all(row["status"] != "infrastructure-failure" for row in validations)


def test_artifact_wire_view_and_provider_rejection_keep_exact_evidence(tmp_path):
    from benchmarks.ascend.training_traces import recorder_factory
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.phase1_provider import (
        DirectActorDriver, DirectCompletion, DirectCompletionProvider, ProviderRejectionError, direct_profile,
    )
    from benchmarks.a5kernels.sft_dataset import _read_trace
    import base64
    raw = b'{"prior":"memory"}'
    digest = hashlib.sha256(raw).hexdigest()
    calls = []
    class Rejected(Exception):
        status_code, code, request_id = 400, "invalid_prompt", "req_fixture"
    def transport(**kwargs):
        calls.append(kwargs["request"])
        if len(calls) == 1:
            return DirectCompletion(json.dumps({"read-artifact": {"sha256": digest}}), 1)
        raise Rejected("credential must not appear")
    provider = DirectCompletionProvider({"OPENAI_API_KEY": "secret", "DEEPSEEK_API_KEY": "secret"},
        transport=transport, sleeper=lambda _: None,
        clock=lambda: max(0, len(calls)-2)*120)
    executor = ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path / "evidence.jsonl"))
    request = _request(tmp_path, executor)
    driver = AdaptiveActorProjectExecutor(executor, DirectActorDriver(provider), direct_profile(MODEL_IDENTITIES[0]),
        trace_recorder_factory=recorder_factory(phase="phase1", round_id="provider-rejection",
            model="fixture", source_manifest_sha256="a" * 64))
    with pytest.raises(ProviderRejectionError):
        driver(request)
    assert len(calls) == 5 and calls[1:] == [calls[1]] * 4
    view = json.loads(calls[1]["messages"][-1]["content"])["host"]
    assert view["text"].encode() == raw and "base64" not in view
    store = ContentAddressedArtifacts(request.branch_root / "artifacts")
    raw_page = store.read_json(view["full_observation_sha256"])
    assert base64.b64decode(raw_page["base64"]) == raw
    assert store.read_bytes(digest) == raw
    trace = _read_trace(next((request.branch_root / "sft-traces").glob("*.jsonl")))
    validations = [row["payload"] for row in trace if row["kind"] == "validation"]
    assert validations[-1]["status"] == "failed"
    assert validations[-1]["details"]["abort_status"] == "provider-rejected"
    assert validations[-1]["details"]["request_id"] == "req_fixture"
    assert "credential must not appear" not in json.dumps(trace)


def test_infrastructure_retry_preserves_failed_trace_and_verified_retry(tmp_path):
    from benchmarks.ascend.training_traces import recorder_factory
    from benchmarks.a5kernels.sft_dataset import _read_trace, _example

    class RetryBackend(SourceBackend):
        fail_once = True

        def compile(self, request):
            if self.fail_once:
                self.fail_once = False
                raise OSError("transient compiler infrastructure failure")
            return super().compile(request)

    executor = ProjectExecutor(RetryBackend(), EvidenceLedger(tmp_path / "evidence.jsonl"))
    request = _request(tmp_path, executor)
    actor = TracedScriptedActor(['{"write":{"slot":"kernel","content":"fixed source"}}',
                                '{"run":{}}', '{"submit":{}}'])
    driver = AdaptiveActorProjectExecutor(executor, actor, object(), trace_recorder_factory=recorder_factory(
        phase="phase1", round_id="retry-round", model="fixture", source_manifest_sha256="a" * 64,
    ))
    with pytest.raises(OSError, match="transient"):
        driver(request)
    failed_path = next((request.branch_root / "sft-traces").glob("*.jsonl"))
    failed_bytes = failed_path.read_bytes()
    driver(request)
    paths = list((request.branch_root / "sft-traces").glob("*.jsonl"))
    assert len(paths) == 2
    assert failed_path.read_bytes() == failed_bytes
    failed = _read_trace(failed_path)
    successful = _read_trace(next(path for path in paths if path != failed_path))
    assert failed[0]["payload"]["trace_id"] != successful[0]["payload"]["trace_id"]
    assert _example(failed) is None
    assert _example(successful)[0]["messages"][-1]["content"] == "fixed source"
    assert [row["payload"]["status"] for row in failed if row["kind"] == "validation"] == ["infrastructure-failure"]


def test_measurement_requires_verified_current_source_and_preserves_comparison_facts(tmp_path):
    from benchmarks.a5kernels.trial import parse_trial_action
    assert parse_trial_action('{"request-timing":{}}').kind == "request-timing"
    executor = ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path / "evidence.jsonl"))
    calls = []

    def timing(payload, context):
        calls.append(context.source)
        digest = context.artifacts.put_json({"timing": context.source})
        return AdaptiveActionResult(
            {"event": "timing", "median_us": 3},
            host_facts=(HostFact("performance-study", "Measured current source", digest),),
            admitted_artifacts={digest: "performance-study"},
        )

    actor = ScriptedActor((
        '{"request-timing":{}}',
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"compile":{}}', '{"request-timing":{}}',
        '{"run":{}}', '{"request-timing":{}}',
        '{"write":{"slot":"kernel","content":"fixed source\\n"}}',
        '{"request-timing":{}}',
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}', '{"submit":{}}',
        '{"request-timing":{}}', '{"submit":{}}',
    ))
    result = AdaptiveActorProjectExecutor(
        executor, actor, object(), request_timing=timing,
    )(_request(tmp_path, executor, _special_spec(timing=True)))
    assert calls == ["fixed source", "fixed source"]
    assert "HOST_VERIFIED" in actor.observations[0]["error"]
    assert "HOST_VERIFIED" in actor.observations[3]["error"]
    assert "HOST_VERIFIED" in actor.observations[7]["error"]
    assert actor.observations[10]["missing"] == ["performance-study"]
    assert len([f for f in result.memory.host_facts if f.category == "performance-study"]) == 1


def test_actor_repairs_semantic_mismatch_before_verified_distillation(tmp_path):
    backend = SourceBackend()
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "evidence.jsonl"))
    actor = ScriptedActor((
        '{"write":{"slot":"kernel","content":"compile failure"}}',
        '{"compile":{}}',
        '{"write":{"slot":"kernel","content":"runtime failure"}}',
        '{"run":{}}',
        '{"write":{"slot":"kernel","content":"broken source"}}',
        '{"run":{}}',
        '{"submit":{}}',
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}',
        '{"submit":{}}',
    ))

    result = AdaptiveActorProjectExecutor(
        executor, actor, object(), immutable_context=b'{"guide":"exact"}',
    )(
        _request(tmp_path, executor)
    )

    assert result.evidence.classification is EvidenceClassification.HOST_VERIFIED
    assert result.memory.project_spec == _spec()
    assert result.memory.source_revision == hashlib.sha256(b"fixed source").hexdigest()
    assert result.memory.actions == (
        "write", "compile", "write", "run", "write", "run", "submit",
        "write", "run", "submit",
    )
    assert result.memory.lessons[0].category == "failure-repair"
    assert actor.distill_calls[0][2][0]["category"] == "host-verification"
    assert actor.observations[1]["event"] == "host-compile"
    assert actor.observations[1]["passed"] is False
    assert actor.observations[1]["artifact"]["payload"][
        "compile_diagnostic"
    ] == "frontend rejected source"
    assert actor.observations[3]["classification"] == "runtime-failure"
    assert actor.observations[3]["artifacts"]["observed"]["payload"][
        "runtime_diagnostic"
    ] == "device launch failed"
    assert actor.observations[5]["classification"] == "host-mismatch"
    assert actor.observations[5]["artifacts"]["observed"]["payload"]["output"]
    assert actor.observations[6]["error"].startswith("submit requires")
    assert actor.observations[8]["classification"] == "host-verified"
    assert '"prior":"memory"' in actor.instruction
    assert 'read-artifact' in actor.instruction
    assert 'Production context (immutable' in actor.instruction
    assert actor.instruction.count('{"guide":"exact"}') == 1
    assert executor.prepare(_spec()).instruction in actor.instruction
    assert len(backend.requests) == 4
    assert len(backend.run_requests) == 3
    assert (tmp_path / "branch" / "artifacts").is_dir()


def test_exhausted_mismatch_never_distills_or_returns_publishable_memory(tmp_path):
    executor = ProjectExecutor(
        SourceBackend(), EvidenceLedger(tmp_path / "evidence.jsonl")
    )
    actor = ScriptedActor((
        '{"write":{"slot":"kernel","content":"still broken"}}',
        '{"run":{}}',
        '{"submit":{}}',
    ))

    with pytest.raises(RuntimeError, match="without host-verified current source"):
        AdaptiveActorProjectExecutor(executor, actor, object())(
            _request(tmp_path, executor)
        )

    assert actor.observations[1]["classification"] == "host-mismatch"
    assert actor.distill_calls == []


@pytest.mark.parametrize("source, classification", [
    ("still broken", "host-mismatch"), ("compile failure", "compile-failure"),
    ("runtime failure", "runtime-failure"),
])
@pytest.mark.parametrize("traced", [False, True])
def test_phase2_retains_semantic_failure_without_distilling(tmp_path, source, classification, traced):
    executor = ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path / "evidence.jsonl"))
    actor = (TracedScriptedActor if traced else ScriptedActor)((json.dumps({"write": {"slot": "kernel", "content": source}}),
                           '{"run":{}}', '{"submit":{}}'))
    factory = None
    if traced:
        from benchmarks.ascend.training_traces import recorder_factory
        factory = recorder_factory(phase="phase2", round_id="semantic-failure", model="fixture",
                                   source_manifest_sha256="a" * 64)
    request = _request(tmp_path, executor)
    result = AdaptiveActorProjectExecutor(executor, actor, object(),
        trace_recorder_factory=factory, retain_semantic_failure=True)(request)
    assert result.evidence.classification.value == classification
    assert not result.evidence.verified
    assert result.memory.no_new_knowledge
    assert actor.distill_calls == []
    if traced:
        from benchmarks.a5kernels.sft_dataset import _read_trace, _example
        rows = _read_trace(next((request.branch_root / "sft-traces").glob("*.jsonl")))
        assert _example(rows) is None
        assert any(row["kind"] == "validation" for row in rows)


def test_adapter_fails_closed_when_request_materialization_differs(tmp_path):
    executor = ProjectExecutor(
        SourceBackend(), EvidenceLedger(tmp_path / "evidence.jsonl")
    )
    request = _request(tmp_path, executor)
    wrong = replace(request.inputs[0], storage=(99.0,) + request.inputs[0].storage[1:])
    actor = ScriptedActor(())

    with pytest.raises(ValueError, match="spec, inputs, or oracle differ"):
        AdaptiveActorProjectExecutor(executor, actor, object())(
            replace(request, inputs=(wrong,))
        )

    assert actor.instruction is None


class ChangedExecutionExecutor:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def prepare(self, spec):
        return self.wrapped.prepare(spec)

    def execute(self, spec, source):
        result = self.wrapped.execute(spec, source)
        changed = replace(result.prepared, instruction=result.prepared.instruction + " changed")
        return replace(result, prepared=changed)


def test_adapter_fails_closed_if_execution_rematerializes_a_new_contract(tmp_path):
    base = ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path / "evidence.jsonl"))
    actor = ScriptedActor((
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}',
    ))

    with pytest.raises(ValueError, match="rematerialized different"):
        AdaptiveActorProjectExecutor(ChangedExecutionExecutor(base), actor, object())(
            _request(tmp_path, base)
        )

    assert actor.distill_calls == []


def test_compile_probe_never_runs_authorizes_submit_or_consumes_action_budget(tmp_path):
    backend = SourceBackend()
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "evidence.jsonl"))
    compile_probes = ('{"compile":{}}',) * 5
    actor = ScriptedActor((
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        *compile_probes,
        '{"submit":{}}',
        '{"run":{}}',
        '{"submit":{}}',
    ))

    result = AdaptiveActorProjectExecutor(
        executor, actor, object(), max_actions=4,
    )(_request(tmp_path, executor))

    assert result.evidence.verified
    assert len(backend.requests) == 6
    assert len(backend.run_requests) == 1
    assert all(item["event"] == "host-compile" for item in actor.observations[1:6])
    assert all(item["passed"] is True for item in actor.observations[1:6])
    assert actor.observations[6]["error"].startswith("submit requires")


def test_optional_treatment_actions_have_deterministic_disabled_observations(tmp_path):
    backend = SourceBackend()
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "evidence.jsonl"))
    actor = ScriptedActor((
        '{"query-knowledge":{"query":"vector movement","limit":3}}',
        '{"request-profile":{}}',
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}',
        '{"submit":{}}',
    ))

    AdaptiveActorProjectExecutor(executor, actor, object())(_request(tmp_path, executor))

    disabled = dict(actor.observations[0])
    explanation, = disabled.pop("status_explanations")
    assert explanation["code"] == "disabled-by-treatment"
    assert "experimental treatment" in explanation["description"]
    assert [disabled, actor.observations[1]] == [
        {
            "event": "knowledge-query", "available": False,
            "reason": "disabled-by-treatment",
        },
        {
            "error": "measurement requires HOST_VERIFIED evidence for current source; run first",
        },
    ]


def test_optional_treatment_actions_use_injected_narrow_dependencies(tmp_path):
    backend = SourceBackend()
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "evidence.jsonl"))
    calls = []

    knowledge_digest = canonical_digest({"knowledge": "fact"})
    profile_digest = canonical_digest({"profile": 7})
    citation = Citation("catlass", "guide.md", 4, 9, knowledge_digest)

    def knowledge(payload, context):
        calls.append(("knowledge", dict(payload), context))
        assert context.artifacts.put_json({"knowledge": "fact"}) == knowledge_digest
        return AdaptiveActionResult(
            {"event": "knowledge-query", "available": True, "results": ["fact"]},
            citations=(citation,),
            admitted_artifacts={knowledge_digest: "knowledge"},
        )

    def profile(payload, context):
        calls.append(("profile", dict(payload), context))
        assert context.artifacts.put_json({"profile": 7}) == profile_digest
        return AdaptiveActionResult(
            {"event": "profile", "available": True, "metric": 7},
            host_facts=(HostFact(
                "profiling", "profile metric captured", profile_digest,
            ),),
            admitted_artifacts={profile_digest: "profiling"},
        )

    actor = ScriptedActor((
        '{"query-knowledge":{"query":"vector movement","limit":3}}',
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}',
        '{"request-profile":{}}',
        '{"submit":{}}',
    ))

    result = AdaptiveActorProjectExecutor(
        executor, actor, object(),
        query_knowledge=knowledge, request_profile=profile,
    )(_request(tmp_path, executor))

    assert calls[0][0:2] == (
        "knowledge", {"query": "vector movement", "limit": 3},
    )
    assert calls[0][2].request.spec == _spec()
    assert calls[0][2].source is None
    assert calls[1][0:2] == ("profile", {})
    assert calls[1][2].source == "fixed source"
    assert calls[1][2].latest_execution.evidence.verified
    assert actor.observations[0]["results"] == ["fact"]
    assert actor.observations[3]["metric"] == 7
    assert result.memory.kdb_citations == (citation,)
    assert result.memory.host_facts[0] == HostFact(
        "profiling", "profile metric captured", profile_digest,
    )
    assert knowledge_digest in result.memory.admitted_artifact_sha256
    assert profile_digest in result.memory.admitted_artifact_sha256
    distill_admitted = actor.distill_calls[0][1]
    assert distill_admitted[knowledge_digest] == "knowledge"
    assert distill_admitted[profile_digest] == "profiling"


@pytest.mark.parametrize(
    ("failure", "bad_source", "bad_action", "classification"),
    [
        ("compile", "compile failure", "compile", None),
        ("host-mismatch", "broken source", "run", "host-mismatch"),
    ],
)
def test_failure_repair_requires_observed_failure_then_verified_source(
    tmp_path, failure, bad_source, bad_action, classification,
):
    backend = SourceBackend()
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "evidence.jsonl"))
    spec = _special_spec(failure=failure)
    actor = ScriptedActor((
        json.dumps({"write": {"slot": "kernel", "content": bad_source}}),
        json.dumps({bad_action: {}}),
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}',
        '{"submit":{}}',
    ))

    result = AdaptiveActorProjectExecutor(executor, actor, object())(
        _request(tmp_path, executor, spec)
    )

    assert result.memory.project_spec == spec
    if classification is None:
        assert actor.observations[1]["passed"] is False
    else:
        assert actor.observations[1]["classification"] == classification


@pytest.mark.parametrize("failure", ["compile", "host-mismatch"])
def test_failure_repair_denies_correctness_when_requested_failure_was_not_seen(
    tmp_path, failure,
):
    executor = ProjectExecutor(
        SourceBackend(), EvidenceLedger(tmp_path / "evidence.jsonl")
    )
    actor = ScriptedActor((
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}',
        '{"submit":{}}',
    ))

    with pytest.raises(RuntimeError, match="without host-verified current source"):
        AdaptiveActorProjectExecutor(executor, actor, object())(
            _request(tmp_path, executor, _special_spec(failure=failure))
        )

    assert actor.observations[-1]["missing"] == [f"observed-{failure}"]
    assert actor.distill_calls == []


@pytest.mark.parametrize(
    ("spec", "missing"),
    [
        (_special_spec(timing=True), "performance-study"),
        (_special_spec(profiling=True), "profiling"),
    ],
)
def test_correctness_alone_cannot_complete_treatment_projects(tmp_path, spec, missing):
    executor = ProjectExecutor(
        SourceBackend(), EvidenceLedger(tmp_path / "evidence.jsonl")
    )
    actor = ScriptedActor((
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}',
        '{"submit":{}}',
    ))

    with pytest.raises(RuntimeError, match="without host-verified current source"):
        AdaptiveActorProjectExecutor(executor, actor, object())(
            _request(tmp_path, executor, spec)
        )

    assert actor.observations[-1]["missing"] == [missing]


def test_retained_performance_and_profile_evidence_complete_treatment_project(
    tmp_path,
):
    executor = ProjectExecutor(
        SourceBackend(), EvidenceLedger(tmp_path / "evidence.jsonl")
    )
    performance = canonical_digest({"timings": [4.0, 7.0]})
    profiling = canonical_digest({"timeline": "captured"})

    def treatment(_payload, context):
        assert context.latest_execution.evidence.verified
        assert context.artifacts.put_json({"timings": [4.0, 7.0]}) == performance
        assert context.artifacts.put_json({"timeline": "captured"}) == profiling
        return AdaptiveActionResult(
            {"event": "profile", "available": True},
            host_facts=(HostFact("profiling", "timeline retained", profiling),),
            admitted_artifacts={
                performance: "performance-study", profiling: "profiling",
            },
        )

    actor = ScriptedActor((
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}',
        '{"request-profile":{}}',
        '{"submit":{}}',
    ))
    spec = _special_spec(timing=True, profiling=True)

    result = AdaptiveActorProjectExecutor(
        executor, actor, object(), request_profile=treatment,
    )(_request(tmp_path, executor, spec))

    assert result.memory.project_spec == spec
    assert performance in result.memory.admitted_artifact_sha256
    assert HostFact("profiling", "timeline retained", profiling) in (
        result.memory.host_facts
    )


def test_two_correct_source_variants_retain_paired_measurements(tmp_path):
    class CorrectVariants(SourceBackend):
        def run(self, request, compilation):
            return RuntimeObservation(True, canonical_digest({"run": request.candidate_source}),
                                      "verified", request.expected.storage)
    executor = ProjectExecutor(CorrectVariants(), EvidenceLedger(tmp_path / "evidence.jsonl"))
    def timing(payload, context):
        digest = context.artifacts.put_json({"source": context.source, "median_us": 3})
        return AdaptiveActionResult({"passed": True},
            host_facts=(HostFact("performance-study", context.source, digest),),
            admitted_artifacts={digest: "performance-study"})
    actor = ScriptedActor((
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}', '{"request-timing":{}}',
        '{"write":{"slot":"kernel","content":"fixed source variant"}}',
        '{"run":{}}', '{"submit":{}}',
        '{"request-timing":{}}', '{"submit":{}}',
    ))
    result = AdaptiveActorProjectExecutor(executor, actor, object(), request_timing=timing)(
        _request(tmp_path, executor, _special_spec(timing=True)))
    assert actor.observations[5]["missing"] == ["performance-study"]
    facts = [fact for fact in result.memory.host_facts if fact.category == "performance-study"]
    assert [fact.statement for fact in facts] == ["fixed source", "fixed source variant"]
    assert all(fact.evidence_sha256 in result.memory.admitted_artifact_sha256 for fact in facts)


@pytest.mark.parametrize("rewrite_same_source", [False, True])
def test_sync_performance_rejects_one_source_and_same_source_rewrites(tmp_path, rewrite_same_source):
    class CorrectVariants(SourceBackend):
        def run(self, request, compilation):
            return RuntimeObservation(True, canonical_digest({"run": request.candidate_source}),
                                      "verified", request.expected.storage)
    executor = ProjectExecutor(CorrectVariants(), EvidenceLedger(tmp_path / "evidence.jsonl"))
    def measure(payload, context):
        digest = context.artifacts.put_json({
            "candidate_sha256": hashlib.sha256(context.source.encode()).hexdigest(),
            "summary": {"median_us": 3}, "supporting_evidence": []})
        return AdaptiveActionResult({"passed": True},
            host_facts=(HostFact("performance-study", context.source, digest),),
            admitted_artifacts={digest: "performance-study"})
    replies = ['{"write":{"slot":"kernel","content":"fixed source"}}',
               '{"run":{}}', '{"request-timing":{}}']
    if rewrite_same_source:
        replies += ['{"write":{"slot":"kernel","content":"fixed source"}}',
                    '{"run":{}}', '{"request-timing":{}}']
    replies += ['{"submit":{}}',
                '{"write":{"slot":"kernel","content":"fixed source variant"}}',
                '{"run":{}}', '{"request-timing":{}}', '{"submit":{}}']
    actor = ScriptedActor(tuple(replies))
    spec = replace(_special_spec(), learning_focus=("sync-performance",))
    result = AdaptiveActorProjectExecutor(executor, actor, object(), request_timing=measure)(
        _request(tmp_path, executor, spec))
    rejected = actor.observations[6 if rewrite_same_source else 3]
    assert "two distinct verified sources" in rejected["missing"][0]
    assert result.evidence.verified


@pytest.mark.parametrize("source", [
    "tla.print(1)", "def unused():\n    tla.print(1)", "if False:\n    tla.print(1)",
])
def test_print_source_without_runtime_trace_does_not_claim_tool_observation(tmp_path, source):
    class CorrectPrint(SourceBackend):
        def run(self, request, compilation):
            return RuntimeObservation(True, canonical_digest({"run": request.candidate_source}),
                                      "verified", request.expected.storage)
    executor = ProjectExecutor(CorrectPrint(), EvidenceLedger(tmp_path / "evidence.jsonl"))
    actor = ScriptedActor((json.dumps({"write": {"slot": "kernel", "content": source}}),
                           '{"run":{}}', '{"submit":{}}'))
    result = AdaptiveActorProjectExecutor(executor, actor, object())(
        _request(tmp_path, executor, _special_spec()))
    assert not any(f.statement.startswith("verified-debug-print:") for f in result.memory.host_facts)


def test_compile_repair_submit_explains_actual_actor_missing_requirement(tmp_path):
    backend = SourceBackend()
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "evidence.jsonl"))
    actor = ScriptedActor((
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}',
        '{"submit":{}}',
        '{"write":{"slot":"kernel","content":"compile failure"}}',
        '{"run":{}}',
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}',
        '{"submit":{}}',
    ))
    result = AdaptiveActorProjectExecutor(executor, actor, object())(
        _request(tmp_path, executor, _special_spec(failure="compile")))
    assert result.evidence.verified
    rejected = actor.observations[2]
    assert rejected["missing"] == ["observed-compile"]
    explanation, = rejected["status_explanations"]
    assert explanation["field"] == "missing"
    assert explanation["code"] == "observed-compile"


def test_subject_lesson_reaches_actual_actor_and_linked_artifact_tools_under_hard_cap(tmp_path):
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.a5kernels.evidence import canonical_bytes
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.phase1_provider import (
        DirectActorDriver, DirectCompletion, DirectCompletionProvider, direct_profile,
    )
    from benchmarks.ascend.agent_context import ContextBudget, request_bytes

    executor = ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path / 'evidence.jsonl'))
    request = _request(tmp_path, executor)
    store = ContentAddressedArtifacts(request.branch_root / 'artifacts')
    # The production curriculum eagerly copies only reachable historical objects
    # into this branch's CAS; the reader must not discover unrelated shared data.
    source_bytes = b'out = max(x, 0)\n'
    source = store.put_bytes(source_bytes)
    proposal = dict(summary='Clamp negatives before storing', applicability='float32 relu',
                    preconditions='Padded input', expectations='Nonnegative logical output',
                    limitations='No timing evidence', before=None,
                    after=dict(sha256=source, start_line=1, end_line=1, text=source_bytes.decode()),
                    evidence=[source])
    record = dict(memory_id='relu', proposal=proposal, provenance={'trace': 'z' * 76665})
    raw = canonical_bytes(dict(schema='subject-memory-snapshot-v1', initial_memory='Use host checks',
                               memories=[record]))
    request = replace(request, pre_wave_memory=raw, publish_learning=False)
    calls, pages = [], []
    def transport(**kwargs):
        wire = kwargs['request']
        assert request_bytes(wire) <= 48 * 1024
        calls.append(wire)
        if len(calls) == 1:
            instruction = wire['messages'][-1]['content']
            start = instruction.index('{"bytes":')
            view, _ = json.JSONDecoder().raw_decode(instruction[start:])
            row = view['memories'][0]
            assert row['summary'] == proposal['summary']
            assert row['after'] == proposal['after']
            assert store.read_json(row['memory_document_sha256']) == record
            action = {'read-artifact': {'sha256': row['proposal_sha256']}}
        else:
            host = json.loads(wire['messages'][-1]['content'])['host']
            if len(calls) <= 3:
                pages.append(host['text'])
            actions = [ {'read-artifact': {'sha256': source}},
                        {'write': {'slot': 'kernel', 'content': 'fixed source'}},
                        {'run': {}}, {'submit': {}} ]
            action = actions[len(calls) - 2]
        return DirectCompletion(json.dumps(action), 1)
    identity = MODEL_IDENTITIES[0]
    provider = DirectCompletionProvider({'OPENAI_API_KEY': 'test', 'DEEPSEEK_API_KEY': 'test'},
        transport=transport, context_budgets={identity.backend_model.value:
            ContextBudget(44 * 1024, 32 * 1024, 48 * 1024)})
    result = AdaptiveActorProjectExecutor(executor, DirectActorDriver(provider), direct_profile(identity))(request)
    assert json.loads(pages[0]) == proposal
    assert pages[1].encode() == source_bytes
    assert result.memory.host_facts
    assert not result.memory.lessons
    assert result.memory.source_revision == hashlib.sha256(b"fixed source").hexdigest()
    assert len(calls) == 5
