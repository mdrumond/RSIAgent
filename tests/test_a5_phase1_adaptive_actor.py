from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import json

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
    encoded_memory = base64.b64encode(b'{"prior":"memory"}').decode("ascii")
    assert encoded_memory in actor.instruction
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

    assert actor.observations[:2] == [
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
