from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import json

import pytest

from benchmarks.a5kernels.evidence import EvidenceLedger, canonical_digest
from benchmarks.a5kernels.phase1_adaptive import AdaptiveProjectRequest
from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActorProjectExecutor
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
from benchmarks.a5kernels.phase1_memory import parse_agent_distillation
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


def _request(tmp_path, executor):
    spec = _spec()
    prepared = executor.prepare(spec)
    return AdaptiveProjectRequest(
        spec, 1, b'{"prior":"memory"}', tmp_path / "branch",
        prepared.inputs, prepared.expected,
    )


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
        '{"compile":{}}',
        '{"submit":{}}',
    ))

    result = AdaptiveActorProjectExecutor(executor, actor, object())(
        _request(tmp_path, executor)
    )

    assert result.evidence.classification is EvidenceClassification.HOST_VERIFIED
    assert result.memory.project_spec == _spec()
    assert result.memory.source_revision == hashlib.sha256(b"fixed source").hexdigest()
    assert result.memory.actions == (
        "write", "compile", "write", "run", "write", "run", "submit",
        "write", "compile", "submit",
    )
    assert result.memory.lessons[0].category == "failure-repair"
    assert actor.distill_calls[0][2][0]["category"] == "host-verification"
    assert actor.observations[1]["classification"] == "compile-failure"
    assert actor.observations[1]["artifacts"]["observed"]["payload"][
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
    assert executor.prepare(_spec()).instruction in actor.instruction
    assert len(backend.requests) == 4
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
