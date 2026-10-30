import json
import hashlib
from pathlib import Path

import pytest

from benchmarks.a3_experiments import (
    BackendModel, KnowledgeMode, ProfilingGuidance, ProgrammingLevel,
    build_a3_experiment_plan,
)
from benchmarks.a3_model_profiles import (
    A3Completion, A3ProviderResponseError, load_a3_model_profile,
)
from benchmarks.a3kernels.candidate import (
    CANDIDATE_SOURCE_CONTRACT,
    CandidateCompilation,
    validate_candidate_source,
)
from benchmarks.a3kernels.knowledge_agent import (
    Citation, KnowledgeQuery, KnowledgeResult,
)
from benchmarks.a3kernels.phase1_evidence import (
    EvidenceKind, EvidenceLedger, canonical_digest,
)
from benchmarks.a3kernels.phase1_memory import (
    AuthoritativeEvidence, Phase1LearningJournal,
)
from benchmarks.a3kernels.phase1_protocol import (
    ExecutionPlan, ExecutionReceipt, FailedEvidence, SourceFile, VerifiedResult, attest,
)
from benchmarks.a3kernels.phase1_registry import DEFAULT_PROPOSALS
from benchmarks.a3kernels.profiling import CompactProfileResult, ProfileMetric
from benchmarks.a3kernels.trial import (
    A3TrialLoop, Action, TrialBudgets, parse_action, trial_protocol_sha256,
)


SOURCE = 'extern "C" __global__ __aicore__ void vector_add(GM_ADDR input_a, GM_ADDR input_b, GM_ADDR output, uint32_t count, uint32_t buffer_bytes) {}'


def _cell(
    *,
    model=BackendModel.GPT_5_6_SOL,
    knowledge=KnowledgeMode.WITHOUT_KDB,
    profiling=ProfilingGuidance.WITHOUT_GUIDANCE,
):
    return next(
        cell for cell in build_a3_experiment_plan().cells
        if cell.backend_model is model
        and cell.knowledge is knowledge and cell.profiling is profiling
        and cell.programming_level is ProgrammingLevel.FOUNDATION
    )


@pytest.mark.parametrize(
    "payload",
    [
        {}, {"action": "write_source"}, {"action": "compile", "argv": ["sh"]},
        {"action": "query", "query": ""}, {"action": "profile", "metric": "Basic"},
        {"action": "submit", "interpretation": "", "supports": []},
    ],
)
def test_action_schema_rejects_missing_extra_or_invalid_fields(payload):
    with pytest.raises(ValueError, match="action"):
        parse_action(json.dumps(payload))


def test_action_schema_has_one_fixed_source_slot_and_no_argv():
    action = parse_action(json.dumps({"action": "write_source", "source": SOURCE}))
    assert action == Action("write_source", source=SOURCE)
    with pytest.raises(ValueError):
        parse_action(json.dumps({"action": "write_source", "path": "other.cpp", "source": SOURCE}))


@pytest.mark.parametrize(
    "payload, expected, received",
    [
        ({"action": "compile", "argv": ["sh"]}, ["action"], ["action", "argv"]),
        ({"action": "write_source"}, ["action", "source"], ["action"]),
    ],
)
def test_action_field_mismatch_reports_expected_and_received_names(
    payload, expected, received,
):
    with pytest.raises(ValueError) as failure:
        parse_action(json.dumps(payload))

    message = str(failure.value)
    assert f"expected={expected!r}" in message
    assert f"received={received!r}" in message


class FakeCandidate:
    def __init__(self, resolver=None): self.source, self.resolver = None, resolver
    def _plan(
        self, source, request_id, attempt_id, length, project_id,
        padded_length=None, block_count=1, execution_profile="gz-a3",
    ):
        padded_length = padded_length or length
        return ExecutionPlan(request_id, attempt_id, project_id, (SourceFile("candidate.cpp", source),),
                             ("python", "host_driver.py", "input.json"),
                             (1.0,) * padded_length, (2.0,) * padded_length,
                             execution_profile=execution_profile,
                             logical_length=length, padded_length=padded_length,
                             block_count=block_count)
    def compile(self, source, _workdir, **kw):
        self.source = source
        plan = self._plan(source, kw["request_id"], kw["attempt_id"], kw["length"], kw["project_id"], kw.get("padded_length"), kw.get("block_count", 1), kw.get("execution_profile", "gz-a3"))
        body = {"plan": plan, "library_sha256": "c" * 64, "stdout": "ok", "stderr": ""}
        result = CandidateCompilation(plan, "c" * 64, "ok", "", attest(body))
        if self.resolver:
            self.resolver.retain(
                "compile", result.attestation_sha256, plan.project_id,
                plan.source_fingerprint,
            )
        return result
    def run(self, source, _workdir, **kw):
        assert source == self.source
        plan = self._plan(source, kw["request_id"], kw["attempt_id"], kw["length"], kw["project_id"], kw.get("padded_length"), kw.get("block_count", 1), kw.get("execution_profile", "gz-a3"))
        result = VerifiedResult.from_receipt(
            plan, ExecutionReceipt(0, (3.0,) * plan.padded_length), max_abs_error=0.0
        )
        if self.resolver:
            self.resolver.retain(
                "host-verification", result.evidence_sha256, plan.project_id,
                plan.source_fingerprint,
            )
        return result


class Resolver:
    def __init__(self): self.records = {}
    def retain(self, kind, digest, project_id, candidate_sha256):
        self.records[digest] = AuthoritativeEvidence(
            kind, digest, project_id, candidate_sha256, True,
        )
    def resolve(self, digest): return self.records.get(digest)


class FakeKnowledge:
    def __init__(self, enabled): self.enabled, self.queries = enabled, []
    def query(self, query):
        assert isinstance(query, KnowledgeQuery)
        self.queries.append(query.query)
        if not self.enabled:
            return ()
        return (
            KnowledgeResult(
                Citation(
                    "a3-reference", "docs/datacopy.md", 10, 14, "1" * 64,
                    "2" * 64, "3" * 64,
                ),
                "Use DataCopyPad for a bounded tail." * 400,
                0.9,
            ),
        )


class FakeProfiler:
    def __init__(self, resolver=None): self.requests, self.resolver = [], resolver
    def profile(self, request):
        self.requests.append(request)
        if request.treatment.value.startswith("without"):
            return None
        result = CompactProfileResult.create(
            request, exported_kernels=("vector_add",),
            metric_values=(("vector_ratio", .75),), timeline=(("kernel_count", 20.0),),
            report_sha256="d" * 64,
        )
        if self.resolver:
            self.resolver.retain(
                "profiling", result.evidence_sha256,
                DEFAULT_PROPOSALS[0].project_id, request.binding.source_fingerprint,
            )
        return result


class FakeClock:
    def __init__(self): self.now = 0.0
    def __call__(self): return self.now
    def advance(self, seconds): self.now += seconds


def _run(
    tmp_path, actions, *, cell=None, knowledge=None, profiler=None,
    candidate=None, budgets=None, completion_tokens=1,
    execution_profile="gz-a3", evidence=None, model_default_budgets=False,
    clock=None, actor_seconds=1, backend_seconds=.25,
):
    clock = clock or FakeClock()
    selected = cell or _cell()
    profile = load_a3_model_profile(selected.backend_model)
    resolver = Resolver()
    queue = list(actions)
    prompts = []
    def actor(actual_profile, context):
        assert actual_profile == profile
        prompts.append(context)
        clock.advance(actor_seconds)
        action = queue.pop(0)
        if isinstance(action, BaseException):
            raise action
        if callable(action):
            action = action(context)
        if isinstance(action, A3Completion):
            return action
        return A3Completion(
            action, completion_tokens,
            {"profile_sha256": profile.fingerprint},
        )
    journal = Phase1LearningJournal(
        tmp_path / "memory.jsonl", (DEFAULT_PROPOSALS[0],),
        cell_id=selected.cell_id, lineage_id="isolated-lineage",
        evidence_resolver=resolver,
        trial_protocol_sha256=trial_protocol_sha256(),
    )
    selected_candidate = candidate or FakeCandidate(resolver)
    selected_profiler = profiler or FakeProfiler(resolver)
    if hasattr(selected_candidate, "resolver"):
        selected_candidate.resolver = resolver
    if hasattr(selected_profiler, "resolver"):
        selected_profiler.resolver = resolver
    def timed(operation):
        def invoke(*args, **kwargs):
            clock.advance(backend_seconds)
            return operation(*args, **kwargs)
        return invoke
    for name in ('compile', 'run'):
        setattr(selected_candidate, name, timed(getattr(selected_candidate, name)))
    selected_profiler.profile = timed(selected_profiler.profile)
    budget_options = (
        {} if model_default_budgets
        else {"budgets": budgets or TrialBudgets(10, 2000)}
    )
    loop = A3TrialLoop(
        cell=selected, proposal=DEFAULT_PROPOSALS[0], profile=profile, actor=actor,
        candidate=selected_candidate, knowledge=knowledge or FakeKnowledge(False),
        profiler=selected_profiler,
        evidence=evidence or EvidenceLedger(tmp_path / "evidence.jsonl"),
        memory=journal, workdir=tmp_path / "work",
        execution_profile=execution_profile, clock=clock,
        **budget_options,
    )
    return loop.run(), prompts, journal


def test_missing_canonical_provider_or_baseline_rejected_before_target_work(tmp_path):
    from rsi.domain_execution import CanonicalConfigurationError
    class NoTargetDispatch(FakeCandidate):
        def compile(self, *args, **kwargs):
            pytest.fail('target compiler ran without mandatory canonical configuration')
        def run(self, *args, **kwargs):
            pytest.fail('target runner ran without mandatory canonical configuration')
    with pytest.raises(CanonicalConfigurationError):
        _run(tmp_path, [], candidate=NoTargetDispatch())
    # The former KDB-disabled/unstructured actor success case is intentionally
    # invalid under the new mandatory baseline/provider contract. The actual
    # canonical default-path success/memory proof lives in production binding tests.
    path = tmp_path / 'memory.jsonl'
    assert not path.exists() or not path.read_bytes()


def test_candidate_contract_serializes_a3_argument_semantics_and_safe_baseline():
    contract = CANDIDATE_SOURCE_CONTRACT.as_dict()

    assert contract["argument_semantics"] == [
        "count is the logical number of float32 elements to process.",
        "Global-memory allocations may be padded, but process only indices in [0, count).",
        "buffer_bytes is the capacity of each independent input-A, input-B, and output local buffer; do not divide it among buffers.",
    ]
    assert contract["safe_baseline"] == [
        "Use three independent TBuf objects for input A, input B, and output, and initialize each one with the full buffer_bytes capacity.",
        "Use DataCopyPad with blockLen = count * sizeof(float) for logical-tail-safe input and output transfers.",
        "Order input copies before vector work with an MTE2-to-V event, and order vector work before the output copy with a V-to-MTE3 event.",
    ]


@pytest.mark.parametrize(
    "cell",
    build_a3_experiment_plan().cells,
    ids=lambda cell: cell.cell_id,
)
def test_candidate_contract_is_identical_in_every_treatment_actor_prompt(
    tmp_path, cell,
):
    actions = [
        json.dumps({"action": "write_source", "source": SOURCE}),
        '{"action":"compile"}',
        '{"action":"run"}',
        *(
            ['{"action":"profile","metric":"PipeUtilization"}']
            if cell.profiling is ProfilingGuidance.WITH_GUIDANCE else []
        ),
        json.dumps({
            "action": "submit",
            "interpretation": "host result passed",
            "supports": [],
        }),
    ]
    result, prompts, _journal = _run(
        tmp_path / cell.cell_id,
        actions,
        cell=cell,
        knowledge=FakeKnowledge(cell.knowledge is KnowledgeMode.WITH_KDB),
    )

    assert result.status == "passed"
    assert all(
        json.loads(prompt)["candidate_source_contract"]
        == CANDIDATE_SOURCE_CONTRACT.as_dict()
        for prompt in prompts
    )


def test_success_observations_expose_hashes_accepted_by_submit(tmp_path):
    submitted_supports = []

    def submit_from_context(context):
        context_value = json.loads(context)
        observations = context_value["observations"]
        authoritative = [
            item["authoritative_evidence"]
            for item in observations
            if isinstance(item, dict) and "authoritative_evidence" in item
        ]
        assert [item["kind"] for item in authoritative] == [
            "compile", "host-verification",
        ]
        assert all(item["status"] == "passed" for item in authoritative)
        assert len({item["source_fingerprint"] for item in authoritative}) == 1
        submitted_supports.extend(
            context_value["action_contract"]["eligible_submit_supports"]
        )
        assert submitted_supports == [
            item["evidence_sha256"] for item in authoritative
        ]
        assert all(len(digest) == 64 for digest in submitted_supports)
        return json.dumps({
            "action": "submit",
            "interpretation": "host compilation and verification passed",
            "supports": submitted_supports,
        })

    result, _, journal = _run(tmp_path, [
        json.dumps({"action": "write_source", "source": SOURCE}),
        '{"action":"compile"}',
        '{"action":"run"}',
        submit_from_context,
    ])

    assert result.status == "passed"
    assert journal.read()[0]["memory"]["agent_interpretations"][0][
        "supports"
    ] == submitted_supports


def test_candidate_failure_retains_structured_attestation_in_context_and_ledger(tmp_path):
    class CompileFailure(FakeCandidate):
        def compile(self, source, _workdir, **kw):
            plan = self._plan(
                source, kw["request_id"], kw["attempt_id"], kw["length"],
                kw["project_id"], kw.get("padded_length"),
                kw.get("block_count", 1), kw.get("execution_profile", "gz-a3"),
            )
            return FailedEvidence.create(
                plan, stage="compile", error_type="CompileError", detail="fixture",
            )

    result, prompts, _ = _run(
        tmp_path,
        [json.dumps({"action": "write_source", "source": SOURCE}),
         '{"action":"compile"}', '{"action":"compile"}'],
        candidate=CompileFailure(), budgets=TrialBudgets(max_seconds=3),
    )
    context = json.loads(prompts[2])
    observation = context["observations"][-1]["candidate_failure"]
    assert observation == {
        "stage": "compile",
        "error_type": "CompileError",
        "diagnostic": "fixture",
        "detail_truncated": False,
        "source_fingerprint": observation["source_fingerprint"],
        "execution_id": observation["execution_id"],
        "attestation_sha256": observation["attestation_sha256"],
        "next_action": "write_source",
    }
    assert all(len(observation[key]) == 64 for key in (
        "source_fingerprint", "execution_id", "attestation_sha256",
    ))
    assert context["action_contract"]["trial_state"] == "rewrite-required"
    assert context["action_contract"]["required_next_action"] == "write_source"
    assert set(context["allowed_actions"]) == {"write_source"}
    assert len(json.dumps(context["observations"][-1])) <= 2048
    failure_entries = EvidenceLedger(tmp_path / "evidence.jsonl").entries
    assert observation["attestation_sha256"] in {
        entry.payload["failed_evidence"]["attestation_sha256"]
        for entry in failure_entries if "failed_evidence" in entry.payload
    }
    assert result.failures == (
        "compile failed: fixture",
        "compile is not valid in rewrite-required",
    )


def test_deterministic_failure_requires_changed_source_without_backend_reentry(tmp_path):
    replacement = SOURCE.replace(" {}", " { return; }")

    class FailOnce(FakeCandidate):
        def __init__(self):
            super().__init__()
            self.compile_calls = []

        def compile(self, source, workdir, **kw):
            self.compile_calls.append(source)
            if len(self.compile_calls) == 1:
                plan = self._plan(
                    source, kw["request_id"], kw["attempt_id"], kw["length"],
                    kw["project_id"], kw.get("padded_length"),
                    kw.get("block_count", 1), kw.get("execution_profile", "gz-a3"),
                )
                return FailedEvidence.create(
                    plan, stage="compile", error_type="CompileError",
                    detail=(
                        "noise one\nERROR: primary failure\n\n"
                        " error:   primary   failure \nFatal: secondary\n"
                        + '"\\\n' * 10000
                    ),
                )
            return super().compile(source, workdir, **kw)

    candidate = FailOnce()
    result, prompts, _ = _run(tmp_path, [
        json.dumps({"action": "write_source", "source": SOURCE}),
        '{"action":"compile"}',
        '{"action":"compile"}',
        json.dumps({"action": "write_source", "source": SOURCE}),
        json.dumps({"action": "write_source", "source": replacement}),
        '{"action":"compile"}', '{"action":"run"}',
        json.dumps({
            "action": "submit", "interpretation": "host facts", "supports": [],
        }),
    ], candidate=candidate)

    assert result.status == "passed"
    assert candidate.compile_calls == [SOURCE, replacement]
    assert "not valid in rewrite-required" in result.failures[1]
    assert "source must change" in result.failures[2]
    failed = json.loads(prompts[2])["observations"][-1]["candidate_failure"]
    assert failed["diagnostic"] == (
        "ERROR: primary failure\nerror: primary failure\nFatal: secondary"
    )
    assert failed["detail_truncated"] is True
    assert len(failed["diagnostic"]) <= 2048
    assert json.loads(prompts[2])["action_contract"] == {
        "eligible_submit_supports": [],
        "profile_metric": "PipeUtilization",
        "required_next_action": "write_source",
        "trial_state": "rewrite-required",
    }


def test_changed_source_requires_compile_before_another_rewrite(tmp_path):
    replacement = SOURCE.replace(" {}", " { return; }")
    later_rewrite = SOURCE.replace(" {}", " { uint32_t unused = count; }")

    class FailOnce(FakeCandidate):
        def __init__(self):
            super().__init__()
            self.compile_calls = []

        def compile(self, source, workdir, **options):
            self.compile_calls.append(source)
            if len(self.compile_calls) == 1:
                plan = self._plan(
                    source, options["request_id"], options["attempt_id"],
                    options["length"], options["project_id"],
                    options.get("padded_length"), options.get("block_count", 1),
                    options.get("execution_profile", "gz-a3"),
                )
                return FailedEvidence.create(
                    plan, stage="compile", error_type="CompileError",
                    detail="ordinary deterministic compile failure",
                )
            return super().compile(source, workdir, **options)

    candidate = FailOnce()
    result, prompts, _ = _run(tmp_path, [
        json.dumps({"action": "write_source", "source": SOURCE}),
        '{"action":"compile"}',
        json.dumps({"action": "write_source", "source": replacement}),
        json.dumps({"action": "write_source", "source": later_rewrite}),
        json.dumps({"action": "write_source", "source": later_rewrite}),
        '{"action":"compile"}', '{"action":"run"}',
        json.dumps({
            "action": "submit", "interpretation": "host facts", "supports": [],
        }),
    ], candidate=candidate)

    assert result.status == "passed"
    assert candidate.compile_calls == [SOURCE, replacement]
    post_rewrite = json.loads(prompts[3])
    assert post_rewrite["action_contract"] == {
        "eligible_submit_supports": [],
        "profile_metric": "PipeUtilization",
        "required_next_action": "compile",
        "trial_state": "compile-required",
    }
    assert set(post_rewrite["allowed_actions"]) == {"compile"}
    assert result.failures[-2:] == (
        "write_source is not valid in compile-required",
        "write_source is not valid in compile-required",
    )


def test_successful_compile_requires_run_before_another_rewrite(tmp_path):
    replacement = SOURCE.replace(" {}", " { return; }")
    later_rewrite = SOURCE.replace(" {}", " { uint32_t unused = count; }")
    candidate = FakeCandidate()

    result, prompts, _ = _run(tmp_path, [
        json.dumps({"action": "write_source", "source": SOURCE}),
        '{"action":"compile"}',
        json.dumps({"action": "write_source", "source": replacement}),
        json.dumps({"action": "write_source", "source": later_rewrite}),
        '{"action":"run"}',
        json.dumps({
            "action": "submit", "interpretation": "host facts", "supports": [],
        }),
        '{"action":"compile"}', '{"action":"run"}',
        json.dumps({
            "action": "submit", "interpretation": "fallback", "supports": [],
        }),
    ], candidate=candidate)

    assert result.status == "passed"
    assert candidate.source == SOURCE
    post_compile = json.loads(prompts[2])
    assert post_compile["action_contract"] == {
        "eligible_submit_supports": [
            post_compile["observations"][-1]["authoritative_evidence"][
                "evidence_sha256"
            ]
        ],
        "profile_metric": "PipeUtilization",
        "required_next_action": "run",
        "trial_state": "run-required",
    }
    assert set(post_compile["allowed_actions"]) == {"run"}
    assert result.failures == (
        "write_source is not valid in run-required",
        "write_source is not valid in run-required",
    )


def test_guided_candidate_requires_profile_then_allows_optimization_rewrite(tmp_path):
    replacement = SOURCE.replace(" {}", " { return; }")
    cell = _cell(profiling=ProfilingGuidance.WITH_GUIDANCE)
    profiler = FakeProfiler()

    result, prompts, _ = _run(tmp_path, [
        json.dumps({"action": "write_source", "source": SOURCE}),
        '{"action":"compile"}', '{"action":"run"}',
        json.dumps({"action": "write_source", "source": replacement}),
        '{"action":"profile","metric":"PipeUtilization"}',
        json.dumps({"action": "write_source", "source": replacement}),
        '{"action":"compile"}', '{"action":"run"}',
        '{"action":"profile","metric":"PipeUtilization"}',
        json.dumps({
            "action": "submit", "interpretation": "profiled", "supports": [],
        }),
    ], cell=cell, profiler=profiler)

    assert result.status == "passed"
    assert len(profiler.requests) == 2
    before_profile = json.loads(prompts[3])
    assert before_profile["action_contract"]["required_next_action"] == "profile"
    assert set(before_profile["allowed_actions"]) == {"profile"}
    after_profile = json.loads(prompts[5])
    assert after_profile["action_contract"]["required_next_action"] is None
    assert set(after_profile["allowed_actions"]) == {"submit", "write_source"}
    assert result.failures == ("write_source is not valid in profile-required",)


@pytest.mark.parametrize(
    "guidance",
    [ProfilingGuidance.WITHOUT_GUIDANCE, ProfilingGuidance.WITH_GUIDANCE],
)
def test_submit_ready_allows_equal_revision_opportunity(tmp_path, guidance):
    replacement = SOURCE.replace(" {}", " { return; }")
    candidate = FakeCandidate()
    cell = _cell(profiling=guidance)
    actions = [
        json.dumps({"action": "write_source", "source": SOURCE}),
        '{"action":"compile"}', '{"action":"run"}',
    ]
    if guidance is ProfilingGuidance.WITH_GUIDANCE:
        actions.append('{"action":"profile","metric":"PipeUtilization"}')
    post_evidence_turn = len(actions)
    actions.extend([
        json.dumps({"action": "write_source", "source": replacement}),
        '{"action":"compile"}', '{"action":"run"}',
    ])
    if guidance is ProfilingGuidance.WITH_GUIDANCE:
        actions.append('{"action":"profile","metric":"PipeUtilization"}')
    actions.append(
        json.dumps({
            "action": "submit", "interpretation": "revised", "supports": [],
        })
    )
    result, prompts, _ = _run(
        tmp_path, actions, cell=cell, candidate=candidate,
    )

    assert result.status == "passed"
    assert candidate.source == replacement
    submit_ready = json.loads(prompts[post_evidence_turn])
    assert submit_ready["action_contract"]["required_next_action"] is None
    assert set(submit_ready["allowed_actions"]) == {"submit", "write_source"}
    assert result.failures == ()


def test_failed_verified_result_is_recoverable_and_requires_rewrite(tmp_path):
    replacement = SOURCE.replace(" {}", " { return; }")

    class MismatchOnce(FakeCandidate):
        def __init__(self):
            super().__init__()
            self.runs = 0

        def run(self, source, workdir, **kw):
            self.runs += 1
            if self.runs == 1:
                plan = self._plan(
                    source, kw["request_id"], kw["attempt_id"], kw["length"],
                    kw["project_id"], kw.get("padded_length"),
                    kw.get("block_count", 1), kw.get("execution_profile", "gz-a3"),
                )
                return VerifiedResult.from_receipt(
                    plan, ExecutionReceipt(0, (0.0,) * plan.padded_length),
                    max_abs_error=3.0,
                )
            return super().run(source, workdir, **kw)

    result, prompts, _ = _run(tmp_path, [
        json.dumps({"action": "write_source", "source": SOURCE}),
        '{"action":"compile"}', '{"action":"run"}',
        json.dumps({"action": "write_source", "source": replacement}),
        '{"action":"compile"}', '{"action":"run"}',
        json.dumps({
            "action": "submit", "interpretation": "host facts", "supports": [],
        }),
    ], candidate=MismatchOnce())

    assert result.status == "passed"
    assert "host verification failed" in result.failures[0]
    context = json.loads(prompts[3])
    assert context["action_contract"]["trial_state"] == "rewrite-required"
    failure = context["observations"][-1]["failed_verification"]
    assert failure["max_abs_error"] == 3.0
    assert failure["tolerance"] == 1e-5
    assert failure["next_action"] == "write_source"
    assert failure["mismatch"] == {
        "logical_index": 0,
        "input_a": 1.0,
        "input_b": 2.0,
        "actual": 0.0,
        "expected": 3.0,
        "absolute_error": 3.0,
    }
    assert set(failure) == {
        "diagnostic", "max_abs_error", "tolerance", "mismatch",
        "source_fingerprint", "execution_id", "evidence_sha256",
        "attestation_sha256", "next_action",
    }
    assert not any(
        term in json.dumps(failure).lower()
        for term in ("datacopy", "setflag", "waitflag", "queue", "query")
    )
    assert all(len(failure[key]) == 64 for key in (
        "source_fingerprint", "execution_id", "evidence_sha256",
        "attestation_sha256",
    ))
    entries = EvidenceLedger(tmp_path / "evidence.jsonl").entries
    assert any("failed_verification" in entry.payload for entry in entries)


def test_failed_verified_result_without_mismatch_reaches_actor_feedback(tmp_path):
    replacement = SOURCE.replace(" {}", " { return; }")

    class ProcessFailureOnce(FakeCandidate):
        def __init__(self):
            super().__init__()
            self.runs = 0

        def run(self, source, workdir, **kw):
            self.runs += 1
            if self.runs == 1:
                plan = self._plan(
                    source, kw["request_id"], kw["attempt_id"], kw["length"],
                    kw["project_id"], kw.get("padded_length"),
                    kw.get("block_count", 1),
                    kw.get("execution_profile", "gz-a3"),
                )
                return VerifiedResult.from_receipt(
                    plan, ExecutionReceipt(2, stderr="runtime failed"),
                    max_abs_error=None,
                )
            return super().run(source, workdir, **kw)

    result, prompts, _ = _run(tmp_path, [
        json.dumps({"action": "write_source", "source": SOURCE}),
        '{"action":"compile"}', '{"action":"run"}',
        json.dumps({"action": "write_source", "source": replacement}),
        '{"action":"compile"}', '{"action":"run"}',
        json.dumps({
            "action": "submit", "interpretation": "host facts", "supports": [],
        }),
    ], candidate=ProcessFailureOnce())

    feedback = json.loads(prompts[3])["observations"][-1][
        "failed_verification"
    ]
    assert result.status == "passed"
    assert feedback["max_abs_error"] is None
    assert feedback["mismatch"] is None
    assert feedback["next_action"] == "write_source"


def test_submit_supports_are_only_for_current_source_revision(tmp_path):
    replacement = SOURCE.replace(" {}", " { return; }")
    cell = _cell(profiling=ProfilingGuidance.WITH_GUIDANCE)

    def rewrite_from_context(context):
        value = json.loads(context)
        old = value["action_contract"]["eligible_submit_supports"]
        assert len(old) == 3
        return json.dumps({"action": "write_source", "source": replacement})

    def submit_current(context):
        value = json.loads(context)
        supports = value["action_contract"]["eligible_submit_supports"]
        assert len(supports) == 3
        observations = value["observations"]
        old = next(
            item["authoritative_evidence"]["evidence_sha256"]
            for item in observations
            if isinstance(item, dict) and "authoritative_evidence" in item
        )
        assert old not in supports
        return json.dumps({
            "action": "submit", "interpretation": "current source only",
            "supports": supports,
        })

    result, _, journal = _run(tmp_path, [
        json.dumps({"action": "write_source", "source": SOURCE}),
        '{"action":"compile"}', '{"action":"run"}',
        '{"action":"profile","metric":"PipeUtilization"}', rewrite_from_context,
        '{"action":"compile"}', '{"action":"run"}',
        '{"action":"profile","metric":"PipeUtilization"}', submit_current,
    ], cell=cell)
    assert result.status == "passed"
    assert len(journal.read()[0]["memory"]["agent_interpretations"][0]["supports"]) == 3


@pytest.mark.parametrize("pending", ["observation", "prepare", "runtime-evidence"])
def test_pending_compile_retry_reuses_the_original_attempt_identity(tmp_path, pending):
    class PendingOnce(FakeCandidate):
        def __init__(self):
            super().__init__()
            self.attempts = []

        def compile(self, source, workdir, **kw):
            self.attempts.append(kw["attempt_id"])
            if len(self.attempts) == 1:
                if pending == "observation":
                    raise RuntimeError("observation unavailable; retry retained handle")
                plan = self._plan(
                    source, kw["request_id"], kw["attempt_id"], kw["length"],
                    kw["project_id"], kw.get("padded_length"),
                    kw.get("block_count", 1), kw.get("execution_profile", "gz-a3"),
                )
                return FailedEvidence.create(
                    plan,
                    stage="prepare" if pending == "prepare" else "compile",
                    error_type=(
                        "TransferPending" if pending == "prepare" else "RuntimeError"
                    ),
                    detail="retry retained handle",
                )
            return super().compile(source, workdir, **kw)

    candidate = PendingOnce()
    result, prompts, _ = _run(
        tmp_path,
        [
            json.dumps({"action": "write_source", "source": SOURCE}),
            '{"action":"compile"}',
            json.dumps({
                "action": "write_source",
                "source": SOURCE.replace(" {}", " { return; }"),
            }),
            '{"action":"compile"}',
            '{"action":"run"}',
            json.dumps({
                "action": "submit", "interpretation": "host facts",
                "supports": [],
            }),
        ],
        candidate=candidate,
    )

    assert result.status == "passed"
    assert candidate.attempts == ["infra-0-turn-2", "infra-0-turn-2"]
    context = json.loads(prompts[2])
    assert context["action_contract"]["trial_state"] == "compile-retry"
    assert context["action_contract"]["required_next_action"] is None
    assert set(context["allowed_actions"]) == {"compile"}
    assert result.failures[-1] == "write_source is not valid in compile-retry"
    if pending != "observation":
        assert context["observations"][-1]["candidate_failure"]["next_action"] == "retry"


def test_prepare_runtime_failure_terminates_for_infrastructure_retry(tmp_path, monkeypatch):
    from test_rsi_production_domain_bindings import canonical_a3_fixture
    result, session, trial = canonical_a3_fixture(tmp_path, monkeypatch, prepare_failure=True)
    assert result.status == 'infrastructure-unverified'
    assert trial.candidate.compile_calls == 1
    assert result.verified is None and result.memory_entry_sha256 is None
    assert not trial.memory.read()
    assert session.contexts[-1]['stage'] == 'test'
    assert not any(context['stage'] == 'debug' for context in session.contexts)
    state = json.loads((trial.workdir / 'canonical' / 'workflow.json').read_text())
    assert state['pending'] and state['executions'] == 1


def test_execute_runtime_observation_retries_same_attempt_without_rewrite(tmp_path):
    class PendingRun(FakeCandidate):
        def __init__(self):
            super().__init__()
            self.attempts = []

        def run(self, source, workdir, **kw):
            self.attempts.append(kw["attempt_id"])
            if len(self.attempts) == 1:
                plan = self._plan(
                    source, kw["request_id"], kw["attempt_id"], kw["length"],
                    kw["project_id"], kw.get("padded_length"),
                    kw.get("block_count", 1), kw.get("execution_profile", "gz-a3"),
                )
                return FailedEvidence.create(
                    plan, stage="execute", error_type="RuntimeError",
                    detail="retained operation observation unavailable",
                )
            return super().run(source, workdir, **kw)

    candidate = PendingRun()
    replacement = SOURCE.replace(" {}", " { return; }")
    result, prompts, _ = _run(tmp_path, [
        json.dumps({"action": "write_source", "source": SOURCE}),
        '{"action":"compile"}', '{"action":"run"}',
        json.dumps({"action": "write_source", "source": replacement}),
        '{"action":"run"}',
        json.dumps({
            "action": "submit", "interpretation": "host facts", "supports": [],
        }),
    ], candidate=candidate)

    context = json.loads(prompts[3])
    assert context["action_contract"]["trial_state"] == "run-retry"
    assert set(context["allowed_actions"]) == {"run"}
    assert context["observations"][-1]["candidate_failure"]["next_action"] == "retry"
    assert candidate.attempts == ["infra-0-turn-3", "infra-0-turn-3"]
    assert result.status == "passed"
    assert result.failures[-1] == "write_source is not valid in run-retry"


def test_verify_runtime_failure_is_deterministic_and_requires_rewrite(tmp_path):
    replacement = SOURCE.replace(" {}", " { return; }")

    class VerifyFailure(FakeCandidate):
        def __init__(self):
            super().__init__()
            self.runs = 0

        def run(self, source, workdir, **kw):
            self.runs += 1
            if self.runs == 1:
                plan = self._plan(
                    source, kw["request_id"], kw["attempt_id"], kw["length"],
                    kw["project_id"], kw.get("padded_length"),
                    kw.get("block_count", 1), kw.get("execution_profile", "gz-a3"),
                )
                return FailedEvidence.create(
                    plan, stage="verify", error_type="RuntimeError",
                    detail="terminal verification failure",
                )
            return super().run(source, workdir, **kw)

    result, prompts, _ = _run(tmp_path, [
        json.dumps({"action": "write_source", "source": SOURCE}),
        '{"action":"compile"}', '{"action":"run"}',
        json.dumps({"action": "write_source", "source": replacement}),
        '{"action":"compile"}', '{"action":"run"}',
        json.dumps({
            "action": "submit", "interpretation": "host facts", "supports": [],
        }),
    ], candidate=VerifyFailure())

    context = json.loads(prompts[3])
    assert context["action_contract"]["trial_state"] == "rewrite-required"
    assert context["observations"][-1]["candidate_failure"]["next_action"] == "write_source"
    assert result.status == "passed"


def test_enabled_kdb_and_profile_are_exactly_cell_bound(tmp_path):
    cell = _cell(knowledge=KnowledgeMode.WITH_KDB, profiling=ProfilingGuidance.WITH_GUIDANCE)
    knowledge, profiler = FakeKnowledge(True), FakeProfiler()
    actions = [
        json.dumps({"action": "write_source", "source": SOURCE}), '{"action":"compile"}',
        '{"action":"run"}', '{"action":"query","query":"DataCopyPad","limit":2}',
        '{"action":"profile","metric":"PipeUtilization"}',
        json.dumps({"action": "submit", "interpretation": "profile supports result", "supports": []}),
    ]
    result, prompts, _ = _run(
        tmp_path, actions, cell=cell, knowledge=knowledge, profiler=profiler
    )
    assert result.status == "passed"
    assert knowledge.queries == ["DataCopyPad"]
    observation = json.loads(prompts[4])["observations"][-1]
    assert observation["knowledge_results"][0]["text"].startswith("Use DataCopyPad")
    assert len(observation["knowledge_results"][0]["text"]) <= 2048
    assert observation["knowledge_results"][0]["citation"] == {
        "chunk_id": "1" * 64,
        "collection": "a3-reference",
        "content_sha256": "3" * 64,
        "end_line": 14,
        "path": "docs/datacopy.md",
        "source_revision": "2" * 64,
        "start_line": 10,
    }
    assert profiler.requests[0].treatment.value == cell.profiling.value
    assert result.profile_result.candidate_execution_id == result.verified.execution_id
    profile_observation = json.loads(prompts[5])["observations"][-1]
    assert profile_observation == {
        "profile_evidence": {
            "evidence_sha256": result.profile_result.evidence_sha256,
            "metric": "PipeUtilization",
            "metric_values": [["vector_ratio", 0.75]],
            "source_fingerprint": result.profile_result.source_fingerprint,
            "timeline": [["kernel_count", 20.0]],
        }
    }
    profile_fact = next(
        fact for fact in json.loads((tmp_path / "memory.jsonl").read_text())["memory"]["host_facts"]
        if fact["category"] == "profiling"
    )
    assert "vector_ratio=0.75" in profile_fact["statement"]
    assert "kernel_count=20" in profile_fact["statement"]
    support_contract = json.loads(prompts[5])["action_contract"]
    assert support_contract["profile_metric"] == "PipeUtilization"
    assert support_contract["eligible_submit_supports"][-1] == (
        result.profile_result.evidence_sha256
    )


def test_trial_ledger_and_candidate_use_explicit_bz_profile(tmp_path):
    result, _, _ = _run(
        tmp_path,
        [json.dumps({"action": "write_source", "source": SOURCE}),
         '{"action":"compile"}', '{"action":"run"}',
         json.dumps({"action": "submit", "interpretation": "host facts", "supports": []})],
        execution_profile="bz-a3-2",
    )
    assert result.status == "passed"
    entries = EvidenceLedger(tmp_path / "evidence.jsonl").entries
    assert {entry.payload["execution_profile"] for entry in entries} == {"bz-a3-2"}


def test_submit_gates_source_identity_verification_and_required_profile(tmp_path):
    # Test the retained host gate directly. Canonical programming control has a
    # separate real default-binding proof; no unstructured actor loop is needed.
    owner = object.__new__(A3TrialLoop)
    owner.cell = _cell(profiling=ProfilingGuidance.WITH_GUIDANCE)
    candidate = FakeCandidate()
    options = dict(request_id=owner.cell.cell_id, attempt_id='gate', length=32, project_id=DEFAULT_PROPOSALS[0].project_id)
    compilation = candidate.compile(SOURCE, tmp_path, **options)
    verified = candidate.run(SOURCE, tmp_path, **options)
    assert owner._submit_gate(None, None, None, None)
    assert owner._submit_gate(SOURCE, compilation, verified, None)
    assert owner._submit_gate(SOURCE, compilation, verified, object()) is None
    other = candidate.compile(SOURCE + '\n// different revision', tmp_path, **options)
    assert owner._submit_gate(SOURCE, other, verified, object())


def test_failed_recompile_invalidates_stale_verification_and_can_recover(tmp_path):
    second_source = SOURCE.replace(" {}", " { return; }")
    third_source = SOURCE.replace(" {}", " { uint32_t unused = count; }")
    cell = _cell(profiling=ProfilingGuidance.WITH_GUIDANCE)
    class FailSecondCompile(FakeCandidate):
        def __init__(self):
            super().__init__()
            self.compile_calls = 0

        def compile(self, source, workdir, **options):
            self.compile_calls += 1
            if self.compile_calls == 2:
                plan = self._plan(
                    source, options["request_id"], options["attempt_id"],
                    options["length"], options["project_id"],
                    options.get("padded_length"), options.get("block_count", 1),
                    options.get("execution_profile", "gz-a3"),
                )
                return FailedEvidence.create(
                    plan, stage="compile", error_type="CompileError",
                    detail="ordinary recompile failure",
                )
            return super().compile(source, workdir, **options)

    result, _, journal = _run(
        tmp_path,
        [
            json.dumps({"action": "write_source", "source": SOURCE}),
            '{"action":"compile"}',
            '{"action":"run"}',
            '{"action":"profile","metric":"PipeUtilization"}',
            json.dumps({"action": "write_source", "source": second_source}),
            '{"action":"compile"}',
            json.dumps({
                "action": "submit", "interpretation": "stale result",
                "supports": [],
            }),
            json.dumps({"action": "write_source", "source": third_source}),
            '{"action":"compile"}',
            '{"action":"run"}',
            '{"action":"profile","metric":"PipeUtilization"}',
            json.dumps({
                "action": "submit", "interpretation": "fresh host result",
                "supports": [],
            }),
        ],
        cell=cell,
        candidate=FailSecondCompile(),
        budgets=TrialBudgets(12, 2000),
    )

    assert result.status == "passed"
    assert any("ordinary recompile failure" in failure for failure in result.failures)
    assert any("submit is not valid in rewrite-required" == failure for failure in result.failures)
    assert len(journal.read()) == 1


def test_elapsed_time_budget_is_terminal_and_isolated(tmp_path):
    result, prompts, journal = _run(
        tmp_path, ['{"action":"compile"}'] * 5,
        budgets=TrialBudgets(max_seconds=2),
    )
    assert result.status == "budget-exhausted" and len(prompts) == 2
    assert journal.read() == ()
    independent, _, other_journal = _run(tmp_path / "other", [
        json.dumps({"action": "write_source", "source": SOURCE}),
        '{"action":"compile"}', '{"action":"run"}',
        json.dumps({"action": "submit", "interpretation": "done", "supports": []}),
    ], budgets=TrialBudgets(max_seconds=10))
    assert independent.status == "passed" and len(other_journal.read()) == 1


@pytest.mark.parametrize("guidance", list(ProfilingGuidance))
def test_legacy_turn_and_token_counts_do_not_prevent_verified_submission(tmp_path, guidance):
    actions = ['not json'] * 30 + [
        json.dumps({"action": "write_source", "source": SOURCE}),
        '{"action":"compile"}', '{"action":"run"}',
    ]
    if guidance is ProfilingGuidance.WITH_GUIDANCE:
        actions.append('{"action":"profile","metric":"PipeUtilization"}')
    actions.append(json.dumps({"action": "submit", "interpretation": "done", "supports": []}))
    result, prompts, _ = _run(tmp_path, actions, cell=_cell(profiling=guidance),
        budgets=TrialBudgets(max_turns=1, max_tokens=1, max_seconds=100), completion_tokens=1000)
    assert result.status == "passed" and result.tokens > 30000
    assert result.completion_turns == 0
    assert all('attempt_completion' not in json.loads(prompt) for prompt in prompts)
    entries = EvidenceLedger(tmp_path / 'evidence.jsonl').entries
    assert not any('attempt_completion_boundary' in entry.payload for entry in entries)
    write = next(entry for entry in entries
                 if entry.kind == 'action' and entry.payload['action'] == 'write_source')
    assert write.payload['source_sha256'] == hashlib.sha256(SOURCE.encode()).hexdigest()


@pytest.mark.parametrize('backend_seconds, expected_calls', [(1, 2), (4, 1)])
def test_slow_backend_work_spends_elapsed_allocation(tmp_path, backend_seconds, expected_calls):
    actions = [json.dumps({"action": "write_source", "source": SOURCE}),
               '{"action":"compile"}', '{"action":"run"}',
               json.dumps({"action": "submit", "interpretation": "done", "supports": []})]
    candidate = FakeCandidate()
    result, prompts, journal = _run(tmp_path, actions, candidate=candidate,
        budgets=TrialBudgets(max_seconds=5), backend_seconds=backend_seconds)
    assert result.status == 'budget-exhausted' and journal.read() == ()
    assert len(prompts) == expected_calls + 1
    assert result.completion_turns == 0


def test_provider_and_invalid_actions_can_recover_while_time_remains(tmp_path):
    actions = [A3ProviderResponseError('invalid-content-or-usage'), 'not json'] * 5 + [
        json.dumps({"action": "write_source", "source": SOURCE}),
        '{"action":"compile"}', '{"action":"run"}',
        json.dumps({"action": "submit", "interpretation": "done", "supports": []})]
    result, _, journal = _run(tmp_path, actions,
        budgets=TrialBudgets(max_turns=1, max_tokens=1, max_seconds=30))
    assert result.status == 'passed' and len(journal.read()) == 1
    assert len(result.failures) == 10
    assert result.completion_turns == 0


@pytest.mark.parametrize('operation', ['compile', 'run'])
def test_validation_failure_allows_changed_source_repair_within_time(tmp_path, operation):
    replacement = SOURCE.replace(' {}', ' { return; }')
    class RejectOnce(FakeCandidate):
        rejected = False
        def invoke(self, name, source, workdir, **kwargs):
            if name == operation and not self.rejected:
                self.rejected = True
                raise ValueError('candidate source rejected by fixture')
            return getattr(super(), name)(source, workdir, **kwargs)
        def compile(self, source, workdir, **kwargs):
            return self.invoke('compile', source, workdir, **kwargs)
        def run(self, source, workdir, **kwargs):
            return self.invoke('run', source, workdir, **kwargs)
    actions = [json.dumps({'action': 'write_source', 'source': SOURCE}), '{"action":"compile"}']
    if operation == 'run':
        actions.append('{"action":"run"}')
    actions += [json.dumps({'action': 'write_source', 'source': replacement}),
                '{"action":"compile"}', '{"action":"run"}',
                json.dumps({'action': 'submit', 'interpretation': 'repaired', 'supports': []})]
    result, prompts, _ = _run(tmp_path, actions, candidate=RejectOnce(),
        budgets=TrialBudgets(max_turns=1, max_seconds=20))
    assert result.status == 'passed'
    after_failure = json.loads(prompts[3 if operation == 'run' else 2])
    assert after_failure['action_contract']['trial_state'] == 'rewrite-required'
    assert result.failures == ('candidate source rejected by fixture',)


def test_provider_response_failure_is_authenticated_and_consumes_one_turn(tmp_path):
    result, prompts, journal = _run(
        tmp_path,
        [
            A3ProviderResponseError("invalid-content-or-usage"),
            json.dumps({"action": "write_source", "source": SOURCE}),
            '{"action":"compile"}',
            '{"action":"run"}',
            json.dumps({
                "action": "submit", "interpretation": "host facts",
                "supports": [],
            }),
        ],
    )

    assert result.status == "passed" and result.turns == 5
    assert result.tokens == 4
    assert result.failures == ("provider response failure: invalid-content-or-usage",)
    assert len(prompts) == 5 and journal.resume_state().completed_projects == 1
    failure = EvidenceLedger(tmp_path / "evidence.jsonl").entries[0]
    assert failure.kind == "failure"
    assert failure.payload["provider_response_failure"] == {
        "code": "invalid-content-or-usage",
        "completion_tokens": None,
    }


def test_provider_failure_usage_is_recorded_until_time_expires(tmp_path):
    calls = []

    def malformed(_context):
        calls.append("provider")
        raise A3ProviderResponseError(
            "invalid-content-or-usage", completion_tokens=7,
        )

    result, _, journal = _run(
        tmp_path, [malformed, malformed, malformed],
        budgets=TrialBudgets(3, 10, max_seconds=3),
    )

    assert result.status == "budget-exhausted"
    assert result.turns == 3 and result.tokens == 21
    assert result.failures == (
        "provider response failure: invalid-content-or-usage",
        "provider response failure: invalid-content-or-usage",
        "provider response failure: invalid-content-or-usage",
    )
    assert calls == ["provider", "provider", "provider"]
    assert journal.read() == ()
    failures = EvidenceLedger(tmp_path / "evidence.jsonl").entries
    assert [
        entry.payload["provider_response_failure"]["completion_tokens"]
        for entry in failures
    ] == [7, 7, 7]


def test_resume_recovers_consumed_provider_failure_turn_without_duplicate_call(
    tmp_path,
):
    class InjectedCrash(RuntimeError):
        pass

    class CrashAfterFirstProviderFailure(EvidenceLedger):
        def append(self, kind, payload):
            entry = super().append(kind, payload)
            if "provider_response_failure" in payload:
                raise InjectedCrash("after retained provider failure")
            return entry

    calls = []

    def malformed(_context):
        calls.append("provider")
        raise A3ProviderResponseError(
            "invalid-content-or-usage", completion_tokens=3,
        )

    path = tmp_path / "evidence.jsonl"
    with pytest.raises(InjectedCrash, match="after retained"):
        _run(
            tmp_path, [malformed], budgets=TrialBudgets(2, 100, max_seconds=2),
            evidence=CrashAfterFirstProviderFailure(path),
        )

    result, prompts, journal = _run(
        tmp_path, [malformed], budgets=TrialBudgets(2, 100, max_seconds=2),
    )

    assert result.status == "budget-exhausted"
    assert result.turns == 2 and result.tokens == 6
    assert calls == ["provider", "provider"]
    assert len(prompts) == 1 and journal.read() == ()
    elapsed = json.loads((tmp_path / 'work/time-budget.json').read_text())
    assert elapsed['elapsed_seconds'] == 2 and elapsed['active_since'] is None
    repeated, repeated_prompts, _ = _run(tmp_path, [], budgets=TrialBudgets(max_seconds=2))
    assert repeated == result and repeated_prompts == []
    assert calls == ['provider', 'provider']
    recovered = json.loads(prompts[0])["observations"]
    assert recovered == [{
        "provider_response_failure": {
            "code": "invalid-content-or-usage",
            "completion_tokens": 3,
        }
    }]
    entries = EvidenceLedger(path).entries
    assert [entry.payload["turn"] for entry in entries] == [1, 2]
    assert all(entry.kind == "failure" for entry in entries)

    reference_calls = []

    def reference_malformed(_context):
        reference_calls.append("provider")
        raise A3ProviderResponseError(
            "invalid-content-or-usage", completion_tokens=3,
        )

    reference, _, _ = _run(
        tmp_path / "reference", [reference_malformed, reference_malformed],
        budgets=TrialBudgets(2, 100, max_seconds=2),
    )
    assert reference_calls == ["provider", "provider"]
    assert result == reference
    assert path.read_bytes() == (
        tmp_path / "reference" / "evidence.jsonl"
    ).read_bytes()


def test_provider_failure_recovery_does_not_hide_mixed_partial_state(tmp_path):
    _run(
        tmp_path, [A3ProviderResponseError("invalid-content-or-usage")],
        budgets=TrialBudgets(1, 100, max_seconds=1),
    )
    cell = _cell()
    EvidenceLedger(tmp_path / "evidence.jsonl").append(EvidenceKind.ACTION, {
        "target": "Ascend910B4", "language": "ascend-c",
        "execution_profile": "gz-a3", "runtime": "native-ascend-c",
        "cell_id": cell.cell_id,
        "project_id": DEFAULT_PROPOSALS[0].project_id,
        "turn": 2, "action": "compile",
    })

    with pytest.raises(ValueError, match="only leading retained failures"):
        _run(tmp_path, [], budgets=TrialBudgets(max_seconds=1))


@pytest.mark.parametrize("error", [ValueError("actor bug"), RuntimeError("actor bug")])
def test_unclassified_actor_errors_still_surface(tmp_path, error):
    with pytest.raises(type(error), match="actor bug"):
        _run(tmp_path, [error])


def test_default_trial_budget_is_bound_to_the_cell_model(tmp_path):
    def build_loop(model, *, budgets=None):
        cell = _cell(model=model)
        profile = load_a3_model_profile(model)
        journal = Phase1LearningJournal(
            tmp_path / model.name / "memory.jsonl", (DEFAULT_PROPOSALS[0],),
            cell_id=cell.cell_id, lineage_id="isolated-lineage",
            evidence_resolver=Resolver(),
            trial_protocol_sha256=trial_protocol_sha256(),
        )
        kwargs = {}
        if budgets is not None:
            kwargs["budgets"] = budgets
        return A3TrialLoop(
            cell=cell, proposal=DEFAULT_PROPOSALS[0], profile=profile,
            actor=lambda *_: None, candidate=FakeCandidate(),
            knowledge=FakeKnowledge(False), profiler=FakeProfiler(),
            evidence=EvidenceLedger(tmp_path / model.name / "evidence.jsonl"),
            memory=journal, workdir=tmp_path / model.name / "work", **kwargs,
        )

    assert build_loop(BackendModel.GPT_5_6_SOL).budgets == TrialBudgets(24, 32768)
    assert build_loop(BackendModel.DEEPSEEK_FLASH).budgets == TrialBudgets(24, 65536)
    override = TrialBudgets(2, 17)
    for model in BackendModel:
        assert build_loop(model, budgets=override).budgets is override


def test_public_no_arg_trial_budget_matches_the_openai_registered_default():
    assert TrialBudgets() == TrialBudgets.for_model(BackendModel.GPT_5_6_SOL)
    assert TrialBudgets() == TrialBudgets(24, 32768)


def test_trial_protocol_versions_structured_mismatch_feedback():
    legacy = canonical_digest({
        "schema": "a3-trial-protocol-v1",
        "verification_feedback_schema": "a3-verification-mismatch-v1",
        "model_budgets": {
            BackendModel.GPT_5_6_SOL.value: {
                "max_turns": 24, "max_tokens": 32768,
            },
            BackendModel.DEEPSEEK_FLASH.value: {
                "max_turns": 24, "max_tokens": 65536,
            },
        },
    })

    assert trial_protocol_sha256() != legacy


def test_trial_protocol_changed_from_v4_candidate_contract():
    assert trial_protocol_sha256() != (
        "9ab21164435b56c95e9b1171ebccd0964102340b96bee8691524dd8b02ed489f"
    )


def test_trial_protocol_versions_prepare_runtime_infrastructure_classification():
    assert trial_protocol_sha256() != (
        "b082a4e71e97f2624869d8cfa280b2ea9eb233bd8fbec84ee5116f453d49e0b4"
    )


def test_deepseek_default_recovers_after_twelve_turns_without_relaxing_gates(tmp_path):
    cell = _cell(model=BackendModel.DEEPSEEK_FLASH)

    class FourMismatches(FakeCandidate):
        def __init__(self):
            super().__init__()
            self.runs = 0

        def run(self, source, workdir, **options):
            self.runs += 1
            if self.runs <= 4:
                plan = self._plan(
                    source, options["request_id"], options["attempt_id"],
                    options["length"], options["project_id"],
                    options.get("padded_length"), options.get("block_count", 1),
                    options.get("execution_profile", "gz-a3"),
                )
                return VerifiedResult.from_receipt(
                    plan, ExecutionReceipt(0, (0.0,) * plan.padded_length),
                    max_abs_error=3.0,
                )
            return super().run(source, workdir, **options)

    sources = [
        SOURCE.replace(" {}", f" {{ uint32_t attempt = {attempt}; }}")
        for attempt in range(5)
    ]
    actions = []
    for source in sources:
        actions.extend([
            json.dumps({"action": "write_source", "source": source}),
            '{"action":"compile"}',
            '{"action":"run"}',
        ])
    actions.append(json.dumps({
        "action": "submit", "interpretation": "recovered", "supports": [],
    }))

    short, _, _ = _run(
        tmp_path / "short", actions, cell=cell, candidate=FourMismatches(),
        budgets=TrialBudgets(12, 65536, max_seconds=14),
    )
    recovered, _, _ = _run(
        tmp_path / "default", actions, cell=cell, candidate=FourMismatches(),
        model_default_budgets=True,
    )

    assert short.status == "budget-exhausted" and short.turns == 12
    assert recovered.status == "passed" and recovered.turns == 16
    assert len(recovered.failures) == 4
    assert all("host verification failed" in item for item in recovered.failures)


def test_openai_default_recovers_after_twelve_turns_without_relaxing_gates(tmp_path):
    cell = _cell(model=BackendModel.GPT_5_6_SOL)

    class FourMismatches(FakeCandidate):
        def __init__(self):
            super().__init__()
            self.runs = 0

        def run(self, source, workdir, **options):
            self.runs += 1
            if self.runs <= 4:
                plan = self._plan(
                    source, options["request_id"], options["attempt_id"],
                    options["length"], options["project_id"],
                    options.get("padded_length"), options.get("block_count", 1),
                    options.get("execution_profile", "gz-a3"),
                )
                return VerifiedResult.from_receipt(
                    plan, ExecutionReceipt(0, (0.0,) * plan.padded_length),
                    max_abs_error=3.0,
                )
            return super().run(source, workdir, **options)

    sources = [
        SOURCE.replace(" {}", f" {{ uint32_t attempt = {attempt}; }}")
        for attempt in range(5)
    ]
    actions = []
    for source in sources:
        actions.extend([
            json.dumps({"action": "write_source", "source": source}),
            '{"action":"compile"}',
            '{"action":"run"}',
        ])
    actions.append(json.dumps({
        "action": "submit", "interpretation": "recovered", "supports": [],
    }))

    short, _, _ = _run(
        tmp_path / "short", actions, cell=cell, candidate=FourMismatches(),
        budgets=TrialBudgets(12, 32768, max_seconds=14),
    )
    recovered, _, _ = _run(
        tmp_path / "default", actions, cell=cell, candidate=FourMismatches(),
        model_default_budgets=True,
    )

    assert short.status == "budget-exhausted" and short.turns == 12
    assert recovered.status == "passed" and recovered.turns == 16
    assert len(recovered.failures) == 4
    assert all("host verification failed" in item for item in recovered.failures)


def test_model_and_treatment_authorities_must_match_cell(tmp_path):
    cell = _cell(knowledge=KnowledgeMode.WITH_KDB)
    with pytest.raises(ValueError, match="Knowledge Agent"):
        _run(tmp_path, [], cell=cell, knowledge=FakeKnowledge(False))


def test_completion_provenance_cannot_switch_the_cell_model(tmp_path):
    cell = _cell()
    profile = load_a3_model_profile(cell.backend_model)
    journal = Phase1LearningJournal(
        tmp_path / "memory.jsonl", (DEFAULT_PROPOSALS[0],),
        cell_id=cell.cell_id, lineage_id="isolated-lineage",
        evidence_resolver=Resolver(),
        trial_protocol_sha256=trial_protocol_sha256(),
    )
    clock = FakeClock()
    def invalid_actor(*_):
        clock.advance(1)
        return A3Completion(
            '{"action":"compile"}', 1, {"profile_sha256": "0" * 64})
    loop = A3TrialLoop(
        cell=cell, proposal=DEFAULT_PROPOSALS[0], profile=profile,
        actor=invalid_actor, clock=clock,
        candidate=FakeCandidate(), knowledge=FakeKnowledge(False), profiler=FakeProfiler(),
        evidence=EvidenceLedger(tmp_path / "evidence.jsonl"), memory=journal,
        workdir=tmp_path / "work", budgets=TrialBudgets(1, 100, max_seconds=1),
    )
    result = loop.run()
    assert result.status == "budget-exhausted"
    assert result.failures == ("completion provenance does not match the cell model",)


def test_structured_completion_bypasses_text_parser_and_keeps_host_validation(tmp_path, monkeypatch):
    profile = load_a3_model_profile(_cell().backend_model)
    selections = [
        {"action": "compile"},  # Rejected while source is required.
        {"action": "write_source", "source": SOURCE},
        {"action": "compile"}, {"action": "run"},
        {"action": "submit", "interpretation": "host result passed", "supports": []},
    ]
    completions = [A3Completion(json.dumps(value), 1, {"profile_sha256": profile.fingerprint},
                               selected_action=value) for value in selections]
    monkeypatch.setattr('benchmarks.a3kernels.trial.parse_action',
                        lambda _: pytest.fail('structured completion must not be reparsed'))
    result, prompts, journal = _run(tmp_path, completions)
    assert result.status == 'passed' and result.verified.passed
    assert len(result.failures) == 1
    assert len(prompts) == 5 and len(journal.read()) == 1
