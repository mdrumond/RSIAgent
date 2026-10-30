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
    def __init__(self, enabled):
        self.enabled, self.queries = enabled, []
        self.canonical_identity = dict(enabled=enabled, test_collection='CPU', manifest='a3-fixture-v1')
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
    from benchmarks.a3kernels.candidate import A3CandidateBackend, CandidateSourceAdmissionError
    contract = json.loads(json.dumps(CANDIDATE_SOURCE_CONTRACT.as_dict()))
    source = contract['exported_signature'] + ' {}'
    backend = A3CandidateBackend(lambda *_args, **_kwargs: pytest.fail('source admission dispatched host'))
    plan = backend.plan(source, request_id='contract', attempt_id='first', project_id='bounded-tail',
                        length=33, padded_length=64, block_count=3, seed=7)
    candidate, = [file for file in plan.files if file.relative_path == contract['source_slot']]
    assert candidate.content == source
    assert (plan.logical_length, plan.padded_length, plan.block_count) == (33, 64, 3)
    assert len(plan.input_a) == len(plan.input_b) == 64
    assert plan.input_a[33:] == plan.input_b[33:] == (0.0,) * 31
    output = tuple(a+b for a,b in zip(plan.input_a, plan.input_b))
    verified = VerifiedResult.from_receipt(plan, ExecutionReceipt(0, output), max_abs_error=0.0)
    assert verified.passed and verified.source_fingerprint == plan.source_fingerprint
    with pytest.raises(CandidateSourceAdmissionError):
        backend.plan(source.replace('uint32_t count', 'uint64_t count'), request_id='contract',
            attempt_id='wrong-abi', project_id='bounded-tail', length=33, padded_length=64,
            block_count=3, seed=7)


@pytest.mark.parametrize(
    "cell",
    build_a3_experiment_plan().cells,
    ids=lambda cell: cell.cell_id,
)
def test_candidate_contract_is_identical_in_every_treatment_actor_prompt(tmp_path, cell):
    from canonical_a3_fixtures import make_trial
    owner, session = make_trial(tmp_path / cell.cell_id, cell=cell)
    result = owner.run()
    assert result.status == 'passed' and result.verified.passed
    presented = [json.loads(context['task']['domain']['instruction']) for context in session.contexts]
    assert presented and all(row['source_contract'] == CANDIDATE_SOURCE_CONTRACT.as_dict() for row in presented)
    queries = owner.baseline_knowledge.journal.read()
    assert queries and all(row['collection'] == 'baseline' for row in queries)
    experience = owner.knowledge.journal.read()
    assert bool(experience) == (cell.knowledge is KnowledgeMode.WITH_KDB)
    if not owner.knowledge.enabled:
        assert owner.knowledge.database is None
        evidence = [row for context in session.contexts for row in context['evidence'] if row['name'] == 'query_kdb']
        assert all(hit['citation']['collection'] == 'baseline'
                   for row in evidence for hit in row['result']['evidence']['payload']['response']['results'])


def test_success_observations_expose_hashes_accepted_by_submit(tmp_path):
    from canonical_a3_fixtures import make_trial
    owner, session = make_trial(tmp_path)
    result = owner.run()
    assert result.status == 'passed'
    runner = owner.canonical.build(owner)
    host = runner.backend.host
    facts = owner.memory.read()[0]['memory']['host_facts']
    assert {row['evidence_sha256'] for row in facts} == {
        host.compilation.attestation_sha256, result.verified.evidence_sha256}
    receipts = [runner.store.read_json(row['reference']) for row in runner.events if row['type'] == 'domain-result']
    assert receipts and all(row['observed_terminal'] for row in receipts)
    assert host.compilation.plan.source_fingerprint == result.verified.source_fingerprint
    assert owner.run() == result
    assert len(owner.memory.read()) == 1


def test_candidate_failure_retains_structured_attestation_in_context_and_ledger(tmp_path):
    from canonical_a3_fixtures import make_trial, recording_candidate
    candidate = recording_candidate(compile_failures=1)
    owner, session = make_trial(tmp_path, candidate=candidate)
    result = owner.run()
    assert result.status == 'passed'
    failures = [row.payload['failed_evidence'] for row in owner.evidence.entries if 'failed_evidence' in row.payload]
    assert len(failures) == 1 and failures[0]['stage'] == 'compile'
    assert failures[0]['error_type'] == 'CompileError'
    assert len(failures[0]['attestation_sha256']) == 64
    debug = [context for context in session.contexts if context['stage'] == 'debug']
    assert debug and debug[0]['task']['domain']['observations']
    feedback = next(row['candidate_failure'] for row in debug[0]['task']['domain']['observations']
                    if 'candidate_failure' in row)
    for key in ('stage', 'error_type', 'source_fingerprint', 'execution_id', 'attestation_sha256'):
        assert feedback[key] == failures[0][key]
    assert feedback['diagnostic'] == failures[0]['detail']
    assert feedback['detail_truncated'] is False
    assert feedback['next_action'] == 'write_source'
    assert any(row.get('passed') is False for row in debug[0]['task']['domain']['observations'])
    assert candidate.compiles == 2 and candidate.calls[0][1] != candidate.calls[1][1]
    assert result.verified.passed and len(owner.memory.read()) == 1


def test_deterministic_failure_requires_changed_source_without_backend_reentry(tmp_path):
    from canonical_a3_fixtures import A3Session, make_trial, recording_candidate

    class UnchangedThenRepair(A3Session):
        attempted_unchanged = False
        genuinely_edited = False

        def next_action(self, context, allowed, **kwargs):
            name, args = super().next_action(context, allowed, **kwargs)
            if name == 'perform' and context['stage'] == 'debug' and not self.attempted_unchanged:
                assert [name for name, _, _ in candidate.calls] == ['compile']
                self.attempted_unchanged = True
                args['edit']['source'] = context['source']['text']
            elif name == 'perform' and self.attempted_unchanged and not self.genuinely_edited:
                assert context['stage'] == 'implementation'
                assert context['source']['text'] == SOURCE
                assert [(name, source) for name, source, _ in candidate.calls] == [('compile', SOURCE)]
                self.genuinely_edited = True
                args['edit'] = dict(path=context['source']['path'], sha256=context['source']['sha256'],
                                    source=self.repaired)
            return name, args

    detail = ('noise one\nERROR: primary failure\n\n error:   primary   failure \nFatal: secondary\n'
              + '\"\\\n' * 10000)
    candidate = recording_candidate(compile_failures=1, compile_detail=detail)
    session = UnchangedThenRepair(initial=SOURCE, repaired=SOURCE + '\n// repaired')
    owner, session = make_trial(tmp_path, candidate=candidate, session=session)
    result = owner.run()
    assert result.status == 'passed'
    compiled = [source for name, source, _ in candidate.calls if name == 'compile']
    assert len(compiled) == 2 and compiled[0] != compiled[1]
    assert session.attempted_unchanged and session.genuinely_edited
    assert compiled == [SOURCE, session.repaired]
    assert candidate.runs == 2
    assert any(context['stage'] == 'debug' for context in session.contexts)
    assert len(owner.memory.read()) == 1

    contexts = [context for context in session.contexts if context['stage'] == 'debug']
    compact = next(row['candidate_failure'] for row in contexts[0]['task']['domain']['observations']
                   if 'candidate_failure' in row)
    assert compact['diagnostic'] == 'ERROR: primary failure\nerror: primary failure\nFatal: secondary'
    assert compact['detail_truncated'] is True and len(compact['diagnostic']) <= 2048


def test_changed_source_requires_compile_before_another_rewrite(tmp_path):
    from canonical_a3_fixtures import make_trial, recording_candidate, RevisionSession
    candidate = recording_candidate()
    session = RevisionSession(initial=SOURCE, repaired=SOURCE + '\n// repaired')
    owner, _ = make_trial(tmp_path, candidate=candidate, session=session)
    result = owner.run()
    assert result.status == 'passed'
    assert [name for name, _, _ in candidate.calls] == ['compile', 'run', 'compile', 'run', 'run']
    initial, revised = candidate.compiled_results
    assert initial.plan.source_fingerprint != revised.plan.source_fingerprint
    assert result.verified.source_fingerprint == revised.plan.source_fingerprint
    assert len(owner.memory.read()) == 1


def test_successful_compile_requires_run_before_another_rewrite(tmp_path):
    from canonical_a3_fixtures import make_trial, recording_candidate, RevisionSession
    candidate = recording_candidate()
    session = RevisionSession(initial=SOURCE, repaired=SOURCE + '\n// repaired')
    owner, _ = make_trial(tmp_path, candidate=candidate, session=session)
    result = owner.run()
    assert result.status == 'passed'
    assert candidate.calls[0][0] == 'compile' and candidate.calls[1][0] == 'run'
    assert candidate.calls[0][1] == candidate.calls[1][1]
    assert candidate.calls[2][0] == 'compile' and candidate.calls[2][1] != candidate.calls[1][1]
    assert all(source == candidate.calls[-1][1] for name, source, _ in candidate.calls[2:])


def test_guided_candidate_requires_profile_then_allows_optimization_rewrite(tmp_path):
    from canonical_a3_fixtures import make_trial, recording_candidate, RevisionSession
    candidate = recording_candidate()
    session = RevisionSession(initial=SOURCE, repaired=SOURCE + '\n// repaired')
    owner, _ = make_trial(tmp_path, candidate=candidate, session=session,
                        cell=_cell(profiling=ProfilingGuidance.WITH_GUIDANCE))
    result = owner.run()
    assert result.status == 'passed' and session.revised
    request, = owner.profiler.requests
    assert request.binding.source_fingerprint == candidate.compiled_results[-1].plan.source_fingerprint
    assert request.binding.source_fingerprint != candidate.compiled_results[0].plan.source_fingerprint
    assert result.profile_result.source_fingerprint == result.verified.source_fingerprint
    assert result.profile_result.candidate_execution_id == result.verified.execution_id


@pytest.mark.parametrize(
    "guidance",
    [ProfilingGuidance.WITHOUT_GUIDANCE, ProfilingGuidance.WITH_GUIDANCE],
)
def test_submit_ready_allows_equal_revision_opportunity(tmp_path, guidance):
    from canonical_a3_fixtures import make_trial, recording_candidate, RevisionSession
    candidate = recording_candidate()
    session = RevisionSession(initial=SOURCE, repaired=SOURCE + '\n// repaired')
    owner, _ = make_trial(tmp_path, candidate=candidate, session=session, cell=_cell(profiling=guidance))
    result = owner.run()
    assert result.status == 'passed' and session.revised
    assert candidate.compiles == 2 and len(owner.memory.read()) == 1
    assert candidate.compiled_results[-1].plan.source_fingerprint == result.verified.source_fingerprint
    assert bool(result.profile_result) == (guidance is ProfilingGuidance.WITH_GUIDANCE)


def test_failed_verified_result_is_recoverable_and_requires_rewrite(tmp_path):
    from canonical_a3_fixtures import make_trial, recording_candidate
    candidate = recording_candidate(run_failures=1)
    owner, session = make_trial(tmp_path, candidate=candidate)
    result = owner.run()
    assert result.status == 'passed' and result.verified.passed
    failures = [row.payload['failed_verification'] for row in owner.evidence.entries if 'failed_verification' in row.payload]
    assert len(failures) == 1 and failures[0]['passed'] is False
    mismatch = failures[0]['mismatch']
    assert mismatch['logical_index'] == 0
    assert abs(mismatch['actual'] - mismatch['expected']) == mismatch['absolute_error'] == 1.0
    assert failures[0]['max_abs_error'] == mismatch['absolute_error']
    debug = next(context for context in session.contexts if context['stage'] == 'debug')
    observed = next(row['failed_verification'] for row in debug['task']['domain']['observations']
                    if 'failed_verification' in row)
    assert observed['mismatch'] == mismatch
    assert observed['evidence_sha256'] == failures[0]['evidence_sha256']
    assert observed['tolerance'] == failures[0]['tolerance']
    assert observed['next_action'] == 'write_source'
    assert candidate.compiles == 2
    sources = [source for name, source, _ in candidate.calls if name == 'compile']
    assert sources[0] != sources[1]
    assert any(context['stage'] == 'debug' for context in session.contexts)
    assert len(owner.memory.read()) == 1


def test_failed_verified_result_without_mismatch_reaches_actor_feedback(tmp_path):
    from canonical_a3_fixtures import make_trial, recording_candidate
    candidate = recording_candidate(run_failures=1, run_exit=1)
    owner, session = make_trial(tmp_path, candidate=candidate)
    result = owner.run()
    assert result.status == 'passed' and result.verified.passed
    debug = [context for context in session.contexts if context['stage'] == 'debug']
    assert debug and debug[0]['task']['domain']['observations']
    feedback = next(row['failed_verification'] for row in debug[0]['task']['domain']['observations']
                    if 'failed_verification' in row)
    failure, = [row.payload['failed_verification'] for row in owner.evidence.entries
                if 'failed_verification' in row.payload]
    assert failure['passed'] is False and failure['exit_code'] == 1
    assert feedback['max_abs_error'] is None and feedback['mismatch'] is None
    assert feedback['next_action'] == 'write_source'
    assert feedback['evidence_sha256'] == failure['evidence_sha256']
    assert feedback['source_fingerprint'] == failure['source_fingerprint']
    compiled = [source for name, source, _ in candidate.calls if name == 'compile']
    assert len(compiled) == 2 and compiled[0] != compiled[1]
    assert result.failures and candidate.compiles == 2
    assert len(owner.memory.read()) == 1


def test_submit_supports_are_only_for_current_source_revision(tmp_path):
    from canonical_a3_fixtures import make_trial, recording_candidate, RevisionSession
    candidate = recording_candidate()
    session = RevisionSession(initial=SOURCE, repaired=SOURCE + '\n// repaired')
    owner, _ = make_trial(tmp_path, candidate=candidate, session=session,
                        cell=_cell(profiling=ProfilingGuidance.WITH_GUIDANCE))
    result = owner.run()
    assert result.status == 'passed'
    facts = owner.memory.read()[0]['memory']['host_facts']
    digests = {row['evidence_sha256'] for row in facts}
    assert candidate.compiled_results[0].attestation_sha256 not in digests
    assert candidate.compiled_results[-1].attestation_sha256 in digests
    assert result.verified.evidence_sha256 in digests and result.profile_result.evidence_sha256 in digests
    assert owner.memory.read()[0]['memory']['source_revision'] == result.verified.source_fingerprint


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
    from canonical_a3_fixtures import make_trial
    class VerifyFailure(FakeCandidate):
        runs = 0
        def __init__(self):
            super().__init__()
            self.run_sources = []
        def run(self, source, workdir, **options):
            self.runs += 1
            self.run_sources.append(source)
            if self.runs == 1:
                plan = self._plan(source, options['request_id'], options['attempt_id'], options['length'],
                    options['project_id'], options.get('padded_length'), options.get('block_count', 1),
                    options.get('execution_profile', 'gz-a3'))
                return FailedEvidence.create(plan, stage='verify', error_type='RuntimeError',
                                            detail='host verifier rejected candidate')
            return super().run(source, workdir, **options)
    candidate = VerifyFailure()
    owner, session = make_trial(tmp_path, candidate=candidate)
    result = owner.run()
    assert result.status == 'passed' and candidate.runs == 3
    debug = next(context for context in session.contexts if context['stage'] == 'debug')
    feedback = next(row['candidate_failure'] for row in debug['task']['domain']['observations']
                    if 'candidate_failure' in row)
    failure, = [row.payload['failed_evidence'] for row in owner.evidence.entries
                if 'failed_evidence' in row.payload]
    assert feedback['stage'] == failure['stage'] == 'verify'
    assert feedback['error_type'] == failure['error_type'] == 'RuntimeError'
    assert feedback['attestation_sha256'] == failure['attestation_sha256']
    assert feedback['next_action'] == 'write_source'
    assert candidate.run_sources[0] != candidate.run_sources[1]
    assert candidate.run_sources[1] == candidate.run_sources[2]
    assert result.verified.passed
    assert len(owner.memory.read()) == 1


def test_enabled_kdb_and_profile_are_exactly_cell_bound(tmp_path):
    from canonical_a3_fixtures import make_trial
    cell = _cell(knowledge=KnowledgeMode.WITH_KDB, profiling=ProfilingGuidance.WITH_GUIDANCE)
    owner, session = make_trial(tmp_path, cell=cell)
    result = owner.run()
    assert result.status == 'passed'
    queries = owner.knowledge.journal.read()
    assert queries and all(row['collection'] == 'experience' for row in queries)
    assert all(row['collection_fingerprint'] == owner.knowledge.manifest.fingerprint for row in queries)
    assert all(row['citations'] for row in queries)
    request, = owner.profiler.requests
    assert request.treatment.value == cell.profiling.value
    assert result.profile_result.candidate_execution_id == result.verified.execution_id
    assert result.profile_result.source_fingerprint == result.verified.source_fingerprint
    facts = owner.memory.read()[0]['memory']['host_facts']
    assert any(row['category'] == 'profiling' and row['evidence_sha256'] == result.profile_result.evidence_sha256 for row in facts)


def test_trial_ledger_and_candidate_use_explicit_bz_profile(tmp_path):
    from canonical_a3_fixtures import make_trial
    owner, session = make_trial(tmp_path, execution_profile='bz-a3-2')
    result = owner.run()
    assert result.status == 'passed'
    assert owner.canonical.build(owner).backend.host.compilation.plan.execution_profile == 'bz-a3-2'
    assert {entry.payload['execution_profile'] for entry in owner.evidence.entries} == {'bz-a3-2'}
    assert all(json.loads(context['task']['domain']['instruction'])['execution_profile'] == 'bz-a3-2'
               for context in session.contexts)


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
    from dataclasses import replace
    from canonical_a3_fixtures import make_trial, recording_candidate, RevisionSession

    class InspectFailedRecompile(RevisionSession):
        prior_verified = None
        inspected_failure = False

        def next_action(self, context, allowed, **kwargs):
            host = self.runner.backend.host
            if context['stage'] == 'optimisation' and host.verified is not None:
                self.prior_verified = host.verified
            if context['stage'] == 'debug' and candidate.compiles == 2:
                assert isinstance(host.compilation, FailedEvidence)
                assert host.compilation.stage == 'compile'
                assert self.prior_verified is not None and self.prior_verified.passed
                assert host.verified is None and host.profile is None
                assert host.compilation.source_fingerprint != self.prior_verified.source_fingerprint
                assert not host.accepted_checkpoint_valid()
                assert owner._submit_gate(host.source, host.compilation, self.prior_verified, host.profile) is not None
                assert self.runner.workflow.outcome != 'accepted'
                assert owner.memory.read() == ()
                self.inspected_failure = True
            return super().next_action(context, allowed, **kwargs)

    candidate = recording_candidate(compile_failure_at=(2,))
    session = InspectFailedRecompile(initial=SOURCE, repaired=SOURCE + '\n// repair after failed recompile')
    owner, _ = make_trial(tmp_path, candidate=candidate, session=session,
                        cell=_cell(profiling=ProfilingGuidance.WITH_GUIDANCE))
    build = owner.canonical.build

    def capture_runner(*args, **kwargs):
        session.runner = build(*args, **kwargs)
        return session.runner

    owner.canonical = replace(owner.canonical, build=capture_runner)
    result = owner.run()
    assert session.inspected_failure
    assert result.status == 'passed' and candidate.compiles == 3
    assert any(row.payload.get('failed_evidence', {}).get('stage') == 'compile' for row in owner.evidence.entries)
    assert candidate.compiled_results[-1].plan.source_fingerprint == result.verified.source_fingerprint
    assert candidate.compiled_results[0].plan.source_fingerprint != result.verified.source_fingerprint
    facts = owner.memory.read()[0]['memory']['host_facts']
    assert candidate.compiled_results[0].attestation_sha256 not in {row['evidence_sha256'] for row in facts}
    assert len(owner.memory.read()) == 1


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
    from canonical_a3_fixtures import make_trial, InvalidDecisionsSession
    session = InvalidDecisionsSession(invalid=30, tokens=1000, initial=SOURCE, repaired=SOURCE + '\n// repaired')
    owner, _ = make_trial(tmp_path, session=session, cell=_cell(profiling=guidance),
                        budgets=TrialBudgets(max_turns=1, max_tokens=1, max_seconds=100))
    result = owner.run()
    assert result.status == 'passed' and result.tokens > 30000
    assert session.invalid == 0 and result.completion_turns == 0
    assert len(owner.memory.read()) == 1 and result.verified.passed
    assert bool(result.profile_result) == (guidance is ProfilingGuidance.WITH_GUIDANCE)


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
@pytest.mark.parametrize('known_admission', [True, False])
def test_validation_failure_allows_changed_source_repair_within_time(
    tmp_path, operation, known_admission,
):
    from benchmarks.a3kernels.candidate import validate_candidate_source
    from canonical_a3_fixtures import A3Session, make_trial
    class RejectOnce(FakeCandidate):
        def __init__(self):
            super().__init__()
            self.rejected = False
            self.calls = []
        def invoke(self, name, source, workdir, **options):
            self.calls.append((name, source))
            if name == operation and not self.rejected:
                self.rejected = True
                if known_admission:
                    # The production validator rejects before planning or dispatch.
                    validate_candidate_source(source)
                    raise AssertionError('malformed source was admitted')
                raise ValueError('candidate source rejected by fixture')
            return getattr(super(), name)(source, workdir, **options)
        def compile(self, source, workdir, **options):
            return self.invoke('compile', source, workdir, **options)
        def run(self, source, workdir, **options):
            return self.invoke('run', source, workdir, **options)
    candidate = RejectOnce()
    initial = SOURCE.replace('vector_add', 'invalid_entrypoint')
    session = A3Session(initial=initial, repaired=SOURCE)
    owner, session = make_trial(tmp_path, candidate=candidate, session=session,
                                budgets=TrialBudgets(max_turns=1, max_seconds=20))
    result = owner.run()
    assert candidate.rejected
    if not known_admission:
        assert result.status == 'suspended' and result.verified is None
        root = owner.canonical.build(owner).workflow
        pending = root.state['pending']
        assert pending and not owner.memory.read()
        before = root.path.read_bytes()
        calls = list(candidate.calls)
        assert owner.run().status == 'suspended'
        assert candidate.calls == calls
        assert root.path.read_bytes() == before
        assert root.state['pending'] == pending
        return
    assert result.status == 'passed' and result.verified.passed
    assert any(context['stage'] == 'debug' for context in session.contexts)
    compiled = [source for name, source in candidate.calls if name == 'compile']
    assert len(compiled) == 2 and compiled[0] != compiled[1]
    assert len(owner.memory.read()) == 1


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
    from canonical_a3_fixtures import make_trial, A3Session, recording_candidate
    class BrokenSession(A3Session):
        def next_action(self, context, allowed, **kwargs):
            raise error
    candidate = recording_candidate()
    owner, _ = make_trial(tmp_path, candidate=candidate, session=BrokenSession(initial=SOURCE))
    if isinstance(error, ValueError):
        assert owner.run().status == 'suspended'
        runner = owner.canonical.build(owner)
        row = runner.store.read_json(runner.latest('project-run-outcome')['reference'])
        assert row['result']['error'] == str(error)
    else:
        with pytest.raises(type(error), match='actor bug'):
            owner.run()
    assert not candidate.calls and not owner.memory.read()


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
    from canonical_a3_fixtures import make_trial, InvalidDecisionsSession, recording_candidate
    session = InvalidDecisionsSession(invalid=13, initial=SOURCE, repaired=SOURCE + '\n// repair')
    candidate = recording_candidate(run_failures=4)
    owner, _ = make_trial(tmp_path, candidate=candidate, session=session, cell=_cell(model=BackendModel.DEEPSEEK_FLASH))
    result = owner.run()
    assert result.status == 'passed' and result.verified.passed
    assert session.invalid == 0 and candidate.compiles == 5 and candidate.runs == 6
    assert len(owner.memory.read()) == 1
    assert len([row for row in owner.evidence.entries if 'failed_verification' in row.payload]) == 4
    assert owner.budgets == TrialBudgets.for_model(BackendModel.DEEPSEEK_FLASH)


def test_openai_default_recovers_after_twelve_turns_without_relaxing_gates(tmp_path):
    from canonical_a3_fixtures import make_trial, InvalidDecisionsSession, recording_candidate
    session = InvalidDecisionsSession(invalid=13, initial=SOURCE, repaired=SOURCE + '\n// repair')
    candidate = recording_candidate(run_failures=4)
    owner, _ = make_trial(tmp_path, candidate=candidate, session=session, cell=_cell(model=BackendModel.GPT_5_6_SOL))
    result = owner.run()
    assert result.status == 'passed' and result.verified.passed
    assert session.invalid == 0 and candidate.compiles == 5 and candidate.runs == 6
    assert len(owner.memory.read()) == 1
    assert len([row for row in owner.evidence.entries if 'failed_verification' in row.payload]) == 4
    assert owner.budgets == TrialBudgets.for_model(BackendModel.GPT_5_6_SOL)


def test_model_and_treatment_authorities_must_match_cell(tmp_path):
    cell = _cell(knowledge=KnowledgeMode.WITH_KDB)
    with pytest.raises(ValueError, match="Knowledge Agent"):
        _run(tmp_path, [], cell=cell, knowledge=FakeKnowledge(False))


def test_completion_provenance_cannot_switch_the_cell_model(tmp_path):
    from canonical_a3_fixtures import make_trial, recording_candidate
    cell = _cell()
    other = next(model for model in BackendModel if model is not cell.backend_model)
    candidate = recording_candidate()
    owner, session = make_trial(tmp_path, cell=cell, candidate=candidate)
    with pytest.raises(ValueError, match='model profile'):
        A3TrialLoop(cell=cell, proposal=owner.proposal, profile=load_a3_model_profile(other),
            actor=owner.actor, candidate=candidate, knowledge=owner.knowledge, profiler=owner.profiler,
            evidence=owner.evidence, memory=owner.memory, workdir=owner.workdir, canonical=owner.canonical)
    assert not session.contexts and not candidate.calls and not owner.memory.read()
    assert owner.run().status == 'passed'
    assert all(json.loads(context['task']['domain']['instruction'])['profile'] == owner.profile.as_dict()
               for context in session.contexts)


def test_structured_completion_bypasses_text_parser_and_keeps_host_validation(tmp_path, monkeypatch):
    from canonical_a3_fixtures import make_trial
    monkeypatch.setattr('benchmarks.a3kernels.trial.parse_action',
                        lambda _: pytest.fail('canonical decisions must not enter the old action parser'))
    owner, session = make_trial(tmp_path)
    result = owner.run()
    assert result.status == 'passed' and result.verified.passed
    assert {context['stage'] for context in session.contexts} >= {'test', 'validation', 'acceptance'}
    assert len(owner.memory.read()) == 1
