"""Observed examples reach repair actors without manufacturing another failure."""
from dataclasses import replace
import json

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import EvidenceLedger, canonical_bytes
from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActorProjectExecutor
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
from benchmarks.ascend.failure_curriculum import FailureCurriculum, select_feedback
from benchmarks.ascend.failure_handoff import FailureContext
from benchmarks.ascend.feedback import FeedbackStore
from test_a5_phase1_adaptive_actor import SourceBackend, ScriptedActor, _request, _special_spec
from test_generic_failure_feedback import phase1


def observed_failure(tmp_path, mode='host-mismatch'):
    spec = _special_spec(failure=mode)
    campaign = tmp_path / 'failed'
    phase1(campaign, spec=spec.as_dict())
    branch = campaign / 'waves/wave-1/branches' / spec.project_id
    executor = ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path / 'ledger'))
    broken = 'compile failure' if mode == 'compile' else 'broken source'
    result = executor.execute(spec, broken)
    classification = result.evidence.classification.value
    assert classification == ('compile-failure' if mode == 'compile' else 'host-mismatch')
    artifacts = ContentAddressedArtifacts(branch / 'artifacts')
    source = artifacts.put_bytes(broken.encode())
    refs = {name: artifacts.put_json(value) for name, value in result.artifacts.items()}
    (branch / 'execution-evidence.bin').write_bytes(canonical_bytes(dict(
        classification=classification, detail=result.evidence.detail,
        source_sha256=source, artifacts=refs)))
    if mode == 'compile':
        from benchmarks.ascend.failure_cases import retain_phase1_failure
        payloads = dict(result.artifacts, diagnostics=result.artifacts['observed'])
        retain_phase1_failure(artifacts, spec, source, classification, result.evidence.detail,
                              payloads, result.prepared.expected.dtype)
    store = FeedbackStore(tmp_path / 'feedback')
    report = store.import_phase1(campaign)
    assert report['imported'] == 1
    assert all(message.endswith('no retained job identity; relocation deduplication unavailable')
               for message in report['warnings'])
    key = store.records()[0]['feedback_id']
    return executor, spec, store, dict(root=str(store.root), feedback_ids=[key])


def test_supplied_case_reads_exact_values_and_authorizes_verified_repair(tmp_path):
    executor, spec, store, context = observed_failure(tmp_path)
    # A new exercise name does not change computation or frozen inputs.
    spec = replace(spec, project_id='repair-with-attached-example')
    failure = FailureContext(context, spec)
    request = replace(_request(tmp_path / 'repair', executor, spec), failure_context=context,
                      inputs=failure.inputs, oracle=failure.expected)
    case = store.failure_cases(context['feedback_ids'][0])[0]
    inputs = case['artifacts']['inputs']['sha256']
    actor = ScriptedActor([
        json.dumps({'read-artifact': {'sha256': inputs, 'offset': 0, 'limit': 8192}}),
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}', '{"submit":{}}',
    ])
    result = AdaptiveActorProjectExecutor(executor, actor, object())(request)
    assert result.evidence.verified
    assert json.loads(actor.observations[0]['text'])['tensors']
    # Only original failed execution and repaired execution; no forced rediscovery.
    assert [row.candidate_source for row in executor._backend.run_requests] == ['broken source', 'fixed source']
    assert executor._backend.run_requests[-1].inputs == request.inputs
    assert executor._backend.run_requests[-1].expected == request.oracle
    assert executor.prepare(spec).inputs != request.inputs  # Shared executor remains unmodified.


def test_case_cannot_replace_verification_of_current_repair(tmp_path):
    executor, spec, _, context = observed_failure(tmp_path)
    request = replace(_request(tmp_path / 'unverified', executor, spec), failure_context=context)
    actor = ScriptedActor(['{"write":{"slot":"kernel","content":"fixed source"}}', '{"submit":{}}'])
    with pytest.raises(RuntimeError, match='without host-verified'):
        AdaptiveActorProjectExecutor(executor, actor, object())(request)
    assert 'current source' in actor.observations[-1]['error']


def test_different_probe_cannot_inherit_negative_credit(tmp_path):
    executor, spec, _, context = observed_failure(tmp_path)
    changed = spec.as_dict()
    changed['graph']['nodes'][0]['op'] = 'copy'
    changed['capabilities'].remove('simd-vector')
    # Choose another supported computation while retaining shape/layout.
    changed = ProjectSpec.from_mapping(changed)
    request = replace(_request(tmp_path / 'other', executor, changed), failure_context=context)
    with pytest.raises(ValueError, match='actor must not invent'):
        AdaptiveActorProjectExecutor(executor, ScriptedActor([]), object())(request)


def test_curriculum_can_read_cases_and_rejects_missing_repair_evidence(tmp_path):
    executor, spec, store, context = observed_failure(tmp_path)
    curriculum = FailureCurriculum(tmp_path, store.root, select_feedback(store.root, 'a3'))
    ref = curriculum.context({})[0]['failure_cases_file']
    descriptor = json.loads(curriculum.read(**ref['read_artifact'])['text'])[0]
    digest = descriptor['artifacts']['expected']['sha256']
    expected = json.loads(curriculum.read(digest)['text'])
    assert expected['storage'] == list(executor.prepare(spec).expected.storage)
    curriculum.lineage = [dict(feedback_id=context['feedback_ids'][0], project_id=spec.project_id)]
    assert curriculum.project_context(spec.project_id) == context
    assert curriculum.project_context('unrelated-new-project') is None
    # Lost retained data has an explicit explanation, and cannot grant credit.
    from pathlib import Path
    Path(descriptor['artifacts']['inputs']['retained_path']).unlink()
    missing = FailureContext(context, spec)
    assert not missing.cases[0]['availability']['complete']
    with pytest.raises(ValueError, match='No compatible complete reproducer'):
        missing.require_repair_evidence(spec)


def test_selection_rejects_incompatible_repair_but_accepts_exploratory_probe(tmp_path):
    from types import SimpleNamespace
    executor, spec, store, context = observed_failure(tmp_path)
    curriculum = FailureCurriculum(tmp_path, store.root, select_feedback(store.root, 'a3'))
    changed = spec.as_dict()
    changed['graph']['nodes'][0]['op'] = 'copy'
    changed['capabilities'].remove('simd-vector')
    changed = ProjectSpec.from_mapping(changed)
    control = replace(_special_spec(), project_id='control')
    wave = SimpleNamespace(projects=[SimpleNamespace(spec=changed), SimpleNamespace(spec=control)])
    row = dict(feedback_id=context['feedback_ids'][0], status='investigating', classification='learnable',
               reason='isolate operation', project_id=spec.project_id, control_project_id='control',
               hypothesis='operation differs', observation='compare against host oracle')
    prior = SimpleNamespace(prior_outcomes=[])
    with pytest.raises(ValueError, match='same computation'):
        curriculum.admit([row], wave, prior)
    with pytest.raises(ValueError, match='No failure cases were linked'):
        curriculum.admit([], wave, prior)
    value = changed.as_dict()
    value['failure_mode'] = 'none'
    value['capabilities'].remove('failure-repair')
    probe = ProjectSpec.from_mapping(value)
    wave.projects[0].spec = probe
    assert curriculum.admit([row], wave, prior) == (row,)
    # Context remains attached even when the negative example cannot grant credit.
    curriculum.lineage = [row]
    attached = FailureContext(curriculum.project_context(probe.project_id), probe)
    assert attached.cases and not attached.compatible


def test_supplied_compile_failure_authorizes_verified_repair(tmp_path):
    executor, spec, _, context = observed_failure(tmp_path, mode='compile')
    spec = replace(spec, project_id='renamed-compile-repair')
    failure = FailureContext(context, spec)
    request = replace(_request(tmp_path / 'repair', executor, spec), failure_context=context,
                      inputs=failure.inputs, oracle=failure.expected)
    actor = ScriptedActor(['{"write":{"slot":"kernel","content":"fixed source"}}',
                           '{"run":{}}', '{"submit":{}}'])
    assert AdaptiveActorProjectExecutor(executor, actor, object())(request).evidence.verified
    assert [x.candidate_source for x in executor._backend.requests] == ['compile failure', 'fixed source']
    assert len(executor._backend.run_requests) == 1


def test_coordinator_passes_frozen_case_to_project_adapter(tmp_path):
    from types import SimpleNamespace
    from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
    executor, spec, store, context = observed_failure(tmp_path)
    renamed = replace(spec, project_id='new-repair-id')
    feedback = FailureCurriculum(tmp_path, store.root, select_feedback(store.root, 'a3'))
    feedback.lineage = [dict(feedback_id=context['feedback_ids'][0], project_id=renamed.project_id)]
    coordinator = object.__new__(AdaptivePhase1Coordinator)
    coordinator.root = tmp_path
    coordinator.failure_feedback = feedback
    coordinator.automatic_remediation = None
    coordinator._remaining_seconds = lambda: 60
    received = []
    def capture(request):
        received.append(request)
        raise RuntimeError('captured adapter boundary')
    coordinator.execute_project = capture
    generated = executor.prepare(renamed)
    assignment = SimpleNamespace(project_id=renamed.project_id, spec=renamed, authored_ordinal=1)
    request = SimpleNamespace(assignment=assignment, pre_wave_memory=b'{}', branch_root=tmp_path / 'branch',
                              inputs=generated.inputs, oracle=generated.expected)
    with pytest.raises(RuntimeError, match='captured adapter boundary'):
        coordinator._execute_bound(request)
    assert received[0].inputs == executor.prepare(spec).inputs != generated.inputs
    assert received[0].oracle == executor.prepare(spec).expected
    assert received[0].failure_context == context


def test_failed_bound_repair_preserves_exact_case_for_next_generation(tmp_path):
    executor, original, store, context = observed_failure(tmp_path)
    first = replace(original, project_id='first-failed-repair')
    failure = FailureContext(context, first)
    request = replace(_request(tmp_path / 'first', executor, first), failure_context=context,
                      inputs=failure.inputs, oracle=failure.expected)
    actor = ScriptedActor(['{"write":{"slot":"kernel","content":"another broken source"}}', '{"run":{}}'])
    failed = AdaptiveActorProjectExecutor(executor, actor, object(), retain_semantic_failure=True)(request)
    assert not failed.evidence.verified
    campaign = tmp_path / 'second-campaign'
    phase1(campaign, spec=first.as_dict())
    branch = campaign / 'waves/wave-1/branches' / first.project_id
    from shutil import copytree
    copytree(request.branch_root / 'artifacts', branch / 'artifacts')
    (branch / 'execution-evidence.bin').write_bytes(failed.evidence_payload)
    assert store.import_phase1(campaign)['imported'] == 1
    key = next(row['feedback_id'] for row in store.records() if row['outcome']['attempt_id'] == first.project_id)
    second = replace(original, project_id='second-successful-repair')
    next_context = dict(root=str(store.root), feedback_ids=[key])
    next_case = FailureContext(next_context, second)
    next_case.require_repair_evidence(second)
    assert next_case.inputs == request.inputs != executor.prepare(first).inputs
    final_request = replace(_request(tmp_path / 'second', executor, second), failure_context=next_context,
                            inputs=next_case.inputs, oracle=next_case.expected)
    fixed_actor = ScriptedActor(['{"write":{"slot":"kernel","content":"fixed source"}}', '{"run":{}}', '{"submit":{}}'])
    assert AdaptiveActorProjectExecutor(executor, fixed_actor, object())(final_request).evidence.verified
    assert executor._backend.run_requests[-1].inputs == request.inputs


@pytest.mark.parametrize('wrong_execution_case', [False, True])
def test_foreign_renamed_failure_retry_keeps_authenticated_inputs_and_original_oracle(tmp_path, wrong_execution_case):
    from benchmarks.a5kernels.phase1_curriculum import FailureMode
    from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
    from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
    from benchmarks.a5kernels.phase1_project import evaluate_reference
    from benchmarks.ascend.automatic_remediation import AutomaticRemediation
    from benchmarks.ascend.failure_handoff import phase1_contract
    from optimizer.jobs import sha
    executor, spec, _, _ = observed_failure(tmp_path/'producer')
    inputs = executor.prepare(spec).inputs
    renamed = replace(spec, project_id='foreign-renamed-repair', failure_mode=FailureMode.NONE,
                      capabilities=tuple(item for item in spec.capabilities if item.value != 'failure-repair'))
    assert inputs != executor.prepare(renamed).inputs
    foreign = tmp_path/'foreign'
    phase1(foreign, spec=renamed.as_dict())
    branch = foreign/'waves/wave-1/branches'/renamed.project_id
    artifacts = ContentAddressedArtifacts(branch/'artifacts')
    result = executor.execute(spec, 'broken source')
    refs = {name: artifacts.put_json(value) for name, value in result.artifacts.items()}
    source = artifacts.put_bytes(b'broken source')
    payload = canonical_bytes(dict(classification='host-mismatch', source_sha256=source, artifacts=refs))
    (branch/'execution-evidence.bin').write_bytes(payload)
    outcome = json.loads((branch/'outcome.json').read_bytes())
    outcome['evidence_sha256'] = sha(payload)
    (branch/'outcome.json').write_bytes(canonical_bytes(outcome))
    store = FeedbackStore(tmp_path/'feedback')
    store.import_phase1(foreign, target='a3', language='ascend-c')
    original, = store.records()
    original_bytes = (store.root/(original['feedback_id']+'.json')).read_bytes()
    received = []
    def execute(request):
        received.append(request)
        assert request.inputs == inputs
        assert request.oracle == evaluate_reference(renamed, inputs)
        bound = request
        if wrong_execution_case:
            wrong = executor.prepare(renamed)
            bound = replace(request, failure_context=None, inputs=wrong.inputs, oracle=wrong.expected)
        actor = ScriptedActor(['{"write":{"slot":"kernel","content":"fixed source"}}',
                               '{"run":{}}', '{"submit":{}}'], memory=bound.pre_wave_memory)
        return AdaptiveActorProjectExecutor(executor, actor, object())(bound)
    runner = AdaptivePhase1Coordinator(tmp_path/'consumer', 'Retry original case.', b'{}',
        CurriculumAgent(lambda _: pytest.fail('no new curriculum project')), execute, max_parallel=1, project_target=1)
    automatic = AutomaticRemediation(runner, target='a3', store=store.root)
    state = runner._load_state()
    automatic.refresh(state)
    directory = runner.root/'original-retries'/original['feedback_id']/'retry-1'
    automatic._phase1_retry(original, directory, state)
    assert len(received) == 1
    plan = json.loads((directory/'waves/original-retry/plan.json').read_bytes())
    expected_contract = phase1_contract(renamed, inputs)
    assert sha(canonical_bytes(plan['case_inputs'][renamed.project_id])) == expected_contract['inputs_sha256']
    retry = next(row for row in store.records() if row['feedback_id'] != original['feedback_id'])
    assert retry['outcome']['verified']
    assert (store.root/(original['feedback_id']+'.json')).read_bytes() == original_bytes
    linked = store.link_original_retry(original['feedback_id'], retry['feedback_id'])
    assert linked['lifecycle'] == ('failed' if wrong_execution_case else 'fixed')
    retained = json.loads((directory/'waves/original-retry/branches'/renamed.project_id/'execution-evidence.bin').read_bytes())
    if not wrong_execution_case:
        assert retained['artifacts']['inputs'] == expected_contract['inputs_sha256']
        assert retained['artifacts']['expected'] == expected_contract['expected_sha256']
        assert executor._backend.run_requests[-1].inputs == inputs
