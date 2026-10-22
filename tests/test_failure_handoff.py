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
from benchmarks.ascend.phase2_feedback import FeedbackStore
from test_a5_phase1_adaptive_actor import SourceBackend, ScriptedActor, _request, _special_spec
from test_generic_failure_feedback import phase1


def observed_failure(tmp_path):
    spec = _special_spec(failure='host-mismatch')
    campaign = tmp_path / 'failed'
    phase1(campaign, spec=spec.as_dict())
    branch = campaign / 'waves/wave-1/branches' / spec.project_id
    executor = ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path / 'ledger'))
    result = executor.execute(spec, 'broken source')
    assert result.evidence.classification.value == 'host-mismatch'
    artifacts = ContentAddressedArtifacts(branch / 'artifacts')
    source = artifacts.put_bytes(b'broken source')
    refs = {name: artifacts.put_json(value) for name, value in result.artifacts.items()}
    (branch / 'execution-evidence.bin').write_bytes(canonical_bytes(dict(
        classification='host-mismatch', detail=result.evidence.detail,
        source_sha256=source, artifacts=refs)))
    store = FeedbackStore(tmp_path / 'feedback')
    assert store.import_phase1(campaign)['warnings'] == []
    key = store.records()[0]['feedback_id']
    return executor, spec, store, dict(root=str(store.root), feedback_ids=[key])


def test_supplied_case_reads_exact_values_and_authorizes_verified_repair(tmp_path):
    executor, spec, store, context = observed_failure(tmp_path)
    # A new exercise name does not change computation or frozen inputs.
    spec = replace(spec, project_id='repair-with-attached-example')
    request = replace(_request(tmp_path / 'repair', executor, spec), failure_context=context)
    case = store.failure_cases(context['feedback_ids'][0])[0]
    inputs = case['artifacts']['inputs']['sha256']
    actor = ScriptedActor([
        json.dumps({'read-artifact': {'sha256': inputs, 'offset': 0, 'limit': 8192}}),
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}', '{"submit":{}}',
    ])
    result = AdaptiveActorProjectExecutor(executor, actor, object())(request)
    assert result.evidence.verified
    assert 'negative-example' in actor.instruction
    assert 'Linked observed failures' in actor.instruction
    assert json.loads(actor.observations[0]['text'])['tensors']
    # Only original failed execution and repaired execution; no forced rediscovery.
    assert [row.candidate_source for row in executor._backend.run_requests] == ['broken source', 'fixed source']


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
    # Choose another supported computation while retaining shape/layout.
    changed = ProjectSpec.from_mapping(changed)
    request = replace(_request(tmp_path / 'other', executor, changed), failure_context=context)
    with pytest.raises(ValueError, match='actor must not invent'):
        AdaptiveActorProjectExecutor(executor, ScriptedActor([]), object())(request)


def test_curriculum_can_read_cases_and_rejects_missing_repair_evidence(tmp_path):
    executor, spec, store, context = observed_failure(tmp_path)
    curriculum = FailureCurriculum(tmp_path, store.root, select_feedback(store.root, 'a3'))
    descriptor = curriculum.context({})[0]['failure_cases'][0]
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
