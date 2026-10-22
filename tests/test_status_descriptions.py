"""Descriptions explain real workflow distinctions without altering evidence."""
import copy

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import canonical_bytes
from benchmarks.ascend.agent_presentation import compact_observation
from benchmarks.ascend.status_descriptions import describe_statuses, explain_status


def test_repair_requirements_are_actionable_and_codes_unchanged(tmp_path):
    raw = {'error': 'submit requires retained project evidence',
           'missing': ['observed-host-mismatch', 'observed-compile-failure',
                       'performance-study', 'profiling']}
    store = ContentAddressedArtifacts(tmp_path)
    view = compact_observation(raw, store)
    assert view['missing'] == raw['missing']
    rows = {row['code']: row for row in view['status_explanations']}
    assert 'host reference' in rows['observed-host-mismatch']['description']
    assert 'do not invent' in rows['observed-host-mismatch']['next_action']
    assert 'compiler diagnostics' in rows['observed-compile-failure']['next_action']
    assert 'timing study' in rows['performance-study']['next_action']
    assert 'metric evidence' in rows['profiling']['next_action']
    assert store.read_json(store.put_json(raw)) == raw


def test_presentation_is_idempotent_and_does_not_rewrite_evidence():
    raw = {'worker': {'status': 'complete'}, 'classification': 'host-verified',
           'artifact': {'payload': {'status': 'failed'}}, 'reason': 'investigating',
           'outcome': {'lifecycle': 'remediated'}}
    original = copy.deepcopy(raw)
    view = describe_statuses(raw)
    assert raw == original
    assert view['artifact'] == raw['artifact']
    assert view['reason'] == 'investigating'
    assert 'status_explanations' not in view['artifact']['payload']
    assert describe_statuses(view) == view
    assert 'Other project requirements' in view['status_explanations'][0]['description']
    assert 'original' in view['outcome']['status_explanations'][0]['next_action']


@pytest.mark.parametrize('code, phrase', [
    ('both-succeeded', 'does not establish'), ('demonstrated-benefit', 'control failed'),
    ('treatment-regressed', 'control passed'), ('neither-succeeded', 'Neither'),
    ('inconclusive', 'does not establish'), ('TIME_BUDGET_EXHAUSTED', 'elapsed time'),
    ('RECOVERY_REQUIRED', 'recovered'), ('waiting-approval', 'approval'),
    ('context-budget-exhausted', 'context limit'), ('could-not-confirm', 'enough evidence'),
    ('infrastructure-failed', 'infrastructure'), ('proposed', 'not yet'),
    ('checking-correctness', 'frozen reference'), ('checking-reference', 'reference'),
    ('succeeded', 'correctness is a separate'), ('practice', 'practice project'),
    ('target', 'original target'), ('review', 'reviewing'), ('qualification', 'qualification'),
])
def test_ordinary_codes_have_specific_explanations(code, phrase):
    explanation = explain_status(code)
    assert phrase in explanation['description']
    assert explanation['next_action']
    assert 'no specific explanation' not in explanation['description']


def test_ambiguous_statuses_are_not_promoted_to_correctness():
    assert 'eligible' in explain_status('ready')['description']
    assert 'verified target' in explain_status('READY')['description']
    assert 'has not yet' in explain_status('failed', field='lifecycle')['description']
    assert 'unsuccessfully' in explain_status('failed')['description']
    assert 'no specific explanation' in explain_status('new-unknown-state')['description']


def test_large_failure_log_keeps_explanation_and_exact_raw_hash(tmp_path):
    raw = {'classification': 'host-mismatch', 'diagnostic': 'x' * 200000,
           'missing': ['observed-host-mismatch']}
    store = ContentAddressedArtifacts(tmp_path)
    view = compact_observation(raw, store)
    assert len(canonical_bytes(view)) <= 12 * 1024
    assert store.read_json(view['full_observation_sha256']) == raw
    assert view['missing'] == raw['missing']
    assert {row['field'] for row in view['status_explanations']} == {'classification', 'missing'}


def test_learner_stage_is_explained_without_promoting_practice_to_target_success():
    view = describe_statuses({'learner_stage': 'practice'})
    row, = view['status_explanations']
    assert row['field'] == 'learner_stage'
    assert row['code'] == view['learner_stage'] == 'practice'
    assert 'practice success alone' in row['next_action']
