"""Descriptions explain real workflow distinctions without altering evidence."""
import copy
from dataclasses import asdict

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import canonical_bytes
from benchmarks.a5kernels.phase1_memory import HostFact
from benchmarks.ascend.agent_presentation import compact_observation
from benchmarks.ascend.status_descriptions import describe_statuses, explain_status


def test_repair_requirements_are_actionable_and_codes_unchanged(tmp_path):
    raw = {'error': 'submit requires retained project evidence',
           'missing': ['observed-host-mismatch', 'observed-compile-failure',
                       'performance-study', 'profiling']}
    store = ContentAddressedArtifacts(tmp_path)
    view = compact_observation(raw, store)
    assert view['missing'] == raw['missing']
    assert {(row['field'], row['code']) for row in view['status_explanations']} == {
        ('missing', code) for code in raw['missing']}
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


@pytest.mark.parametrize('code', [
    'disabled-by-treatment', 'diagnostic-backend-not-bound', 'correctness-failed',
])
def test_registered_reason_codes_are_explained_without_interpreting_prose(code):
    raw = {'reason': code}
    view = describe_statuses(raw)
    row, = view['status_explanations']
    assert row['field'] == 'reason'
    assert row['code'] == view['reason'] == raw['reason']
    assert row['description'] and row['next_action']
    prose = {'reason': 'The diagnostic was unavailable after an interrupted request.'}
    assert describe_statuses(prose) == prose


@pytest.mark.parametrize('code', ['profiling', 'performance-study'])
def test_measurement_fact_and_missing_requirement_have_distinct_explanations(code, tmp_path):
    fact = asdict(HostFact(code, 'Completed host measurement', 'a' * 64))
    raw = {'facts': [fact], 'missing': [code]}
    store = ContentAddressedArtifacts(tmp_path)
    view = compact_observation(raw, store)
    retained, = view['facts'][0]['status_explanations']
    missing, = view['status_explanations']
    assert retained['field'] == 'category'
    assert missing['field'] == 'missing'
    assert view['facts'][0]['category'] == retained['code'] == code
    assert view['missing'] == [missing['code']] == [code]
    assert retained['description'] != missing['description']
    assert retained['next_action'] != missing['next_action']
    assert retained['description'] != explain_status(code)['description']
    assert missing['description'] != explain_status(code)['description']
    assert {key: value for key, value in view['facts'][0].items()
            if key != 'status_explanations'} == fact
    assert store.read_json(store.put_json(raw)) == raw
    assert describe_statuses(view) == view
