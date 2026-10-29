"""Aggregate paging preserves exact evidence and visible host admission state."""
import hashlib
import json

import pytest

from benchmarks.ascend.review_context import ReviewContextRecords
from coding.pi.context import ContextRecords, ContextLimitError, encoded


def fixture():
    digests = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(12)]
    details = {f'case_{i}': {'expectation': 'Observed é correctness for case ' + str(i),
               'before': 'short source line ' * 12, 'limitations': 'shape-specific evidence ' * 8}
               for i in range(35)}
    controls = dict(host_checks_passed=True, knowledge_queried=True,
        allowed_evidence=digests[:8], read_artifacts=digests[:8],
        missing_required_artifacts=[{'sha256': h, 'roles': [f'host-result:{i}']}
                                    for i, h in enumerate(digests[8:])], citations=[])
    return dict(host=dict(project=dict(project=dict(project_id='retained-copy',
        source_sha256=digests[0], host_verified=True, artifacts={'source': digests[0]}),
        rewrite_feedback={'prior_draft': details, 'task': 'Audit and retain only supported lessons',
            'known_counterevidence': {'input': '2^-40', 'epsilon': '2^-32', 'formula_result': '1/256'}}, tools={'knowledge_query': True}),
        review_state=controls, artifact_pages=[], validation_feedback=None,
        tool_observations={'knowledge_query': {'hits': [], 'citations': []}}),
        remaining_seconds=60, source_sha256=None)


def reconstruct(records, reference):
    chunks, offset = [], 0
    while offset is not None:
        page = records.read(reference['record'], offset, 509)
        chunks.append(page['text'])
        offset = page['next_offset']
    return ''.join(chunks)


def test_many_small_fields_page_losslessly_and_keep_pending_reads_visible():
    context = fixture()
    original = encoded(context)
    with pytest.raises(ContextLimitError):
        ContextRecords().snapshot(context, target=16384)
    records = ReviewContextRecords()
    packed = records.snapshot(context, target=16384)
    assert len(encoded(packed)) <= 16384
    ref = packed['host']['project']['rewrite_feedback']['prior_draft']
    assert json.loads(reconstruct(records, ref)) == context['host']['project']['rewrite_feedback']['prior_draft']
    assert packed['host']['project']['rewrite_feedback']['task'] == context['host']['project']['rewrite_feedback']['task']
    assert packed['host']['project']['rewrite_feedback']['known_counterevidence'] == context['host']['project']['rewrite_feedback']['known_counterevidence']
    assert packed['host']['review_state'] == context['host']['review_state']
    assert packed['host']['project']['project'] == context['host']['project']['project']
    assert encoded(context) == original


def test_tool_observation_aggregate_is_losslessly_paged():
    context = fixture()
    details = context['host']['project']['rewrite_feedback'].pop('prior_draft')
    context['host']['tool_observations'] = details
    records = ReviewContextRecords()
    packed = records.snapshot(context, target=16384)
    assert len(encoded(packed)) <= 16384
    assert json.loads(reconstruct(records, packed['host']['tool_observations'])) == details
    assert packed['host']['review_state'] == context['host']['review_state']


def test_validation_feedback_keeps_error_and_pending_reads_while_paging_citation_details():
    context = fixture()
    context['host']['project']['rewrite_feedback'].pop('prior_draft')
    controls = context['host']['review_state']
    citations = [dict(collection='baseline', path=('long_source_directory/' * 10) + f'{i}.cpp',
        start_line=1, end_line=4, source_sha256=controls['allowed_evidence'][i]) for i in range(8)]
    context['host']['validation_feedback'] = dict(error_code='unread-evidence',
        allowed_citations=citations, allowed_evidence=controls['allowed_evidence'],
        missing_required_artifacts=controls['missing_required_artifacts'])
    with pytest.raises(ContextLimitError):
        ContextRecords().snapshot(context, target=6144)
    records = ReviewContextRecords()
    packed = records.snapshot(context, target=6144)
    assert len(encoded(packed)) <= 6144
    feedback = packed['host']['validation_feedback']
    assert feedback['error_code'] == 'unread-evidence'
    assert feedback['missing_required_artifacts'] == controls['missing_required_artifacts']
    assert json.loads(reconstruct(records, feedback['allowed_citations'])) == citations
    assert packed['host']['review_state'] == controls


def test_small_review_and_actor_default_keep_existing_direct_views():
    context = fixture()
    context['host']['project']['rewrite_feedback'] = {'scope': 'Only the retained shapes'}
    assert ReviewContextRecords().snapshot(context, target=16384) == ContextRecords().snapshot(context, target=16384)
    with pytest.raises(ContextLimitError):
        ContextRecords().snapshot(fixture(), target=16384)


def test_large_readable_registry_keeps_original_read_controls_within_native_budget():
    from dataclasses import replace
    from benchmarks.ascend.project_verifier import ReviewState
    from test_project_verifier_agent import Harness, retained_request, decision, memory, B
    req = retained_request()
    extra = {f'retained:{i:04d}':hashlib.sha256(f'large-{i}'.encode()).hexdigest() for i in range(300)}
    req = replace(req,artifacts={**req.artifacts,**extra})
    h = Harness([])
    state = ReviewState(h.agent,req,lambda:60,None)
    state.native = True
    feedback = state.execute(decision(memory()))
    controls = state.summary()
    assert controls['readable_artifact_count']==304
    assert len(controls['readable_artifacts'])<=16
    assert len(feedback['readable_artifacts'])<=16
    assert B in {row['sha256'] for row in controls['missing_required_artifacts']}
    context = fixture()
    context['host'].update(project=state.context,review_state=controls,validation_feedback=feedback)
    packed = ReviewContextRecords().snapshot(context,target=16384)
    assert len(encoded(packed))<=16384
    assert packed['host']['review_state']['missing_required_artifacts']==controls['missing_required_artifacts']
    recovered,offset={},0
    while offset is not None:
        page = state.execute({'list_artifacts':{'offset':offset,'limit':50}})
        recovered.update(page['artifacts']);offset=page['next_offset']
    assert recovered==state.registry
    assert set(extra.values())<=set(recovered.values())
