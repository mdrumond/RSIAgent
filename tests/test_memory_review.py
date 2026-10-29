"""Memory decisions are grounded in delivered versions and independent evidence."""
from dataclasses import replace
import json

import pytest

from benchmarks.ascend.memory_review import validate_memory_review
from benchmarks.ascend.project_verifier import ProjectVerifierAgent
from test_project_verifier_agent import Harness, S, R, L, sha, request, read, decision, prerequisites, memory


DOCUMENT = b'{"summary":"Every vector load is aligned"}'
D = sha(DOCUMENT)
EXPOSURE = dict(exposure_id='exposure', memory_id='lesson', version=D, document_sha256=D)


def judgment(**changes):
    return dict(dict(exposure_id='exposure', usefulness='not-useful', correctness='contradicted',
        investigate=False, reason='Project result disproves the alignment claim', evidence=[R]), **changes)


def test_contradiction_requires_read_exact_memory_and_independent_evidence():
    bad = decision()
    bad['memory_judgments'] = [judgment(evidence=[D])]
    good = decision()
    good['memory_judgments'] = [judgment()]
    h = Harness([*prerequisites(), good, read(D), bad, good])
    h.artifacts[D] = DOCUMENT
    result = h.agent.review(replace(request(), memory_exposures=(EXPOSURE,)))
    assert result['memory_judgments'] == [judgment()]
    feedback = [event['payload']['error'] for event in h.agent.last_trace if event['kind'] == 'feedback']
    assert len(feedback) == 2
    assert any('exact accessed memory' in error for error in feedback)
    assert any('independent' in error for error in feedback)


def test_every_delivered_exposure_requires_a_distinct_judgment():
    value = decision()
    req = replace(request(), memory_exposures=(EXPOSURE,))
    with pytest.raises(ValueError, match='every accessed'):
        validate_memory_review(value, req, {S,R,L,D}, ProjectVerifierAgent._evidence, {S,R,L,D})
    value['memory_judgments'] = [judgment(), judgment()]
    with pytest.raises(ValueError, match='every accessed'):
        validate_memory_review(value, req, {S,R,L,D}, ProjectVerifierAgent._evidence, {S,R,L,D})


def test_unknown_is_valid_without_evidence_but_is_not_an_automatic_investigation():
    value = decision()
    value['memory_judgments'] = [judgment(usefulness='unknown', correctness='unknown', evidence=[])]
    req = replace(request(), memory_exposures=(EXPOSURE,))
    validate_memory_review(value, req, {S,R,L}, ProjectVerifierAgent._evidence, {S,R,L,D})
    value['memory_judgments'][0]['investigate'] = True
    with pytest.raises(ValueError, match='explicit uncertain concern'):
        validate_memory_review(value, req, {S,R,L}, ProjectVerifierAgent._evidence, {S,R,L,D})


def test_passing_project_cannot_silently_resolve_memory_repair():
    req = replace(request(), memory_repairs=(dict(repair_id='repair', memory_id='lesson', version=D,
        document_sha256=D),))
    value = decision()
    with pytest.raises(ValueError, match='explicitly resolve'):
        validate_memory_review(value, req, {S,R,L,D}, ProjectVerifierAgent._evidence, {S,R,L,D})
    value['memory_resolutions'] = [dict(repair_id='repair', action='replace', reason='Correct aligned-only precondition', evidence=[R])]
    with pytest.raises(ValueError, match='explicitly replace'):
        validate_memory_review(value, req, {S,R,L,D}, ProjectVerifierAgent._evidence, {S,R,L,D})
    value['memory'] = dict(memory(), operation='append', base_versions=[dict(memory_id='lesson',version=D)])
    validate_memory_review(value, req, {S,R,L,D}, ProjectVerifierAgent._evidence, {S,R,L,D})


def test_retired_claim_needs_fresh_evidence_and_explicit_reconciliation():
    proposed = memory()
    identity = [' '.join(str(proposed.get(k, '')).casefold().split()) for k in
                ('summary','applicability','preconditions','expectations','limitations')]
    retired = dict(tombstone_id='retired', memory_id='lesson', version=D, document_sha256=D,
                   evidence=[L], claim_identity=identity)
    req = replace(request(), retired_memories=(retired,))
    value = decision(proposed)
    value['retirement_checks'] = [dict(tombstone_id='retired', relation='unrelated', reason='Different wording', evidence=[])]
    with pytest.raises(ValueError, match='exact retired claim'):
        validate_memory_review(value, req, {S,R,L,D}, ProjectVerifierAgent._evidence, {S,R,L,D})
    check = value['retirement_checks'][0]
    check.update(relation='reconciles', reason='New result resolves recorded limitation', evidence=[L])
    with pytest.raises(ValueError, match='fresh resolving'):
        validate_memory_review(value, req, {S,R,L,D}, ProjectVerifierAgent._evidence, {S,R,L,D})
    check['evidence'] = [R]
    validate_memory_review(value, req, {S,R,L,D}, ProjectVerifierAgent._evidence, {S,R,L,D})


def test_partial_memory_document_read_cannot_support_correctness_claim():
    good = decision()
    good['memory_judgments'] = [judgment(correctness='supported', usefulness='useful')]
    h = Harness([*prerequisites(), {'read_artifact':{'sha256':D,'limit':5}}, good, read(D), good])
    h.artifacts[D] = DOCUMENT
    assert h.agent.review(replace(request(), memory_exposures=(EXPOSURE,))) == good
    assert len([row for row in h.agent.last_trace if row['kind']=='feedback']) == 1


def test_historical_memory_evidence_cannot_prove_a_new_contradiction():
    value = decision()
    value['memory_judgments'] = [judgment()]
    req = replace(request(), memory_exposures=(dict(EXPOSURE, artifact_sha256=[R]),))
    with pytest.raises(ValueError, match='independent'):
        validate_memory_review(value, req, {S,R,L,D}, ProjectVerifierAgent._evidence, {S,R,L,D})
