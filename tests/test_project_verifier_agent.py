"""Observable verifier decisions, tools, provenance and elapsed allocations."""
import hashlib
import json

import pytest

from benchmarks.ascend.project_verifier import ProjectReview, ProjectVerifierAgent


def sha(data):
    return hashlib.sha256(data).hexdigest()


SOURCE = b'def kernel(x):\n    return x + 1\n'
BEFORE = b'def kernel(x):\n    return x\n'
RESULT = b'{"verified":true,"cases":3}'
LOG = b'Compiled source; ran three independent input cases.\n'
S, B, R, L = map(sha, (SOURCE, BEFORE, RESULT, LOG))


def request(verified=True):
    return ProjectReview('increment', S, verified, {'result': R, 'actor-log': L},
                         ({'category': 'host-verification', 'evidence_sha256': R},),
                         'Learn reliable increments')


def read(digest):
    return {'read_artifact': {'sha256': digest}}


def snippet(digest=S, text=SOURCE.decode()):
    return dict(sha256=digest, start_line=1, end_line=2, text=text)


def memory(operation='create', bases=None):
    return dict(operation=operation, base_versions=bases or [], summary='Increment preserves shape',
                applicability='Elementwise integer increment', preconditions='No overflow',
                expectations='Each output equals input plus one', limitations='No overflow proof',
                before=None, before_unavailable_reason='No retained earlier candidate',
                after=snippet(), evidence=[R], citations=[])


def decision(proposal=None):
    return dict(decision='accept', reason='Independent cases and source agree',
                evidence=[R], repair=None, memory=proposal)


def repair():
    return dict(decision='repair', reason='A constant special case is not sufficient', evidence=[R],
                repair=dict(instructions='Test another input distribution',
                            expected_evidence='Host-validated output for negative inputs'), memory=None)


class Harness:
    def __init__(self, responses, *, knowledge=None, check=None, clock=None):
        self.responses = iter(responses)
        self.calls, self.reads, self.queries, self.checks = [], [], [], []
        self.artifacts = {S: SOURCE, B: BEFORE, R: RESULT, L: LOG}
        self.knowledge = knowledge or {'memories': [], 'citations': []}
        self.check_result = check or {'host_verified': True, 'artifacts': {}}
        self.agent = ProjectVerifierAgent(self.complete, self.reader,
            query_knowledge=self.query, request_check=self.check, **({'clock': clock} if clock else {}))

    def complete(self, messages, *, remaining_seconds):
        self.calls.append((messages, remaining_seconds))
        return json.dumps(next(self.responses))

    def reader(self, digest, *, remaining_seconds):
        self.reads.append((digest, remaining_seconds))
        return self.artifacts[digest]

    def query(self, payload, *, remaining_seconds):
        self.queries.append((payload, remaining_seconds))
        return self.knowledge

    def check(self, payload, *, remaining_seconds):
        self.checks.append((payload, remaining_seconds))
        return self.check_result


def prerequisites():
    return [read(S), read(R), read(L), {'knowledge_query': {'query': 'increment overflow lessons'}}]


def test_accept_requires_actual_source_results_logs_and_kdb_calls():
    h = Harness([decision(), read(S), read(R), decision(), read(L), decision(),
                 {'knowledge_query': {'query': 'increment'}}, decision(memory())])
    result = h.agent.review(request())
    assert result['memory']['expectations'] == memory()['expectations']
    assert [d for d, _ in h.reads[:3]] == [S, R, L]
    assert [q for q, _ in h.queries] == [{'query': 'increment'}]
    assert len([e for e in h.agent.last_trace if e['kind'] == 'feedback']) == 3
    assert h.agent.last_trace[-1]['payload'] == result


@pytest.mark.parametrize('host_verified', [True, False])
def test_failed_host_or_additional_check_cannot_be_overridden(host_verified):
    h = Harness([*prerequisites(),
                 {'request_check': {'hypothesis': 'Negative inputs expose accidental pass',
                                    'check': {'variant': 'negative'}}},
                 decision(memory()), repair()], check={'host_verified': False, 'artifacts': {}})
    result = h.agent.review(request(host_verified))
    assert result == repair()
    assert result['memory'] is None
    assert h.checks[0][0]['check'] == {'variant': 'negative'}
    assert [event['kind'] for event in h.agent.last_trace][-2:] == ['response', 'decision']


@pytest.mark.parametrize('operation', ['append', 'merge'])
def test_memory_reuses_observed_versions_and_exact_before_after_code(operation):
    citation = {'collection': 'memory-v1', 'path': 'lesson.json', 'start_line': 1, 'end_line': 3}
    base = {'memory_id': 'increment-lesson', 'version': 'v1'}
    proposed = memory(operation, [base])
    proposed.update(before=snippet(B, BEFORE.decode()), before_unavailable_reason=None,
                    citations=[citation])
    h = Harness([*prerequisites(), read(B), decision(proposed)], knowledge={
        'memories': [{**base, 'artifacts': {'before-source': B}}], 'citations': [citation]})
    result = h.agent.review(request())
    assert result['memory']['base_versions'] == [base]
    assert result['memory']['before']['text'] == BEFORE.decode()
    assert result['memory']['after']['sha256'] == S
    assert result['memory']['citations'] == [citation]
    assert B in [digest for digest, _ in h.reads]


def test_invented_snippet_and_unseen_base_and_citation_are_rejected():
    wrong_snippet = memory()
    wrong_snippet['after'] = snippet(text='def kernel(x):\n    return 42\n')
    wrong_base = memory('append', [{'memory_id': 'unseen', 'version': 'v1'}])
    wrong_citation = memory()
    wrong_citation['citations'] = [{'collection': 'unseen'}]
    h = Harness([*prerequisites(), decision(wrong_snippet), decision(wrong_base),
                 decision(wrong_citation), decision(memory())])
    result = h.agent.review(request())
    assert result['memory'] == memory()
    assert len([e for e in h.agent.last_trace if e['kind'] == 'feedback']) == 3


def test_verifier_can_return_successful_project_for_curriculum_repair():
    h = Harness([read(R), repair()])
    assert h.agent.review(request(True)) == repair()
    assert not h.queries


def test_rewrite_uses_same_evidence_in_fresh_context_and_preserves_feedback():
    feedback = {'previous_proposal': memory(), 'instructions': 'Explain overflow preconditions'}
    h = Harness([*prerequisites(), decision(memory()), *prerequisites(), decision(memory())])
    original = h.agent.review(request())
    revised = h.agent.review(request(), feedback=feedback)
    assert original == revised
    starts = [json.loads(calls[0][1]['content']) for calls in h.calls if len(calls[0]) == 2]
    assert [item['rewrite_feedback'] for item in starts] == [None, feedback]
    assert starts[0]['project'] == starts[1]['project']
    assert len(h.checks) == 0


def test_elapsed_budget_forwarded_and_late_provider_decision_rejected():
    now = [0.0]
    h = Harness([*prerequisites(), decision()], clock=lambda: now[0])
    original_reader = h.agent.read_artifact

    def reader(*args, **kwargs):
        now[0] += 1
        return original_reader(*args, **kwargs)

    h.agent.read_artifact = reader
    h.agent.review(request(), max_seconds=10)
    assert [seconds for _, seconds in h.reads] == [10, 9, 8]
    assert h.queries[0][1] == 7
    assert h.calls[-1][1] == 7

    def late(messages, *, remaining_seconds):
        now[0] += remaining_seconds + 1
        return json.dumps(decision())

    h.agent.complete = late
    with pytest.raises(TimeoutError):
        h.agent.review(request(), max_seconds=10)
    assert not any(event['kind'] == 'decision' for event in h.agent.last_trace)
