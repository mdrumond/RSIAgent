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


def test_project_failing_before_source_gets_evidence_backed_repair():
    failed = ProjectReview('no-source', None, False, {'execution': R, 'actor-log': L})
    h = Harness([read(R), repair(), read(L), decision(), repair()])
    result = h.agent.review(failed)
    assert result == repair()
    assert [digest for digest, _ in h.reads] == [R, L]
    assert len([event for event in h.agent.last_trace if event['kind'] == 'feedback']) == 2
    with pytest.raises(ValueError):
        ProjectReview('invalid-pass', None, True, {'result': R})


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


def test_duplicate_merge_bases_receive_feedback_before_publication():
    base=dict(memory_id='increment',version='v1')
    h=Harness([*prerequisites(),decision(memory('merge',[base,base])),
               decision(memory('merge',[base]))],knowledge={'memories':[base],'citations':[]})
    result=h.agent.review(request())
    assert result['memory']['base_versions']==[base]
    assert any(e['kind']=='feedback' for e in h.agent.last_trace)


@pytest.mark.parametrize('source_present', [True, False])
def test_empty_required_log_can_be_inspected_for_acceptance_or_sourceless_repair(source_present):
    empty = sha(b'')
    req = ProjectReview('empty-log', S if source_present else None, source_present,
                        {'result': R, 'actor-log': empty})
    responses = [read(R), read(empty)]
    if source_present:
        responses += [read(S), {'knowledge_query': {'query': 'increment'}}, decision(memory())]
    else:
        responses += [repair()]
    h = Harness(responses)
    h.artifacts[empty] = b''
    result = h.agent.review(req)
    assert result['decision'] == ('accept' if source_present else 'repair')
    empty_reads = [event['payload']['observation'] for event in h.agent.last_trace
                   if event['kind'] == 'tool' and event['payload']['action'] == read(empty)]
    assert empty_reads == [dict(sha256=empty, offset=0, size=0, text='', next_offset=None)]


@pytest.mark.parametrize('quoted_source', ['before', 'after'])
def test_snippet_requires_all_quoted_bytes_observed_across_pages(quoted_source):
    target = B if quoted_source == 'before' else S
    data = BEFORE if quoted_source == 'before' else SOURCE
    proposal = memory()
    if quoted_source == 'before':
        proposal.update(before=snippet(B, BEFORE.decode()), before_unavailable_reason=None)
    first = {'read_artifact': {'sha256': target, 'offset': 0, 'limit': 5}}
    tail = {'read_artifact': {'sha256': target, 'offset': 10, 'limit': len(data) - 10}}
    gap = {'read_artifact': {'sha256': target, 'offset': 5, 'limit': 5}}
    setup = [read(R), read(L), {'knowledge_query': {'query': 'increment'}}]
    if quoted_source == 'before':
        setup.insert(0, read(S))
    h = Harness([*setup, first, decision(proposal), tail, decision(proposal), gap, decision(proposal)],
                knowledge={'memories': [{'memory_id': 'old', 'version': 'v1',
                                        'artifacts': {'before-source': B}}], 'citations': []})
    result = h.agent.review(request())
    assert result['memory'] == proposal
    assert len([event for event in h.agent.last_trace if event['kind'] == 'feedback']) == 2
    observed_pages = [event['payload']['action'] for event in h.agent.last_trace
                      if event['kind'] == 'tool' and 'read_artifact' in event['payload']['action']
                      and event['payload']['action']['read_artifact']['sha256'] == target]
    assert observed_pages == [first, tail, gap]


def test_provider_metadata_is_corrected_before_executing_valid_tool():
    malformed=dict(read(R),type='json_object')
    h=Harness([malformed,*prerequisites(),decision(memory())])
    result=h.agent.review(request())
    assert result['decision']=='accept'
    assert len([event for event in h.agent.last_trace if event['kind']=='feedback'])==1
    assert [sha for sha,_ in h.reads].count(R)==1


def test_large_artifact_pages_fit_context_without_losing_admission_state():
    from dataclasses import replace
    large=b'x'*50000
    digest=sha(large)
    first={'read_artifact':{'sha256':digest,'offset':0,'limit':20000}}
    second={'read_artifact':{'sha256':digest,'offset':20000,'limit':20000}}
    pages=[{'read_artifact':{'sha256':digest,'offset':offset,'limit':8192}} for offset in range(0,49152,8192)]
    h=Harness([*prerequisites(),*pages,read(S),decision(memory())])
    h.artifacts[digest]=large
    req=replace(request(),artifacts=dict(request().artifacts,large_log=digest))
    assert h.agent.review(req)['memory']==memory()
    assert max(len(json.dumps(messages,ensure_ascii=False).encode()) for messages,_ in h.calls)<=40000
    assert len([sha for sha,_ in h.reads if sha==digest])==6


def test_large_registry_pages_hidden_before_code_and_preserves_exact_snippets():
    refs = {f'retained:{i:04}': sha(str(i).encode()) for i in range(500)}
    refs.update({'result': R, 'actor-log': L, 'retained:before-variant': B})
    proposal = memory()
    proposal.update(before=snippet(B, BEFORE.decode()), before_unavailable_reason=None)
    h = Harness([{'list_artifacts': {'offset': -1}},
                 {'list_artifacts': {'prefix': 'retained:before', 'limit': 1}},
                 *prerequisites(), read(B), decision(proposal)])
    review = ProjectReview('many-artifacts', S, True, refs)
    result = h.agent.review(review)
    assert result['memory']['before']['sha256'] == B
    initial = json.loads(h.calls[0][0][1]['content'])['project']
    assert initial['artifact_count'] == 504
    assert initial['artifacts']['source'] == S
    assert 'retained:before-variant' not in initial['artifacts']
    assert len(json.dumps(h.calls[0][0]).encode()) < 40000
    assert any(event['kind'] == 'feedback' for event in h.agent.last_trace)
    listing = next(event['payload']['observation'] for event in h.agent.last_trace
                   if event['kind'] == 'tool' and 'list_artifacts' in event['payload']['action'])
    assert listing == {'artifacts': {'retained:before-variant': B}, 'total': 1, 'next_offset': None}
    assert h.agent.last_trace[0]['payload']['project']['artifacts'] == refs


def test_rich_knowledge_observation_is_bounded_and_new_artifacts_are_pageable():
    base = {'memory_id': 'old', 'version': 'v1'}
    proposal = memory('append', [base])
    proposal.update(before=snippet(B, BEFORE.decode()), before_unavailable_reason=None)
    h = Harness([*prerequisites(),
                 {'list_artifacts': {'prefix': 'memory:old:v1:'}}, read(B), decision(proposal)],
                knowledge={'memories': [{**base, 'proposal': {'summary': 'x' * 80000},
                            'artifacts': {'before-source': B}}], 'citations': [],
                           'hits': [{'text': 'y' * 80000, 'citation': {'path': 'old'}}]})
    assert h.agent.review(request())['memory']['base_versions'] == [base]
    query_event = next(e['payload']['observation'] for e in h.agent.last_trace
                       if e['kind'] == 'tool' and 'knowledge_query' in e['payload']['action'])
    assert len(json.dumps(query_event)) < 3000
    assert query_event['memories'][0]['artifact_prefix'] == 'memory:old:v1:'
    assert B in [digest for digest, _ in h.reads]


def test_host_extracts_agent_selected_observed_lines_without_json_transcription():
    proposal = memory()
    del proposal['after']['text']
    h = Harness([*prerequisites(), decision(proposal)])
    result = h.agent.review(request())
    assert result['memory']['after']['text'] == SOURCE.decode()
    assert result['memory']['after']['start_line'] == 1
    assert result['memory']['after']['end_line'] == 2


def test_host_extraction_still_requires_observation_of_all_selected_bytes():
    proposal = memory()
    del proposal['after']['text']
    h = Harness([{'read_artifact': {'sha256': S, 'limit': 3}}, read(R), read(L),
                 {'knowledge_query': {'query': 'increment'}}, decision(proposal),
                 read(S), decision(proposal)])
    result = h.agent.review(request())
    assert result['memory']['after']['text'] == SOURCE.decode()
    assert len([e for e in h.agent.last_trace if e['kind'] == 'feedback']) == 1


def test_missing_evidence_feedback_names_exact_roles_and_digest():
    h = Harness([read(S), read(R), decision(), read(L),
                 {'knowledge_query': {'query': 'increment'}}, decision(memory())])
    h.agent.review(request())
    feedback = next(event['payload'] for event in h.agent.last_trace if event['kind'] == 'feedback')
    assert feedback['missing_required_artifacts'] == [{'roles': ['actor-log'], 'sha256': L}]
    assert feedback['knowledge_query_required'] is True


class TypedSession:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.contexts = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def next_action(self, context, allowed, record):
        self.contexts.append(context)
        action = next(self.actions)
        name, args = (('submit_review', action) if 'decision' in action else next(iter(action.items())))
        assert name in allowed
        record('pi-action', {'name': name, 'arguments': args})
        return name, args


def test_typed_tools_preserve_gate_and_expose_missing_log_for_next_action():
    h = Harness([])
    session = TypedSession([read(S), read(R), decision(), read(L),
        {'knowledge_query': {'query': 'increment'}}, decision(memory())])
    h.agent.session_factory = lambda **_: session
    result = h.agent.review(request())
    assert result == decision(memory())
    assert not h.calls  # Pi actions, not JSON completion requests.
    assert session.closed
    assert session.contexts[3]['review_state']['missing_required_artifacts'] == [
        {'roles': ['actor-log'], 'sha256': L}]
    assert [digest for digest, _ in h.reads[:3]] == [S, R, L]
    assert h.queries[0][0] == {'query': 'increment'}


def test_typed_tools_cannot_override_failed_host_and_close_on_cancellation():
    from coding.pi.rpc import PiCancelled
    h = Harness([])
    session = TypedSession([*prerequisites(), decision(memory()), repair()])
    h.agent.session_factory = lambda **_: session
    assert h.agent.review(request(False)) == repair()
    assert session.closed

    class Cancelled(TypedSession):
        def next_action(self, *args, **kwargs):
            raise PiCancelled('cancelled')
    session = Cancelled([])
    h.agent.session_factory = lambda **_: session
    with pytest.raises(PiCancelled):
        h.agent.review(request())
    assert session.closed


def test_typed_tools_cannot_admit_late_action():
    seconds = [0]
    h = Harness([], clock=lambda: seconds[0])
    class Late(TypedSession):
        def next_action(self, *args, **kwargs):
            seconds[0] = 6
            return 'submit_review', decision()
    session = Late([])
    h.agent.session_factory = lambda **_: session
    with pytest.raises(TimeoutError):
        h.agent.review(request(), max_seconds=5)
    assert session.closed
    assert not any(event['kind'] == 'decision' for event in h.agent.last_trace)


def test_prose_evidence_is_a_format_error_not_a_missing_artifact_read():
    bad = decision(memory())
    bad['evidence'] = [f'actor-log {L}: compiled successfully']
    h = Harness([*prerequisites(), bad, decision(memory())])
    assert h.agent.review(request()) == decision(memory())
    feedback = next(e['payload'] for e in h.agent.last_trace if e['kind'] == 'feedback')
    assert feedback['missing_required_artifacts'] == []
    assert set(feedback['allowed_evidence']) == {S, R, L}
    assert feedback['allowed_citations'] == []
    assert feedback['error_code'] == 'malformed-evidence'
    assert [digest for digest, _ in h.reads].count(R) == 1
    assert [digest for digest, _ in h.reads].count(L) == 1


def test_typed_context_deduplicates_repeated_pages_and_keeps_current_feedback():
    bad = decision(memory())
    bad['evidence'] = [f'source artifact {S}']
    h = Harness([])
    session = TypedSession([*prerequisites(), bad, *[read(S)] * 30, decision(memory())])
    h.agent.session_factory = lambda **_: session
    assert h.agent.review(request()) == decision(memory())
    sizes = [len(json.dumps(context).encode()) for context in session.contexts]
    assert max(sizes) < 10000
    assert all('history' not in context for context in session.contexts)
    assert len(session.contexts[-1]['artifact_pages']) == 3
    feedback = session.contexts[-1]['validation_feedback']
    assert feedback['missing_required_artifacts'] == []
    assert set(feedback['allowed_evidence']) == {S, R, L}


def retained_request(original=B):
    from dataclasses import replace
    req = request()
    return replace(req, artifacts={**req.artifacts, 'original-source': original})


def contrasted_memory(original=B, text=BEFORE.decode()):
    proposal = memory()
    proposal.update(before=snippet(original, text), before_unavailable_reason=None)
    return proposal


@pytest.mark.parametrize('rewrite_feedback', [None, {'instructions':'Preserve the retained code contrast'}])
def test_distinct_original_requires_full_read_and_real_before_including_rewrite(rewrite_feedback):
    partial = {'read_artifact': {'sha256':B,'offset':0,'limit':4}}
    rest = {'read_artifact': {'sha256':B,'offset':4}}
    h = Harness([*prerequisites(), decision(memory()), partial, decision(memory()),
                 rest, decision(memory()), decision(contrasted_memory())])
    result = h.agent.review(retained_request(), feedback=rewrite_feedback)
    assert result['memory']['before']['sha256'] == B
    feedback = [e['payload'] for e in h.agent.last_trace if e['kind']=='feedback']
    assert [e['error_code'] for e in feedback] == [
        'incomplete-original-source','incomplete-original-source','retained-before-required']
    first = next(row for row in feedback[0]['missing_required_artifacts'] if row['sha256']==B)
    assert first['complete_read_required'] is True and first['next_read']['offset'] == 0
    second = next(row for row in feedback[1]['missing_required_artifacts'] if row['sha256']==B)
    assert second['missing_ranges'] == [{'offset':4,'end':len(BEFORE)}]
    assert second['next_read']['offset'] == 4
    assert B in feedback[0]['readable_artifacts'] and B not in feedback[0]['allowed_evidence']
    assert B in feedback[1]['allowed_evidence']  # Compatibility: one observed page remains citable.
    assert feedback[2]['missing_required_artifacts'] == []


def test_coverage_requires_gap_read_even_if_last_original_page_was_read():
    from benchmarks.ascend.project_verifier import ReviewState
    h = Harness([])
    state = ReviewState(h.agent, retained_request(), lambda:10, None)
    state.execute({'read_artifact': {'sha256':B,'offset':8}})
    state.execute({'read_artifact': {'sha256':B,'offset':0,'limit':4}})
    summary = state.summary()
    assert B in summary['readable_artifacts'] and B in summary['read_artifacts']
    assert B not in summary['fully_read_artifacts']
    missing = next(row for row in summary['missing_required_artifacts'] if row['sha256']==B)
    assert missing['missing_ranges'] == [{'offset':4,'end':8}]
    state.execute({'read_artifact': {'sha256':B,'offset':4,'limit':4}})
    assert B in state.summary()['fully_read_artifacts']
    assert B not in {row['sha256'] for row in state.summary()['missing_required_artifacts']}


@pytest.mark.parametrize('original', [None, S])
def test_null_before_remains_valid_for_absent_or_identical_original(original):
    req = request() if original is None else retained_request(original)
    h = Harness([*prerequisites(), decision(memory())])
    result = h.agent.review(req)
    assert result['memory']['before'] is None
    assert not [e for e in h.agent.last_trace if e['kind']=='feedback']


@pytest.mark.parametrize('original', [None, S, B])
def test_identical_before_after_rejected_without_inventing_distinct_source(original):
    bad = memory()
    bad.update(before=snippet(), before_unavailable_reason=None)
    corrected = contrasted_memory() if original == B else memory()
    req = request() if original is None else retained_request(original)
    h = Harness([*prerequisites(), *([read(B)] if original==B else []),
                 decision(bad), decision(corrected)])
    assert h.agent.review(req)['memory'] == corrected
    feedback = [e['payload'] for e in h.agent.last_trace if e['kind']=='feedback']
    assert len(feedback)==1 and feedback[0]['error_code']=='identical-before-after'


def test_other_kdb_source_cannot_replace_this_projects_original():
    other = b'def kernel(x):\n    return 0\n'
    digest = sha(other)
    base = {'memory_id':'other','version':'v1','artifacts':{'before-source':digest}}
    h = Harness([*prerequisites(),read(B),read(digest),decision(contrasted_memory(digest,other.decode())),
                 decision(contrasted_memory())],knowledge={'memories':[base],'citations':[]})
    h.artifacts[digest] = other
    assert h.agent.review(retained_request())['memory']['before']['sha256']==B
    assert any(e['kind']=='feedback' and e['payload']['error_code']=='wrong-before-source' for e in h.agent.last_trace)


def test_unreadable_retained_original_is_not_evidence_of_absence():
    h = Harness([*prerequisites(),read(B),decision(memory()),repair()])
    original_reader = h.agent.read_artifact
    def unavailable(digest, **options):
        if digest==B:raise FileNotFoundError('retained object temporarily unavailable')
        return original_reader(digest, **options)
    h.agent.read_artifact = unavailable
    assert h.agent.review(retained_request())==repair()
    feedback = [e['payload'] for e in h.agent.last_trace if e['kind']=='feedback']
    assert [e['error_code'] for e in feedback]==['artifact-unavailable','incomplete-original-source']
    assert all(B in e['readable_artifacts'] and B not in e['allowed_evidence'] for e in feedback)
    assert all(B in {row['sha256'] for row in e['missing_required_artifacts']} for e in feedback)


def test_passing_original_is_preserved_as_history_without_relabeling_it_failed():
    from dataclasses import replace
    passing = b'{"role":"original-control","host_verified":true}'
    p = sha(passing)
    req = retained_request()
    req = replace(req, artifacts={**req.artifacts,'original-result':p},
                  host_facts=({'category':'host-verification','source_sha256':B,'host_verified':True,'evidence_sha256':p},))
    proposal = contrasted_memory()
    proposal.update(summary='Two host-verified increment implementations',limitations='Both versions passed; no failure reproduction or repair attribution is established')
    h = Harness([*prerequisites(),read(B),read(p),decision(proposal)])
    h.artifacts[p] = passing
    result = h.agent.review(req)
    assert result['memory']==proposal and result['memory']['before']['sha256']==B
    initial = next(e['payload'] for e in h.agent.last_trace if e['kind']=='request')
    assert initial['project']['host_facts'][0]['host_verified'] is True
    assert not [e for e in h.agent.last_trace if e['kind']=='feedback']
