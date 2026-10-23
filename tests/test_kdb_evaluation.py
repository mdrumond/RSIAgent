import json

import pytest

from benchmarks.ascend.kdb_evaluation import Budget, Case, Pilot, objective_checks


class Clock:
    value = 1000
    def __call__(self):
        return self.value


def test_budget_resume_charges_interrupted_work_and_preserves_cap(tmp_path):
    clock = Clock()
    budget = Budget(tmp_path / 'state.json', 'frozen', seconds=30, clock=clock)
    clock.value += 7
    resumed = Budget(budget.path, 'frozen', seconds=30, clock=clock)
    assert resumed.remaining() == 23
    clock.value += 4
    resumed.close()
    clock.value += 100
    resumed = Budget(budget.path, 'frozen', seconds=30, clock=clock)
    assert resumed.remaining() == 19
    with pytest.raises(ValueError):
        Budget(budget.path, 'changed', seconds=30, clock=clock)


def test_fresh_arms_and_direct_memory_isolation(tmp_path):
    seen, gates = [], []
    class Gate:
        def query(self, request, seconds):
            assert request.query == 'queue lifetime'
            return ()
    def factory(arm):
        gates.append(arm)
        return Gate()
    def complete(messages, seconds):
        seen.append(messages[:])
        assert 0 < seconds <= 45
        if len(messages) == 2 and 'No retrieval' not in messages[0]['content']:
            return dict(text=json.dumps({'knowledge_query': {'query': 'queue lifetime'}}), tokens=2)
        return dict(text=json.dumps({'answer': 'Use a verified transfer example.', 'citations': [], 'numerical_claims': []}), tokens=3)
    pilot = Pilot(complete, factory, 'a complete lesson snapshot', Budget(tmp_path/'state.json', 'id'))
    case = Case('one', 'writer', 'transfer', 'Find useful code', {})
    results = [pilot.run_arm(case, arm) for arm in 'ABCD']
    assert all(r['status'] == 'complete' for r in results)
    assert gates == ['B', 'C', 'D']
    starts = [m for m in seen if len(m) == 2]
    assert [len(m) for m in starts] == [2]*4
    assert ['snapshot' in m[-1]['content'] for m in starts] == [False, False, True, False]
    assert [len(r['queries']) for r in results] == [0, 1, 1, 1]
    pilot.budget.close()


def test_invalid_numeric_claims_and_citations_are_observed():
    checks = objective_checks({'citations': [{'path': 'invented'}], 'numerical_claims': [
        {'fact': 'latency', 'value': 1, 'unit': 'ms'},
        {'fact': 'missing', 'value': 0, 'unit': '%'},
        {'fact': 'latency', 'value': 38.5, 'unit': 'us'},
    ]}, [], [{'fact': 'latency', 'value': 38.5, 'unit': 'us'}])
    assert checks['invalid_citations'] == 1
    assert checks['unsupported_numerical_claims'] == 1
    assert checks['unregistered_numerical_claims'] == 1


def test_transport_receives_remaining_elapsed_allowance(tmp_path):
    clock = Clock()
    calls = []
    def complete(messages, seconds):
        calls.append(seconds)
        clock.value += 5
        raise TimeoutError()
    budget = Budget(tmp_path/'state.json', 'id', seconds=3, clock=clock)
    result = Pilot(complete, None, '', budget).run_arm(Case('x', 'writer', 'x', 'task', {}), 'A')
    assert calls == [3]
    assert result['status'] == 'timeout'
    assert budget.remaining() == 0
    budget.close()


def test_curriculum_uses_production_admission_and_retrieval(tmp_path):
    seen = []
    class Gate:
        def query(self, request, seconds):
            seen.append(request.query)
            return ()
    clock = Clock()
    replies = iter(['{"knowledge_query":{"query":"Ascend C movement"}}',
                    'DECISION: STALLED\n{"reason":"Need target evidence"}'])
    def complete(messages, seconds):
        clock.value += 1
        try:
            text = next(replies)
        except StopIteration:
            raise TimeoutError() from None
        return dict(text=text, tokens=2)
    context = dict(objective='Study A3 movement', wave_index=1,
                   pre_wave_memory_sha256='0'*64, pre_wave_coverage=[], prior_outcomes=[],
                   phase2_feedback=({'observed_attempt': 'Prior target run unavailable'},))
    pilot = Pilot(complete, lambda _: Gate(), 'lesson', Budget(tmp_path/'state.json', 'id', clock=clock))
    result = pilot.run_arm(Case('c', 'curriculum', 'movement', 'task', {}, context=context), 'D')
    assert result['status'] == 'complete'
    assert result['checks']['admitted']
    assert seen == ['Ascend C movement']
    assert len(result['transcript']) == 5
    pilot.budget.close()


def test_artifact_pages_decode_and_gate_closes(tmp_path):
    from dataclasses import dataclass
    @dataclass
    class Page:
        data: bytes
    closed = []
    class Gate:
        def read_artifact(self, **request):
            return Page(b'code\xff')
        def close(self):
            closed.append(True)
    replies = iter(['{"read_artifact":{"collection":"lessons","digest":"abc"}}',
                    '{"answer":"Useful source","citations":[],"numerical_claims":[]}'])
    messages_seen = []
    def complete(messages, seconds):
        messages_seen.append(messages[:])
        return dict(text=next(replies), tokens=2)
    pilot = Pilot(complete, lambda _: Gate(), '', Budget(tmp_path/'state.json', 'id'))
    result = pilot.run_arm(Case('w', 'writer', 'copy', 'task', {}), 'D')
    assert result['status'] == 'complete'
    assert json.loads(messages_seen[-1][-1]['content'])['data'] == 'code\ufffd'
    assert closed == [True]
    pilot.budget.close()


def test_curriculum_timeout_retains_query_and_provider_transcript(tmp_path):
    closed = []
    class Gate:
        def query(self, request, seconds):
            raise TimeoutError()
        def close(self):
            closed.append(True)
    def complete(messages, seconds):
        assert 'Specific case instruction' in messages[-1]['content']
        return dict(text='{"knowledge_query":{"query":"movement"}}', tokens=2)
    context = dict(objective='Study A3 movement', wave_index=1,
                   pre_wave_memory_sha256='0'*64, pre_wave_coverage=[], prior_outcomes=[],
                   phase2_feedback=({'observed_attempt': 'Prior target run unavailable'},))
    pilot = Pilot(complete, lambda _: Gate(), '', Budget(tmp_path/'state.json', 'id'))
    result = pilot.run_arm(Case('c', 'curriculum', 'copy', 'Specific case instruction', {}, context=context), 'D')
    assert result['status'] == 'timeout'
    assert result['queries'][0]['query'] == 'movement'
    assert result['transcript'][-1]['role'] == 'assistant'
    assert closed == [True]
    pilot.budget.close()


def test_malformed_output_and_nonfinite_numbers():
    with pytest.raises(ValueError):
        objective_checks({'citations': ['bad'], 'numerical_claims': []}, [], [])
    check = objective_checks({'citations': [], 'numerical_claims': [
        {'fact': 'latency', 'value': float('nan'), 'unit': 'us'},
    ]}, [], [{'fact': 'latency', 'value': 4, 'unit': 'us'}])
    assert check['unsupported_numerical_claims'] == 1


def test_writer_tool_request_receives_real_evidence_before_final_answer(tmp_path):
    from types import SimpleNamespace
    from benchmarks.a5kernels.knowledge_agent import Citation
    citation = Citation('reference', 'copy.cpp', 1, 3, 'a'*64)
    source = 'DataCopy(destination, source, count);'
    class Gate:
        def query(self, request, seconds):
            assert request.query == 'Ascend C copy example'
            return (SimpleNamespace(citation=citation, text=source, score=1.0),)
    def complete(messages, seconds):
        if messages[-1]['role'] == 'user' and len(messages) == 2:
            return dict(text='{"knowledge_query":{"query":"Ascend C copy example"}}', tokens=3)
        observed = json.loads(messages[-1]['content'])
        assert observed[0]['text'] == source
        return dict(text=json.dumps(dict(answer='Adapt the inspected transfer count to the logical shape.',
                                         citations=[observed[0]['citation']], numerical_claims=[])), tokens=5)
    pilot = Pilot(complete, lambda _: Gate(), '', Budget(tmp_path/'state.json', 'id'))
    result = pilot.run_arm(Case('w', 'writer', 'copy', 'Retrieve an actual copy example', {}), 'B')
    assert result['status'] == 'complete'
    assert result['checks']['citation_count'] == 1
    assert result['checks']['invalid_citations'] == 0
    assert result['queries'][0]['query'] == 'Ascend C copy example'
    pilot.budget.close()


def test_judge_receives_blinded_evidence_context_and_shared_memory(tmp_path):
    budget = Budget(tmp_path/'state.json', 'id')
    case = Case('judge', 'curriculum', 'copy', 'Choose next task', {'support': 'Ground sources'},
                context={'prior_outcomes': [{'project_id': 'seen-copy'}]})
    expected = {}
    for arm in 'ABCD':
        packet = dict(status='complete', output='candidate-'+arm, retrieved=[{'text': 'source-'+arm}],
                      checks={'invalid_citations': ord(arm)-65})
        budget.state['runs'][case.case_id+':'+arm] = packet
        expected[packet['output']] = {k: packet[k] for k in ('output', 'retrieved', 'checks')}
    def complete(messages, seconds):
        payload = json.loads(messages[-1]['content'])
        assert payload['context'] == case.context
        assert payload['shared_memory'] == 'complete producer snapshot'
        assert set(payload['candidates']) == {'0', '1', '2', '3'}
        for packet in payload['candidates'].values():
            assert packet == expected[packet['output']]
        return dict(text=json.dumps(dict(
            scores={k: dict.fromkeys(['relevance', 'support', 'applicability', 'uncertainty', 'next_actions'], 3)
                    for k in payload['candidates']},
            reasons={k: 'Supported by inspected evidence' for k in payload['candidates']})), tokens=10)
    pilot = Pilot(complete, None, 'complete producer snapshot', budget)
    pilot.judge(case)
    assert budget.state['judgments']['judge']['status'] == 'complete'
    budget.close()


def test_judgment_transport_failure_preserves_run_and_next_judgment(tmp_path):
    budget = Budget(tmp_path/'state.json', 'id')
    for case_id in ['first', 'next']:
        for arm in 'ABCD':
            budget.state['runs'][case_id+':'+arm] = dict(status='complete', output='answer', retrieved=[], checks={})
    calls = []
    def complete(messages, seconds):
        calls.append(messages)
        if len(calls) == 1:
            raise OSError('bounded provider transport unavailable')
        ids = json.loads(messages[-1]['content'])['candidates']
        return dict(text=json.dumps(dict(
            scores={k: dict.fromkeys(['relevance', 'support', 'applicability', 'uncertainty', 'next_actions'], 2) for k in ids},
            reasons={k: 'Evidence assessed' for k in ids})), tokens=5)
    pilot = Pilot(complete, None, '', budget)
    pilot.judge(Case('first', 'writer', 'copy', 'task', {}))
    pilot.judge(Case('next', 'writer', 'copy', 'task', {}))
    assert budget.state['judgments']['first']['status'] == 'incomplete'
    assert budget.state['judgments']['next']['status'] == 'complete'
    assert len(budget.state['runs']) == 8
    budget.close()


@pytest.mark.parametrize('status', ['invalid', 'timeout'])
def test_incomplete_arm_skips_judge_provider(tmp_path, status):
    budget = Budget(tmp_path/'state.json', 'id')
    for arm in 'ABCD':
        budget.state['runs']['case:'+arm] = dict(status='complete', output='answer')
    budget.state['runs']['case:B']['status'] = status
    def forbidden(*args):
        pytest.fail('Incomplete matched group must not spend provider allowance')
    pilot = Pilot(forbidden, None, '', budget)
    pilot.judge(Case('case', 'writer', 'copy', 'task', {}))
    assert budget.state['judgments']['case'] == {'status': 'incomplete', 'reason': 'arm-incomplete'}
    budget.close()


def test_judge_receives_unchecked_report_claim_separate_from_registered_error(tmp_path):
    # Pipe percentage is available in the raw report but not in frozen fact IDs.
    raw_report = {'vector_time_percent': 12.5}
    checks = objective_checks({'citations': [], 'numerical_claims': [
        {'fact': 'vector_time_percent', 'value': 12.5, 'unit': '%'},
        {'fact': 'ordinary_median_us', 'value': 999, 'unit': 'us'},
    ]}, [], [{'fact': 'ordinary_median_us', 'value': 38.5, 'unit': 'us'}])
    assert checks['unregistered_numerical_claims'] == 1
    assert checks['unsupported_numerical_claims'] == 1
    budget = Budget(tmp_path/'state.json', 'id')
    for arm in 'ABCD':
        budget.state['runs']['numbers:'+arm] = dict(status='complete', output='analysis',
                                                  retrieved=[], checks=checks)
    def complete(messages, seconds):
        packet = json.loads(messages[-1]['content'])
        assert json.loads(packet['task']) == raw_report
        for candidate in packet['candidates'].values():
            assert candidate['checks']['unregistered_numerical_claims'] == 1
            assert candidate['checks']['unsupported_numerical_claims'] == 1
        keys = packet['candidates']
        return dict(text=json.dumps(dict(
            scores={k: dict.fromkeys(['relevance', 'support', 'applicability', 'uncertainty', 'next_actions'], 2) for k in keys},
            reasons={k: 'One registered mismatch; report percentage checked against raw evidence' for k in keys})), tokens=5)
    pilot = Pilot(complete, None, '', budget)
    pilot.judge(Case('numbers', 'analyzer', 'copy', json.dumps(raw_report), {}))
    assert budget.state['judgments']['numbers']['status'] == 'complete'
    budget.close()
