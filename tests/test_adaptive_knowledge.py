from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.adaptive_knowledge import AdaptiveKnowledgeAgent
from benchmarks.a5kernels.knowledge_agent import Citation, KnowledgeQuery, KnowledgeResult
from benchmarks.a5kernels.knowledge_candidates import CandidatePool


class Clock:
    value = 0
    def __call__(self):
        return self.value
    def advance(self, seconds):
        self.value += seconds


def row(index):
    return KnowledgeResult(Citation('refs', f'{index}.cpp', 1, 2, str(index)),
                           f'original evidence {index}', .8)


def identity(index):
    return 'c' + str(index + 1)


class Gate:
    enabled = True
    collections = ('refs',)
    manifests = {'refs': object()}
    def __init__(self, clock, pools):
        self.clock, self.pools = clock, pools
        self.calls, self.entries, self.reads = [], [], []
        self.journal = SimpleNamespace(append=lambda **entry: self.entries.append(entry))
        self.closed = False
    def candidate_pool(self, action, **options):
        options['check_remaining']()
        self.clock.advance(2)
        self.calls.append(action.query)
        return CandidatePool(tuple(self.pools.get(action.query, ())))
    def read_artifact(self, collection, digest, offset=0, limit=16384):
        if collection != 'refs' or digest != 'admitted':
            raise ValueError('artifact not admitted')
        self.reads.append((collection, digest, offset, limit))
        self.clock.advance(3)
        return Page(b'verified artifact', 0)
    def close(self):
        self.closed = True


@dataclass
class Page:
    data: bytes
    offset: int


def setup(decisions, *, pools=None, seconds=90):
    clock = Clock()
    gate = Gate(clock, pools or {'query': [row(i) for i in range(25)]})
    calls, audit = [], []
    def complete(messages, allowance):
        calls.append((messages.copy(), allowance))
        clock.advance(4)
        decision = decisions.pop(0)
        if isinstance(decision, Exception):
            raise decision
        return {'text': decision if isinstance(decision, str) else json.dumps(decision), 'tokens': 10}
    agent = AdaptiveKnowledgeAgent(gate, complete, clock=clock, seconds=seconds, audit=audit.append)
    return agent, gate, clock, calls, audit


def pick(index, action='finish', **kwargs):
    return dict(action=action, selected=[dict(id=identity(index), reason='Useful for the requested target')], **kwargs)


def test_multiple_pages_and_wrong_target_rejection_preserve_original_citations():
    agent, gate, _, calls, audit = setup([
        dict(action='next', rejected=[dict(id=identity(0), reason='Wrong target')]), pick(12)])
    result = agent.search(KnowledgeQuery('query'))
    assert result.status == 'complete' and result.results == (row(12),)
    assert result.results[0].citation == row(12).citation
    assert gate.calls == ['query']
    assert len(audit[0]['inspections']) == 2
    assert audit[0]['rejected'] == {identity(0): 'Wrong target'}
    assert gate.entries[0]['citations'] == (row(12).citation,)
    assert len(calls) == 2 and result.audit['tokens'] == 20


def test_early_finish_never_reads_another_page_and_actor_receives_status():
    agent, gate, _, calls, _ = setup([pick(1, gaps=['CANN version unverified'])])
    result = agent.execute(KnowledgeQuery('query'))
    value = json.loads(result.observation)['knowledge_query']
    assert value['status'] == 'partial' and value['gaps'] == ['CANN version unverified']
    assert value['results'][0]['text'] == row(1).text
    assert value['selection_reasons'] == [dict(citation=row(1).citation.as_dict(),
        reason='Useful for the requested target')]
    assert len(calls) == len(agent.last_search_details.audit['inspections']) == 1
    agent.close()
    assert gate.closed


def test_reformulate_can_select_new_evidence_and_abstention_is_valid():
    agent, gate, _, calls, _ = setup([dict(action='rewrite', query='precise'), pick(1)],
                                    pools={'query': [row(0)], 'precise': [row(99)]})
    agent.configure_context(task='cube example', role='writer', context={'target': 'a3'})
    result = agent.search(KnowledgeQuery('query'))
    assert result.results == (row(99),) and gate.calls == ['query', 'precise']
    assert json.loads(calls[0][0][1]['content'])['context'] == {'target': 'a3'}
    agent, _, _, _, _ = setup([dict(action='finish', gaps=['No compatible example'])])
    result = agent.search(KnowledgeQuery('query'))
    assert result.status == 'insufficient' and not result.results


@pytest.mark.parametrize('failure', [TimeoutError(), {'action': 'finish', 'selected': [dict(id='invented', reason='bad')]}])
def test_failure_retains_only_prior_validated_selections(failure):
    agent, _, _, _, audit = setup([pick(1, action='next'), failure] + ([] if isinstance(failure, Exception) else [TimeoutError()]))
    result = agent.search(KnowledgeQuery('query'))
    assert result.status == 'partial' and result.results == (row(1),)
    assert audit[0]['error_type'] in {'TimeoutError', 'ValueError'}
    assert len(audit[0]['requests']) == (2 if isinstance(failure, Exception) else 3)


def test_queries_and_external_reads_share_elapsed_budget():
    agent, gate, clock, calls, _ = setup([pick(1), pick(2)], seconds=15)
    assert agent.query(KnowledgeQuery('query')) == (row(1),)  # six seconds
    agent.read_artifact('refs', 'admitted', seconds=20)  # three more
    assert agent.query(KnowledgeQuery('query'), seconds=20) == ()  # six seconds exhausts allocation
    assert agent.remaining() == 0 and agent.spent_seconds == 15
    assert calls[0][1] == 13 and calls[1][1] == 4
    assert agent.last_search_details.status == 'insufficient'
    with pytest.raises(TimeoutError):
        agent.read_artifact('refs', 'admitted')
    assert len(gate.reads) == 1


def test_internal_artifact_read_is_gated_and_retained_in_audit():
    agent, gate, _, _, audit = setup([
        dict(action='artifact', artifact=dict(collection='refs', digest='admitted')), pick(1)])
    result = agent.search(KnowledgeQuery('query'))
    assert result.status == 'complete'
    assert audit[0]['artifacts'][0]['response']['data'] == 'verified artifact'
    agent, gate, _, _, _ = setup([pick(1, action='artifact', artifact=dict(collection='refs', digest='unknown'))])
    result = agent.search(KnowledgeQuery('query'))
    assert result.status == 'partial' and result.results == (row(1),) and not gate.reads


def test_inspection_workload_bound_and_empty_candidate_pool():
    decisions = [dict(action='next') for _ in range(10)]
    agent, _, _, calls, _ = setup(decisions, pools={'query': [row(i) for i in range(130)]}, seconds=1000)
    result = agent.search(KnowledgeQuery('query'))
    assert sum(len(batch['candidates']) for batch in result.audit['inspections']) == 100
    assert len(calls) == 10 and result.status == 'insufficient'
    agent, _, _, calls, _ = setup([], pools={'query': []})
    assert agent.search(KnowledgeQuery('query')).status == 'insufficient' and not calls


@pytest.mark.parametrize('seconds', [0, -1, float('inf'), float('nan'), True])
def test_invalid_elapsed_allocation(seconds):
    with pytest.raises(ValueError):
        AdaptiveKnowledgeAgent(None, None, seconds=seconds)


def test_rolling_context_retains_selected_evidence_without_previous_batches():
    agent, _, _, calls, audit = setup([pick(1, action='next'), pick(12, action='next'), pick(22)])
    result = agent.search(KnowledgeQuery('query'))
    assert result.results == (row(1), row(12), row(22))
    assert [len(messages) for messages, _ in calls] == [3, 3, 3]
    third = json.loads(calls[2][0][-1]['content'])
    assert [entry['id'] for entry in third['state']['selected']] == [identity(1), identity(12)]
    assert {entry['id'] for entry in third['observation']['candidates']} == {identity(i) for i in range(20, 25)}
    assert identity(0) not in {entry['id'] for entry in third['state']['selected']}
    assert audit[0]['inspections'][0]['candidates'][0]['text'] == row(0).text
    assert result.audit['selected_citations'][identity(22)] == row(22).citation.as_dict()


def test_selection_limit_rejection_preserves_previously_valid_progress():
    too_many = dict(action='finish', selected=[dict(id=identity(i), reason='useful') for i in range(10, 16)])
    agent, _, _, _, _ = setup([pick(1, action='next'), too_many, TimeoutError()])
    result = agent.search(KnowledgeQuery('query', 5))
    assert result.status == 'partial' and result.results == (row(1),)
    assert result.audit['requests'][1]['validation_error']['type'] == 'ValueError'


def test_malformed_decision_is_corrected_without_rescanning_or_accepting_invalid_ids():
    malformed = '{"action":"finish","selected":[{"id":"invented","reason":"bad"}],"gaps":{}'
    agent, gate, _, calls, audit = setup([malformed, pick(1)])
    result = agent.search(KnowledgeQuery('query'))
    assert result.status == 'complete' and result.results == (row(1),)
    assert gate.calls == ['query'] and result.audit['tokens'] == 20
    assert audit[0]['requests'][0]['response']['text'] == malformed
    assert audit[0]['requests'][0]['validation_error']['type'] == 'JSONDecodeError'
    correction = json.loads(calls[1][0][-1]['content'])['correction']
    assert correction['last_invalid_response'] == malformed
    assert len(calls[1][0]) == 3


def test_repeated_malformed_responses_spend_original_deadline_and_preserve_progress():
    agent, _, _, calls, audit = setup([pick(1, action='next')] + ['{bad'] * 5, seconds=18)
    result = agent.search(KnowledgeQuery('query'))
    assert result.status == 'partial' and result.results == (row(1),)
    assert result.audit['error_type'] == 'TimeoutError' and agent.spent_seconds == 18
    assert [allowance for _, allowance in calls] == [16, 12, 8, 4]
    assert len(audit[0]['requests']) == 4
    assert all(len(messages) == 3 for messages, _ in calls)
