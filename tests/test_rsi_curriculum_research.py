"""Typed research uses actual retrieval, elapsed receipts, and durable decisions."""
from dataclasses import replace
import json

import pytest

from core.artifacts import ContentAddressedArtifacts
from rsi.contracts import ProjectSpec
from rsi.curriculum_research import (DiagnosticDecision, DiagnosticProposal, NegativeDecision,
    QueryPlan, ResearchQuery, Researcher, ResearchPending, ResearchRejected, parse_proposal)
from rsi.roles import _proposal


class Clock:
    now = 100.
    def __call__(self):
        return self.now


class Roles:
    def __init__(self, clock):
        self.clock, self.calls, self.delivered, self.deadlines = clock, [], [], []
        self.observer = lambda *_: None
        self.omit_negative = self.empty_queries = self.unknown = False
        self.parent = None

    def start(self, name, deadline):
        self.calls.append(name)
        self.deadlines.append(deadline)
        self.clock.now += 40  # Separate worker setup, never actor thinking.
        self.observer('role_started', {'role': name})
        self.clock.now += 2
        if self.unknown:
            raise RuntimeError('worker completion could not be observed')
        self.observer('role_finished', {'role': name, 'elapsed_seconds': 2., 'success': True})

    def research_queries(self, prompt, evidence, *, deadline, wall_deadline):
        self.start('research_queries', deadline)
        return QueryPlan('Investigate the observed failure', tuple(e.evidence_id for e in evidence),
            () if self.empty_queries else (ResearchQuery('reduction reference', 'Find documented constraints'),))

    def diagnostic(self, prompt, evidence, *, deadline, wall_deadline):
        self.start('diagnostic', deadline)
        self.delivered = evidence
        queries = [e for e in evidence if 'negative_records' in e.data]
        decisions = [NegativeDecision(e.evidence_id, r['index'], 'rejected', 'Observed failure applies')
                     for e in queries for r in e.data['negative_records']]
        return DiagnosticDecision('Isolate one retained case', tuple(e.evidence_id for e in evidence),
            DiagnosticProposal('one-case', 'Investigate documented behavior', 'Independent oracle passes',
                'cpu.case-subset', {'case_indices': [0]}, self.parent),
            tuple(decisions[:-1] if self.omit_negative else decisions))


@pytest.fixture
def fixture(tmp_path):
    clock = Clock()
    roles = Roles(clock)
    store = ContentAddressedArtifacts(tmp_path / 'artifacts')
    queried = []
    response = dict(results=[dict(text='Actual retained document fact', path='reference-skills/guide.md')],
                    negative_records=[dict(reason='Prior numeric mismatch', outcome='failed')],
                    provenance={'collection': 'frozen-docs'})
    def query(payload, *, deadline):
        queried.append((payload, deadline))
        clock.now += 3
        return response
    parent = ProjectSpec('original', 'Original unchanged goal', ('python', 'verify.py'))
    template = dict(template_id='cpu.case-subset', parent_project_id='original', parameters={
        'case_indices': dict(type='array', items={'type': 'integer', 'enum': [0, 1, 2]},
                             minItems=1, maxItems=2, uniqueItems=True)})
    context = dict(operation_id='decision-1', root=tmp_path, deadline=200.,
                   templates=[template], memory_ref='accepted-memory')
    researcher = Researcher(roles, query, store, clock=clock)
    return researcher, parent, context, roles, queried, response, clock, store


def test_actual_docs_and_negatives_precede_plan_and_setup_is_not_actor_time(fixture):
    researcher, parent, context, roles, queried, response, clock, store = fixture
    result = researcher.research(parent, {'outcome': 'budget_exhausted'}, context)
    assert roles.calls == ['research_queries', 'diagnostic']
    assert queried[0][0]['query'] == 'reduction reference'
    assert result.actor_seconds == 7  # Two measured roles plus actual retrieval; not 80 setup seconds.
    assert clock.now == 187
    assert queried[0][1] == 240
    evidence = next(e for e in roles.delivered if e.evidence_id == result.evidence_refs[0])
    assert evidence.data['results'][0]['facts']['text'] == response['results'][0]['text']
    assert evidence.data['negative_records'][0]['facts']['reason'] == response['negative_records'][0]['reason']
    retained = store.read_json(result.evidence_refs[0])
    assert retained['response'] == response
    assert store.read_json(result.receipt_ref)['status'] == 'accepted'
    clock.now = 10000
    again = researcher.research(parent, {'outcome': 'budget_exhausted'}, {**context, 'deadline': 10001})
    assert again == result and roles.calls == ['research_queries', 'diagnostic']
    assert parent.goal == 'Original unchanged goal'


def test_empty_research_and_unconsidered_negative_are_rejected_with_measured_receipts(fixture):
    researcher, parent, context, roles, queried, _, _, store = fixture
    roles.empty_queries = True
    with pytest.raises(ResearchRejected) as first:
        researcher.research(parent, {}, context)
    assert first.value.actor_seconds == 2 and not queried
    assert store.read_json(first.value.receipt_ref)['status'] == 'rejected'
    roles.empty_queries, roles.omit_negative = False, True
    with pytest.raises(ResearchRejected) as second:
        researcher.research(parent, {}, {**context, 'operation_id': 'decision-2'})
    assert second.value.actor_seconds == 7
    calls = list(roles.calls)
    with pytest.raises(ResearchRejected):
        researcher.research(parent, {}, {**context, 'operation_id': 'decision-2'})
    assert roles.calls == calls


def test_worker_unknown_is_durable_and_never_replayed(fixture):
    researcher, parent, context, roles, queried, _, _, store = fixture
    roles.unknown = True
    with pytest.raises(ResearchPending) as pending:
        researcher.research(parent, {}, context)
    assert pending.value.actor_seconds is None
    assert store.read_json(pending.value.receipt_ref)['status'] == 'pending'
    with pytest.raises(ResearchPending):
        researcher.research(parent, {}, context)
    assert roles.calls == ['research_queries'] and not queried


def test_only_exhausted_elapsed_returns_no_proposal(fixture):
    researcher, parent, context, roles, queried, _, clock, store = fixture
    result = researcher.research(parent, {}, {**context, 'deadline': clock.now})
    assert result.proposal is None and result.actor_seconds == 0
    assert store.read_json(result.receipt_ref)['status'] == 'budget_exhausted'
    assert not roles.calls and not queried


def test_selects_another_admitted_parent_with_its_template(fixture):
    researcher, parent, context, roles, _, _, _, _ = fixture
    other = replace(parent, project_id='other-failed-parent')
    roles.parent = other.project_id
    context['failed_parents'] = [dict(project_id=p.project_id, spec=p.to_dict(), failure={'outcome': 'failed'})
                                 for p in (parent, other)]
    context['templates'].append({**context['templates'][0], 'parent_project_id': other.project_id})
    result = researcher.research(parent, {}, context)
    assert result.proposal.parent_project_id == other.project_id
    assert len([e for e in roles.delivered if e.data.get('project_id') in (parent.project_id, other.project_id)]) == 2


@pytest.mark.parametrize('parameters', [{'case_indices': []}, {'case_indices': [0, 1, 2]},
    {'case_indices': [0, 0]}, {'case_indices': [99]}, {'command': 'arbitrary executable'}])
def test_model_cannot_exceed_host_advertised_template(fixture, parameters):
    _, _, context, _, _, _, _, _ = fixture
    proposal = DiagnosticProposal('child', 'goal', 'oracle', 'cpu.case-subset', parameters)
    with pytest.raises(ValueError):
        Researcher._validate_template(proposal, 'original', context['templates'])


def test_typed_role_wire_is_compatible_and_negative_reuse_needs_redirection():
    legacy = _proposal('curriculum', dict(rationale='Observed', evidence_ids=['failure'],
        projects=[dict(name='old', goal='goal', acceptance='oracle')]), {'failure'})
    assert legacy.projects[0].name == 'old'
    data = dict(rationale='Read docs', evidence_ids=['query'], proposal=dict(name='child', goal='goal',
        acceptance='oracle', template_id='cpu.case-subset', parameters={'case_indices': [0]}),
        negative_decisions=[dict(query_ref='query', index=0, action='used', reason='Change applicability')])
    with pytest.raises(ValueError):
        parse_proposal('diagnostic', data, {'query'})
    data['negative_decisions'][0]['redirection'] = 'Use a smaller isolated shape'
    assert _proposal('diagnostic', data, {'query'}).proposal.parameters == {'case_indices': [0]}


def test_historical_envelope_is_archival_but_actual_failure_proofs_are_retained(fixture):
    researcher, parent, context, roles, _, _, _, store = fixture
    old_event = 'f' * 64  # Kept in the original task, unavailable in this child CAS.
    envelope = store.put_json({'result': {'events': [old_event]}})
    with pytest.raises(ValueError):
        researcher.research(parent, {'artifact_ref': envelope}, context)
    assert not roles.calls
    proof = store.put_json({'stage': 'test', 'classification': 'numeric_failure'})
    failure = dict(outcome='failed', error='retained numeric mismatch', artifact_refs=[proof],
        origin_task_archive={'scope': 'origin-task-archive', 'service_result': envelope})
    result = researcher.research(parent, failure, context)
    from rsi.evidence_io import _references
    assert proof in set(_references(failure))
    assert envelope not in set(_references(failure))
    assert store.read_json(result.receipt_ref)['status'] == 'accepted'
