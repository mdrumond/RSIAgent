"""Semantic curriculum decisions leave machine binding and provenance to the host."""
import json
from dataclasses import replace

import pytest

from core.artifacts import ContentAddressedArtifacts
from rsi.contracts import ProjectSpec
from rsi.curriculum_research import (InvestigationIdea, Researcher, ResearchPending,
                                    ResearchRejected)
from rsi.roles import _proposal


class Clock:
    now = 100.
    def __call__(self):
        return self.now


class Roles:
    def __init__(self, clock):
        self.clock = clock
        self.observer = lambda *_: None
        self.calls = []
        self.unknown = False
        self.error = None

    def investigation(self, prompt, evidence, *, deadline, wall_deadline):
        self.calls.append((prompt, evidence, deadline))
        self.clock.now += 40  # Worker setup is separate from measured actor time.
        self.observer('role_started', {'role': 'investigation'})
        if self.unknown:
            raise RuntimeError('completion unavailable')
        self.clock.now += 2
        self.observer('role_finished', dict(role='investigation', elapsed_seconds=2,
                                           success=self.error is None))
        if self.error:
            raise self.error
        return InvestigationIdea('Which small case isolates the mismatch?',
                                 'Investigate the observed discrepancy', 'Separate behavior from setup')


@pytest.fixture
def fixture(tmp_path):
    clock = Clock()
    roles = Roles(clock)
    store = ContentAddressedArtifacts(tmp_path / 'artifacts')
    calls = []
    response = dict(results=[dict(text='Actual document result', path='guide.md')],
                    negative_records=[dict(reason='Observed numeric mismatch', outcome='failed')],
                    provenance={'collection': 'latest-memory-and-docs'})
    def query(payload, *, deadline):
        assert not roles.calls  # Actual retrieval precedes the semantic decision.
        calls.append((payload, deadline))
        clock.now += 3
        return response
    parent = ProjectSpec('original', 'Preserved original goal', ('python', 'verify.py'))
    template = dict(template_id='cpu.case-subset', parent_project_id='original', parameters={
        'case_indices': dict(type='array', items={'type': 'integer', 'enum': [0, 1, 2]},
                             minItems=1, maxItems=2, uniqueItems=True)})
    context = dict(operation_id='decision-1', root=tmp_path, deadline=200.,
                   templates=[template], memory_ref='latest-memory')
    return Researcher(roles, query, store, clock=clock), parent, context, roles, calls, response, clock, store


def test_retrieval_then_semantic_idea_host_binding_retention_and_idempotent_resume(fixture):
    researcher, parent, context, roles, calls, response, clock, store = fixture
    result = researcher.research(parent, {'error': 'actual numeric failure'}, context)
    assert len(roles.calls) == len(calls) == 1
    assert json.loads(calls[0][0]['query'])['failure']['error'] == 'actual numeric failure'
    assert result.actor_seconds == 5
    assert clock.now == 145
    assert result.proposal.parent_project_id == parent.project_id
    assert result.proposal.template_id == 'cpu.case-subset'
    assert result.proposal.parameters == {'case_indices': [0]}
    assert store.read_json(result.evidence_refs[0])['response'] == response
    shown = roles.calls[0][1][-1].data
    assert shown['results'][0]['facts']['text'] == response['results'][0]['text']
    assert shown['negative_records'][0]['facts']['reason'] == response['negative_records'][0]['reason']
    receipt = store.read_json(result.receipt_ref)
    events = [store.read_json(ref) for ref in receipt['artifact_refs']]
    binding = next(e['payload'] for e in events if e.get('event') == 'investigation_bound')
    assert binding['negative_coverage'] == [dict(query_ref=result.evidence_refs[0], index=0)]
    clock.now = 10000
    assert researcher.research(parent, {'error': 'actual numeric failure'},
                               {**context, 'deadline': 10001}) == result
    assert len(roles.calls) == len(calls) == 1
    assert parent.goal == 'Preserved original goal'
    assert researcher.actor_context(result.receipt_ref)['rationale'] == binding['idea']['why']


def test_runtime_guidance_and_prior_feedback_reach_semantic_input(fixture):
    researcher, parent, context, roles, _, _, _, store = fixture
    guidance = {'curriculum': 'Investigate operator-selected behavior', 'waves': 'Preserve original cases'}
    prior = store.put_json(dict(status='rejected', error='retained actual earlier error'))
    researcher.research(parent, {}, {**context, 'user_guidance': guidance, 'previous_rejections': [prior]})
    evidence = roles.calls[0][1]
    assert evidence[0].data['user_guidance'] == guidance
    assert next(e for e in evidence if e.evidence_id == prior).data['error'] == 'retained actual earlier error'


def test_hot_repeated_negative_facts_need_no_model_ledger(fixture):
    researcher, parent, context, roles, _, response, _, store = fixture
    response['negative_records'] *= 8
    result = researcher.research(parent, {}, context)
    assert result.proposal is not None and len(roles.calls) == 1
    receipt = store.read_json(result.receipt_ref)
    event = next(store.read_json(r) for r in receipt['artifact_refs']
                 if store.read_json(r).get('event') == 'investigation_bound')
    assert len(event['payload']['negative_coverage']) == 8


def test_unknown_completion_retains_known_query_cost_but_never_replays(fixture):
    researcher, parent, context, roles, calls, _, _, store = fixture
    roles.unknown = True
    with pytest.raises(ResearchPending) as error:
        researcher.research(parent, {}, context)
    assert error.value.actor_seconds is None
    receipt = store.read_json(error.value.receipt_ref)
    assert receipt['measured_actor_seconds'] == 3
    with pytest.raises(ResearchPending):
        researcher.research(parent, {}, context)
    assert len(roles.calls) == len(calls) == 1


def test_closed_role_error_is_retained_with_actual_elapsed(fixture):
    researcher, parent, context, roles, _, _, _, store = fixture
    roles.error = ValueError('actual structured role failure')
    with pytest.raises(ResearchRejected) as error:
        researcher.research(parent, {}, context)
    assert error.value.actor_seconds == 5
    assert store.read_json(error.value.receipt_ref)['error'] == str(roles.error)


def test_elapsed_exhaustion_after_query_starts_no_role(fixture):
    researcher, parent, context, roles, calls, _, _, store = fixture
    result = researcher.research(parent, {}, {**context, 'deadline': 102})
    assert result.proposal is None and result.actor_seconds == 3
    assert calls and not roles.calls
    assert store.read_json(result.receipt_ref)['status'] == 'budget_exhausted'


def test_no_capability_is_honest_closed_rejection(fixture):
    researcher, parent, context, _, _, _, _, _ = fixture
    with pytest.raises(ResearchRejected) as error:
        researcher.research(parent, {}, {**context, 'templates': []})
    assert error.value.actor_seconds == 5


def test_controller_prefers_registered_tiny_shape_over_case_subset(fixture):
    researcher, parent, context, _, _, _, _, _ = fixture
    shape = dict(template_id='triton.shape-reduction.case-0', parent_project_id=parent.project_id,
                 shape_constraints={'maximum_per_dimension': [32, 8], 'at_least_one_strictly_smaller': True},
                 parameters={'shape': dict(type='array', minItems=2, maxItems=2, uniqueItems=False,
                                          items={'type': 'integer', 'enum': [1, 2, 4, 8, 32]})})
    result = researcher.research(parent, {}, {**context, 'templates': [*context['templates'], shape]})
    assert result.proposal.template_id == shape['template_id']
    assert result.proposal.parameters == {'shape': [1, 1]}


def test_changed_request_cannot_reuse_completed_binding(fixture):
    researcher, parent, context, _, _, _, _, _ = fixture
    researcher.research(parent, {}, context)
    with pytest.raises(ValueError):
        researcher.research(replace(parent, goal='Changed logical task'), {}, context)


def test_historical_service_archive_is_not_a_required_child_payload(fixture):
    researcher, parent, context, roles, _, _, _, store = fixture
    archive = 'a' * 64
    result = researcher.research(parent, {'origin_task_archive': {'service_result': archive}}, context)
    assert result.proposal is not None
    other = {**context, 'operation_id': 'unsafe-envelope'}
    with pytest.raises(ValueError):
        researcher.research(parent, {'result': {'events': ['unavailable-history']}}, other)


@pytest.mark.parametrize('data', [
    {'question': 'What?', 'goal': 'Investigate', 'why': 'Observed mismatch', 'evidence_ids': []},
    {'question': '', 'goal': 'Investigate', 'why': 'Observed mismatch'},
])
def test_semantic_wire_rejects_machine_bookkeeping_or_empty_intent(data):
    with pytest.raises(ValueError):
        _proposal('investigation', data, set())


def test_semantic_wire_requires_no_machine_identifiers():
    idea = _proposal('investigation', dict(question='Which smaller case?', goal='Isolate behavior',
                                          why='Separate two possible causes'), set())
    assert idea == InvestigationIdea('Which smaller case?', 'Isolate behavior', 'Separate two possible causes')
