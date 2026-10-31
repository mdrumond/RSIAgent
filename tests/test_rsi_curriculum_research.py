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


def test_rank_fixed_shape_catalog_uses_generic_array_parameters(fixture):
    _, parent, _, _, _, _, _, _ = fixture
    template = dict(template_id='triton.shape-reduction.case-0', parent_project_id=parent.project_id,
        parameters={'shape': dict(type='array', items={'type': 'integer', 'enum': [1, 2, 4, 8]},
                                 minItems=2, maxItems=2, uniqueItems=False)})
    proposal = DiagnosticProposal('tiny', 'Investigate a tiny shape', 'Same independent oracle',
                                 template['template_id'], {'shape': [1, 1]})
    Researcher._validate_template(proposal, parent.project_id, [template])
    with pytest.raises(ValueError):
        Researcher._validate_template(replace(proposal, parameters={'shape': [1]}), parent.project_id, [template])


def test_real_nooa_typed_research_consumes_actual_retrieval_with_mock_transport(fixture):
    pytest.importorskip('nooa')
    from test_rsi_roles import make_roles
    from benchmarks.a5kernels.phase1_provider import DirectCompletion, DirectCompletionProvider
    _, parent, context, _, queried, _, clock, store = fixture
    from uuid import uuid4
    guidance = dict(curriculum=uuid4().hex, waves=uuid4().hex)
    context.update(user_guidance=guidance, configuration_ref=store.put_json({'prompts': guidance}),
                   skills_ref='retained-skills', skill_guidance={'debug': uuid4().hex},
                   kdb_guidance={'collections': ['frozen-docs']})
    roles, _, events, _ = make_roles([], clock=clock)
    def transport(**kwargs):
        clock.now += 1
        started = next(payload for event, payload in reversed(events) if event == 'role_started')
        evidence = started['evidence']
        assert evidence[0]['data']['user_guidance'] == guidance
        assert evidence[0]['data']['configuration_ref'] == context['configuration_ref']
        assert evidence[0]['data']['skills_ref'] == context['skills_ref']
        assert evidence[0]['data']['skill_guidance'] == context['skill_guidance']
        assert evidence[0]['data']['kdb_guidance'] == context['kdb_guidance']
        ids = [row['evidence_id'] for row in evidence]
        if started['role'] == 'research_queries':
            value = dict(rationale='Read retained documentation', evidence_ids=ids,
                queries=[dict(query='reduction constraints', purpose='Investigate the numeric failure')])
        else:
            retrieved = [r for r in evidence if 'negative_records' in r['data']]
            assert retrieved and retrieved[0]['data']['results'][0]['facts']['path'] == 'reference-skills/guide.md'
            value = dict(rationale='Isolate a documented case', evidence_ids=ids,
                proposal=dict(name='small', goal='Investigate one case', acceptance='Existing independent oracle',
                    template_id='cpu.case-subset', parameters={'case_indices': [0]}),
                negative_decisions=[dict(query_ref=r['evidence_id'], index=n['index'], action='rejected',
                    reason='The retained failure must not be repeated')
                    for r in retrieved for n in r['data']['negative_records']])
        return DirectCompletion(json.dumps(value), 20)
    roles.provider = DirectCompletionProvider({'DEEPSEEK_API_KEY': 'test-only-secret'},
        transport=transport, models=[roles.profile.identity.backend_model], clock=clock, invalid_prompt_retries=0)
    def query(payload, *, deadline):
        queried.append(payload)
        clock.now += 3
        return dict(results=[dict(path='reference-skills/guide.md', text='Retained documentation')],
                    negative_records=[dict(reason='Numeric mismatch')], provenance={'backend': 'TEST'})
    result = Researcher(roles, query, store, clock=clock).research(parent, {}, context)
    assert result.actor_seconds == 5 and len(queried) == 1
    assert [p['role'] for e, p in events if e == 'role_finished'] == ['research_queries', 'diagnostic']


def test_host_command_diagnostics_reach_research_and_child_despite_provider_abort(fixture):
    researcher, parent, context, roles, _, response, _, store = fixture
    compiler_error = 'AttributeError: compiler API build_kernel is unavailable'
    numeric = dict(case_id='frozen-case-2', max_absolute_error=0.125, expected=1.0, actual=0.875)
    observations = dict(schema='rsi-project-observations-v1', outcome='budget_exhausted',
        applicability={'project_id': parent.project_id, 'final_source_hash': 'a' * 64},
        commands=dict(items=[
            dict(command='compile', classification='execution_failure', returncode=1, numeric=False,
                 source_hash='a' * 64, final_source_scope=True,
                 stderr={'text': compiler_error}, stdout={'text': ''}),
            dict(command='test', classification='numeric_failure', returncode=1, numeric=True,
                 source_hash='a' * 64, final_source_scope=True,
                 stderr={'text': ''}, stdout={'text': json.dumps(numeric)})], total=2))
    observation_ref = store.put_json(observations)
    archive = store.put_json({'result': {'events': ['f' * 64]}})
    failure = dict(error='provider aborted after command execution', outcome='budget_exhausted',
        source_hash='a' * 64, observations_ref=observation_ref,
        origin_task_archive={'scope': 'origin-task-archive', 'service_result': archive})
    response['negative_records'] = [dict(error=failure['error'], observations_ref=observation_ref)]
    selected_inputs = []
    original_queries = roles.research_queries
    def select_query(prompt, evidence, **kwargs):
        selected_inputs.extend(evidence)
        plan = original_queries(prompt, evidence, **kwargs)
        facts = evidence[0].data['failure']
        assert facts['error'] == failure['error']
        commands = facts['observations']['commands']['items']
        assert commands[0]['stderr']['text'] == compiler_error
        assert commands[1]['structured_stdout'] == numeric
        return replace(plan, queries=(ResearchQuery(commands[0]['stderr']['text'], 'Investigate the host API error'),))
    roles.research_queries = select_query
    result = researcher.research(parent, failure, context)
    received = next(e for e in roles.delivered if 'negative_records' in e.data)
    negative = received.data['negative_records'][0]['facts']['observations']
    assert negative['commands']['items'][1]['structured_stdout']['case_id'] == numeric['case_id']
    assert negative['artifact_ref'] == observation_ref
    assert store.read_json(observation_ref) == observations
    from rsi.evidence_io import _references
    assert observation_ref in set(_references(selected_inputs[0].data))
    assert archive not in set(_references(selected_inputs[0].data))
    child = researcher.actor_context(result.receipt_ref)
    assert len(json.dumps(child).encode()) <= 2500
    assert child['rationale'] == 'Isolate one retained case'
    assert child['documents'][0]['facts']['text'] == response['results'][0]['text']
    assert child['commands'][0]['stderr']['text'] == compiler_error
    assert child['commands'][1]['structured_stdout']['max_absolute_error'] == numeric['max_absolute_error']


@pytest.mark.parametrize('missing', [True, False])
def test_missing_or_non_observation_payload_never_starts_research(fixture, missing):
    researcher, parent, context, roles, _, _, _, store = fixture
    reference = 'e' * 64 if missing else store.put_json({'result': {'events': ['f' * 64]}})
    with pytest.raises(ResearchRejected) as error:
        researcher.research(parent, {'observations_ref': reference}, context)
    assert error.value.actor_seconds == 0 and not roles.calls


def test_runtime_curriculum_guidance_reaches_both_typed_stages_and_is_frozen(fixture):
    from uuid import uuid4
    researcher, parent, context, roles, _, _, _, store = fixture
    guidance = {'curriculum': uuid4().hex, 'waves': uuid4().hex}
    context.update(user_guidance=guidance, configuration_ref=store.put_json({'prompts': guidance}))
    seen = []
    for name in ('research_queries', 'diagnostic'):
        original = getattr(roles, name)
        def receive(prompt, evidence, _original=original, **kwargs):
            seen.append(evidence[0].data['user_guidance'])
            assert evidence[0].data['configuration_ref'] == context['configuration_ref']
            return _original(prompt, evidence, **kwargs)
        setattr(roles, name, receive)
    result = researcher.research(parent, {}, context)
    assert seen == [guidance, guidance]
    request = store.read_json(store.read_json(result.receipt_ref)['request_ref'])
    assert request['context']['user_guidance'] == guidance
    with pytest.raises(ValueError):
        researcher.research(parent, {}, {**context, 'user_guidance': {'curriculum': uuid4().hex}})
