"""Native role proposals, bounded evidence and authoritative settlement behavior."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from coding.pi.native import NativeError, NativeSettlement
from core.artifacts import ContentAddressedArtifacts
from rsi.roles import AdmittedEvidence, NativeRoles, RoleFailure, _proposal


class Clock:
    now = 100.
    def __call__(self):
        return self.now


class Files:
    def __init__(self, root, *, store, read_paths):
        self.paths = read_paths
    def session_options(self):
        return {}


def fixture_roles(tmp_path, *, responses=None, fault=None, settled=True, setup=40.):
    clock, events, sessions = Clock(), [], []
    store = ContentAddressedArtifacts(tmp_path/'artifacts')
    profile = SimpleNamespace(identity=SimpleNamespace(profile_id='deepseek-flash-native-v1',
                                                       credential_env='DEEPSEEK_API_KEY'))
    class Session:
        provider_settled = settled
        pending_tools = set() if settled else {'pending-native-call'}
        session_file = 'native-session.jsonl'
        usage = []
        def __init__(self, root, **kwargs):
            self.root, self.options = root, kwargs
            self.closed = False
            sessions.append(self)
        def prepare(self, timeout):
            clock.now += setup
        def prompt(self, text, *, operation_id, remaining):
            self.prompt_text = text
            self.options['observer'](dict(type='agent_start'))
            clock.now += 2
            if fault:
                if fault is KeyboardInterrupt:
                    raise KeyboardInterrupt()
                raise fault('actual failure', settlement=NativeSettlement('failed', settled,
                    tuple(self.pending_tools), self.session_file, ()))
            for data in responses or []:
                try:
                    self.result = self.options['tools'][0].callback(data,
                        dict(operation_id=operation_id, tool_call_id='native-proposal', remaining=remaining))
                except ValueError as exc:
                    self.options['observer'](dict(type='host_callback_error', message=str(exc)))
                    continue
                assert self.result['terminate'] is True
                break
            return NativeSettlement('settled', True, (), self.session_file, ())
        def close(self):
            self.closed = True
    roles = NativeRoles(tmp_path/'roles', profile, observer=lambda *event: events.append(event),
        environment={'DEEPSEEK_API_KEY': 'fixture-secret'}, store=store, operation_id=lambda: 'operation',
        session_factory=Session, files_factory=Files, clock=clock)
    return roles, clock, events, sessions, store


IDEA = dict(question='Which input isolates the mismatch?', goal='Investigate a smaller input',
            why='Distinguish compilation from numerical behavior')


def test_semantic_proposal_uses_native_tool_and_setup_is_not_actor_time(tmp_path):
    roles, clock, events, sessions, store = fixture_roles(tmp_path, responses=[IDEA])
    idea = roles.investigation('Read documentation before proposing.',
        [AdmittedEvidence('failure', {'error': 'actual mismatch'})], deadline=110)
    assert asdict(idea) == IDEA and clock.now == 142
    finished = [data for event, data in events if event == 'role_finished']
    assert finished == [dict(role='investigation', success=True, elapsed_seconds=2)]
    started = next(data for event, data in events if event == 'role_started')
    assert store.read_json(started['request_ref'])['evidence'][0]['data']['error'] == 'actual mismatch'
    assert all(session.closed for session in sessions)
    assert sessions[0].options['tools'][0].name == 'submit_proposal'
    assert 'fixture-secret' not in json.dumps(events)


@pytest.mark.parametrize('role,data,field', [
    ('curriculum', dict(rationale='Useful next project', evidence_ids=['fact'],
        projects=[dict(name='tiny', goal='Diagnose', acceptance='Independent oracle passes')]), 'projects'),
    ('skillwriter', dict(rationale='Retained failure', evidence_ids=['fact'],
        edits=[dict(skill='debug', operation='append', content='Investigate before changing code')]), 'edits'),
    ('review', dict(rationale='No regression', evidence_ids=['fact'], findings=[]), 'findings'),
])
def test_typed_callers_keep_domain_values_without_applying_them(tmp_path, role, data, field):
    roles, _, _, _, _ = fixture_roles(tmp_path, responses=[data])
    result = getattr(roles, role)('Perform the role', [AdmittedEvidence('fact', {})], deadline=110)
    assert asdict(result)[field] == tuple(data[field])
    assert not list(tmp_path.glob('*.py'))


def test_native_validation_feedback_can_precede_one_committed_proposal(tmp_path):
    roles, _, events, sessions, _ = fixture_roles(tmp_path,
        responses=[dict(IDEA, evidence_ids=['invented']), IDEA])
    assert asdict(roles.investigation('Investigate', [], deadline=110)) == IDEA
    assert len([e for e, _ in events if e == 'role_proposal']) == 1
    assert sessions[0].result['terminate']
    assert len([e for e, _ in events if e == 'role_native_event']) == 2


def test_large_evidence_retained_as_files_without_full_lifecycle_echo(tmp_path):
    evidence = {'text': 'document line\n'*400000, 'tail': 'important final diagnostic'}
    roles, _, events, sessions, store = fixture_roles(tmp_path, responses=[IDEA])
    roles.investigation('Read the full admitted document.', [AdmittedEvidence('document', evidence)], deadline=110)
    started = next(data for event, data in events if event == 'role_started')
    manifest = store.read_json(started['evidence_manifest_ref'])
    assert len(manifest) == 1 and manifest[0]['bytes'] > 4_000_000
    full = Path(manifest[0]['path']).read_bytes()
    assert full == store.read_bytes(manifest[0]['reference'])
    assert json.loads(full)['data'] == evidence
    assert len(json.dumps(events)) < 4096 and len(sessions[0].prompt_text) < 1024


@pytest.mark.parametrize('settled', [True, False])
def test_known_failure_bills_once_and_unknown_never_replays(tmp_path, settled):
    roles, _, events, sessions, _ = fixture_roles(tmp_path, fault=NativeError, settled=settled)
    with pytest.raises(NativeError):
        roles.investigation('Investigate', [], deadline=110)
    before = len(events)
    with pytest.raises(RoleFailure if settled else NativeError):
        roles.investigation('Investigate', [], deadline=200)
    assert len(sessions) == 1 and sessions[0].closed
    finished = [row for event, row in events if event == 'role_finished']
    assert [row['elapsed_seconds'] for row in finished] == ([2.] if settled else [])
    if settled:
        assert len(events) == before
    else:
        assert events[-1][0] == 'accounting_incomplete'
        assert events[-1][1]['start_receipt_received']


def test_completed_operation_returns_retained_domain_value_without_dispatch(tmp_path):
    roles, _, events, sessions, _ = fixture_roles(tmp_path, responses=[IDEA])
    first = roles.investigation('Investigate', [], deadline=110)
    before = len(events)
    assert roles.investigation('Investigate', [], deadline=200) == first
    assert len(sessions) == 1 and len(events) == before
    with pytest.raises(ValueError):
        roles.investigation('Different instructions', [], deadline=200)


@pytest.mark.parametrize('settled', [True, False])
def test_interrupt_reports_only_observed_native_settlement(tmp_path, settled):
    roles, _, events, _, _ = fixture_roles(tmp_path, fault=KeyboardInterrupt, settled=settled)
    with pytest.raises(KeyboardInterrupt):
        roles.investigation('Investigate', [], deadline=110)
    assert any(event == 'role_finished' for event, _ in events) is settled
    assert any(event == 'accounting_incomplete' for event, _ in events) is not settled


def test_settled_without_proposal_is_closed_failure(tmp_path):
    roles, _, events, _, _ = fixture_roles(tmp_path)
    with pytest.raises(RoleFailure):
        roles.investigation('Investigate', [], deadline=110)
    assert events[-1][0] == 'role_finished' and events[-1][1]['success'] is False


def test_deadline_duplicate_evidence_and_secret_are_rejected_before_session(tmp_path):
    roles, _, _, sessions, _ = fixture_roles(tmp_path)
    for evidence, deadline, prompt, error in [
        ([], 100, 'Investigate', TimeoutError),
        ([AdmittedEvidence('x', {}), AdmittedEvidence('x', {})], 110, 'Investigate', ValueError),
        ([], 110, 'fixture-secret', ValueError),
    ]:
        with pytest.raises(error):
            roles.investigation(prompt, evidence, deadline=deadline)
    assert not sessions


def test_legacy_diagnostic_reader_preserves_conditional_redirection_contract():
    data = dict(rationale='Read docs', evidence_ids=['query'], proposal=dict(name='child', goal='goal',
        acceptance='oracle', template_id='cpu.case-subset', parameters={'case_indices': [0]}),
        negative_decisions=[dict(query_ref='query', index=0, action='used', reason='Changed applicability')])
    with pytest.raises(ValueError):
        _proposal('diagnostic', data, {'query'})
    data['negative_decisions'][0]['redirection'] = 'Use a smaller admitted input'
    assert _proposal('diagnostic', data, {'query'}).proposal.parameters == {'case_indices': [0]}
