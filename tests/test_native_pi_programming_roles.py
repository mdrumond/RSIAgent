"""Actual native SDK proposal/verification roles; no legacy completion transport."""
import hashlib
import json

import pytest

from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
from benchmarks.a5kernels.phase1_memory import parse_agent_distillation
from benchmarks.a5kernels.phase1_provider import direct_profile
from benchmarks.ascend.native_curriculum import NativeSubjectRole
from benchmarks.ascend.native_review import PiReviewSession
from benchmarks.ascend.pi_project_verifier import SCHEMAS
from coding.pi.native import NativeError
from core.artifacts import canonical_bytes
from test_native_pi_filesystem import scripted_provider
from test_project_verifier_agent import Harness, request, decision, memory, S, R, L


def registered_profile():
    return direct_profile(next(item for item in MODEL_IDENTITIES
                               if item.backend_model.value == 'deepseek-flash'))


def redirect(monkeypatch, url):
    import coding.pi.native as native
    profile = native.model_profile
    monkeypatch.setattr(native, 'model_profile', lambda identity: dict(profile(identity), base_url=url))


def test_native_terminal_role_reads_context_repairs_domain_error_and_reopens(tmp_path, monkeypatch, scripted_provider):
    url, calls, requests = scripted_provider
    redirect(monkeypatch, url)
    messages = [dict(role='user', content='Distill only supported lessons; no new evidence established.')]
    role = NativeSubjectRole(tmp_path, registered_profile(), {'DEEPSEEK_API_KEY':'fixture-secret'},
        validator=lambda value:parse_agent_distillation(json.dumps(value), {}))
    identity = hashlib.sha256(canonical_bytes(dict(messages=messages, wave=False, model=role.model))).hexdigest()
    valid = dict(distillation=dict(no_new_knowledge=True, lessons=[]))
    calls.extend([('read', dict(path=str(tmp_path/identity/'context.json'))),
        ('submit_curriculum_proposal', dict(value=dict(distillation=dict(no_new_knowledge=False, lessons=[])))),
        ('submit_curriculum_proposal', dict(value=valid))])
    assert json.loads(role(messages, remaining_seconds=30)) == valid
    assert len(requests) == 3
    assert role.last_receipt['actor_seconds'] > 0
    assert all('output' in item for item in role.last_receipt['usage'])
    assert any(message['role']=='tool' and 'no_new_knowledge' in message['content']
               for message in requests[-1]['messages'])
    assert json.loads(role(messages, remaining_seconds=30)) == valid
    assert len(requests) == 3  # The completed durable proposal is never replayed.
    closure = json.loads((tmp_path/identity/'closure.json').read_text())
    assert closure['provider_settled'] and not closure['pending_tools'] and closure['process_exit'] == 0
    assert closure['artifact_refs']


def test_retained_unknown_curriculum_intent_does_not_start_native_session(tmp_path):
    role = NativeSubjectRole(tmp_path, registered_profile(), {'DEEPSEEK_API_KEY':'fixture-secret'})
    messages = [dict(role='user', content='Unknown retained role request')]
    identity = hashlib.sha256(canonical_bytes(dict(messages=messages, wave=False, model=role.model))).hexdigest()
    root = tmp_path/identity
    root.mkdir()
    (root/'operation.json').write_text(json.dumps(dict(status='pending')))
    with pytest.raises(NativeError):
        role(messages, remaining_seconds=30)
    assert not (root/'session').exists()


def test_native_review_reads_actual_artifacts_queries_and_admits_memory(tmp_path, monkeypatch, scripted_provider):
    url, calls, requests = scripted_provider
    redirect(monkeypatch, url)
    harness = Harness([])
    session = PiReviewSession(model='deepseek-flash', schemas=SCHEMAS,
        environ={'DEEPSEEK_API_KEY':'fixture-secret'}, root=tmp_path)
    calls.append(('submit_review', decision(memory())))  # Actual missing-read admission failure.
    calls.extend([('read', dict(path=str(session.root/'evidence'/(reference+'.txt'))))
                  for reference in (S, R, L)])
    calls.extend([('knowledge_query', dict(query='increment overflow lessons')),
                  ('submit_review', decision(memory()))])
    harness.agent.session_factory = lambda **kwargs: session
    assert harness.agent.review(request(), max_seconds=30) == decision(memory())
    assert {reference for reference, _ in harness.reads} >= {S, R, L}
    assert harness.queries and not harness.calls  # No legacy text completion path.
    observed = [row for row in harness.agent.last_trace if row['kind']=='native-review-read']
    assert {row['payload']['sha256'] for row in observed} == {S, R, L}
    assert len(requests) == 6
    retained = next(row['payload'] for row in harness.agent.last_trace if row['kind']=='native-review-session')
    assert retained['artifact_refs']
