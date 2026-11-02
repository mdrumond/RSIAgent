"""Elapsed audits retain incremental evidence without inventing closure."""
import json

import pytest

from core.artifacts import ContentAddressedArtifacts
from rsi.event_journal import EventJournal
from rsi.python_execution import atomic_json
from rsi.supervision import ProjectSupervision, RoleSupervision
from test_native_pi_session import provider


@pytest.fixture
def project(tmp_path):
    store = ContentAddressedArtifacts(tmp_path / 'artifacts')
    contract = store.put_json(dict(spec=dict(project_id='affected-project')))
    atomic_json(tmp_path / 'runner.json', dict(contract=contract))
    atomic_json(tmp_path / 'workflow.json', dict(stage='implementation', source='frozen-source',
        pending=dict(id='operation', handle='original-handle')))
    return tmp_path, store, EventJournal(tmp_path)


def append(store, journal, event):
    reference = store.put_json(event)
    journal.append(reference)
    return reference


def test_elapsed_audit_carries_across_resume_and_retains_only_complete_new_records(project):
    root, store, journal = project
    now = [0.]
    audit = ProjectSupervision(root, clock=lambda: now[0])
    assert audit.poll() is None
    path = root / 'native' / 'session' / 'native-events.jsonl'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"type":"first"}\n{"type":')
    first = append(store, journal, dict(type='host-command', classification='compile_failure'))
    now[0] = 599
    assert audit.poll() is None
    now[0] = 600
    resumed = ProjectSupervision(root, clock=lambda: now[0])
    reference = resumed.poll()
    report = store.read_json(reference)
    assert report['host_refs'] == [first]
    assert [store.read_bytes(ref) for ref in report['native_refs']] == [b'{"type":"first"}\n']
    assert report['elapsed_since_audit'] == 600 and report['closure'] == 'unknown'
    assert resumed.poll() is None
    with path.open('ab') as stream:
        stream.write(b'"second"}\n')
    second = append(store, journal, dict(type='host-command', classification='numeric_failure'))
    now[0] = 1200
    report = store.read_json(resumed.poll())
    assert report['host_refs'] == [second]
    assert [store.read_bytes(ref) for ref in report['native_refs']] == [b'{"type":"second"}\n']
    assert resumed.terminal_audit(json.loads((root / 'workflow.json').read_text())['pending']) is None


@pytest.mark.parametrize('origin', ['controller', 'protocol', 'transport'])
def test_typed_fault_reports_immediately_without_claiming_native_or_external_closure(project, origin):
    root, store, journal = project
    event = dict(type='pi-native-event', event=dict(type='native_fault', fault_id='fault',
        origin=origin, kind='eof' if origin == 'transport' else 'fatal', operation_id='operation',
        session_file='retained-session', settlement=dict(provider_settled=False, pending_tools=['tool'])))
    reference = append(store, journal, event)
    audit = ProjectSupervision(root, clock=lambda: 1.)
    report_ref = audit.poll(fault_ref=reference)
    report = store.read_json(report_ref)
    assert report['fault_ref'] == reference and report['fault'] == event['event']
    assert report['closure'] == 'unknown' and report['replay_allowed'] is False
    assert audit.poll(fault_ref=reference) == report_ref
    state = json.loads(audit.path.read_text())
    assert state['reports'] == [report_ref]
    pending = json.loads((root / 'workflow.json').read_text())['pending']
    assert audit.terminal_audit(pending) is None  # A fault alone is never terminal authority.
    assert audit.diagnostic()['audit_ref'] == report_ref
    assert pending['handle'] == 'original-handle'


def test_ordinary_tool_errors_and_wrong_operation_cannot_become_harness_facts(project):
    root, store, journal = project
    audit = ProjectSupervision(root, clock=lambda: 1.)
    ordinary = append(store, journal, dict(type='pi-native-event', event=dict(
        type='host_callback_error', origin='host', error_type='FileNotFoundError',
        message='ordinary missing model-selected path')))
    with pytest.raises(ValueError):
        audit.poll(fault_ref=ordinary)
    foreign = append(store, journal, dict(type='pi-native-event', event=dict(type='native_fault',
        fault_id='other', operation_id='another-operation', origin='controller')))
    with pytest.raises(ValueError):
        audit.poll(fault_ref=foreign)
    assert audit.diagnostic() is None


def test_role_wait_without_project_runner_keeps_timer_and_cursors_across_resume(tmp_path):
    store = ContentAddressedArtifacts(tmp_path / 'artifacts')
    root = tmp_path / 'role'
    root.mkdir()
    scope = dict(cycle_id='cycle', role='skillwriter', operation_id='role-operation', request_ref='request')
    atomic_json(root / 'role.json', dict(scope, status='pending'))
    rows = [dict(operation_id='role-operation', event='role_started', payload=dict(role='skillwriter'))]
    now = [0.]
    audit = RoleSupervision(root, store=store, scope=scope, host_rows=rows, clock=lambda: now[0])
    assert audit.poll() is None
    path = root / 'native' / 'native-events.jsonl'
    path.parent.mkdir()
    path.write_bytes(b'{"type":"provider_stream_event"}\n')
    now[0] = 600
    resumed = RoleSupervision(root, store=store, scope=scope, host_rows=rows, clock=lambda: now[0])
    report = store.read_json(resumed.poll())
    assert report['scope'] == scope
    assert [store.read_json(ref) for ref in report['host_refs']] == rows
    assert store.read_bytes(report['native_refs'][0]) == path.read_bytes()
    assert report['closure'] == 'role_lifecycle_authority'
    assert not (root / 'workflow.json').exists() and not (root / 'runner.json').exists()
    assert resumed.poll() is None
    with pytest.raises(ValueError):
        resumed.terminal_audit({'id': 'role-operation'})


def test_role_typed_fault_reports_without_project_classification(tmp_path):
    store = ContentAddressedArtifacts(tmp_path / 'artifacts')
    scope = dict(cycle_id='cycle', role='curriculum', operation_id='role-operation', request_ref='request')
    atomic_json(tmp_path / 'role.json', dict(scope, status='pending'))
    fact = dict(type='native_fault', fault_id='role-fault', origin='protocol',
                operation_id='role-operation', settlement=dict(provider_settled=False, pending_tools=[]))
    reference = store.put_json(fact)
    rows = [dict(event='role_native_event', payload=dict(artifact_ref=reference))]
    audit = RoleSupervision(tmp_path, store=store, scope=scope, host_rows=rows, clock=lambda: 1.)
    report_ref = audit.poll(fault_ref=reference)
    assert store.read_json(report_ref)['fault'] == fact
    assert audit.poll(fault_ref=reference) == report_ref
    assert json.loads((tmp_path / 'role.json').read_text())['status'] == 'pending'


def test_canonical_failed_runner_authenticates_audit_only_after_known_closure(tmp_path):
    from coding.pi.native import NativeError
    from test_rsi_project_runner import Session, make
    class Fault(Session):
        def configure(self, runner):
            self.runner = runner
        def next_action(self, context, allowed, **kwargs):
            if context['stage'] == 'design' and any(row['name'] == 'query_kdb' for row in context['evidence']):
                self.runner.actor_event('pi-native-event', dict(event=dict(type='native_fault',
                    fault_id='actual-controller-fault', origin='controller', kind='fatal',
                    operation_id=self.runner.workflow.pending['id'], session_file='test-native-session',
                    settlement=dict(provider_settled=False, pending_tools=[]))))
                raise NativeError('explicit trusted controller fault fixture')
            return super().next_action(context, allowed, **kwargs)
    runner, _, database = make(tmp_path, Fault())
    try:
        result = runner.run()
        assert result['status'] == 'suspended' and result['actor_closure_ref']
        proof = runner.terminal_failure_proof()
        assert proof['classification'] == 'harness_failure'
        audit = runner.store.read_json(proof['audit_ref'])
        assert audit['contract_ref'] == runner.contract_ref
        assert audit['operation'] == runner.workflow.pending
        assert proof['pending'] == runner.workflow.pending
        assert proof['replay_allowed'] is False and proof['stage_completion_claim'] is False
        assert proof['retrieval_receipts']
        from rsi.job_inspection import inspect_task
        unknown_task = inspect_task(runner.root)
        assert unknown_task['supervision']['closure'] == 'unknown'
        from rsi.supervision import retain_terminal
        retain_terminal(runner, proof)
        task = inspect_task(runner.root)
        assert task['supervision']['audit_ref'] == proof['audit_ref']
        assert task['supervision']['closure'] == 'known'
        assert task['supervision']['audit_closure'] == 'unknown'
        role_root = runner.root / 'role-wait'
        role_root.mkdir()
        scope = dict(cycle_id='actual-cycle', operation_id='writer-wait', role='skillwriter',
                     request_ref=runner.store.put_json({'actual': 'role-input'}))
        atomic_json(role_root / 'role.json', dict(scope, status='pending'))
        now = [0.]
        role = RoleSupervision(role_root, store=runner.store, scope=scope, host_rows=[], clock=lambda: now[0])
        role.poll()
        now[0] = 600
        role.poll()
        assert_dashboard_detail(tmp_path, unknown_task, role.diagnostic())
        assert_dashboard_detail(tmp_path, task, role.diagnostic())
    finally:
        database.close()


def assert_dashboard_detail(tmp_path, task, role):
    """Real HTTP projection plus the existing browser detail-view handler."""
    from flask import Flask
    from types import SimpleNamespace
    from dashboard.rsi_execution import register_rsi_execution
    import shutil
    import subprocess
    from pathlib import Path
    job = dict(job_id=role['scope']['request_ref'], state='running',
        request=dict(cycle_id=role['scope']['cycle_id'], mode='phase1'))
    evidence = dict(job=job, inspection=dict(tasks=[task], waves=[], role_supervision=[role]))
    app = Flask(__name__)
    register_rsi_execution(app, SimpleNamespace(handlers={}, enabled=False), 'test', lambda *args: evidence)
    response = app.test_client().get('/api/rsi/jobs/' + job['job_id'])
    assert response.status_code == 200
    detail = response.get_json()
    projected = detail['inspection']['tasks'][0]['supervision']
    assert projected.get('terminal_failure_ref') == task['supervision'].get('terminal_failure_ref')
    assert projected['audit_ref'] == task['supervision']['audit_ref']
    assert projected['closure'] == task['supervision']['closure']
    assert projected['audit_closure'] == 'unknown'
    assert detail['inspection']['role_supervision'][0] == role
    node = shutil.which('node')
    assert node, 'Browser behavior validation requires the installed Node runtime'
    path = tmp_path / 'http-detail.json'
    path.write_text(json.dumps(detail))
    script = r"""
const assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
const detail=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
class E {constructor(){this.children=[];this.listeners={};this.dataset={};this.value='';}
addEventListener(k,f){this.listeners[k]=f;} append(...n){this.children.push(...n);}
replaceChildren(...n){this.children=n;}}
const nodes=new Map(),get=id=>{if(!nodes.has(id))nodes.set(id,new E());return nodes.get(id);};
get('phase-form').elements={operation:new E()};get('rsi-execution').dataset.enabled='true';
const context={document:{getElementById:get,createElement:()=>new E(),body:{dataset:{}}},
Option:class extends E{constructor(t,v){super();this.textContent=t;this.value=v;}},
fetch:async url=>({ok:true,json:async()=>url==='/api/rsi/capabilities'?{}:
url==='/api/rsi/jobs'?{jobs:[detail.job]}:detail}),setTimeout,console};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),context);
setImmediate(async()=>{try{await get('jobs-list').children[0].listeners.click();
assert.deepEqual(JSON.parse(get('job-inspection').textContent),detail);
}catch(e){console.error(e);process.exitCode=1;}});
"""
    subprocess.run([node, '-e', script, str(Path(__file__).parents[1] / 'dashboard/static/rsi_execution.js'), str(path)],
                   check=True, capture_output=True, text=True, timeout=20)


def test_late_fault_cannot_reclassify_an_earlier_closed_semantic_failure(tmp_path):
    from test_rsi_project_runner import failed_after_query
    runner, _, database, _ = failed_after_query(tmp_path)
    try:
        before = runner.terminal_failure_proof()
        runner.actor_event('pi-native-event', dict(event=dict(type='native_fault',
            fault_id='late-fault', origin='protocol', kind='protocol_error',
            operation_id=runner.workflow.pending['id'])))
        after = runner.terminal_failure_proof()
        assert after == before and after['classification'] == 'provider_timeout'
    finally:
        database.close()


def test_unscoped_native_startup_fault_retains_evidence_without_masking_transport_failure(tmp_path):
    from test_rsi_project_runner import make
    runner, _, database = make(tmp_path)
    try:
        assert runner.workflow.pending is None
        runner.actor_event('pi-native-event', dict(event=dict(type='native_fault',
            fault_id='startup-fault', origin='transport', kind='eof', operation_id=None)))
        incomplete = runner.latest('supervision-audit-incomplete')
        retained = runner.store.read_json(incomplete['fault_ref'])
        assert retained['event']['fault_id'] == 'startup-fault'
        assert retained['event']['operation_id'] is None
        assert runner.workflow.pending is None and runner.workflow.outcome is None
    finally:
        database.close()


@pytest.mark.parametrize('known', [False, True])
def test_real_sdk_protocol_fault_retains_native_logs_and_requires_observed_settlement(tmp_path, monkeypatch, provider, known):
    import coding.pi.native as native
    from coding.pi.workflow import NativeWorkflowSession
    from rsi.actor import PiSessionAdapter
    from rsi.contracts import BudgetLimits
    from rsi.runtime import WorkflowError
    from rsi.supervision import retain_terminal
    from test_rsi_project_runner import make
    url, requests = provider
    profile = native.model_profile
    monkeypatch.setattr(native, 'model_profile', lambda name: dict(profile(name), base_url=url))
    observe = NativeWorkflowSession._observe
    injected = []
    def protocol_fault(session, event):
        observe(session, event)
        trigger = (event['type'] == 'agent_settled' if known else
                   event['type'] == 'message_start' and event.get('message', {}).get('role') == 'assistant')
        if not injected and trigger:
            injected.append(event['type'])
            # Inject a controller-channel parse fault at the actual SDK boundary,
            # after either real settlement or the first real in-flight response.
            with session.session.events.mutex:
                session.session.events.queue.appendleft(dict(type='protocol_error', message='injected parse fault'))
                session.session.events.not_empty.notify()
    monkeypatch.setattr(NativeWorkflowSession, '_observe', protocol_fault)
    runner, _, database = make(tmp_path, budgets=BudgetLimits(actor_seconds=30, wall_seconds=45),
        session_factory=lambda: PiSessionAdapter(model='deepseek-flash', environ={'DEEPSEEK_API_KEY': 'fixture-secret'}))
    try:
        result = runner.run()
        assert injected and requests and result['status'] == 'suspended'
        pending = dict(runner.workflow.pending)
        diagnostic = ProjectSupervision(runner.root).diagnostic()
        audit = runner.store.read_json(diagnostic['audit_ref'])
        assert audit['fault']['origin'] == 'protocol' and audit['native_refs']
        assert audit['fault']['settlement']['provider_settled'] is known
        assert all(b'fixture-secret' not in runner.store.read_bytes(ref) for ref in audit['native_refs'])
        if known:
            proof = runner.terminal_failure_proof()
            assert proof['classification'] == 'harness_failure' and proof['audit_ref'] == diagnostic['audit_ref']
            retain_terminal(runner, proof)
            from rsi.job_inspection import inspect_task
            task = inspect_task(runner.root)
            assert task['supervision']['closure'] == 'known'
        else:
            with pytest.raises(WorkflowError):
                runner.terminal_failure_proof()
            assert diagnostic['closure'] == 'unknown'
        before = len(requests)
        runner.run()  # Observation of the original pending operation never dispatches again.
        assert len(requests) == before and runner.workflow.pending == pending
    finally:
        runner.actor.close()
        database.close()
