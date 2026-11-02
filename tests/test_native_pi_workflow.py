"""Actual SDK tools execute the canonical Python workflow and independent oracle."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import pytest

from rsi.actor import PiSessionAdapter
from rsi.contracts import BudgetLimits, Stage
from test_rsi_project_runner import make


def text_blocks(content):
    return [content] if isinstance(content, str) else [row['text'] for row in content if row.get('type') == 'text']


def object_lines(message):
    for text in text_blocks(message.get('content') or ''):
        for line in text.splitlines():
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value
            except ValueError:
                pass


@pytest.fixture
def canonical_provider():
    class Observations(list):
        overflow = False
        overflow_sent = False
        history_reads = 0
        compactions = 0
        summary_output = 0
        guidance_nonce = None
        guidance_armed = False
        guidance_queued = False
        post_success = False
        post_success_queued = False
        post_success_operation = None
    requests, emitted, repaired = [], Observations(), False
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            nonlocal repaired
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            messages = body['messages']
            if not body.get('tools'):
                emitted.compactions += 1
                payload = dict(id='summary', object='chat.completion.chunk', model='deepseek-flash',
                    choices=[dict(index=0, delta=dict(role='assistant', content=
                        '## Goal\nComplete the retained canonical Python task.\n## Progress\n'
                        'The host retained actual tool results and the numeric repair source transition. '
                        'Keep exact receipt IDs in the recent tool state and finish the pending host stage.\n'), finish_reason='stop')],
                    usage=dict(prompt_tokens=10, completion_tokens=40, total_tokens=50))
                emitted.summary_output += payload['usage']['completion_tokens']
                self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
                self.wfile.write(('data: '+json.dumps(payload)+'\n\ndata: [DONE]\n\n').encode())
                return
            if emitted.overflow and repaired and not emitted.overflow_sent:
                emitted.overflow_sent = True
                self.send_response(400); self.send_header('Content-Type', 'application/json'); self.end_headers()
                self.wfile.write(json.dumps(dict(error=dict(type='invalid_request_error',
                    code='context_length_exceeded', message='Maximum context length exceeded; compact the conversation.'))).encode())
                return
            prompts = [(i, value) for i, message in enumerate(messages) if message['role'] == 'user'
                       for value in object_lines(message) if 'mapped_skills' in value]
            assert prompts, 'native provider must receive actual canonical stage context'
            start, prompt = prompts[-1]
            results = [value for message in messages[start+1:] if message['role'] == 'tool'
                       for value in object_lines(message)]
            state = next((r['workflow_state'] for r in reversed(results) if 'workflow_state' in r),
                         dict(skill_receipts=[], evidence_receipts=[]))
            read = {r['skill'] for r in state['skill_receipts']}
            missing = next((r for r in prompt['mapped_skills'] if r['name'] not in read), None)
            evidence = {r['name']: r['reference'] for r in state['evidence_receipts']}
            observations = [r['observation'] for r in results if isinstance(r.get('observation'), dict)]
            name, args = 'perform', dict(proposal='Use stable accumulation')
            if state.get('stale') and state.get('task'):
                repaired = False
                name, args = 'read', dict(path=state['task']['path'])
            elif missing:
                name, args = 'read', dict(path=missing['path'])
            elif emitted.overflow and prompt['stage'] == 'design' and emitted.history_reads < 3:
                from pathlib import Path
                name, args = 'read', dict(path=str(Path(prompt['editable_files'][0]).parent/'history-reference.md'),
                                          offset=1+emitted.history_reads*1000, limit=1000)
                emitted.history_reads += 1
            elif prompt['stage'] in ('design', 'optimisation'):
                if 'query_kdb' not in evidence:
                    name, args = 'query_kdb', dict(query='stable sum')
                elif 'retrospect' not in evidence:
                    name, args = 'retrospect', dict(supplement='Retain observed numeric failure')
                elif 'select_strategy' not in evidence:
                    query = next(row for row in observations if row.get('reference') == evidence['query_kdb'])
                    response = query['evidence']['payload']['response']
                    decisions = [dict(group=group, index=index,
                        action='used' if group == 'results' else 'rejected',
                        reason='Use stable reference' if group == 'results' else 'Avoid observed cancellation')
                        for group in ('results', 'negative_records') for index in range(len(response[group]))]
                    name, args = 'select_strategy', dict(proposal='Use stable accumulation',
                        query_ref=evidence['query_kdb'], retrospective_ref=evidence['retrospect'],
                        skill_refs=[row['receipt_id'] for row in state['skill_receipts']], decisions=decisions)
                elif prompt['stage'] == 'optimisation' and 'optimisation_decision' not in evidence:
                    name, args = 'optimisation_decision', dict(action='noop', reason='Preserve verified stable sum')
            elif prompt['stage'] == 'debug' and not repaired:
                name, args = 'edit', dict(path=prompt['editable_files'][0], edits=[dict(
                    oldText='def solve(xs):\n    total = 0.0\n    for x in xs: total += x\n    return total',
                    newText='from math import fsum\ndef solve(xs): return fsum(xs)')])
                repaired = True
                if emitted.guidance_nonce:
                    emitted.guidance_armed = True
            emitted.append(dict(operation_id=prompt['operation_id'], stage=prompt['stage'], name=name))
            delta = dict(role='assistant', content=None, tool_calls=[dict(index=0,
                id=f'call-{len(requests)}', type='function', function=dict(name=name, arguments=json.dumps(args)))])
            if name == 'perform' and prompt['stage']=='implementation' and emitted.post_success and not emitted.post_success_queued:
                emitted.post_success_queued = True
                emitted.post_success_operation = prompt['operation_id']
                delta['tool_calls'].append(dict(index=1, id=f'post-success-{len(requests)}', type='function',
                    function=dict(name='write', arguments=json.dumps(dict(path=prompt['editable_files'][0],
                        content='raise RuntimeError("queued mutation after host success")\n')))))
            if name == 'edit' and emitted.guidance_nonce and not emitted.guidance_queued:
                emitted.guidance_queued = True
                delta['tool_calls'].append(dict(index=1, id=f'stale-perform-{len(requests)}', type='function',
                    function=dict(name='perform', arguments=json.dumps(dict(proposal='Use stable accumulation')))))
            payload = dict(id='fixture', object='chat.completion.chunk', model='deepseek-flash',
                choices=[dict(index=0, delta=delta, finish_reason='tool_calls')],
                usage=dict(prompt_tokens=10, completion_tokens=2, total_tokens=12))
            time.sleep(.01)  # Actual observed provider wait is actor time.
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
            self.wfile.write(('data: '+json.dumps(payload)+'\n\ndata: [DONE]\n\n').encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', requests, emitted
    finally:
        server.shutdown(); server.server_close(); thread.join()


@pytest.mark.parametrize('scenario', ['normal', 'overflow', 'guidance', 'post_success'])
def test_native_sdk_canonical_nine_stages_repair_and_retained_evidence(tmp_path, monkeypatch, canonical_provider, scenario):
    import coding.pi.native as native
    url, requests, emitted = canonical_provider
    overflow = emitted.overflow = scenario == 'overflow'
    emitted.post_success = scenario == 'post_success'
    if scenario == 'guidance':
        from uuid import uuid4
        emitted.guidance_nonce = uuid4().hex
    profile = native.model_profile
    monkeypatch.setattr(native, 'model_profile', lambda identity: dict(profile(identity), base_url=url))
    runner, _, database = make(tmp_path, budgets=BudgetLimits(actor_seconds=20 if emitted.post_success else 120, wall_seconds=180),
        session_factory=lambda: PiSessionAdapter(model='deepseek-flash', environ={'DEEPSEEK_API_KEY': 'fixture-secret'}),
        extra_files={'history-reference.md': ('Python math.fsum preserves stable accumulation. Keep the frozen oracle and host evidence.\n' * 4000).encode()} if overflow else None)
    if emitted.guidance_nonce:
        original_context = runner.context
        def current_context():
            return dict(original_context(), user_guidance=dict(waves=
                emitted.guidance_nonce if emitted.guidance_armed else 'initial operator value'))
        runner.context = current_context
        runner.actor.context_refresh = lambda delivered: (current_context()
            if current_context()['user_guidance'] != delivered.get('user_guidance') else None)
    verifier = runner.backend.verifier.read_bytes()
    try:
        result = runner.run()
        assert result['status'] == 'accepted', (result['status'], result.get('error'), result.get('stage'))
        assert result['correctness'] is True
        assert {row['evidence']['stage'] for row in result['history']} == {stage.value for stage in Stage}
        assert any(row['evidence']['stage'] == 'test' and not row['passed'] for row in result['history'])
        assert runner.backend.verifier.read_bytes() == verifier
        if emitted.post_success:
            assert emitted.post_success_queued
            assert 'queued mutation after host success' not in runner.backend.read('solution.py')['content']
            commits = [row for row in result['history']
                       if row['evidence']['operation_id']==emitted.post_success_operation]
            assert len(commits) == 1
            operation = commits[0]['evidence']['operation_id']
            assert len([row for row in emitted if row['operation_id']==operation and row['name']=='perform']) == 1
            assert not any(row['type']=='source-edit' and row.get('operation_id')==operation for row in runner.events)
            native_events = [json.loads(line) for line in
                (runner.root/'native/session/native-events.jsonl').read_text().splitlines()]
            scoped = [event for event in native_events if event.get('operation_id')==operation]
            completed = next(index for index,event in enumerate(scoped)
                             if event['type']=='host_callback_end' and event.get('name')=='perform')
            assert not any(event['type']=='before_provider_request' for event in scoped[completed+1:])
            assert any(event['type']=='agent_settled' for event in scoped[completed+1:])
        assert runner.backend.read('solution.py')['content'].startswith('from math import fsum')
        assert {'read', 'edit', 'query_kdb', 'retrospect', 'select_strategy', 'perform'} <= {r['name'] for r in emitted}
        transitions = [runner.store.read_json(e['reference']) for e in runner.events if e['type'] == 'native-source-transition']
        assert len(transitions) == 1 and transitions[0]['status'] == 'applied'
        assert runner.store.read_bytes(transitions[0]['artifact_ref']).startswith(b'from math import fsum')
        selections = [runner.store.read_json(e['reference'])['payload'] for e in runner.events
                      if e['type'] == 'project-tool' and e['name'] == 'select_strategy']
        assert selections and all(any(d['group'] == 'negative_records' and d['action'] == 'rejected'
                                      for d in row['decisions']) for row in selections)
        for category in ('setup', 'thinking', 'retrieval', 'compilation', 'execution'):
            assert runner.workflow.accounting.seconds(category) > 0
        assert runner.workflow.accounting.seconds() < sum(
            end-start for kind, start, end in runner.workflow.accounting.records.values() if end is not None)
        closure = runner.store.read_json(result['actor_closure_ref'])['observation']
        assert closure['provider_settled'] and closure['actor_closed']
        assert any(any(m['role'] == 'tool' for m in request['messages']) for request in requests)
        from rsi.evidence_io import export_task
        exported = tmp_path/'evidence-export'
        manifest = export_task(runner.root, exported)
        rows = [json.loads(line) for line in (exported/'evidence.jsonl').read_text().splitlines()]
        assert len(rows) == manifest['records']
        session_refs = next(e['artifact_refs'] for e in runner.events if e['type'] == 'native-session-artifacts')
        assert session_refs and all(b'fixture-secret' not in runner.store.read_bytes(reference) for reference in session_refs)
        assert set(session_refs) <= {row['id'] for row in rows if row['kind'] == 'artifact'}
        assert any(row['kind'] == 'event' and row['data'].get('type') == 'native-session-artifacts' for row in rows)
        if emitted.guidance_nonce:
            superseded = [event for event in runner.events if event['type'] == 'actor-action-superseded']
            assert len(superseded) == 1
            operation = superseded[0]['operation_id']
            assert len([event for event in runner.events if event['type'] == 'host-dispatch'
                        and event['operation_id'] == operation]) == 1
            assert len([event for event in emitted if event['name'] == 'edit']) == 2
            assert any(emitted.guidance_nonce in json.dumps(request['messages']) for request in requests)
        if overflow:
            assert emitted.overflow_sent and emitted.compactions > 0
            native = [event['event'] for event in runner.events if event['type'] == 'pi-native-event']
            assert any(event['type'] == 'compaction_start' for event in native)
            assert any(event['type'] == 'compaction_end' for event in native)
            entries = [json.loads(line) for path in (runner.root/'native/session/sessions').glob('*.jsonl')
                       for line in path.read_text().splitlines()]
            assert any(entry['type'] == 'compaction' for entry in entries)
            assert any(event['type'] == 'session_before_compact' and event['reason'] == 'overflow' for event in native)
            assert any(event['type'] == 'compaction_end' and event['result']['usage']['output'] == emitted.summary_output for event in native)
            assert len([entry for entry in emitted if entry['name'] == 'edit']) == 1

    finally:
        runner.actor.close()
        database.close()


@pytest.mark.parametrize('boundary', ['transport_unknown', 'elapsed_after_host'])
def test_canonical_host_result_requires_settlement_and_elapsed_admission(tmp_path, monkeypatch, canonical_provider, boundary):
    import coding.pi.native as native
    url, requests, emitted = canonical_provider
    profile = native.model_profile
    monkeypatch.setattr(native, 'model_profile', lambda identity:dict(profile(identity), base_url=url))
    original = native.NativeSession._record
    intercepted = []
    def record(session, event):
        original(session, event)
        if event['type']=='host_callback_end' and event.get('name')=='perform' and not intercepted:
            intercepted.append(event['tool_call_id'])
            if boundary == 'transport_unknown':
                with session.events.mutex:
                    session.events.queue.appendleft(dict(type='protocol_error', message='Injected lost settlement after host result'))
            else:
                time.sleep(2.1)  # Measured post-host thinking/settlement time, not a fake charge.
    monkeypatch.setattr(native.NativeSession, '_record', record)
    runner, _, database = make(tmp_path, budgets=BudgetLimits(actor_seconds=2, wall_seconds=20),
        session_factory=lambda:PiSessionAdapter(model='deepseek-flash', environ={'DEEPSEEK_API_KEY':'fixture-secret'}))
    try:
        result = runner.run()
        assert intercepted
        assert runner.workflow.pending['stage'] == 'task_construction'
        operation = runner.workflow.pending['id']
        assert all(row['evidence']['operation_id'] != operation for row in runner.workflow.state['history'])
        proposals = [event for event in runner.events if event['type']=='stage-result-proposed'
                     and event['operation_id']==operation]
        assert len(proposals) == 1
        assert result['status'] != 'accepted'
        if boundary == 'transport_unknown':
            assert any(event['type']=='pi-native-event' and event['event']['type']=='native_fault'
                       for event in runner.events)
        else:
            assert runner.workflow.accounting.seconds() >= 2.1
    finally:
        runner.actor.close()
        database.close()


@pytest.mark.parametrize('partial', ['limit', 'offset', 'truncation'])
def test_native_skill_requires_actual_full_read_before_one_stage_commit(tmp_path, monkeypatch, scripted_provider, partial):
    from rsi.skills import SkillDocument, SkillBundle
    from test_native_pi_programming_roles import redirect
    from test_pi_runtime import _native_wire_objects
    import test_rsi_project_runner as fixture
    content='# Retained skill\n'+''.join(f'Instruction line {i}\n' for i in range(2500 if partial=='truncation' else 20))
    document=SkillDocument('retained-guide',content)
    bundle=SkillBundle((document,),tuple((stage,(document.name,)) for stage in Stage))
    monkeypatch.setattr(fixture,'builtin_bundle',lambda:bundle)
    url,calls,requests=scripted_provider
    redirect(monkeypatch,url)
    selections=[]
    runner,_,database=make(tmp_path, budgets=BudgetLimits(actor_seconds=30,wall_seconds=60),
        session_factory=lambda:PiSessionAdapter(model='deepseek-flash',environ={'DEEPSEEK_API_KEY':'fixture-secret'}))
    original_complete=runner.workflow.complete
    def complete(evidence,**kwargs):
        result=original_complete(evidence,**kwargs)
        if evidence.stage==Stage.TASK_CONSTRUCTION:runner.workflow.stop()
        return result
    monkeypatch.setattr(runner.workflow,'complete',complete)
    def choose(wire):
        prompt=next(value for role,value in reversed(list(_native_wire_objects([wire])))
                    if role=='user' and 'mapped_skills' in value)
        path=prompt['mapped_skills'][0]['path']
        turn=len(selections)
        selections.append(turn)
        if turn in (1,2):
            assert not any(row['evidence']['stage']=='task_construction' for row in runner.workflow.state['history'])
            assert not any(event['type']=='host-dispatch' and event['operation_id']==prompt['operation_id'] for event in runner.events)
        if turn==0:
            return 'read',dict(path=path,**({'limit':1} if partial=='limit' else {'offset':2} if partial=='offset' else {}))
        if turn==2:
            return 'read',dict(path=path,**({'offset':2} if partial=='limit' else {'limit':1} if partial=='offset' else {'offset':2001}))
        return 'perform',dict(proposal='Apply the completely observed frozen guide')
    calls.responder=choose
    try:
        runner.run()
        commits=[row for row in runner.workflow.state['history'] if row['evidence']['stage']=='task_construction']
        assert len(commits)==1 and len(requests)==4
        operation=commits[0]['evidence']['operation_id']
        assert len([event for event in runner.events if event['type']=='host-dispatch' and event['operation_id']==operation])==1
        reads=[runner.store.read_json(event['reference']) for event in runner.events if event['type']=='native-skill-read']
        assert [row['complete'] for row in reads]==[False,True]
        if partial=='truncation':assert reads[0]['truncation']['truncated'] is True
        assert runner.workflow.pending is None
    finally:
        runner.actor.close()
        database.close()
