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
    requests, emitted, repaired = [], [], False
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            nonlocal repaired
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            messages = body['messages']
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
            if missing:
                name, args = 'read', dict(path=missing['path'])
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
            emitted.append(dict(operation_id=prompt['operation_id'], stage=prompt['stage'], name=name))
            delta = dict(role='assistant', content=None, tool_calls=[dict(index=0,
                id=f'call-{len(requests)}', type='function', function=dict(name=name, arguments=json.dumps(args)))])
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


def test_native_sdk_canonical_nine_stages_repair_and_retained_evidence(tmp_path, monkeypatch, canonical_provider):
    import coding.pi.native as native
    url, requests, emitted = canonical_provider
    profile = native.model_profile
    monkeypatch.setattr(native, 'model_profile', lambda identity: dict(profile(identity), base_url=url))
    runner, _, database = make(tmp_path, budgets=BudgetLimits(actor_seconds=120, wall_seconds=180),
        session_factory=lambda: PiSessionAdapter(model='deepseek-flash', environ={'DEEPSEEK_API_KEY': 'fixture-secret'}))
    verifier = runner.backend.verifier.read_bytes()
    try:
        result = runner.run()
        assert result['status'] == 'accepted', (result['status'], result.get('error'), result.get('stage'))
        assert result['correctness'] is True
        assert {row['evidence']['stage'] for row in result['history']} == {stage.value for stage in Stage}
        assert any(row['evidence']['stage'] == 'test' and not row['passed'] for row in result['history'])
        assert runner.backend.verifier.read_bytes() == verifier
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
        closure = runner.store.read_json(result['closure_ref'])['observation']
        assert closure['provider_settled'] and closure['actor_closed']
        assert any(any(m['role'] == 'tool' for m in request['messages']) for request in requests)
    finally:
        runner.actor.close()
        database.close()
