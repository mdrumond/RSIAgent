"""Real installed Pi SDK role tools over a deterministic streaming provider fixture."""
from dataclasses import asdict
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

import pytest

from core.artifacts import ContentAddressedArtifacts
from rsi.roles import AdmittedEvidence, NativeRoles


@pytest.mark.parametrize('large', [False, True])
def test_real_native_reads_full_evidence_and_submits_typed_proposal(tmp_path, monkeypatch, large):
    import coding.pi.native as native
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.phase1_provider import direct_profile
    profile = direct_profile(next(row for row in MODEL_IDENTITIES
                                  if row.profile_id == 'deepseek-flash-native-v1'))
    operation = 'actual-native-role'
    root = tmp_path/'roles'/hashlib.sha256((operation+':investigation').encode()).hexdigest()
    manifest = root/'evidence/manifest.json'
    original = {'documents': [dict(text='Retained source evidence '+('x'*1000))
                              for _ in range(4200 if large else 1)], 'tail': 'actual-tail-observation'}
    requests, events, errors = [], [], []
    idea = dict(question='Which input explains the observed tail?', goal='Investigate the retained failure',
                why='The last admitted observation narrows the investigation')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            try:
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(request)
                number = len(requests)
                if number == 1:
                    name, arguments = 'read', dict(path=str(manifest))
                elif number == 2:
                    evidence = json.loads(manifest.read_text())[0]
                    path = Path(evidence['path'])
                    # The fixture chooses a native line offset; Pi owns pagination.
                    offset = max(1, len(path.read_text().splitlines())-5)
                    name, arguments = 'read', dict(path=str(path), offset=offset, limit=10)
                elif number == 3:
                    name, arguments = 'submit_proposal', dict(idea, goal='')
                else:
                    name, arguments = 'submit_proposal', idea
                delta = dict(role='assistant', content=None, tool_calls=[dict(index=0,
                    id='native-role-call-'+str(number), type='function',
                    function=dict(name=name, arguments=json.dumps(arguments)))])
                chunk = dict(id='role-fixture', object='chat.completion.chunk', model='deepseek-flash',
                    choices=[dict(index=0, delta=delta, finish_reason='tool_calls')],
                    usage=dict(prompt_tokens=10, completion_tokens=3, total_tokens=13))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(('data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n').encode())
            except Exception as error:
                errors.append(error)
                raise
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model_profile = native.model_profile
    monkeypatch.setattr(native, 'model_profile', lambda identity: dict(model_profile(identity),
        base_url=f'http://127.0.0.1:{server.server_port}/v1'))
    store = ContentAddressedArtifacts(tmp_path/'artifacts')
    roles = NativeRoles(tmp_path/'roles', profile, observer=lambda *event: events.append(event),
        environment={'DEEPSEEK_API_KEY': 'native-role-fixture-secret'}, store=store,
        operation_id=lambda: operation)
    try:
        result = roles.investigation('Read the admitted manifest and final observations before proposing.',
            [AdmittedEvidence('failure', original)], deadline=time.monotonic()+45)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert not errors and asdict(result) == idea
    assert len(requests) == 4  # Invalid native proposal corrected; committed proposal terminates.
    assert 'actual-tail-observation' in json.dumps(requests[2]['messages'])
    state = json.loads((root/'role.json').read_text())
    assert state['status'] == 'completed' and state['actor_seconds'] > 0
    retained = json.loads(manifest.read_text())[0]
    assert json.loads(store.read_bytes(retained['reference']))['data'] == original
    if large:
        assert retained['bytes'] > 4_000_000
    file_events = [row for kind, row in events if kind == 'role_native_file']
    assert len(file_events) == 2
    read = store.read_json(file_events[-1]['reference'])
    assert store.read_bytes(read['artifact_ref']) == Path(retained['path']).read_bytes()
    assert [kind for kind, _ in events].count('role_proposal') == 1
    assert [kind for kind, _ in events].count('role_finished') == 1
    finished = next(row for kind, row in events if kind == 'role_finished')
    assert finished['success']
    assert max(len(json.dumps(row)) for _, row in events) < 4096
    assert 'native-role-fixture-secret' not in json.dumps(events)
    assert list((root/'native/sessions').glob('*.jsonl'))
