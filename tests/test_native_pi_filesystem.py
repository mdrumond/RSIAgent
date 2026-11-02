"""Native SDK filesystem behavior and durable host source ownership."""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest

from coding.pi.filesystem import NativeFiles
from coding.pi.native import NativeSession
from rsi.contracts import Stage, StepEvidence
from test_rsi_python_execution import backend


def pending(tmp_path):
    execution = backend(tmp_path)
    execution.workflow.state['stage'] = Stage.IMPLEMENTATION.value
    operation = execution.workflow.begin()
    files = NativeFiles(tmp_path/'native-files', store=execution.store, backend=execution)
    return execution, operation, files


def context(operation, call='edit-1'):
    return dict(operation_id=operation, tool_call_id=call, remaining=lambda: 30.)


def test_multiple_pending_edits_retain_content_and_commit_once(tmp_path):
    execution, operation, files = pending(tmp_path)
    original = execution.workflow.source_hash
    for index, name in enumerate(('helper.py', 'solution.py')):
        files.write(dict(path=str(execution.project/name), content=f'value = {index + 2}\n'),
                    context(operation, f'write-{index}'))
        assert execution.workflow.source_hash == original
    final = files.pending_source(operation)
    chain = execution.store.read_json(final['reference'])
    assert chain['base'] == original and len(chain['artifact_refs']) == 2
    previous = original
    for reference in chain['artifact_refs']:
        receipt = execution.store.read_json(reference)
        assert receipt['before'] == previous and receipt['status'] == 'applied'
        assert execution.store.read_bytes(receipt['artifact_ref']).startswith(b'value = ')
        previous = receipt['after']
    assert previous == final['source_hash'] == execution.project_hash()
    execution.workflow.complete(StepEvidence(Stage.IMPLEMENTATION, operation,
        artifact_refs=(final['reference'],), skill_refs=('host-native-skill-read',), source_hash=original),
        new_source_hash=final['source_hash'])
    assert execution.workflow.source_hash == previous and execution.workflow.pending is None
    assert execution.workflow.stage == Stage.TEST


@pytest.mark.parametrize('applied', [False, True])
def test_crash_intent_reconciles_without_replaying_write(tmp_path, monkeypatch, applied):
    execution, operation, files = pending(tmp_path)
    target = execution.project/'helper.py'
    original = target.read_bytes()
    with monkeypatch.context() as crash:
        if applied:
            crash.setattr(files, '_settle', lambda *args: (_ for _ in ()).throw(RuntimeError('crash')))
        else:
            import coding.pi.filesystem as module
            replace = module.os.replace
            def interrupted(source, destination):
                if Path(destination) == target:
                    raise RuntimeError('crash')
                return replace(source, destination)
            crash.setattr(module.os, 'replace', interrupted)
        with pytest.raises(RuntimeError):
            files.write(dict(path=str(target), content='value = 9\n'), context(operation))
    reopened = NativeFiles(files.root, store=execution.store, backend=execution)
    result = reopened.pending_source(operation)
    chain = execution.store.read_json(result['reference'])
    receipt = execution.store.read_json(chain['artifact_refs'][0])
    assert receipt['status'] == ('applied' if applied else 'not_applied')
    assert target.read_bytes() == (b'value = 9\n' if applied else original)
    with pytest.raises(ValueError):
        reopened.write(dict(path=str(target), content='value = 9\n'), context(operation))
    assert len(execution.store.read_json(reopened.pending_source(operation)['reference'])['artifact_refs']) == 1


def test_readonly_resource_scope_and_pending_source_guard(tmp_path):
    execution, operation, files = pending(tmp_path)
    resource = tmp_path/'reference.md'
    resource.write_text('immutable resource\n')
    readonly = NativeFiles(tmp_path/'role-files', store=execution.store, read_paths=[resource])
    result = readonly.read(dict(path=str(resource)), context(operation))['details']
    receipt = execution.store.read_json(result['reference'])
    assert base64.b64decode(result['data']) == execution.store.read_bytes(receipt['artifact_ref'])
    assert readonly.session_options()['tools_options'] == {'writable': False}
    for path in (execution.verifier, tmp_path/'private.json'):
        with pytest.raises(PermissionError):
            files.read(dict(path=str(path)), context(operation))
    with pytest.raises(PermissionError):
        readonly.write(dict(path=str(resource), content='changed'), context(operation))
    resource.write_text('changed outside host admission')
    with pytest.raises(ValueError):
        readonly.read(dict(path=str(resource)), context(operation))
    (execution.project/'helper.py').write_text('unreceipted = True\n')
    with pytest.raises(ValueError):
        files.write(dict(path=str(execution.project/'solution.py'), content='x = 1\n'), context(operation))


@pytest.mark.parametrize('reason', ['stopped', 'deadline', 'other_operation'])
def test_mutation_requires_live_admitted_operation(tmp_path, reason):
    execution, operation, files = pending(tmp_path)
    scope = context(operation)
    if reason == 'stopped':
        execution.workflow.stop()
    elif reason == 'deadline':
        scope['remaining'] = lambda: 0.
    else:
        scope['operation_id'] = 'other-operation'
    before = execution.project_hash()
    with pytest.raises(PermissionError):
        files.write(dict(path=str(execution.project/'helper.py'), content='changed\n'), scope)
    assert execution.project_hash() == before


def test_unbounded_deadline_allows_native_read_and_pending_write(tmp_path):
    execution, operation, files = pending(tmp_path)
    calls = []
    scope = dict(context(operation), remaining=lambda: calls.append(True))
    target = execution.project/'helper.py'
    result = files.read(dict(path=str(target)), scope)['details']
    assert base64.b64decode(result['data']) == target.read_bytes()
    files.write(dict(path=str(target), content='value = 9\n'), scope)
    assert target.read_text() == 'value = 9\n'
    assert len(calls) == 2


@pytest.fixture
def scripted_provider():
    class Calls(list):
        responder = None
        include_usage = True
    calls, requests = Calls(), []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            delta = dict(role='assistant', content='Finished.')
            action = calls.responder(requests[-1]) if calls.responder is not None else (calls.pop(0) if calls else None)
            if action is not None:
                name, arguments = action
                delta.update(content=None, tool_calls=[dict(index=0, id=f'call-{len(requests)}', type='function',
                    function=dict(name=name, arguments=json.dumps(arguments)))])
            payload = dict(id='fixture', object='chat.completion.chunk', model='deepseek-flash',
                choices=[dict(index=0, delta=delta, finish_reason='tool_calls' if 'tool_calls' in delta else 'stop')],
                usage=dict(prompt_tokens=10, completion_tokens=2, total_tokens=12))
            if not calls.include_usage:
                payload.pop('usage')
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
            self.wfile.write(('data: '+json.dumps(payload)+'\n\ndata: [DONE]\n\n').encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', calls, requests
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_real_native_tools_paginate_edit_and_reject_protected_paths(tmp_path, monkeypatch, scripted_provider):
    import coding.pi.native as native
    url, calls, requests = scripted_provider
    profile = native.model_profile
    monkeypatch.setattr(native, 'model_profile', lambda name: dict(profile(name), base_url=url))
    execution, operation, files = pending(tmp_path)
    resource = tmp_path/'long-reference.md'
    resource.write_text(''.join(f'line {i}\n' for i in range(1, 2501)))
    files = NativeFiles(files.root, store=execution.store, backend=execution, read_paths=[resource])
    calls.extend([
        ('read', dict(path=str(resource), offset=2001, limit=2)),
        ('edit', dict(path='helper.py', edits=[dict(oldText='value = 1', newText='value = 7')])),
        ('write', dict(path='solution.py', content='value = 8\n')),
        ('read', dict(path=str(execution.verifier)))])
    verifier = execution.verifier.read_bytes()
    with NativeSession(tmp_path/'session', cwd=execution.project,
                       environ={'DEEPSEEK_API_KEY': 'fixture-secret'}, **files.session_options()) as session:
        result = session.prompt('Use the admitted native tools.', operation_id=operation, remaining=lambda: 30.)
        assert result.provider_settled and not result.pending_tools
    assert len(requests) == 5 and not calls
    feedback = [m for request in requests for m in request['messages'] if m['role'] == 'tool']
    assert any('line 2001\nline 2002' in m['content'] for m in feedback)
    assert (execution.project/'helper.py').read_text() == 'value = 7\n'
    assert (execution.project/'solution.py').read_text() == 'value = 8\n'
    assert execution.verifier.read_bytes() == verifier
    assert files.pending_source(operation)['source_hash'] == execution.project_hash()
    assert any('outside admitted' in m['content'] for m in feedback)


def hard_crash_before_replace(root, workflow_path, files_root, operation):
    """Importable spawn target: os._exit bypasses every Python finally block."""
    import os
    import coding.pi.filesystem as module
    from rsi.contracts import BudgetLimits
    from rsi.python_execution import PythonExecution
    from rsi.runtime import Workflow
    execution = PythonExecution(Path(root))
    state = json.loads(Path(workflow_path).read_text())
    execution.workflow = Workflow(Path(workflow_path), lambda *args: True,
        source_hash=state['source'], limits=BudgetLimits.from_dict(state['limits']))
    files = NativeFiles(Path(files_root), store=execution.store, backend=execution)
    replace = module.os.replace
    def abrupt(source, destination):
        if Path(destination) == execution.project/'helper.py':
            os._exit(37)
        return replace(source, destination)
    module.os.replace = abrupt
    files.write(dict(path=str(execution.project/'helper.py'), content='value = 77\n'), context(operation))


def test_hard_process_exit_leaves_staging_outside_source_and_reconciles(tmp_path):
    import multiprocessing
    execution, operation, files = pending(tmp_path)
    before = execution.project_hash()
    child = multiprocessing.get_context('spawn').Process(target=hard_crash_before_replace,
        args=(str(execution.root), str(tmp_path/'workflow.json'), str(files.root), operation))
    child.start()
    child.join(timeout=15)
    try:
        assert not child.is_alive() and child.exitcode == 37
        assert execution.project_hash() == before
        assert (execution.project/'helper.py').read_text() == 'value = 1\n'
        assert len(list((files.root/'staging').iterdir())) == 1
        reopened = NativeFiles(files.root, store=execution.store, backend=execution)
        closure = execution.store.read_json(reopened.pending_source(operation)['reference'])
        receipt = execution.store.read_json(closure['artifact_refs'][0])
        assert receipt['status'] == 'not_applied' and closure['head'] == before
    finally:
        if child.is_alive():
            child.terminate(); child.join(timeout=5)
        child.close()
