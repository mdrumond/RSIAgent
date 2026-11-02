"""Legacy fixture wire helpers; production SDK behaviors live in test_native_pi_* tests."""
from rsi.event_journal import read_event_refs


from core.artifacts import canonical_bytes


import json


import threading


import pytest


from coding.pi.schema import obj, STRING


from rsi.actor import EvidenceTool, PiSessionAdapter, StageResult, WorkflowActor


from rsi.contracts import Stage


from rsi.runtime import Workflow, WorkflowError


from rsi.skills import builtin_bundle


def response(actions, index, tokens=100, *, summary=None, finish=None, completion_tokens=10):
    from test_pi_runtime import stream
    result = stream(*(actions[0] if actions else ('unused', {})), index, finish=finish or ('tool_calls' if actions else 'stop'))
    original = result.iter_lines
    def lines():
        for line in original():
            if line.startswith(b'data: {'):
                item = json.loads(line[6:])
                delta = item['choices'][0]['delta']
                if delta.get('tool_calls'):
                    if actions:
                        delta['tool_calls'] = [dict(index=i,id=f'call_{index}_{i}',type='function',
                            function=dict(name=name,arguments=json.dumps(args))) for i,(name,args) in enumerate(actions)]
                    else:
                        delta.pop('tool_calls')
                        delta['content'] = summary or '# Retained context\nSkills read; retain actual receipts and host evidence. Continue current stage.'
                if 'usage' in item:
                    item['usage']['prompt_tokens'] = tokens
                    item['usage']['completion_tokens'] = completion_tokens
                line = b'data: '+json.dumps(item).encode()
            yield line
    result.iter_lines = lines
    return result


def native_actor(tmp_path, events, perform, tools=None, *, clock=None, limits=None,
                 context_window=16384, request_bytes=49152):
    from coding.pi.session import runtime_preflight, RuntimeUnavailable
    try: runtime_preflight()
    except RuntimeUnavailable: pytest.skip('cached pinned Pi/Node22 required')
    workflow = Workflow(tmp_path/'workflow.json', lambda *_:False, source_hash='project-v1',
                        limits=limits, **({'clock':clock} if clock is not None else {}))
    actor = WorkflowActor(workflow, builtin_bundle(),
        lambda:PiSessionAdapter(model='deepseek-flash',environ={'DEEPSEEK_API_KEY':'fake-provider-secret'},
                               context_window=context_window,request_bytes=request_bytes),
        perform=perform,preflight=lambda *_:StageResult(),evidence_tools=tools,
        emit=lambda kind,data:events.append((kind,data)))
    return actor, workflow

