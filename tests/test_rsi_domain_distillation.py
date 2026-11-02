"""Canonical kernel terminal distillation uses native Pi without Python OpenAI."""
import builtins
import json

import pytest

from test_native_pi_filesystem import scripted_provider
from test_native_pi_programming_roles import redirect
from test_pi_runtime import _canonical_kernel_case


@pytest.mark.parametrize('large', [False, True])
def test_kernel_terminal_native_repair_retains_host_evidence_without_openai(tmp_path, monkeypatch, scripted_provider, large):
    url, calls, requests = scripted_provider
    redirect(monkeypatch, url)
    original_import = builtins.__import__
    def no_openai(name, *args, **kwargs):
        if name == 'openai' or name.startswith('openai.'):
            raise ImportError('Python OpenAI deliberately unavailable')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_openai)
    case = _canonical_kernel_case(tmp_path, monkeypatch).run()
    before = list(case.calls)
    summaries = [dict(tensor=list(range(10000)))] if large else []
    valid = dict(distillation=dict(no_new_knowledge=True, lessons=[]))
    calls.extend([('submit_curriculum_proposal', dict(value=dict(distillation=dict(no_new_knowledge=False, lessons=[])))),
                  ('submit_curriculum_proposal', dict(value=valid))])
    result = case.driver.distill(case.outcome, host_summaries=summaries)
    assert result.distillation is not None and result.distillation_failure is None
    assert case.calls == before
    assert len(requests) == 2
    assert any(message['role']=='tool' and 'no_new_knowledge' in message['content']
               for message in requests[-1]['messages'])
    roots = list((case.runner.root/'native-distillation').glob('*/context.json'))
    assert len(roots) == 1
    context = json.loads(roots[0].read_text())[-1]['content']
    context = json.loads(context)
    assert context['host_outcome'] == 'host-verified'
    assert context['host_summaries'] == summaries
    archive = context['historical_workflow']
    assert archive['scope'] == 'origin-task-archive'
    assert case.runner.store.read_json(archive['reference']) == case.runner.workflow.state['history']
    artifacts = [event for event in case.runner.events if event['type']=='domain-distillation-artifacts']
    assert artifacts[-1]['artifact_refs']
    for reference in artifacts[-1]['artifact_refs']:
        case.runner.store.read_bytes(reference)
