"""Actual canonical CPU host, bounded terminal provider and retained evidence."""
import json

import pytest

from benchmarks.ascend.agent_context import ContextBudget, request_bytes
from benchmarks.a5kernels.phase1_provider import DirectCompletion


@pytest.mark.parametrize('hard',[None,7000])
def test_terminal_contract_stays_pinned_and_provider_copy_is_bounded(tmp_path,monkeypatch,hard):
    from test_pi_runtime import _canonical_kernel_case
    case=_canonical_kernel_case(tmp_path,monkeypatch).run()
    outcome=case.outcome;driver=case.driver;runner=case.runner
    profile=outcome.continuation.profile
    if hard:driver.provider.context_budgets[profile.identity.backend_model.value]=ContextBudget(5000,3500,hard)
    original=driver.provider.context_budget(profile)
    wires=[];terminals=[];usage=[]
    def transport(**kwargs):
        wire=kwargs['request'];wires.append(wire)
        if wire['messages'][0]['content'].startswith('Update a compact working log'):
            usage.append(7);return DirectCompletion('Retain actual host results and exact citations.',7)
        terminals.append(wire);usage.append(2)
        return DirectCompletion('x'*3500 if len(terminals)==1 else
            '{"distillation":{"no_new_knowledge":true,"lessons":[]}}',2)
    driver.provider._transport=transport
    calls=list(case.calls)
    summaries=[dict(tensor=list(range(10000)))]*16 if not hard else []
    result=driver.distill(outcome,host_summaries=summaries)
    assert result.distillation is not None and len(terminals)==2
    assert result.tokens==sum(usage) and case.calls==calls
    assert driver.provider.context_budget(profile)==original
    assert all(request_bytes(wire)<=min(hard or original.hard,48*1024) for wire in wires)
    packets=[json.loads(wire['messages'][-1]['content'].split('\n')[-1]) for wire in terminals]
    assert packets[0]['admitted_artifacts']==outcome.admitted_artifacts==packets[1]['admitted_artifacts']
    assert packets[0]['current_host_state']==packets[1]['current_host_state']
    assert packets[0]['workflow_history'][-1]['passed'] is True
    assert runner.store.read_json(packets[0]['history_ref'])==runner.workflow.state['history']
    if summaries:assert runner.store.read_json(packets[0]['host_summaries'][0]['record'])==summaries[0]
