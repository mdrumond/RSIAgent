"""Bound noncoding terminal context while retaining exact evidence in the CAS."""
from copy import copy
import json

from benchmarks.ascend.agent_context import ContextBudget, ContextBudgetError, WorkingContext, request_bytes


def terminal_budget(provider, profile):
    configured = provider.context_budget(profile)
    hard = min(configured.hard, 48 * 1024)
    scale = hard / configured.hard
    return ContextBudget(max(1025, int(configured.high * scale)),
                         max(1024, int(configured.target * scale)), hard)


def bounded_driver(owner, seconds, profile, budget):
    driver = owner.with_time_budget(seconds)
    driver.provider = copy(owner.provider)
    driver.provider.context_budgets = dict(owner.provider.context_budgets)
    driver.provider.context_budgets[profile.identity.backend_model.value] = budget
    return driver


def bounded_message(store, message, limit):
    if len(json.dumps(message, ensure_ascii=False).encode()) <= limit:
        return message
    raw = message['content'].encode()
    reference = store.put_bytes(raw)
    preview = raw[:max(0, limit // 3)].decode(errors='ignore')
    return dict(role=message['role'], content=json.dumps(dict(record=reference,
        size_bytes=len(raw), preview=preview, payload_omitted=True), ensure_ascii=False))


def terminal_context(owner, runner, profile, instruction, admitted, payloads, host_summaries, budget, prompt):
    store, host = runner.store, runner.backend.host
    history = runner.workflow.state['history']
    final = history[-1] if history else {}
    evidence = final.get('evidence', {})
    compact = dict(passed=final.get('passed'), evidence={key:evidence[key] for key in
        ('stage','operation_id','source_hash') if key in evidence})
    facts, summary_refs = [], []
    for summary in host_summaries:
        reference = store.put_json(summary)
        summary_refs.append(reference)
        fact = dict(record=reference)
        if isinstance(summary, dict):
            fact.update({key:summary[key] for key in
                ('category','evidence_sha256','source_sha256','candidate_sha256','passed','status') if key in summary})
            if isinstance(summary.get('detail'), str) and len(summary['detail'].encode()) <= 256:
                fact['detail'] = summary['detail']
        facts.append(fact)
    data = dict(admitted_artifacts=admitted, artifact_payloads={}, host_summaries=facts,
        workflow_history=[compact] if history else [], history_ref=store.put_json(history),
        instruction_ref=store.put_bytes(instruction.encode()),
        current_host_state=dict(source_hash=host.project_hash(), outcome=runner.workflow.outcome))
    system = dict(role='system', content=owner.system_prompt)
    goal = bounded_message(store, dict(role='user', content=instruction), max(256, budget.target // 6))
    feedback = [dict(role=row['role'], content=row['content']) for row in host.feedback]
    def terminal():
        return dict(role='user', content=prompt + '\n' + json.dumps(data, ensure_ascii=False))
    def size():
        return request_bytes(profile.request([system, goal, *feedback, terminal()]))
    reserve = min(1024, max(256, budget.hard // 10))
    if size() > budget.hard - reserve:
        raise ContextBudgetError('pinned terminal schema, citations and host facts exceed configured context budget')
    allowance = min(8192, budget.target // 4, budget.hard - reserve - size())
    for reference, raw in payloads.items():
        if len(raw) > allowance:
            continue
        text = raw.decode(errors='replace')
        data['artifact_payloads'][reference] = text
        if size() > budget.hard - reserve or len(json.dumps(data['artifact_payloads']).encode()) > allowance:
            del data['artifact_payloads'][reference]
    working = WorkingContext([system, goal])
    limit = max(256, budget.target // 4)
    for row in history:
        working.append(bounded_message(store, dict(role='user', content=json.dumps(dict(workflow_step=row))), limit))
    for summary in host_summaries:
        working.append(bounded_message(store, dict(role='user', content=json.dumps(dict(host_summary=summary))), limit))
    working.pinned = feedback
    working.tail = [terminal()]  # Never folded into an agent-written work log.
    runner.append(dict(type='domain-distillation-context', reference=store.put_json(dict(
        terminal=working.tail[0], history=data['history_ref'], summaries=summary_refs,
        instruction=data['instruction_ref'], budget=dict(high=budget.high,target=budget.target,hard=budget.hard)))))
    return working
