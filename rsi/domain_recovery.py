"""Settle an observed domain workload boundary, never replay an actor or host."""
from rsi.actor import StageResult
from rsi.contracts import StepEvidence
from rsi.domain_checkpoint import decode
from rsi.runtime import WorkflowError


def recover_completed_boundary(runner):
    flow, host = runner.workflow, runner.backend
    recover = getattr(host, 'recover_terminal_boundary', None)
    if recover is None or getattr(host, 'boundary_stage', None) != flow.stage:
        return False
    pending = flow.pending
    operation = pending['id']
    event = runner.latest('domain-stage-intent', operation)
    if event is None:
        return False  # Historical effects without an admitted intent stay pending.
    intent = runner.store.read_json(event['reference'])
    if (intent['operation_id'] != operation or intent['contract_ref'] != runner.contract_ref
            or intent['stage'] != flow.stage.value or intent['source_hash'] != host.project_hash()
            or host.project_hash() != flow.source_hash or not intent['request_id']):
        raise WorkflowError('retained domain stage intent differs from current pending work')
    requests = [row for row in runner.events if row['type'] == 'actor-request'
                and row['operation_id'] == operation and row['request_id'] == intent['request_id']]
    if not requests:
        raise WorkflowError('retained domain intent has no original actor request')
    closed = runner.latest('actor-closure')
    if closed is None:
        raise WorkflowError('terminal domain recovery requires observed actor closure')
    closure = runner.store.read_json(closed['reference'])
    observed = closure['observation']
    if (closure['pending'] != pending or closure['contract_ref'] != runner.contract_ref
            or closure['source_hash'] != host.project_hash() or observed.get('provider_settled') is not True
            or not (observed.get('actor_closed') is True or observed.get('actor_absent') is True)):
        raise WorkflowError('terminal domain actor closure differs')
    # Validates every original dispatch/reservation/result and closed clock row.
    # Any unresolved native activity or uncertain placement remains a hard stop.
    proof = runner._completed_effects()
    receipts = []
    for reference in proof['command_receipts']:
        receipt = runner.store.read_json(reference)
        if receipt['operation_id'] == operation:
            dispatch = runner.store.read_json(receipt['dispatch_ref'])
            if dispatch['stage'] != flow.stage.value or receipt['source_hash'] != flow.source_hash:
                raise WorkflowError('terminal domain activity scope differs')
            receipts.append(dict(reference=reference, dispatch=dispatch, result=decode(receipt['result'])))
    if not receipts:
        return False
    skill_refs = tuple(dict.fromkeys(runner.service.read_receipt(row['receipt_id'])
        for row in runner.events if row['type'] == 'skill-usage' and row.get('kind') == 'read'
        and row['operation_id'] == operation and row.get('section') is None))
    if not skill_refs:
        raise WorkflowError('terminal domain recovery lacks original mapped skill reads')
    # Host may reconstruct its public history only from these authenticated
    # returned values. It must never invoke a compiler/provider/backend here.
    observation = recover(flow.stage, operation, intent['arguments'], receipts)
    if observation is None:
        return False
    checkpoint = runner.capture_checkpoint(operation, scope='stage')
    result = StageResult(passed=observation.passed,
        artifact_refs=(*observation.artifact_refs, *(row['reference'] for row in receipts), checkpoint))
    evidence = StepEvidence(stage=flow.stage, operation_id=operation, source_hash=flow.source_hash,
                            skill_refs=skill_refs, artifact_refs=result.artifact_refs)
    from dataclasses import asdict
    runner.append(dict(type='stage-result', stage=flow.stage.value, operation_id=operation,
                       result=asdict(result), evidence=evidence.to_dict(), settlement='observed-host-recovery'))
    flow.reconcile(operation, evidence, passed=result.passed)
    flow.state['domain_boundary'] = evidence.stage.value
    flow.stop()  # Workload boundary, not acceptance or a reset of elapsed usage.
    return True
