"""Read-only A3 retry admission from observed canonical receipts."""
import json
from rsi.event_journal import read_event_refs
from hashlib import sha256
from pathlib import Path

from core.artifacts import ContentAddressedArtifacts
from rsi.domain_checkpoint import decode
from rsi.domain_evidence import completed_effects
from rsi.runtime import WorkflowError


def known_prelaunch_failure(value):
    from benchmarks.a3kernels.phase1_protocol import FailedEvidence, attest
    if not isinstance(value, FailedEvidence):
        return False
    if attest(value.attestation_payload()) != value.attestation_sha256:
        raise ValueError('A3 failure attestation differs from its actual fields')
    return value.stage == 'prepare' and value.error_type == 'RuntimeError'


def observed_prelaunch_root(root):
    root = Path(root)
    if not (root/'domain.json').exists():
        return False
    pointer = json.loads((root/'domain.json').read_text())
    location = Path(pointer['store_root'])
    store = ContentAddressedArtifacts(location if location.is_absolute() else root/location)
    frozen = store.read_json(pointer['contract'])
    state = json.loads((root/'workflow.json').read_text())
    events = [store.read_json(ref) for ref in read_event_refs(root)]
    if not state['pending']:
        return False
    if sha256((root.parent/'canonical-source').read_bytes()).hexdigest() != state['source']:
        raise ValueError('A3 failed-attempt source changed outside retained canonical state')
    closure = next((row for row in reversed(events) if row['type']=='actor-closure'), None)
    if closure is None:
        return False
    closed = store.read_json(closure['reference'])
    observation = closed['observation']
    if (closed['contract_ref'] != pointer['contract'] or closed['pending'] != state['pending']
            or closed['source_hash'] != state['source'] or observation.get('provider_settled') is not True
            or not (observation.get('actor_closed') is True or observation.get('actor_absent') is True)):
        return False
    try:
        proof = completed_effects(root, store, pointer['contract'], state, events,
            frozen['backend_identity'], frozen['test_backend'], frozen['identity'], state['source'])
    except WorkflowError:
        return False  # Unknown work is never fresh-attempt authority.
    current = [store.read_json(ref) for ref in proof['command_receipts']
               if store.read_json(ref)['operation_id']==state['pending']['id']]
    if not current:
        return False
    failure = decode(current[-1]['result'])
    if not known_prelaunch_failure(failure):
        return False
    if failure.attempt_id != state['pending']['id'] or failure.project_id != frozen['spec']['project_id']:
        raise ValueError('A3 pre-launch failure belongs to a different native attempt/project')
    return True


def retry_spec(host, spec, clock):
    """Retain the original elapsed allocation across a known-safe fresh attempt."""
    from dataclasses import replace
    prior = Path(host.canonical_prior_root)
    if not observed_prelaunch_root(prior):
        raise WorkflowError('fresh A3 attempt lacks observed pre-launch failure authority')
    state = json.loads((prior/'workflow.json').read_text())
    if state['limits']['actor_seconds'] != spec.budgets.actor_seconds:
        raise WorkflowError('fresh A3 retry changed the parent actor allocation')
    wall = state['limits']['wall_seconds']
    if wall is not None:
        remaining = wall - max(0., clock()-state['created'])
        if remaining <= 0:
            raise WorkflowError('parent A3 retry wall allocation exhausted')
        cap = spec.budgets.wall_seconds
        spec = replace(spec, budgets=replace(spec.budgets, wall_seconds=min(cap, remaining) if cap else remaining))
    return spec


def carry_retry_elapsed(runner, prior):
    from hashlib import sha256
    from rsi.runtime import ActorClock
    prior = Path(prior)
    raw = (prior/'workflow.json').read_bytes()
    state = json.loads(raw)
    contract = json.loads((prior/'domain.json').read_text())['contract']
    imported = {'prior-'+contract+':'+key:list(value) for key,value in state['intervals'].items()
                if value[0] in ActorClock.BILLABLE}
    runner.workflow.state['intervals'].update(imported)
    runner.workflow._save()
    runner.append(dict(type='domain-elapsed-import', reference=runner.store.put_json(dict(
        prior_contract=contract, prior_workflow_sha256=sha256(raw).hexdigest(),
        intervals=imported, actor_seconds=ActorClock(imported,lambda:None,runner.workflow.clock).seconds(),
        setup_transfer_excluded=True))))
