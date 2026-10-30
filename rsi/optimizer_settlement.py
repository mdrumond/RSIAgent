"""Read-only admission of task results from the existing canonical CAS."""
from hashlib import sha256
import json
from pathlib import Path

from core.artifacts import ContentAddressedArtifacts
from optimizer.control import ObservationUnavailable, ObservedBudgetStop, WorkerStopped
from rsi.domain_checkpoint import decode
from rsi.domain_evidence import completed_effects
from rsi.runtime import WorkflowError


def require_settled_candidate(execution, state, *, stop_requested, historical=False):
    """Candidate history alone never closes a provider or a native operation."""
    if stop_requested:
        raise WorkerStopped('explicit stop prevents cached candidate publication')
    active = state.get('canonical_active')
    if not active:
        raise ObservationUnavailable('task result lacks canonical settlement authority')
    root = Path(execution) / ('canonical-' + str(active['index']))
    pointer = json.loads((root/'domain.json').read_text())
    workflow = json.loads((root/'workflow.json').read_text())
    location = Path(pointer['store_root'])
    store = ContentAddressedArtifacts(location if location.is_absolute() else root/location)
    contract = pointer['contract']
    frozen = store.read_json(contract)
    events = [store.read_json(ref) for ref in json.loads((root/'events.json').read_text())]
    actual = (workflow['source'] if historical else
              sha256((Path(execution)/'canonical-candidate.json').read_bytes()).hexdigest())
    try:
        proof = completed_effects(root, store, contract, workflow, events,
            frozen['backend_identity'], frozen['test_backend'], frozen['identity'], actual)
        closure_index = next(i for i in range(len(events)-1, -1, -1) if events[i]['type']=='actor-closure')
        closure = store.read_json(events[closure_index]['reference'])
        observed = closure['observation']
        if (closure['contract_ref'] != contract or closure['source_hash'] != actual
                or observed.get('provider_settled') is not True
                or not (observed.get('actor_closed') is True or observed.get('actor_absent') is True)
                or any(row['type']=='actor-dispatch' for row in events[closure_index+1:])):
            raise WorkflowError('task publication lacks current observed provider closure')
        if closure['pending'] != workflow['pending']:
            old = closure['pending']
            recovered = next((row for row in reversed(events) if row['type']=='stage-result'
                and row.get('settlement')=='observed-host-recovery'
                and old and row['operation_id']==old['id']), None)
            if workflow['pending'] is not None or recovered is None:
                raise WorkflowError('task publication closure differs from retained pending operation')
        pending = workflow['pending']
        if pending:
            # A known optional-profile deadline may leave the stage uncommitted
            # after its terminal result. Only its real cancellation proof can
            # authorize publishing the independently verified evaluation.
            receipts = [store.read_json(ref) for ref in proof['command_receipts']]
            current = [row for row in receipts if row['operation_id']==pending['id']]
            if pending['stage']!='optimisation' or not current:
                raise WorkflowError('canonical candidate is awaiting stage settlement')
            for row in current:
                dispatch = store.read_json(row['dispatch_ref'])
                value = decode(row['result'])
                if (dispatch['name']!='profile' or row['status']!='failed'
                        or value.get('error_type')!='ObservedBudgetStop'):
                    raise WorkflowError('pending optional diagnostics lack observed budget cancellation')
                retained = value['terminal_receipt']
                stopped = ObservedBudgetStop('retained observed profile stop', retained['path'])
                if stopped.terminal_receipt != retained:
                    raise WorkflowError('retained optional profile cancellation differs')
                stopped.validate(execution)
    except (WorkflowError, StopIteration) as error:
        raise ObservationUnavailable('canonical task remains observation-required; no cached result published') from error
