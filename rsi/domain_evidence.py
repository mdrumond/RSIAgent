"""Observed domain lifecycle/proofs and adapter into the shared evidence exporter."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile

from core.artifacts import ContentAddressedArtifacts, canonical_bytes
from rsi.runtime import WorkflowError


def completed_effects(root, store, contract, state, events, backend, test_backend, identity, actual, *, actor_active=False):
    root = Path(root)
    if actor_active or state.get('timing_uncertainties') or any(row[2] is None for row in state['intervals'].values()):
        raise WorkflowError('native actor or activity remains unobserved')
    pointer = json.loads((root/'domain.json').read_text())
    frozen = store.read_json(contract)
    from rsi.domain_allocations import check_grants
    check_grants(state, store, contract, events)
    if (pointer['contract'] != contract or frozen['identity'] != identity
            or frozen['backend_identity'] != backend or frozen.get('test_backend', False) != test_backend
            or frozen['spec']['budgets'] != state['limits']):
        raise WorkflowError('native frozen contract/target/budget differs')
    dispatch = {e['activity_id']:e for e in events if e['type']=='domain-dispatch'}
    results = {e['activity_id']:e for e in events if e['type']=='domain-result'}
    if (len(dispatch) != sum(e['type']=='domain-dispatch' for e in events)
            or len(results) != sum(e['type']=='domain-result' for e in events) or set(dispatch) != set(results)):
        raise WorkflowError('native activity has no unique terminal receipt')
    commands, reservations = [], {}
    known_sources = set()
    for activity, event in dispatch.items():
        intent = store.read_json(event['reference'])
        receipt = store.read_json(results[activity]['reference'])
        if (intent['activity_id'] != activity or receipt['activity_id'] != activity
                or receipt['dispatch_ref'] != event['reference']
                or intent['contract_ref'] != contract or receipt['contract_ref'] != contract
                or intent['identity'] != identity or intent['backend_identity'] != backend
                or intent['test_backend'] != test_backend or receipt['observed_terminal'] is not True
                or intent['operation_id'] != receipt['operation_id'] or intent['stage'] != receipt['stage']):
            raise WorkflowError('native activity terminal scope/authority differs')
        intervals = store.read_json(receipt['accounting_ref'])
        if any(value[2] is None or state['intervals'].get(key) != value for key,value in intervals.items()):
            raise WorkflowError('native activity accounting receipt differs')
        known_sources.update((intent['source_hash'], receipt['source_hash']))
        if intent['execution']:
            if intent['reservation'] in reservations:
                raise WorkflowError('native execution reservation reused')
            reservations[intent['reservation']] = intent['operation_id']
            commands.append(results[activity]['reference'])
    if set(reservations) != set(state.get('execution_reservations', {})):
        raise WorkflowError('native execution admission is not fully observed')
    for key, operation in reservations.items():
        if state['execution_reservations'][key]['stage_operation'] != operation:
            raise WorkflowError('native execution reservation operation differs')
    for event in events:
        if event['type'] == 'domain-checkpoint':
            row = store.read_json(event['reference'])
            if row['contract_ref'] != contract or row['identity'] != identity:
                raise WorkflowError('native source checkpoint scope differs')
            for ref in row['sources'].values():
                store.read_bytes(ref)
            known_sources.add(row['source_hash'])
    if actual not in known_sources:
        raise WorkflowError('native current source has no observed checkpoint')
    queries = {e['call_id']:e for e in events if e['type']=='query-dispatch'}
    finished = {e['call_id']:e for e in events if e['type']=='query-finished'}
    if set(queries) != set(finished) or any(e['status'] != 'complete' for e in finished.values()):
        raise WorkflowError('native retrieval remains unavailable/unobserved')
    retrievals = set()
    for key,event in finished.items():
        row = store.read_json(event['reference'])
        intent = queries[key]
        if (event['operation_id'] != intent['operation_id'] or row['context']['workflow'] != contract
                or row['context']['operation'] != event['operation_id']
                or row['payload']['request'] != store.read_json(intent['request_ref'])
                or row['payload']['backend'] != backend or row['payload']['test_backend'] != test_backend):
            raise WorkflowError('native retrieval receipt scope differs')
        retrievals.add(event['reference'])
    retained = {e['reference'] for e in events if e['type']=='strategy-evidence' and e['kind']=='retrieval'}
    if retrievals != retained or any(e['type']=='strategy-evidence' and e['kind']=='query-unavailable' for e in events):
        raise WorkflowError('native retrieval completion coverage differs')
    from rsi.runtime import ActorClock
    elapsed = ActorClock(state['intervals'], lambda: None, lambda: max(
        [state['created']] + [row[2] for row in state['intervals'].values()])).seconds()
    return dict(command_receipts=commands, retrieval_receipts=sorted(retrievals),
                actor_seconds_total=elapsed, source_hash=actual)


class DomainExportAdapter:
    """Read-only normalized input view for the existing export_task serializer.

    Objects remain the original CAS bytes, hard-linked where possible. This
    creates no new logical evidence store and no fabricated execution manifest.
    The normalized manifest explicitly declares its domain contract/schema.
    """
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)

    @contextmanager
    def view(self):
        names = ('domain.json','workflow.json','events.json')
        before = {name:(self.root/name).read_bytes() for name in names}
        pointer, state, refs = [json.loads(before[name]) for name in names]
        if 'store_root' not in pointer:
            raise WorkflowError('native task predates typed CAS export; explicit checkpoint migration required')
        location = Path(pointer['store_root'])
        store = ContentAddressedArtifacts(location if location.is_absolute() else self.root/location)
        frozen = store.read_json(pointer['contract'])
        events = [store.read_json(ref) for ref in refs]
        # A pending actor needs a genuine matching closed-provider observation.
        outcomes = [store.read_json(e['reference']) for e in events if e['type']=='project-run-outcome']
        outcome = next((row for row in reversed(outcomes)
            if store.read_json(row['workflow_state_ref']) == state and row.get('closure_ref')), None)
        if outcome is None:
            raise WorkflowError('native export requires observed quiescent actor closure')
        closure = store.read_json(outcome['closure_ref'])
        observation = closure['observation']
        if (closure['pending'] != state['pending'] or closure['contract_ref'] != pointer['contract']
                or observation.get('provider_settled') is not True
                or not (observation.get('actor_closed') is True or observation.get('actor_absent') is True)):
            raise WorkflowError('native export actor closure differs')
        completed_effects(self.root, store, pointer['contract'], state, events,
            frozen['backend_identity'], frozen['test_backend'], frozen['identity'], closure['source_hash'])
        objects = {path.parent.name+path.name:store.read_bytes(path.parent.name+path.name)
                   for path in store.objects.glob('*/*')}
        with tempfile.TemporaryDirectory(prefix='rsi-domain-export-') as temporary:
            view = Path(temporary)
            metadata = dict(contract=pointer['contract'], domain_contract='rsi-domain-contract-v2')
            for name,value in (('runner.json',metadata),('manifest.json',metadata),('workflow.json',state),('events.json',refs)):
                (view/name).write_bytes(canonical_bytes(value))
            for ref,data in objects.items():
                path = view/'artifacts'/'objects'/ref[:2]/ref[2:]
                path.parent.mkdir(parents=True,exist_ok=True)
                try:
                    os.link(store.objects/ref[:2]/ref[2:],path)
                except OSError:
                    path.write_bytes(data)  # immutable export staging only
            if any((self.root/name).read_bytes()!=data for name,data in before.items()) or any(
                    store.read_bytes(ref)!=data for ref,data in objects.items()):
                raise WorkflowError('native task changed during export snapshot')
            yield view


def export_domain_task(root, destination):
    from rsi.evidence_io import export_task
    return export_task(root, destination, adapter=DomainExportAdapter(root))
