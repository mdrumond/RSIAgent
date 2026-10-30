"""Additional controller allocations in the existing canonical CAS/event log."""
from rsi.runtime import WorkflowError


def retained_grants(store, contract, events):
    expected = {}
    for event in events:
        if event['type'] != 'actor-allocation-grant':
            continue
        row = store.read_json(event['reference'])
        if (row['kind'] != 'actor-allocation-grant-v1' or row['contract'] != contract
                or not isinstance(row['authority'], dict) or not row['authority']):
            raise WorkflowError('additional actor allocation authority differs')
        grant = dict(seconds=row['seconds'], receipt=event['reference'])
        if row['key'] in expected and expected[row['key']] != grant:
            raise WorkflowError('additional actor allocation key was reused')
        expected[row['key']] = grant
    return expected


def check_grants(state, store, contract, events, *, allow_unapplied=False):
    expected = retained_grants(store, contract, events)
    actual = state.get('actor_grants', {})
    if (any(key not in expected or expected[key] != value for key,value in actual.items())
            or not allow_unapplied and actual != expected):
        raise WorkflowError('actor allocation lacks its retained controller grant')
    return expected


def restore_grants(runner):
    # The grant event commits authority before the workflow cache is updated.
    # Reopen can finish this write, never infer a new allocation from elapsed time.
    expected = check_grants(runner.workflow.state, runner.store, runner.contract_ref,
                            runner.events, allow_unapplied=True)
    for key, row in expected.items():
        runner.workflow.grant_actor_seconds(key, row['seconds'], row['receipt'])


def optimizer_resume_grants(owner, runner):
    """Import only explicit, chained JobStore budget increases after admission."""
    events = owner.store.events(owner.key)  # JobStore verifies the full hash chain.
    index = owner.state['canonical_active']['index']
    previous, admitted = None, False
    for event in events:
        payload = event['payload']
        manifest = payload.get('manifest')
        if (event['kind'] == 'optimizer-state'
                and payload.get('canonical_active', {}).get('index') == index):
            admitted = True
        if manifest is None:
            continue
        current = manifest['max_seconds']
        if admitted and event['kind'] in ('resumed', 'manual-continuation'):
            if previous is None or current < previous:
                raise WorkflowError('optimizer resume allocation lacks its admitted parent budget')
            if current > previous:
                runner.grant_actor_seconds(event['sha256'], current-previous,
                    dict(kind='optimizer-jobstore-allocation-v1', job_id=owner.key,
                         previous_max_seconds=previous, event=event))
        previous = current
