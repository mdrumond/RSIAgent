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


def optimizer_allocation_anchor(owner):
    """Snapshot the parent budget actually used to freeze the new contract."""
    events = owner.store.events(owner.key)
    parent = next((event['payload']['manifest'] for event in reversed(events)
                   if 'manifest' in event['payload']), None)
    if parent is None:
        raise WorkflowError('optimizer contract has no retained parent budget admission')
    return parent, dict(job_id=owner.key, sequence=events[-1]['sequence'],
                        event_sha256=events[-1]['sha256'], max_seconds=parent['max_seconds'])


def optimizer_resume_grants(owner, runner):
    """Import only increases after the immutable contract's budget boundary."""
    events = owner.store.events(owner.key)  # JobStore verifies the full hash chain.
    anchor = runner.store.read_json(runner.contract_ref).get('actor_allocation_anchor')
    if anchor is None:
        # Old contracts can spend their existing allocation. A resume label is
        # not evidence of extra time, and cannot establish a new grant boundary.
        previous, increased = None, False
        for event in events:
            manifest = event['payload'].get('manifest')
            if manifest is None:
                continue
            current = manifest['max_seconds']
            if event['kind'] in ('resumed', 'manual-continuation'):
                increased |= previous is None or current > previous
            previous = current
        if increased:
            raise WorkflowError('optimizer contract needs an explicit allocation-anchor migration')
        return  # Never import unassigned extra time into a legacy contract.
    sequence = anchor['sequence']
    if (anchor['job_id'] != owner.key or type(sequence) is not int or not 0 <= sequence < len(events)
            or events[sequence]['sha256'] != anchor['event_sha256']):
        raise WorkflowError('optimizer contract parent budget boundary differs')
    parent = next(event['payload']['manifest'] for event in reversed(events[:sequence+1])
                  if 'manifest' in event['payload'])
    if parent['max_seconds'] != anchor['max_seconds']:
        raise WorkflowError('optimizer contract admitted parent budget differs')
    previous = anchor['max_seconds']
    for event in events[sequence+1:]:
        manifest = event['payload'].get('manifest')
        if manifest is None:
            continue
        current = manifest['max_seconds']
        if event['kind'] in ('resumed', 'manual-continuation'):
            if current < previous:
                raise WorkflowError('optimizer resume allocation lacks its admitted parent budget')
            if current > previous:
                runner.grant_actor_seconds(event['sha256'], current-previous,
                    dict(kind='optimizer-jobstore-allocation-v1', job_id=owner.key,
                         previous_max_seconds=previous, event=event))
        previous = current
