"""Separate execution allocation from frozen subject experiment semantics."""
import hashlib
import json
from pathlib import Path
import time

from optimizer.jobs import atomic, canonical

ALLOCATION_FIELDS = {'physical_device', 'physical_devices'}
TERMINAL = {'completed', 'failed', 'cancelled'}


def configuration(raw):
    value = json.loads(raw)
    allocation = {key: value.pop(key) for key in ALLOCATION_FIELDS if key in value}
    return hashlib.sha256(canonical(value)).hexdigest(), allocation


def unsettled_operations(root):
    """A terminal controller observation can settle an older transport checkpoint."""
    root = Path(root)
    observed = {}
    for path in root.rglob('remote-observations.json'):
        for row in json.loads(path.read_bytes()):
            observed[row.get('handle')] = row.get('state')
    pending = []
    for path in root.rglob('operation.json'):
        row = json.loads(path.read_bytes())
        if 'result' in row or row.get('state') in TERMINAL:
            continue
        if row.get('handle') and observed.get(row['handle']) in TERMINAL:
            continue
        if row.get('state') not in ('new', None):
            pending.append(row.get('handle') or str(path))
    return pending


def admit_identity(root, identity, referenced, *, previous=None):
    """Migrate legacy hashes only with exact authenticated prior configuration bytes."""
    root = Path(root)
    path = root / 'subject-runtime.json'
    old = json.loads(path.read_bytes()) if path.exists() else None
    semantics, allocation = {}, {}
    for role, raw in referenced.items():
        semantics[role], allocation[role] = configuration(raw)
    new = dict(identity, schema='subject-runtime-v2', referenced_semantic_sha256=semantics)
    if old is not None:
        if old.get('schema') == 'subject-runtime-v2':
            compatible = old == new
        else:
            expected = {key: value for key, value in old.items() if key != 'referenced_json_sha256'}
            compatible = expected == identity and set(old.get('referenced_json_sha256', {})) == set(referenced)
            for role, raw in referenced.items():
                before = (previous or {}).get(role, raw)
                compatible &= (hashlib.sha256(before).hexdigest() == old.get('referenced_json_sha256', {}).get(role)
                               and configuration(before)[0] == semantics[role])
        if not compatible:
            raise ValueError('subject runtime settings or target changed on resume')
    history = root / 'execution-allocations.json'
    entries = json.loads(history.read_bytes()) if history.exists() else []
    if old is not None and (old != new or not entries or entries[-1]['allocation'] != allocation):
        pending = unsettled_operations(root)
        if pending:
            raise ValueError('execution allocation cannot change with unsettled operations: ' + ', '.join(pending))
    if old is not None and old != new:
        archive = root / 'identity-history' / (hashlib.sha256(canonical(old)).hexdigest() + '.json')
        archive.parent.mkdir(parents=True, exist_ok=True)
        atomic(archive, canonical(old))
    if not entries or entries[-1]['allocation'] != allocation:
        entries.append(dict(epoch=time.time(), allocation=allocation))
        atomic(history, canonical(entries))
    atomic(path, canonical(new))
    return new


def permitted_devices(target, eligible):
    """Campaign allocation policy; eligibility must come from fresh device checks."""
    return [device for device in eligible if target != 'gz-a3' or device in (0, 1, 2)]
