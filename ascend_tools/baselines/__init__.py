"""Frozen PyTorch baseline contracts and fail-closed qualification receipts."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent / 'packs'
WORKLOADS = ('fa', 'gdn', 'bsa', 'kimi_delta_attention', 'lact_ttt_chunk',
             'sparse_lightning_indexer_grad_kl_loss_enhance', 'tensor_program_vm')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack(workload):
    spec = json.loads((ROOT / 'manifest.json').read_text())[workload]
    directory = ROOT / spec['directory']
    # Check original bytes before interpreting any vendored source or case data.
    provenance = json.loads((ROOT / 'upstream/provenance.json').read_text()) if spec['origin'] == 'cann-bench' else (
        json.loads((directory / 'provenance.json').read_text()) if (directory / 'provenance.json').exists() else None)
    if provenance:
        base = ROOT / 'upstream' if spec['origin'] == 'cann-bench' else directory
        for name, expected in provenance['files'].items():
            if file_hash(base / name) != expected:
                raise ValueError('Frozen reference changed: ' + name)
    cases = json.loads((directory / spec['cases']).read_text())
    return spec, directory, cases


def describe(workload):
    spec, directory, cases = pack(workload)
    files = {p.name: file_hash(p) for p in sorted(directory.iterdir()) if p.is_file() and p.suffix != '.pyc'}
    return dict(workload=workload, sourceHash=digest({'files':files,'contract':spec}), caseHash=digest(cases),
                verifierHash=digest({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}),
                caseIds=[c['case_id'] for c in cases], origin=spec['origin'],
                independentCheckHash=file_hash(Path(__file__).parent/'checks.py'))


def validate_receipt(receipt, expected):
    for key in ('workload', 'target', 'sourceHash', 'caseHash', 'verifierHash'):
        if receipt.get(key) != expected[key]:
            raise ValueError('Qualification mismatch: ' + key)
    if receipt.get('format') != 1 or receipt.get('qualified') is not True:
        raise ValueError('Unqualified baseline')
    if receipt.get('runtime', {}).get('fingerprint') != expected['runtimeFingerprint']:
        raise ValueError('Qualification runtime mismatch')
    independent = receipt.get('independentChecks', {})
    if independent.get('passed') is not True or independent.get('checkerHash') != expected['independentCheckHash']:
        raise ValueError('Missing independent numerical checks')
    rows = receipt.get('cases', [])
    ids = expected['caseIds']
    if sorted(row['id'] for row in rows) != sorted(ids) or not all(row.get('passed') is True for row in rows):
        raise ValueError('Incomplete numerical qualification')
    timing = receipt.get('timing', {})
    samples = timing.get('samples', [])
    if timing.get('passed') is not True or sorted(row['id'] for row in samples) != sorted(ids):
        raise ValueError('Incomplete timing qualification')
    for row in samples:
        if not row.get('seconds') or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in row['seconds']):
            raise ValueError('Invalid timing sample')
    for row in rows:
        if not row.get('referenceHash') or not row.get('outputHash') or not math.isfinite(row.get('maxAbsError', float('nan'))) or row['maxAbsError'] < 0:
            raise ValueError('Missing numerical evidence')
    return receipt
