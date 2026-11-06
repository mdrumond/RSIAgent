"""Inspect the files actually copied into a fresh qualified campaign."""
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tarfile
from . import baselines
from .baselines.projects import _LAUNCHER
from .phase2_focus import FOCUSED_LAUNCHER, focused_files, select_focus


def json_equivalent(left, right):
    """JSON numbers retain value across a Node bridge; booleans remain distinct."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(json_equivalent(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(json_equivalent(a, b) for a, b in zip(left, right))
    return left == right


def inspect_launch(request):
    target = request['target']
    options = request['profile']['environment']['options']
    if not options.get('target') or not options.get('runtime'):
        raise ValueError('A registered transport target and runtime are required')
    for workload in baselines.WORKLOADS:
        receipt = next(r for r in request['qualification']['receipts']
                       if r['target'] == target and r['workload'] == workload)
        project = request['phase2Projects'][workload]
        verifier = project['verifier']
        paths = [Path(p).resolve() for p in verifier['files']]
        if len(paths) != 2 or {p.name for p in paths} != {'verify.py', 'verifier.tar.gz'}:
            raise ValueError('Expected the owned two-file verifier bundle')
        launcher = next(p for p in paths if p.name == 'verify.py')
        archive = next(p for p in paths if p.name == 'verifier.tar.gz')
        if archive.parent != launcher.parent:
            raise ValueError('Verifier archive must accompany its launcher')
        focus = project.get('benchmark_focus')
        template = FOCUSED_LAUNCHER if focus is not None else _LAUNCHER
        if launcher.read_text() != template.replace('ARCHIVE_HASH', baselines.file_hash(archive)):
            raise ValueError('Frozen verifier launcher changed')
        with tarfile.open(archive) as tar:
            try:
                member = tar.getmember('contract.json')
            except KeyError as error:
                raise ValueError('Frozen verifier contract is missing') from error
            if not member.isfile():
                raise ValueError('Frozen verifier contract must be a file')
            contract_bytes = tar.extractfile(member).read()
        archived = json.loads(contract_bytes)
        if not isinstance(archived, dict) or not all(isinstance(archived.get(k), dict) for k in ('qualification', 'identity')):
            raise ValueError('Frozen verifier contract is malformed')
        original_receipt = archived['qualification']
        identity = archived['identity']
        binding = {k: v for k, v in identity.items() if k != 'fingerprint'}
        fingerprint = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':'),
                                                allow_nan=False).encode()).hexdigest()
        if (identity.get('target') != target or identity.get('transportTarget') != options['target']
                or identity.get('runtimeName') != options['runtime'] or identity['fingerprint'] != fingerprint):
            raise ValueError('Actual dispatch profile differs from qualified runtime')
        expected = {**baselines.describe(workload), 'target': target, 'runtimeFingerprint': fingerprint}
        baselines.validate_receipt(original_receipt, expected)
        baselines.validate_receipt(receipt, expected)
        contract = {'language': 'Triton-Ascend' if target == 'a3' else 'Catlass DSL', 'format': 1,
                    'workload': workload, 'target': target, 'expected': expected,
                    'qualification': receipt, 'identity': receipt['runtime'], 'warmup': 5, 'repeats': 20}
        if not json_equivalent(archived, contract):
            raise ValueError('Frozen verifier contract differs from qualified request')
        files = {'ascend_tools/__init__.py': b'', 'contract.json': contract_bytes}
        package = Path(baselines.__file__).parent
        for path in package.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                files['ascend_tools/baselines/' + path.relative_to(package).as_posix()] = path.read_bytes()
        if focus is not None:
            original_focus = select_focus(original_receipt)
            if not json_equivalent(focus, original_focus):
                raise ValueError('Project focus differs from qualified timing evidence')
            files.update(focused_files(original_focus))
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            if len(members) != len(files) or {m.name for m in members} != files.keys():
                raise ValueError('Frozen verifier package membership differs')
            if any(not m.isfile() or tar.extractfile(m).read() != files[m.name] for m in members):
                raise ValueError('Frozen verifier contract/reference differs from qualified machinery')
        base = [options.get('python', 'python3'), str(launcher), '--candidate',
                str(Path(project['source']).resolve() / 'kernel.py'), '--artifacts', '../verification-artifacts']
        for key, mode in [('command', 'correctness'), ('performance_command', 'performance')]:
            if shlex.split(verifier[key]) != base + ['--mode', mode]:
                raise ValueError('Verifier execution command differs from owned contract')
    return {'verified': len(baselines.WORKLOADS), 'target': target}


if __name__ == '__main__':
    print(json.dumps(inspect_launch(json.load(sys.stdin))))
