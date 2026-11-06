"""Inspect the files actually copied into a fresh qualified campaign."""
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tarfile
from . import baselines
from .baselines.projects import _LAUNCHER


def inspect_launch(request):
    target = request['target']
    options = request['profile']['environment']['options']
    if not options.get('target') or not options.get('runtime'):
        raise ValueError('A registered transport target and runtime are required')
    for workload in baselines.WORKLOADS:
        receipt = next(r for r in request['qualification']['receipts']
                       if r['target'] == target and r['workload'] == workload)
        identity = receipt['runtime']
        binding = {k: v for k, v in identity.items() if k != 'fingerprint'}
        fingerprint = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':'),
                                                allow_nan=False).encode()).hexdigest()
        if (identity.get('target') != target or identity.get('transportTarget') != options['target']
                or identity.get('runtimeName') != options['runtime'] or identity['fingerprint'] != fingerprint):
            raise ValueError('Actual dispatch profile differs from qualified runtime')
        expected = {**baselines.describe(workload), 'target': target, 'runtimeFingerprint': fingerprint}
        baselines.validate_receipt(receipt, expected)
        project = request['phase2Projects'][workload]
        verifier = project['verifier']
        paths = [Path(p).resolve() for p in verifier['files']]
        if len(paths) != 2 or {p.name for p in paths} != {'verify.py', 'verifier.tar.gz'}:
            raise ValueError('Expected the owned two-file verifier bundle')
        launcher = next(p for p in paths if p.name == 'verify.py')
        archive = next(p for p in paths if p.name == 'verifier.tar.gz')
        if archive.parent != launcher.parent:
            raise ValueError('Verifier archive must accompany its launcher')
        if launcher.read_text() != _LAUNCHER.replace('ARCHIVE_HASH', baselines.file_hash(archive)):
            raise ValueError('Frozen verifier launcher changed')
        contract = {'language': 'Triton-Ascend' if target == 'a3' else 'Catlass DSL', 'format': 1,
                    'workload': workload, 'target': target, 'expected': expected,
                    'qualification': receipt, 'identity': identity, 'warmup': 5, 'repeats': 20}
        files = {'ascend_tools/__init__.py': b'',
                 'contract.json': json.dumps(contract, sort_keys=True).encode()}
        package = Path(baselines.__file__).parent
        for path in package.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                files['ascend_tools/baselines/' + path.relative_to(package).as_posix()] = path.read_bytes()
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
