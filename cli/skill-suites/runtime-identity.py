"""Fresh identity from the runtime-owned setup report, never candidate claims."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def identity():
    report = json.loads(Path(os.environ['RSI_SETUP_REPORT']).read_text())['identity']
    if not all(name in report['packages'] for name in ('torch', 'torch-npu', 'triton')):
        raise RuntimeError('A3 Torch-NPU and Triton packages required')
    digests = {}
    def digest(path):
        path = Path(path).resolve()
        if path not in digests:
            with path.open('rb') as stream:
                digests[path] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return digests[path]

    packages = {name: {'version': data[0], 'record': data[2],
                      'sources': [digest(stamp[0]) for stamp in data[3]]}
                for name, data in report['packages'].items()}
    compilers = {name: digest(stamp[0]) for name, stamp in report['compilers'].items()}
    versions = set()
    for root in {value for value in report['cann'].values() if value}:
        for relative in ('ascend_toolkit_install.info', 'version.cfg', 'version.info',
                         'aarch64-linux/ascend_toolkit_install.info', 'x86_64-linux/ascend_toolkit_install.info'):
            path = Path(root) / relative
            if path.is_file():
                versions.add((relative, digest(path)))
    board = subprocess.run(['npu-smi', 'info', '-t', 'board', '-i', os.environ['ASCEND_PHYSICAL_DEVICE'], '-c', '0'],
                           capture_output=True, text=True, timeout=5, check=True).stdout
    product = re.search(r'Chip Name\s*:\s*(\S+)', board)
    if not product or not versions or not report['compilers']:
        raise RuntimeError('Device, CANN version and compiler identity required')
    return {'setup': {'binding': report['binding'], 'python': report['python'][0],
                      'machine': report['machine'], 'packages': packages, 'compilers': compilers},
            'cannVersions': [list(version) for version in sorted(versions)], 'product': product.group(1)}
