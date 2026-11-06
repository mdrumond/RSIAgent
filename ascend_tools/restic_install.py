"""Resolve Restic, or explicitly install the pinned official release."""
import argparse
import bz2
import hashlib
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import urllib.request

VERSION = '0.19.1'
HASHES = {
    'amd64': 'f415415624dcc452f2a02b8c33641791a8c6d6d3b65bbb3543fcf9a25151585c',
    'arm64': 'a5f64aaab53d51e311fa3829124c5b703f2d14cf187d8640b6be3b2b49376465',
}


def resolve_restic():
    override = os.environ.get('RSI_RESTIC')
    if override:
        return override
    installed = shutil.which('restic')
    if installed:
        return installed
    cached = cache_path()
    if cached.is_file():
        return str(cached)
    raise RuntimeError('Restic is required: run python -m ascend_tools.restic_install --install')


def cache_path():
    root = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    return root / 'ascend-rsi' / 'tools' / ('restic-' + VERSION) / 'restic'


def check_version(binary):
    result = subprocess.run([binary, 'version'], check=True, stdout=subprocess.PIPE,
                            text=True, timeout=15)
    match = re.match(r'restic (\d+)\.(\d+)\.(\d+)', result.stdout)
    if not match or tuple(map(int, match.groups())) < (0, 17, 0):
        raise RuntimeError('Restic >= 0.17.0 is required; choose a newer executable with RSI_RESTIC')


def install():
    try:
        binary = resolve_restic()
    except RuntimeError:
        binary = None
    if binary:
        check_version(binary)
        return binary
    arch = {'x86_64': 'amd64', 'aarch64': 'arm64'}.get(platform.machine())
    if platform.system() != 'Linux' or arch not in HASHES:
        raise RuntimeError('Install Restic using your system package manager on this platform')
    target = cache_path()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    url = f'https://github.com/restic/restic/releases/download/v{VERSION}/restic_{VERSION}_linux_{arch}.bz2'
    with tempfile.TemporaryDirectory(prefix='.install-', dir=target.parent) as scratch:
        archive = Path(scratch) / 'restic.bz2'
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=60) as response, archive.open('wb') as output:
            while block := response.read(65536):
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != HASHES[arch]:
            raise RuntimeError('Restic release checksum mismatch')
        unpacked = Path(scratch) / 'restic'
        with bz2.open(archive, 'rb') as source, unpacked.open('wb') as output:
            shutil.copyfileobj(source, output, length=65536)
        unpacked.chmod(0o700)
        check_version(str(unpacked))
        os.replace(unpacked, target)
    return str(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true', help='Download the pinned official binary if needed')
    args = parser.parse_args()
    print(install() if args.install else resolve_restic())
