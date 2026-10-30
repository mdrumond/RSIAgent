"""Dispatch instance commands through that instance's immutable machinery."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def dispatch(instance, argv):
    root = Path(instance).expanduser().resolve(strict=True)
    manifest = json.loads((root / 'instance.json').read_text())
    if 'machinery' not in manifest:
        # Configuration-only legacy records have no executable snapshot.
        return None
    machinery = root / 'machinery'
    if machinery.resolve() != machinery:
        raise ValueError('pinned machinery directory changed')
    if Path(__file__).resolve().parent.parent == machinery:
        return None
    for relative, expected in manifest['files'].items():
        if relative.startswith('machinery/'):
            path = root / relative
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'pinned machinery integrity failure: {relative}')
    if 'machinery/rsi/cli.py' not in manifest['files']:
        raise ValueError('instance does not include a CLI release')
    bootstrap = ('import sys; sys.path.insert(0, sys.argv.pop(1)); '
                 'from rsi.cli import main; raise SystemExit(main())')
    return subprocess.run([sys.executable, '-I', '-c', bootstrap,
                           str(machinery), *argv], cwd=Path.cwd()).returncode
