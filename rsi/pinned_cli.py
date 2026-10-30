"""Dispatch instance commands through that instance's immutable machinery."""
import json
import os
from pathlib import Path
import subprocess
import sys


def dispatch(instance, argv):
    root = Path(instance).expanduser().resolve(strict=True)
    manifest = json.loads((root / 'instance.json').read_text())
    if 'machinery' not in manifest:
        # Configuration-only legacy records have no executable snapshot.
        return None
    from .instance_dispatch import launch_context, command_policy
    required, repair = command_policy(argv)
    root, machinery, interpreter, env = launch_context(root, required=required, repair=repair)
    if 'machinery/rsi/cli.py' not in manifest['files']:
        raise ValueError('instance does not include a CLI release')
    if (Path(__file__).resolve().parent.parent == machinery and
            interpreter == sys.executable and env.get('PATH') == os.environ.get('PATH')):
        return None
    bootstrap = ('import sys; sys.path.insert(0, sys.argv.pop(1)); '
                 'from rsi.cli import main; raise SystemExit(main())')
    return subprocess.run([interpreter, '-I', '-c', bootstrap,
                           str(machinery), *argv], cwd=Path.cwd(), env=env).returncode
