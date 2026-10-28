"""Readable local exports of completed matched-memory trials.

The manifest is an inspection view, not an execution or admission contract.
Evidence stays in the original trial; only task and memory snapshots are copied.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


MODEL_FIELDS = ('model', 'backend_model', 'provider', 'backend', 'coding_engine')
RUNTIME_FIELDS = ('target', 'product', 'architecture', 'cann_version',
                  'compiler_version', 'catlass_revision', 'triton_revision')


def _summary(identity, fields):
    if isinstance(identity, str):
        identity = {fields[0]: identity}
    if not isinstance(identity, dict):
        return {}
    return {key: value for key in fields if isinstance(value := identity.get(key), (str, int, float, bool))
            and not (isinstance(value, str) and
                     (value.startswith(('/', '~')) or '://' in value or re.match(r'^[A-Za-z]:[\\/]', value)))}


def _artifact(trial, digest):
    if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
        raise ValueError('invalid retained artifact identifier')
    path = trial / 'artifacts' / 'objects' / digest[:2] / digest[2:]
    if not path.is_file():
        raise ValueError(f'missing retained artifact: {digest}')
    return path


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def export_trial(trial, output):
    """Export completed trial metadata without changing any retained source file."""
    trial, output = Path(trial).resolve(), Path(output).resolve()
    if output == trial or trial in output.parents:
        raise ValueError('output must be outside the retained trial')
    if output.exists():
        raise ValueError('output already exists')
    if not output.parent.is_dir():
        raise ValueError('output parent must exist')
    # Open the existing lock read-only: inspection must not create source files.
    with (trial / '.evaluation.lock').open('rb') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('trial is active; export after its judgment is retained') from None
        state = json.loads((trial / 'evaluation.json').read_bytes())
        decision = state.get('decision')
        if not isinstance(decision, dict) or decision.get('decision') not in {
                'accept', 'rewrite', 'reject', 'inconclusive'}:
            raise ValueError('trial has no completed judgment')
        identity = state['identity']
        if not decision.get('evaluation_id'):
            raise ValueError('completed judgment has no evaluation ID')
        if state.get('task') != decision.get('trial_task'):
            raise ValueError('retained task and judgment disagree')
        evidence = dict(decision['artifacts'])
        for name, arm in state['arms'].items():
            for role, digest in (arm.get('outcome') or {}).get('artifacts', {}).items():
                key = name + ':' + role
                if key in evidence and evidence[key] != digest:
                    raise ValueError('retained arm evidence and judgment disagree')
                evidence[key] = digest
        paths = {role: _artifact(trial, digest) for role, digest in evidence.items()}
        memories = {name: _artifact(trial, identity[name + '_sha256'])
                    for name in ('baseline', 'treatment')}
        proposal = _artifact(trial, identity['proposal_sha256'])
        if any(digest not in evidence.values() for digest in decision.get('evidence', [])):
            raise ValueError('judgment cites missing evidence')
        staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=output.parent))
        try:
            memory_refs = {}
            for name, path in memories.items():
                label = 'control' if name == 'baseline' else 'treatment'
                data = path.read_bytes()
                try:
                    parsed = json.loads(data)
                except (ValueError, UnicodeDecodeError):
                    filename = label + '-memory.txt'
                    (staging / filename).write_bytes(data)
                else:
                    filename = label + '-memory.json'
                    _write_json(staging / filename, parsed)
                memory_refs[label] = filename
            task = None
            if state.get('task') is not None:
                _write_json(staging / 'task.json', state['task'])
                task = 'task.json'
            manifest = dict(
                schema='memory-trial-manifest-v1', evaluation_id=decision['evaluation_id'],
                source=dict(evaluation=os.path.relpath(trial / 'evaluation.json', output),
                            proposal=os.path.relpath(proposal, output)),
                model=_summary(identity.get('model'), MODEL_FIELDS),
                runtime=_summary(identity.get('runtime'), RUNTIME_FIELDS),
                task=task, memory=memory_refs, sequence=identity['sequence'],
                order=state.get('order', []),
                allocation_seconds=dict(trial=state['allocation'],
                    configured_trial=identity['max_seconds'], configured_arm=identity['arm_seconds'],
                    arm=state.get('arm_allocation')),
                result={key: decision[key] for key in ('decision', 'reason', 'validity',
                    'assessment', 'error_type', 'instructions', 'next_probe') if key in decision},
                evidence={role: os.path.relpath(path, output)
                          for role, path in sorted(paths.items())},
                cited_evidence=[role for role, digest in sorted(evidence.items())
                                if digest in decision.get('evidence', [])],
                evidence_scope='Local view; evidence paths require the original retained trial. '
                    'Only task and memory snapshots are copied; runtime details are summarized.')
            _write_json(staging / 'manifest.json', manifest)
            if output.exists():
                raise ValueError('output already exists')
            staging.rename(output)
            return output / 'manifest.json'
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export', help='Export a completed memory trial for local inspection')
    export.add_argument('--trial', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        path = export_trial(args.trial, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f'Cannot export memory trial: {error}\n')
    print(path)


if __name__ == '__main__':
    main()
