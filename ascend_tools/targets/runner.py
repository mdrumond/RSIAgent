"""Public verifier boundary; inputs are frozen by the caller, never by candidates."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen(path, digest):
    if sha(path) != digest:
        raise ValueError('frozen input identity mismatch')
    return json.loads(Path(path).read_text())


def compare(actual, expected, *, atol=0., rtol=0.):
    """Independent finite scalar comparison used by the CPU control."""
    if not all(math.isfinite(x) and x >= 0 for x in (atol, rtol)):
        raise ValueError('invalid tolerances')
    if len(actual) != len(expected):
        raise ValueError('output extent mismatch')
    if any(not math.isfinite(a) or not math.isfinite(b) or
           abs(a-b) > atol + rtol*abs(b) for a, b in zip(actual, expected)):
        raise ValueError('numeric mismatch')


@contextmanager
def raw_output(directory):
    """Retain Python and native stdout/stderr; leave stdout free for one report."""
    with (directory/'stdout.log').open('w') as out, (directory/'stderr.log').open('w') as err:
        sys.stdout.flush()
        sys.stderr.flush()
        saved = [os.dup(1), os.dup(2)]
        try:
            os.dup2(out.fileno(), 1)
            os.dup2(err.fileno(), 2)
            with redirect_stdout(out), redirect_stderr(err):
                yield
        finally:
            import ctypes
            out.flush()
            err.flush()
            ctypes.CDLL(None).fflush(None)
            for fd, original in zip((1, 2), saved):
                os.dup2(original, fd)
                os.close(original)


def runtime_check(identity, target):
    if identity['target'] != target or identity['python'] != sys.version.split()[0]:
        raise ValueError('target/Python runtime identity mismatch')
    if target != 'cpu' and sys.version_info[:2] != (3, 11):
        raise ValueError('device target requires Python 3.11')
    if target != 'cpu' and (not identity.get('compiler_consumer_files') or not identity.get('cann_metadata')):
        raise ValueError('exact compiler and CANN identities required')
    for item in identity.get('compiler_consumer_files', []) + identity.get('cann_metadata', []):
        if sha(item['path']) != item['sha256']:
            raise ValueError('runtime consumer identity mismatch')


def verify(args):
    directory = Path(args.artifacts).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(target=args.target, mode=args.mode, status='failed', device_evidence=False,
                  compile='not-run', launch='not-run', correctness='not-run', performance='not-run',
                  source_sha256=args.source_sha256, cases_sha256=args.cases_sha256,
                  runtime_sha256=args.runtime_sha256,
                  verifier_sha256={p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    started = time.monotonic()
    def save():
        (directory/'result.json').write_text(json.dumps(report, sort_keys=True, allow_nan=False)+'\n')
    save()
    try:
        with raw_output(directory):
            try:
                editable = Path(args.candidate).resolve().parent
                for trusted in (Path(args.cases), Path(args.runtime), Path(__file__).parent, directory):
                    if trusted.resolve().is_relative_to(editable):
                        raise ValueError('trusted inputs/verifier/artifacts must be outside editable directory')
                cases = frozen(args.cases, args.cases_sha256)
                identity = frozen(args.runtime, args.runtime_sha256)
                if sha(args.candidate) != args.source_sha256:
                    raise ValueError('candidate source identity mismatch')
                runtime_check(identity, args.target)
                if not cases['cases']:
                    raise ValueError('nonempty frozen cases required')
                # Retain exact inputs, not only the hashes, on ordinary failures too.
                for name, path in [('candidate.py', args.candidate), ('cases.json', args.cases), ('runtime.json', args.runtime)]:
                    (directory/name).write_bytes(Path(path).read_bytes())
                if args.target == 'cpu':
                    from .triton import load_module
                    model = load_module('_trusted_cpu_candidate', args.candidate)
                    report['cases'] = []
                    if args.mode != 'test':
                        raise ValueError('CPU control provides numerical testing only')
                    for case in cases['cases']:
                        left, right = case['left'], case['right']
                        if cases['operation'] != 'add':
                            raise ValueError('unknown CPU control operation')
                        expected = [a+b for a, b in zip(left, right, strict=True)]
                        report['correctness'] = 'running'
                        actual = list(model.run(left, right))
                        display = lambda values: [x if math.isfinite(x) else repr(x) for x in values]
                        report['cases'].append(dict(actual=display(actual), expected=display(expected)))
                        save()
                        compare(actual, expected, atol=case.get('atol', 0), rtol=case.get('rtol', 0))
                    report['correctness'] = 'passed'
                else:
                    if args.target == 'a5-catlass':
                        from .catlass import execute
                    else:
                        from .triton import execute
                    execute(args, cases, identity, directory, report, save)
                if sha(args.candidate) != args.source_sha256:
                    raise ValueError('candidate changed during verification')
                frozen(args.cases, args.cases_sha256)
                frozen(args.runtime, args.runtime_sha256)
                report['status'] = 'passed'
            except Exception as exc:
                for stage in ('compile', 'launch', 'correctness', 'performance'):
                    if report[stage] == 'running':
                        report[stage] = 'failed'
                report.update(error_type=type(exc).__name__, error=str(exc))
                traceback.print_exc()
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        save()
    return report
