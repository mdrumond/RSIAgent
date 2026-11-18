"""Trusted oracle process guard; embedded by the failure-first runtime."""
import ctypes, hashlib, importlib.util, json, os, pathlib, signal, subprocess, sys, tempfile, time

PAYLOAD = __PAYLOAD__

def receipt(path):
    found = None
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        for line in stream.read().decode('utf8', errors='replace').splitlines():
            try:
                value = json.loads(line)
                if isinstance(value, dict) and 'kernelSha256' in value: found = value
            except ValueError: pass
    return found

source = pathlib.Path(sys.argv[1]).resolve()
digest = hashlib.sha256((source / 'kernel.py').read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='rsi-failure-oracle-') as temporary:
    root = pathlib.Path(temporary) / 'checker'; root.mkdir()
    for name, text in {**PAYLOAD.get('helpers', {}), 'oracle.py': PAYLOAD['checker']}.items():
        path = root / name
        if not path.resolve().is_relative_to(root) or path == root: raise ValueError('Invalid oracle helper path')
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
    if PAYLOAD.get('runtime') is not None:
        path = root / 'runtime_identity.py'; path.write_text(PAYLOAD['runtimeHelper'])
        spec = importlib.util.spec_from_file_location('trusted_identity', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        if module.identity() != PAYLOAD['runtime']:
            print(json.dumps({'correct': None, 'kernelSha256': digest, 'evidence': {'reason': 'Qualified runtime identity changed'}})); sys.exit(125)
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0: raise RuntimeError('Subreaper unavailable')
    out = pathlib.Path(temporary) / 'stdout'; err = pathlib.Path(temporary) / 'stderr'; out.touch(); err.touch()
    env = {**os.environ, 'PYTHONPATH': str(source) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    begun = time.monotonic(); own = os.getpid(); known = {}; reason = None; terminating = None; peak = 0; observed_at = None; code = None
    def interrupted(sig, frame):
        global reason
        reason = reason or 'signal:' + str(sig)
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, interrupted)
    with out.open('wb') as stdout, err.open('wb') as stderr:
        child = subprocess.Popen([sys.executable, str(root / 'oracle.py'), str(source)], stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        while True:
            rows = {}
            for path in pathlib.Path('/proc').iterdir():
                if not path.name.isdecimal(): continue
                try:
                    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
                    rows[int(path.name)] = (int(fields[1]), fields[19], int(fields[21]) * os.sysconf('SC_PAGE_SIZE'), fields[0])
                except (OSError, ValueError, IndexError): pass
            parents = {own}; seen = {own}
            while parents:
                children = {pid for pid, row in rows.items() if row[0] in parents and pid != own}
                parents = children - seen; seen.update(children)
                for pid in children: known[pid] = rows[pid][1]
            live = {pid: row for pid, row in rows.items() if known.get(pid) == row[1] and row[3] not in ('Z', 'X')}
            rss = sum(row[2] for row in live.values()); peak = max(peak, rss); value = receipt(out)
            if value is not None: observed_at = observed_at or time.monotonic()
            if rss > PAYLOAD.get('rssBytes', 4096 * 1024**2): reason = reason or 'rss-limit'
            if out.stat().st_size + err.stat().st_size > 16 * 1024**2: reason = reason or 'output-limit'
            if time.monotonic() - begun > PAYLOAD.get('seconds', 180): reason = reason or 'elapsed-limit'
            if observed_at is not None and time.monotonic() - observed_at > PAYLOAD.get('cleanupSeconds', 5): reason = reason or 'cleanup-elapsed'
            if reason:
                terminating = terminating or time.monotonic()
                for pid, row in live.items():
                    try:
                        if pathlib.Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19] == row[1]:
                            os.kill(pid, signal.SIGKILL if time.monotonic() - terminating > 5 else signal.SIGTERM)
                    except OSError: pass
            complete = False
            while True:
                try: pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError: complete = True; break
                if not pid: break
                if pid == child.pid: code = os.waitstatus_to_exitcode(status); child.returncode = code
            if complete: break
            time.sleep(.05)
    value = receipt(out)
    valid = isinstance(value, dict) and value.get('kernelSha256') == digest and hashlib.sha256((source / 'kernel.py').read_bytes()).hexdigest() == digest and isinstance(value.get('evidence'), dict)
    cleanup_only = reason in ('elapsed-limit', 'cleanup-elapsed') and valid and value.get('correct') is False
    guard = {'reason': reason, 'commandExit': code, 'allChildrenReaped': complete, 'peakRssBytes': peak, 'elapsedSeconds': time.monotonic() - begun}
    if not valid or not complete or reason and not cleanup_only:
        value = {'correct': None, 'kernelSha256': digest, 'evidence': {'reason': 'Verification guard refused or receipt missing', 'processGuard': guard}}
    else:
        value['evidence']['processGuard'] = guard
        if cleanup_only: value['evidence']['cleanupElapsedAfterFailure'] = True
    sys.stderr.write(err.read_text(errors='replace')[-2048:]); print(json.dumps(value), flush=True)
    sys.exit(0 if value.get('correct') is True else 1 if value.get('correct') is False else 125)
