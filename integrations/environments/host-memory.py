"""Read-only aggregate cgroup-v2 proof; no transport or provisioning."""
import hashlib
import json
import posixpath
import re
from pathlib import Path


def host_memory_proof(configuration, read=lambda p: Path(p).read_text(),
                      stat=lambda p: Path(p).stat()):
    try:
        cap = configuration['max_bytes']
        if type(cap) is not int or cap <= 0:
            raise ValueError('Invalid host cap')
        membership = next(line[3:] for line in read('/proc/self/cgroup').splitlines()
                          if line.startswith('0::'))
        mounts = [line.split() for line in read('/proc/self/mountinfo').splitlines()
                  if ' - cgroup2 ' in line]
        # Multiple or escaped mount mappings need operator correction, not guesses.
        if len(mounts) != 1 or '\\' in mounts[0][3] + mounts[0][4]:
            raise ValueError('Unambiguous cgroup-v2 mount required')
        mount_root, mountpoint = mounts[0][3:5]
        slice_name = configuration.get('slice', 'rsi.slice')
        if not isinstance(slice_name, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*\.slice', slice_name):
            raise ValueError('Invalid shared slice unit')
        parts = membership.split('/')
        if slice_name in parts:
            scope = '/'.join(parts[:parts.index(slice_name) + 1])
            if not re.fullmatch(r'/(?:user\.slice/user-\d+\.slice/user@\d+\.service/)?' + re.escape(slice_name), scope):
                raise ValueError('RSI aggregate must be the shared system or user slice, not a job subtree')
            kind = 'rsi.slice'
        else:
            # Explicit registered container context only. A job subtree is never
            # inferred to be an aggregate, even when its memory.max is finite.
            scope = configuration.get('container_root')
            if not scope or scope != mount_root or scope == '/':
                raise ValueError('Register a memory-bounded container context or shared rsi.slice')
            kind = 'container-root'
        if not (membership == scope or membership.startswith(scope.rstrip('/') + '/')):
            raise ValueError('Command is outside aggregate scope')
        relative = posixpath.relpath(scope, mount_root)
        if relative == '..' or relative.startswith('../'):
            raise ValueError('Aggregate scope is outside visible cgroup mount')
        path = posixpath.normpath(mountpoint + '/' + relative)
        maximum = read(path + '/memory.max').strip()
        current = read(path + '/memory.current').strip()
        if not maximum.isdecimal() or not current.isdecimal() or not 0 < int(maximum) <= cap:
            raise ValueError('Actual aggregate memory.max is unavailable or exceeds configured host cap')
        identity = [read('/proc/sys/kernel/random/boot_id').strip(), scope,
                    stat(path).st_dev, stat(path).st_ino]
        fingerprint = hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()
        if configuration.get('fingerprint') not in (None, fingerprint):
            raise ValueError('Aggregate host scope changed between commands')
        return dict(available=True, kind=kind, scope=scope, fingerprint=fingerprint,
                    max_bytes=int(maximum), current_bytes=int(current))
    except (OSError, ValueError, KeyError, StopIteration, TypeError) as exc:
        return dict(available=False, code='MEMORY_ENFORCEMENT_UNAVAILABLE', detail=str(exc))
