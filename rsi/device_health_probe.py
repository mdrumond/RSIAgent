"""Stdlib-only device health probe for controllers and frozen Python 3.11 verifiers.

Run local_probe ONCE per command before Torch/NPU imports. All process occupancy
is retained and counted; no PID exclusion configuration exists. Numeric children
must exit before a fresh profiling command probe. This module supplies health
facts, not CPL permission: callers must validate their actual outer admission.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time


class AdmissionFailure(Exception):
    def __init__(self, kind, detail):
        self.kind = kind
        super().__init__(detail)


def _require(condition, kind, detail):
    if not condition:
        raise AdmissionFailure(kind, detail)


def _limit(remaining, ceiling=60):
    value = remaining()
    _require(value is None or (type(value) in (int, float) and
             math.isfinite(value) and value > 0), 'budget_exhausted',
             'Device admission elapsed allocation exhausted')
    return ceiling if value is None else min(value, ceiling)


def _fields(text):
    return {key.strip().lower(): value.strip() for key, value in
            (line.split(':', 1) for line in text.splitlines() if ':' in line)}


def _number(fields, name):
    value = fields.get(name.lower(), '')
    _require(bool(re.fullmatch(r'\d+(?:\.\d+)?', value)), 'metric_unavailable',
             'Missing or unsupported metric: ' + name)
    return float(value)


def _measure(raw, product, device):
    _require(isinstance(raw, dict), 'metric_unavailable', 'Malformed probe measurements')
    _require(raw.get('physical_device') == device, 'identity_mismatch', 'Probe device changed')
    queries = raw.get('queries', {})
    _require(isinstance(queries, dict), 'metric_unavailable', 'Malformed probe query set')
    names = ['board', 'health', 'usages', 'proc-mem']
    if product == 'a5':
        names.append('current-fault-event')
    for name in names:
        item = queries.get(name, {})
        _require(isinstance(item, dict), 'metric_unavailable', 'Malformed query: ' + name)
        _require(item.get('exit') == 0 and isinstance(item.get('stdout'), str),
                 item.get('failure', 'metric_unavailable'), 'Unavailable query: ' + name)
    board = _fields(queries['board']['stdout'])
    _require(_number(board, 'NPU ID') == device, 'identity_mismatch', 'Board device ID changed')
    _require(('950' if product == 'a5' else '910') in board.get('chip name', ''),
             'identity_mismatch', 'Measured chip family differs from profile product')
    health = _fields(queries['health']['stdout'])
    status = health.get('health status', health.get('hw health status'))
    _require(status is not None, 'metric_unavailable', 'Health status unavailable')
    _require(status.upper() == 'OK', 'unhealthy', 'Device health: ' + status)
    # A3 exposes hardware errors in health, rather than current-fault-event.
    error = health.get('error code', health.get('hw error code', health.get('hw health code')))
    _require(error is not None, 'metric_unavailable', 'Hardware error code unavailable')
    _require(error.upper() in ('NA', '0', '0X0', 'NONE'), 'unhealthy',
             'Device hardware error: ' + error)
    if product == 'a5':
        faults = _number(_fields(queries['current-fault-event']['stdout']), 'fault event count')
        _require(faults == 0, 'unhealthy', 'Current fault events present')
    usages = _fields(queries['usages']['stdout'])
    capacity = _number(usages, 'HBM Capacity(MB)')
    memory = _number(usages, 'HBM Usage Rate(%)')
    core = _number(usages, 'Aicore Usage Rate(%)')
    vector = _number(usages, 'Aivector Usage Rate(%)')
    utilization = max(core, vector)
    if product == 'a5':
        utilization = max(utilization, _number(usages, 'NPU Utilization(%)'),
                          _number(usages, 'Aicube Usage Rate(%)'))
    _require(capacity > 0 and all(0 <= n <= 100 for n in (memory, utilization)),
             'metric_unavailable', 'Invalid memory/utilization measurement')
    processes = queries['proc-mem']['stdout']
    pids = re.findall(r'Process id\s*:\s*(\d+)', processes, flags=re.I)
    _require(bool(pids) or 'No process in device.' in processes,
             'metric_unavailable', 'Process occupancy unavailable')
    return dict(health=status, memory_percent=memory, memory_capacity_mb=capacity,
                utilization_percent=utilization, process_ids=[int(pid) for pid in pids],
                idle=not pids and utilization == 0)



def _configuration(configuration):
    product, chip = configuration['product'], configuration['chip']
    _require((product == 'a3' and type(chip) is int and chip == 0) or
             (product == 'a5' and chip is None), 'configuration_invalid', 'Product/chip syntax mismatch')
    timeout = configuration.get('query_timeout_seconds', 8)
    _require(type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= 30,
             'configuration_invalid', 'Invalid per-query timeout')
    return product, chip, timeout


def assess(raw, *, physical_device, configuration):
    """Apply the same strict parser and admission policy to retained raw queries."""
    product, _, _ = _configuration(configuration)
    measurements = _measure(raw, product, physical_device)
    _require(measurements['idle'] or (not configuration.get('require_idle', False) and
             configuration.get('allow_busy') is True and measurements['memory_percent'] < 50),
             'device_busy', 'Healthy idle device required; configured ordinary sharing requires memory below 50%')
    return measurements


def local_probe(*, physical_device, configuration, remaining):
    """Bounded selected-device queries and a JSON-safe decision with exact raw output.

    remaining() is the caller's actual elapsed allocation (or None for unlimited).
    Settings: product='a3'/'a5', chip=0/None, query_timeout_seconds (default 8),
    allow_busy=False, require_idle=False. No registry, CPL or controller imports.
    """
    started = time.monotonic()
    raw = dict(physical_device=physical_device, queries={})
    result = dict(admitted=False, physical_device=physical_device, raw_measurements=raw,
                  observation='Device health unavailable')
    try:
        _require(type(physical_device) is int and physical_device >= 0,
                 'identity_mismatch', 'Explicit physical ID required')
        product, chip, timeout = _configuration(configuration)
        names = ['board', 'health', 'usages', 'proc-mem']
        if product == 'a5':
            names.append('current-fault-event')
        for query in names:
            allowance = _limit(remaining, timeout)
            argv = ['npu-smi', 'info', '-t', query, '-i', str(physical_device)]
            if chip is not None:
                argv += ['-c', str(chip)]
            try:
                p = subprocess.run(argv, capture_output=True, timeout=allowance)
                raw['queries'][query] = dict(argv=argv, exit=p.returncode,
                    stdout=p.stdout.decode(errors='replace'), stderr=p.stderr.decode(errors='replace'))
            except subprocess.TimeoutExpired as exc:
                raw['queries'][query] = dict(argv=argv, failure='query_timeout',
                    stdout=(exc.stdout or b'').decode(errors='replace'),
                    stderr=(exc.stderr or b'').decode(errors='replace'))
                raise AdmissionFailure('query_timeout', 'Unavailable query: ' + query)
            except OSError as exc:
                raw['queries'][query] = dict(argv=argv, failure='query_unavailable', error=str(exc))
                raise AdmissionFailure('query_unavailable', 'Unavailable query: ' + query)
            _limit(remaining)
        result['measurements'] = assess(raw, physical_device=physical_device, configuration=configuration)
        result.update(admitted=True, observation='Fresh bounded device measurements admitted')
    except AdmissionFailure as exc:
        result.update(failure=dict(type=exc.kind, detail=str(exc)), observation=str(exc))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        result.update(failure=dict(type='metric_unavailable', detail=str(exc)),
                      observation='Device health unavailable: ' + str(exc))
    result['elapsed_seconds'] = time.monotonic() - started
    return result


def standalone_source():
    """Exact UTF-8 module source for freezing via exec(source, private_namespace)."""
    return Path(__file__).read_text(encoding='utf-8')


def standalone_hash():
    """SHA256 of standalone_source().encode('utf-8'); freeze this with the verifier."""
    return hashlib.sha256(standalone_source().encode('utf-8')).hexdigest()
