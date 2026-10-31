"""Built-in host device admission; no provider calls and no workload imports.

Profiles supply settings, never health callbacks. ``evidence_root`` must be an
existing host-owned directory outside the future Runner root. ``probe_id`` is a
stable host request identity; reusing it reconciles, never resubmits. Every new
admission/command needs a new identity to obtain fresh measurements.
"""
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from rsi.cpl_execution import CplExecutor
from rsi.python_execution import atomic_json, digest
from rsi.runtime import WorkflowError


# Export precisely the same stdlib-only query engine/parser used by verifiers.
from rsi.device_health_probe import AdmissionFailure, _require, _limit, assess, standalone_source, standalone_hash

_PROBE = standalone_source() + r"""
import sys
spec = json.loads(sys.argv[1])
deadline = time.monotonic() + spec['remaining']
result = local_probe(physical_device=spec['device'], configuration=dict(
    product=spec.get('product', 'a3' if spec['chip'] == 0 else 'a5'), chip=spec['chip'],
    query_timeout_seconds=spec['query_timeout']), remaining=lambda: max(0., deadline-time.monotonic()))
print(json.dumps(result['raw_measurements']), flush=True)
"""


def _outer(metadata, configuration):
    # Receipt comes from the host's actual outer submission, not actor metadata.
    receipt = json.loads(Path(configuration['outer_receipt']).read_text())
    handle = metadata.get('outer_handle')
    _require(isinstance(handle, str) and
             handle.startswith('remote:' + metadata['target'] + ':job:') and
             receipt.get('handle') == handle and receipt.get('target') == metadata['target'],
             'outer_admission_invalid', 'Actual outer CPL handle/target mismatch')
    npu = receipt.get('npu', {})
    _require(metadata['physical_device'] in npu.get('devices', []) and
             bool(npu.get('checked_at')) and bool(npu.get('pool')),
             'outer_admission_invalid', 'Outer CPL receipt lacks physical device admission')
    _require(configuration.get('outer_runtime') == metadata['runtime'] == receipt.get('runtime'),
             'outer_admission_invalid', 'Outer named runtime mismatch')
    _require(os.environ.get('RSI_OUTER_CPL_HANDLE') == handle and
             os.environ.get('RSI_CPL_RUNTIME') == metadata['runtime'],
             'outer_admission_invalid', 'Current worker context differs from actual outer admission')
    return receipt


def _remote(executor, spec, directory, remaining, raw):
    receipt = directory / 'submission.json'
    response = directory / 'submission-response.json'
    intent = directory / 'intent.json'
    if not receipt.exists() and response.exists():
        value = json.loads(response.read_text())
        if value.get('handle'):
            executor._handle(value)
            atomic_json(receipt, value)
    if not receipt.exists():
        _require(not intent.exists(), 'submission_unknown',
                 'Probe submission unknown; reconcile retained evidence without replay')
        script = directory / 'probe.sh'
        script.write_text('#!/bin/bash\nexec ' + shlex.join([
            executor.config.remote_python, '-c', _PROBE, json.dumps(spec)]) + '\n')
        atomic_json(intent, dict(status='dispatching'))
        try:
            value = executor._cli(['run', executor.config.target, '--runtime', executor.config.runtime,
                '--cwd', executor.config.command_cwd, '--npu', spec['device'], '--file', script,
                '--receipt-file', receipt], remaining, result_path=response)
            if not receipt.exists() and value.get('handle'):
                executor._handle(value)
                atomic_json(receipt, value)
        finally:
            if receipt.exists():
                raw['submission'] = json.loads(receipt.read_text())
    _require(receipt.exists(), 'submission_unknown', 'CPL probe returned no durable receipt')
    raw['submission'] = json.loads(receipt.read_text())
    handle = executor._handle(raw['submission'])
    _require(spec['device'] in raw['submission'].get('npu', {}).get('devices', []),
             'policy_denied', 'CPL probe receipt lacks declared physical device')
    _require(raw['submission'].get('runtime') == executor.config.runtime,
             'identity_mismatch', 'CPL receipt named runtime differs from profile')
    # Follow the exact retained handle after any disconnect. No replacement run.
    args = ['observe', handle]
    seconds = executor._observation_seconds(remaining)
    if seconds >= 1:
        args += ['--wait', '--timeout', seconds]
    terminal = executor._cli(args, remaining, result_path=directory / 'observe.json')
    raw['terminal'] = terminal
    _require(executor._handle(terminal) == handle, 'identity_mismatch', 'Probe handle changed')
    _require(terminal.get('state') in ('completed', 'failed', 'cancelled'),
             'probe_pending', 'Probe still observable; resume the same probe identity')
    _require(terminal.get('state') == 'completed' and terminal.get('exit') == 0,
             'probe_failed', 'CPL probe did not complete successfully')
    logs = executor._cli(['logs', handle, '--stream', 'stdout', '--tail', 100],
                         remaining, result_path=directory / 'logs.json')
    raw['logs'] = logs
    _require(executor._handle(logs) == handle, 'identity_mismatch', 'Probe log handle changed')
    return json.loads(logs['content'])


def guard(request, *, phase='admission', metadata, configuration, remaining):
    """Return a persisted decision. Host wiring owns CAS/SFT ingestion of it.

    Configuration: evidence_root, probe_id (admission), product, chip,
    executor (HERE), outer_receipt/outer_runtime (ON), query_timeout_seconds,
    allow_busy. Metadata: target/runtime/physical_device/executor_identity,
    controller_placement (here/on), outer_handle. CommandRequest supplies the
    command operation_id and category; admission may receive a host mapping.
    """
    started = time.monotonic()
    raw = {}
    result = dict(admitted=False, observation='Device admission unavailable',
                  physical_device=metadata.get('physical_device'), raw_measurements=raw,
                  probe_source_sha256=standalone_hash())
    directory = None
    try:
        _require(phase in ('admission', 'command'), 'configuration_invalid', 'Invalid guard phase')
        device = metadata.get('physical_device')
        _require(type(device) is int and device >= 0, 'identity_mismatch', 'Explicit physical ID required')
        _require(all(isinstance(metadata.get(k), str) and metadata[k] for k in ('target', 'runtime')),
                 'identity_mismatch', 'Target and named runtime required')
        product, chip = configuration['product'], configuration['chip']
        _require((product == 'a3' and type(chip) is int and chip == 0) or
                 (product == 'a5' and chip is None), 'configuration_invalid', 'Product/chip syntax mismatch')
        host_path = (getattr(request, 'root', None) if phase == 'admission' else
                     getattr(request, 'project', None))
        host_parent = (host_path.parent if phase == 'admission' else host_path.parent.parent) if host_path else None
        root = Path(configuration.get('evidence_root') or host_parent or '')
        _require(configuration.get('evidence_root') or host_parent is not None,
                 'configuration_invalid', 'Host-owned evidence scope required')
        _require(root.is_dir(), 'configuration_invalid', 'Existing host evidence root required')
        identity = (getattr(request, 'operation_id', None) if phase == 'command' else
                    configuration.get('probe_id', getattr(request, 'dispatch_id', None)))
        _require(isinstance(identity, str) and bool(identity), 'configuration_invalid', 'Stable probe identity required')
        binding = dict(phase=phase, identity=identity, metadata=dict(metadata), product=product, chip=chip)
        directory = root / ('device-' + digest(json.dumps(binding, sort_keys=True).encode()))
        directory.mkdir(exist_ok=True)
        atomic_json(directory / 'identity.json', binding)
        result['evidence_path'] = str(directory / 'decision.json')
        timeout = configuration.get('query_timeout_seconds', 8)
        _require(type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= 30,
                 'configuration_invalid', 'Invalid per-query timeout')
        queries = ['board', 'health', 'usages', 'proc-mem']
        if product == 'a5':
            queries.append('current-fault-event')
        spec = dict(device=device, product=product, chip=chip, queries=queries, query_timeout=timeout,
                    remaining=_limit(remaining, timeout * len(queries)))
        placement = metadata.get('controller_placement', 'remote')
        if placement in ('here', 'remote'):
            executor = configuration['executor']
            _require(isinstance(executor, CplExecutor) and all(
                executor.metadata.get(k) == metadata.get(k) for k in
                ('target', 'runtime', 'physical_device', 'executor_identity')),
                'identity_mismatch', 'CPL executor differs from admission metadata')
            raw['probe'] = _remote(executor, spec, directory, remaining, raw)
        elif placement in ('on', 'local'):
            _require(metadata.get('executor_identity') is None, 'identity_mismatch',
                     'On-target local executor identity must be None')
            raw['outer_submission'] = _outer(metadata, configuration)
            _require(not (directory / 'local-started.json').exists(), 'probe_unknown',
                     'Local probe already started; use a new explicit admission identity')
            atomic_json(directory / 'local-started.json', dict(started=True))
            p = subprocess.run([sys.executable, '-c', _PROBE, json.dumps(spec)],
                               capture_output=True, timeout=_limit(remaining, spec['remaining']))
            raw['local'] = dict(exit=p.returncode, stdout=p.stdout.decode(errors='replace'),
                                stderr=p.stderr.decode(errors='replace'))
            _require(p.returncode == 0, 'probe_failed', 'Local probe failed')
            raw['probe'] = json.loads(raw['local']['stdout'])
        else:
            raise AdmissionFailure('configuration_invalid', 'Controller placement must be here or on')
        atomic_json(directory / 'measurements.json', raw)
        _limit(remaining)
        category = getattr(request, 'category', '')
        policy = dict(configuration)
        policy['require_idle'] = (configuration.get('require_idle', False) or
            category in ('profile', 'profiling', 'timing', 'benchmark') or
            getattr(request, 'command', '') in ('profile', 'profiling', 'timing', 'benchmark') or
            bool(getattr(request, 'profile_path', None)))
        result['measurements'] = assess(raw['probe'], physical_device=device, configuration=policy)
        result.update(admitted=True, degraded=result['measurements']['degraded'],
            observation=('Explicitly authorized Alarm device admitted for ordinary work (degraded)'
                         if result['measurements']['degraded'] else 'Fresh bounded device measurements admitted'))
    except AdmissionFailure as exc:
        result.update(failure=dict(type=exc.kind, detail=str(exc)), observation=str(exc))
    except (KeyError, TypeError, ValueError, OSError, WorkflowError, subprocess.SubprocessError) as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            raw['timeout'] = dict(stdout=(exc.stdout or b'').decode(errors='replace'),
                                  stderr=(exc.stderr or b'').decode(errors='replace'))
        result.update(failure=dict(type='admission_unavailable', detail=str(exc)),
                      observation='Device admission unavailable: ' + str(exc))
    result['elapsed_seconds'] = time.monotonic() - started
    if directory is not None:
        raw['client_results'] = [dict(json.loads(path.read_text()),
            stdout=Path(str(path)[:-len('.client.json')] + '.stdout').read_text(errors='replace'),
            stderr=Path(str(path)[:-len('.client.json')] + '.stderr').read_text(errors='replace'))
            for path in sorted(directory.glob('*.client.json'))]
        atomic_json(directory / 'decision.json', result)
    return result
