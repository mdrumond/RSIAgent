"""Local Catlass bindings using the same host-owned drivers and evidence contracts."""
from dataclasses import replace
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import sys
import tarfile

from benchmarks.a5kernels.bz import BZSessionAdapter, _parse_output
from benchmarks.a5kernels.evidence import canonical_bytes
from benchmarks.a5kernels.fixtures import _CATLASS_DRIVER
from benchmarks.a5kernels.phase1_project_bz import CatlassProjectBackend, PROJECT_RESULT_MARKER, ProjectInfrastructureError
from benchmarks.a5kernels.phase2_attention_bz import (CatlassAttentionBackend, _ATTENTION_DRIVER,
    _ATTENTION_COMPILE_DRIVER, _input_payload, _parse_output as attention_output, validate_attention_candidate)
from benchmarks.a5kernels.phase2_attention import AttentionReceipt, AttentionInfrastructureError
from benchmarks.a5kernels.protocol import ExecutionReceipt
from benchmarks.a5kernels.profiling import EvidenceArchive, ProfileCapture, ProfileMetric
from benchmarks.a5kernels.profiling_bz import BZProfileBackend, _archive_entries, _read_metric_archive
from .execution import LocalProcessRunner, local_identity, preflight_local


def _stage(root, files):
    root.mkdir(parents=True, exist_ok=True)
    for source in files:
        path = root / source.relative_path
        if Path(source.relative_path).is_absolute() or '..' in Path(source.relative_path).parts:
            raise ValueError('kernel bundle path must be relative')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != source.content:
            raise ValueError('retained local kernel bundle differs')
        path.write_text(source.content)


class LocalCatlassExecutor:
    def __init__(self, worker_config):
        self.config = worker_config
        self.root = Path(worker_config['workspace'])
        self.runner = LocalProcessRunner(worker_config)
        self.identity = preflight_local(worker_config, catlass=True)
        source = Path(worker_config['catlass_source'])
        manifest = json.loads((source / 'python/tla_dsl/csrc/mlir/build/.codex-native-provenance.json').read_text())
        self.arguments = (self.identity['catlass_revision'],
            hashlib.sha256(canonical_bytes(manifest)).hexdigest(), manifest['artifacts'][0]['sha256'],
            manifest['ascendnpu_ir_gitlink'], manifest['ascendnpu_ir_install_commit'], manifest['cann_version'])
        # Reuse the exact artifact/dependency checks embedded in the host driver.
        namespace = {'__name__': 'rsi_local_catlass_provenance'}
        exec(_CATLASS_DRIVER, namespace)
        namespace['_native_artifact'](source, *self.arguments)
        self.runtime_provenance = (('catlass_revision', self.arguments[0]),
            ('catlass_source', str(source)), ('execution_profile', 'local'),
            *tuple(zip(('manifest_sha256', 'bridge_sha256', 'ascendnpu_ir_gitlink',
                        'ascendnpu_ir_install_commit', 'cann_version'), self.arguments[1:])))

    def probe_device(self, device):
        if device != self.config['physical_device']:
            raise ValueError('local device differs from worker configuration')

    def execute(self, argv, operation, *, environment=None, direct=False):
        environment = dict(environment or {})
        environment.update(CATLASS_SRC=self.config['catlass_source'], PYTHONDONTWRITEBYTECODE='1',
                           BZ_A5_PROFILE_PHYSICAL_DEVICE=str(self.config['physical_device']))
        environment['PYTHONPYCACHEPREFIX'] = None
        if direct:
            # Profilers and the application must see the same physical ordinal.
            environment['ASCEND_RT_VISIBLE_DEVICES'] = ''
            environment['A5KERNEL_PROFILE_DIRECT_PHYSICAL_DEVICE'] = '1'
        else:
            environment['A5KERNEL_PROFILE_DIRECT_PHYSICAL_DEVICE'] = '0'
        argv = [sys.executable if arg == 'python' else str(arg) for arg in argv]
        return self.runner.run(argv, operation, cwd=self.root, env=environment)


class LocalSessionAdapter(BZSessionAdapter):
    def __init__(self, worker_config):
        self._executor = LocalCatlassExecutor(worker_config)
        self.config = worker_config

    def prepare_execution(self, plan):
        plan = replace(plan, environment=plan.environment.with_binding(
            'BZ_A5_PROFILE_PHYSICAL_DEVICE', str(self.config['physical_device'])))
        return super().prepare_execution(plan)

    def execute(self, plan):
        plan = self.prepare_execution(plan)
        if plan.argv is None:
            raise ValueError('no executable runtime registered for this kernel')
        root = self._executor.root / '.a5kernels' / plan.execution_id / plan.attempt_id
        _stage(root, plan.files)
        (root / 'input.json').write_text(json.dumps({'input_a': plan.input_a, 'input_b': plan.input_b}))
        files = {source.relative_path for source in plan.files} | {'input.json'}
        argv = [str(root / arg) if arg in files else arg for arg in plan.argv]
        environment = {**dict(plan.environment.bindings), **dict.fromkeys(plan.environment.unset)}
        result = self._executor.execute([*argv, *self._executor.arguments], root / 'operation', environment=environment)
        return ExecutionReceipt(result.returncode, _parse_output(result.stdout) if result.returncode == 0 else (),
            result.stdout, result.stderr, getattr(result, 'local_handle', None) or json.loads((root / 'operation/transport.json').read_text())['handle'])


class LocalCatlassProjectBackend(CatlassProjectBackend):
    def __init__(self, worker_config, *, evidence_root):
        self.config = worker_config
        self._executor = LocalCatlassExecutor(worker_config)
        self._root = Path(evidence_root)
        self._device = worker_config['physical_device']

    def _execute(self, identity, files, mode, output_elements):
        root = self._executor.root / '.a5kernels/projects' / identity
        _stage(root, files)
        operation = root / 'operations' / mode
        result = self._executor.execute([sys.executable, '-B', str(root / 'host_driver.py'),
            str(root / 'kernel.py'), str(root / 'manifest.json'), mode, *self._executor.arguments], operation)
        records = [line[len(PROJECT_RESULT_MARKER):] for line in result.stdout.splitlines()
                   if line.startswith(PROJECT_RESULT_MARKER)]
        if result.returncode or len(records) != 1:
            raise ProjectInfrastructureError('local project driver failed: ' + result.stderr[-4000:])
        record = json.loads(records[0])
        if not isinstance(record.get('status'), str) or not record.get('diagnostic'):
            raise ProjectInfrastructureError('local project driver returned an invalid record')
        receipt = json.loads((operation / 'transport.json').read_text())
        evidence = self._retain_evidence({'schema': 'a5-phase1-project-observation-v1',
            'identity': identity, 'mode': mode, 'record': record, 'handle': getattr(result, 'local_handle', receipt['handle']),
            'execution_identity': local_identity(self.config),
            'stdout': result.stdout, 'stderr': result.stderr, 'exit_code': result.returncode})
        return record, evidence


class LocalCatlassAttentionBackend(CatlassAttentionBackend):
    def __init__(self, worker_config):
        self._executor = LocalCatlassExecutor(worker_config)
        self._device = worker_config['physical_device']

    def _dispatch(self, candidate_source, execution, workdir, *, compile_only):
        validate_attention_candidate(candidate_source)
        from benchmarks.a5kernels.protocol import SourceFile
        payload = json.dumps(_input_payload(execution), sort_keys=True, separators=(',', ':'))
        driver = _ATTENTION_COMPILE_DRIVER if compile_only else _ATTENTION_DRIVER
        identity = hashlib.sha256(canonical_bytes({'candidate': candidate_source,
            'input': payload, 'compile_only': compile_only, 'provenance': self._executor.runtime_provenance})).hexdigest()
        root = self._executor.root / '.a5kernels/attention' / identity
        _stage(root, (SourceFile('kernel.py', candidate_source), SourceFile('host_driver.py', driver), SourceFile('input.json', payload)))
        result = self._executor.execute([sys.executable, '-B', str(root / 'host_driver.py'),
            str(root / 'kernel.py'), str(root / 'input.json'), *self._executor.arguments], root / 'operation',
            environment={'A5KERNEL_BLOCK_NUM': '1'})
        if compile_only and result.returncode == 0 and result.stdout.splitlines().count('A5ATTENTION_COMPILED=1') != 1:
            raise AttentionInfrastructureError('local compile receipt is unavailable')
        output = attention_output(result.stdout, execution.case.output_elements) if result.returncode == 0 and not compile_only else ()
        return AttentionReceipt(result.returncode, output, result.stdout, result.stderr,
            getattr(result, 'local_handle', None) or json.loads((root / 'operation/transport.json').read_text())['handle'],
            (('case_id', execution.case.case_id), ('execution', 'local')))


class LocalProfileBackend(BZProfileBackend):
    def __init__(self, worker_config, *, evidence_directory):
        self._executor = LocalCatlassExecutor(worker_config)
        self._evidence_directory = Path(evidence_directory)

    def _workload(self, request):
        root = self._executor.root
        project = dict(request.plan.environment.bindings).get('A5PROJECT_MASKED_TIMING') == '1'
        bundle = root / (f'.a5kernels/projects/{request.plan.request_id}' if project else
                         f'.a5kernels/{request.execution_id}/{request.attempt_id}')
        _stage(bundle, request.plan.files)
        if not project:
            (bundle / 'input.json').write_text(json.dumps({'input_a': request.plan.input_a, 'input_b': request.plan.input_b}))
        names = {s.relative_path for s in request.plan.files} | {'input.json'}
        argv = [str(bundle / arg) if arg in names else arg for arg in request.workload_argv]
        provenance = dict(request.plan.runtime_provenance)
        if any(provenance.get(k) != v for k, v in self._executor.runtime_provenance):
            raise ValueError('local profiling runtime provenance differs')
        return [*argv, *self._executor.arguments]

    def _timing_replay(self, command):
        request = command.request
        environment = dict(request.plan.environment.bindings)
        environment.update(A5KERNEL_EMIT_TIMING='1', A5KERNEL_WARM_UP=str(request.warm_up),
                           A5KERNEL_LAUNCH_COUNT=str(request.launch_count))
        result = self._executor.execute(self._workload(request),
            self._evidence_directory / command.replay_id / 'operation', environment=environment)
        values = [line.split('=', 1)[1] for line in result.stdout.splitlines() if line.startswith('A5KERNEL_TIMING_US=')]
        if result.returncode or len(values) != 1:
            raise RuntimeError('local timing replay failed: ' + result.stderr[-4000:])
        return float(values[0]), self._retain_timing_evidence(command.replay_id, result.stdout)

    def capture(self, command):
        request = command.request
        root = self._evidence_directory / command.replay_id
        root.mkdir(parents=True, exist_ok=True)
        application = root / 'application.sh'
        argv = [sys.executable if arg == 'python' else str(arg) for arg in self._workload(request)]
        application.write_text('#!/bin/sh\nexec ' + shlex.join(argv) + '\n')
        application.chmod(0o700)
        raw = root / 'raw'
        args = ['msprof', 'op', '--application=' + str(application.resolve()), '--output=' + str(raw.resolve()),
                '--aic-metrics=' + command.metric.value, '--warm-up=' + str(request.warm_up),
                '--launch-count=' + str(request.launch_count), '--replay-mode=kernel']
        if command.kernel_name:
            args.append('--kernel-name=' + command.kernel_name)
        result = self._executor.execute(args, root / 'operation',
            environment=dict(request.plan.environment.bindings), direct=True)
        if result.returncode:
            raise RuntimeError('local profiling failed: ' + result.stdout[-2000:] + result.stderr[-2000:])
        archive_path = root / 'ascend-profile-summary.tar.gz'
        with tarfile.open(archive_path, 'w:gz') as archive:
            reports = sorted(raw.rglob('*.csv'))
            for i, path in enumerate(reports):
                if not path.name.startswith(('OpBasicInfo', 'PipeUtilization', 'Memory', 'Arithmetic', 'Resource')):
                    continue
                data = gzip.compress(path.read_bytes(), mtime=0)
                info = tarfile.TarInfo(f'{i}/{path.name}.gz'); info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            if command.metric is ProfileMetric.PIPE_TIMELINE:
                traces = sorted(raw.rglob('trace.json'))
                if len(traces) != 1:
                    raise ValueError('local PipeTimeline requires one trace.json')
                trace = json.loads(traces[0].read_text())
                events = trace if isinstance(trace, list) else trace['traceEvents']
                durations = [e for e in events if e.get('ph') == 'X' and isinstance(e.get('dur'), (float, int))]
                if not durations:
                    raise ValueError('local PipeTimeline contains no duration events')
                data = canonical_bytes({'events': len(durations), 'duration_us': sum(e['dur'] for e in durations)})
                info = tarfile.TarInfo('timeline-summary.json'); info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        data = archive_path.read_bytes()
        kernels, summary = _read_metric_archive(archive_path, command.metric)
        evidence = EvidenceArchive(archive_path.name, hashlib.sha256(data).hexdigest(), len(data),
            str(raw.resolve()), True, _archive_entries(data))
        return ProfileCapture(command.metric, request.source_fingerprint, request.execution_id,
            command.replay_id, kernels if command.metric is ProfileMetric.BASIC_INFO else (command.kernel_name,), summary, evidence)
