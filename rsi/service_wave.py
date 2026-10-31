"""Instance wave binding: canonical child composition, parent-only publication."""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time

from core.artifacts import ContentAddressedArtifacts
from rsi.contracts import CycleSpec, ProjectSpec
from rsi.executor_profiles import FrozenExecutorFactory, ProfileCommandGuard
from rsi.jobs import _alive
from rsi.project_composition import ProjectComposition, file_identity, verify_kdb
from rsi.project_wave import ProjectWave
from rsi.project_binding import admission_hash, source_identity, validate_resolution, attach_execution_context
from rsi.python_execution import atomic_json, ingest_elapsed
from rsi.setup_telemetry import SetupTelemetry
from rsi.skills import SkillBundle


class WorkerArtifacts(ContentAddressedArtifacts):
    """Read immutable parent objects; all child writes remain in its own store."""
    def __init__(self, root, parent):
        super().__init__(root)
        self.parent = ContentAddressedArtifacts(parent)

    def read_bytes(self, reference):
        if self._path(reference).exists():
            return super().read_bytes(reference)
        return self.parent.read_bytes(reference)


def admission_intervals(branch):
    """Original attempts and bounded observations, each retained under its identity."""
    records = [json.loads(path.read_text()) for path in branch.glob('worker-admission-*.json')]
    for path in branch.glob('admission-observations-*.json'):
        records.extend(json.loads(path.read_text()))
    return records


@dataclass(frozen=True)
class ProjectWorkerFactory:
    instance: str
    cycle_root: str
    admission_json: str
    test_bindings: object = None
    executor_factory: object = None

    def prepare_worker(self, request):
        """Apply target context only in the dedicated child, before runner imports."""
        worker = json.loads((request.root.parent / 'worker.json').read_text())
        if worker['pid'] != os.getpid() or worker['dispatch_id'] != request.dispatch_id:
            raise ValueError('worker environment may only be prepared by its owning child')
        path = request.root.parent / 'worker-admission.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if previous['dispatch_id'] == request.dispatch_id and previous['pid'] == os.getpid():
                if previous['receipt'].get('admitted') is True:
                    return
                raise ValueError('original worker admission was denied; no implicit retry')
        admitted = json.loads(self.admission_json)
        device = next(d for d in admitted['devices'] if d['device'] == request.device)
        identity = device.get('worker_profile') or device.get('executor_identity')
        if identity is None:
            if self.test_bindings is None:
                raise ValueError('production NPU worker requires a trusted host admission profile')
            return
        if self.executor_factory is None:
            raise ValueError('trusted NPU worker environment/guard resolver required')
        inherited = dict(os.environ)
        environment = self.executor_factory.worker_environment(identity, request, inherited=inherited)
        # This process belongs exclusively to the project; parent env is untouched.
        os.environ.clear()
        os.environ.update(environment)
        source_elapsed = (admitted['operations'][request.project_id].get('source_interval', {}).get('elapsed', 0.)
                          if not request.reopen and request.actor_elapsed == 0 else 0.)
        deadline = time.monotonic() + max(0., request.actor_seconds - request.actor_elapsed - source_elapsed)
        def remaining():
            limits = [max(0., deadline - time.monotonic())]
            if admitted['wall_deadline'] is not None:
                limits.append(max(0., admitted['wall_deadline'] - time.time()))
            return min(limits)
        started, monotonic_start = time.time(), time.monotonic()
        receipt = None
        try:
            receipt = self.executor_factory.guard(identity, request, phase='admission', remaining=remaining)
        except Exception as exc:
            receipt = getattr(exc, 'admission_receipt', dict(admitted=False,
                observation='worker admission failed', error_type=type(exc).__name__))
            raise
        finally:
            elapsed = time.monotonic() - monotonic_start
            record = dict(dispatch_id=request.dispatch_id, pid=os.getpid(),
                profile=identity, receipt=receipt, started=started, elapsed=elapsed,
                ended=max(time.time(), started + elapsed))
            atomic_json(path.with_name('worker-admission-' + request.dispatch_id + '.json'), record)
            atomic_json(path, record)

    def __call__(self, request):
        self.prepare_worker(request)
        from rsi.services import InstanceServices
        admitted = json.loads(self.admission_json)
        operation = admitted['operations'][request.project_id]
        if request.memory != bytes.fromhex(admitted['memory_bytes']) or request.skills != bytes.fromhex(admitted['skills_bytes']):
            raise ValueError('worker snapshots differ from frozen service admission')
        services = InstanceServices(self.instance, test_bindings=self.test_bindings,
                                   executor_factory=self.executor_factory)
        services.store = WorkerArtifacts(request.root.parent / 'service-artifacts', services.store.root)
        services.setup = SetupTelemetry(services.store)
        binding = admitted['binding']
        composition = ProjectComposition(services, Path(self.cycle_root), binding,
                                         CycleSpec.from_dict(binding['spec']))
        composition.kdb_path = Path(self.cycle_root) / 'knowledge.sqlite3'
        composition.kdb_stat = admitted['kdb_stat']
        project_binding = dict(operation['binding'])
        device = next(d for d in admitted['devices'] if d['device'] == request.device)
        profile = device.get('worker_profile') or device.get('executor_identity')
        project_binding['executor_identity'] = device.get('executor_identity')
        project_binding['worker_profile'] = profile
        project_binding['executor_provenance'] = admitted['identities'].get(profile)
        logical = dict(operation_id=request.project_id, spec=operation['spec'], config=operation['config'],
            memory_ref=admitted['memory'], skills_ref=admitted['skills'], configuration_ref=admitted['configuration'])
        resolution = None
        binding_interval = None
        if request.reopen:
            retained = json.loads((request.root / 'service-binding.json').read_text())
            if any(retained.get(k) != v for k, v in project_binding.items()):
                raise ValueError('continuation differs from original service/executor binding')
            if retained.get('resolved_binding_ref'):
                store = ContentAddressedArtifacts(request.root / 'artifacts')
                resolution = validate_resolution(logical, store.read_json(retained['resolved_binding_ref']),
                    device=request.device, executor_identity=device.get('executor_identity'))
            elif admitted.get('binding_contracts', {}).get(profile):
                raise ValueError('existing runner lacks a frozen lease binding; never regenerate on resume')
            project_binding = retained
        elif admitted.get('binding_contracts', {}).get(profile):
            started, monotonic_start = time.time(), time.monotonic()
            admission_path = request.root.parent / 'worker-admission.json'
            admission_elapsed = json.loads(admission_path.read_text())['elapsed'] if admission_path.exists() else 0.
            source_elapsed = (operation.get('source_interval', {}).get('elapsed', 0.)
                              if request.actor_elapsed == 0 else 0.)
            deadline = monotonic_start + max(0., request.actor_seconds - request.actor_elapsed - admission_elapsed - source_elapsed)
            def remaining():
                limits = [max(0., deadline - time.monotonic())]
                if admitted['wall_deadline'] is not None:
                    limits.append(max(0., admitted['wall_deadline'] - time.time()))
                return min(limits)
            resolution = self.executor_factory.bind_project(profile, request, logical, remaining=remaining)
            if request.root.exists():
                raise ValueError('host binder must not precreate the canonical runner root')
            resolution = validate_resolution(logical, resolution, device=request.device,
                                               executor_identity=device.get('executor_identity'))
            evidence = resolution['evidence']
            if evidence['logical_source_identity'] != operation['logical_source_identity']:
                raise ValueError('host binder differs from the parent-frozen logical source identity')
            public = evidence.get('public_runtime', {})
            metadata = admitted['identities'][profile]
            if any(k in public and str(public[k]) != str(metadata[k]) for k in
                   ('target', 'runtime', 'physical_device', 'logical_device', 'outer_handle', 'controller_placement')
                   if k in metadata):
                raise ValueError('public runtime facts differ from the actual leased profile')
            for config, key in ((operation['config'], 'logical_source_identity'),
                                (resolution['config'], 'resolved_source_identity')):
                if source_identity(config['source'], verifier=config['verifier'],
                                   generated_paths=config.get('generated_paths', ())) != evidence[key]:
                    raise ValueError('host binder source identity differs from actual input tree')
            elapsed = time.monotonic() - monotonic_start
            binding_interval = dict(started=started, elapsed=elapsed, ended=max(time.time(), started + elapsed))
            project_binding['resolved_binding_ref'] = services.store.put_json(resolution)
            project_binding['logical_admission_hash'] = admission_hash(logical)
        config = (resolution['config'] if resolution else
                  dict(operation['config'], executor_identity=device.get('executor_identity')))
        spec = resolution['spec'] if resolution else operation['spec']
        if resolution and resolution['evidence']['logical_source_identity'] != operation['logical_source_identity']:
            raise ValueError('resolved source no longer authenticates the frozen logical admission')
        backend = services.store.put_json(project_binding)
        runner = composition.runner(request.root, spec=ProjectSpec.from_dict(spec),
            bundle=SkillBundle.from_dict(json.loads(request.skills)), config=config,
            memory_ref=admitted['memory'], backend_identity=backend, reopen=request.reopen,
            configuration_ref=admitted['configuration'], command_guard=(
                ProfileCommandGuard(self.executor_factory, profile)
                if profile and device.get('executor_identity') is None else None))
        if resolution:
            from core.artifacts import canonical_bytes
            if (runner.spec.to_dict() != resolution['spec']
                    or canonical_bytes(asdict(runner.config)) != canonical_bytes(resolution['config'])):
                raise ValueError('canonical runner differs from the exact retained lease binding')
        if not request.reopen:
            from rsi.evaluation import case_identity
            if resolution:
                if runner.backend.project_hash() != resolution['evidence']['resolved_source_identity']:
                    raise ValueError('canonical source snapshot differs from trusted lease resolution')
                runner.store.put_json(project_binding)
                runner.store.put_json(resolution)
                runner.append(dict(type='lease-binding', reference=project_binding['resolved_binding_ref'],
                                   artifact_ref=backend, source_hash=runner.backend.project_hash()))
            atomic_json(request.root / 'service-binding.json', project_binding)
            atomic_json(request.root / 'training-identity.json', case_identity(runner))
        admission_path = request.root.parent / 'worker-admission.json'
        admission = json.loads(admission_path.read_text()) if admission_path.exists() else None
        for interval in admission_intervals(request.root.parent):
            ingest_elapsed(runner.workflow.accounting, interval.get('interval_id',
                'worker-admission:' + interval['dispatch_id']),
                {'elapsed': interval['elapsed']}, category='execution',
                observed_start=interval['started'], observed_end=interval['ended'],
                observed_execution_start=interval['started'],
                observed_execution_end=interval['started'] + interval['elapsed'])
        if binding_interval:
            ingest_elapsed(runner.workflow.accounting, 'lease-binding:' + request.dispatch_id,
                {'elapsed': binding_interval['elapsed']}, category='execution',
                observed_start=binding_interval['started'], observed_end=binding_interval['ended'],
                observed_execution_start=binding_interval['started'],
                observed_execution_end=binding_interval['started'] + binding_interval['elapsed'])
        source_interval = operation.get('source_interval')
        if source_interval:
            ingest_elapsed(runner.workflow.accounting, 'logical-source:' + request.project_id,
                {'elapsed': source_interval['elapsed']}, category='execution',
                observed_start=source_interval['started'], observed_end=source_interval['ended'],
                observed_execution_start=source_interval['started'],
                observed_execution_end=source_interval['started'] + source_interval['elapsed'])
        attach_execution_context(runner, project_binding, resolution,
                                 admission['receipt'] if admission else None)
        if admitted['wall_deadline'] is not None:
            runner.workflow.constrain_wall_deadline('instance-wave', admitted['wall_deadline'])
        if request.recovery_receipt:
            authority = services.store.put_json(json.loads(request.recovery_receipt))
            runner.workflow.grant_actor_seconds('wave-harness-recovery', 7200., authority)
        if request.reopen and not runner.workflow.pending:
            runner.workflow.resume()
        canonical_run = runner.run
        def run():
            try:
                return canonical_run()
            finally:
                if composition.database:
                    composition.database.close()
        runner.run = run
        return runner


class WaveBindings:
    def __init__(self, cycle):
        self.cycle = cycle
        self.policy = cycle.config['project_wave']
        self.active_pool = None

    def _pool(self, wave):
        cycle = self.cycle
        return ProjectWave(cycle.root / 'waves' / wave['id'],
            [op['id'] for op in wave['operations']], [d['device'] for d in self.policy['devices']],
            cycle.store.read_bytes(wave['memory']), cycle.store.read_bytes(wave['skills']),
            max_parallel=self.policy['max_parallel'])

    def _prepare(self, maximum):
        cycle = self.cycle
        state = cycle.scheduler.state
        wave = cycle.scheduler.reserve_original_wave(maximum)
        if 'admission_ref' not in wave:
            kdb_path = cycle.root / 'knowledge.sqlite3'
            admission = cycle.root / 'kdb-admission.json'
            if admission.exists():
                identity = json.loads(admission.read_text())
                if identity['sha256'] != cycle.binding['kdb'] or identity['stat'] != file_identity(kdb_path):
                    raise ValueError('shared read-only wave KDB differs from admitted identity')
                kdb_stat = identity['stat']
            else:
                kdb_stat = verify_kdb(kdb_path, cycle.binding['kdb'])
                atomic_json(admission, dict(sha256=cycle.binding['kdb'], stat=kdb_stat))
            identities, binding_contracts = {}, {}
            for device in self.policy['devices']:
                identity = device.get('worker_profile') or device.get('executor_identity')
                if identity is None:
                    if cycle.services.test is None:
                        raise ValueError('production wave devices require trusted executor bindings')
                    continue
                if cycle.services.executor_factory is None:
                    raise ValueError('ignored host executor profile resolver required')
                resolver = cycle.services.executor_factory
                if not all(callable(getattr(resolver, name, None)) for name in
                           ('metadata', 'worker_environment', 'guard')):
                    raise ValueError('NPU profiles must provide trusted worker context and health/policy guards')
                metadata = resolver.metadata(identity)
                if str(metadata['physical_device']) != device['device']:
                    raise ValueError('executor profile physical device differs from admitted lease')
                if metadata['executor_identity'] != device.get('executor_identity'):
                    raise ValueError('worker placement differs from configured local/target execution binding')
                identities[identity] = metadata
                contract = resolver.binding_contract(identity) if callable(getattr(resolver, 'binding_contract', None)) else None
                if contract:
                    binding_contracts[identity] = contract
                elif cycle.services.test is None:
                    raise ValueError('production NPU profiles require a shipped lease-bound project binder')
            admissions = {}
            for op in wave['operations']:
                spec = ProjectSpec.from_dict(op['spec'])
                binding = dict(memory=wave['memory'], skills=wave['skills'],
                    configuration=cycle.spec.configuration_ref, kdb=cycle.binding['kdb'],
                    provider=cycle.config['provider'], test_identity=cycle.binding['test_identity'],
                    actor_seconds_baseline=0.)
                config = asdict(cycle.services.project_config(cycle.config, spec))
                started, monotonic_start = time.time(), time.monotonic()
                source_hash = source_identity(config['source'], verifier=config['verifier'],
                                              generated_paths=config.get('generated_paths', ()))
                elapsed = time.monotonic() - monotonic_start
                admissions[op['id']] = dict(spec=op['spec'], binding=binding, config=config,
                    logical_source_identity=source_hash, source_interval=dict(started=started,
                    elapsed=elapsed, ended=max(time.time(), started + elapsed)))
                relative = Path('waves') / wave['id'] / 'projects' / op['id'] / 'runner'
                state.setdefault('project_roots', {})[op['id']] = relative.as_posix()
            admitted = dict(binding=cycle.binding, operations=admissions,
                configuration=cycle.spec.configuration_ref, memory=wave['memory'], skills=wave['skills'],
                memory_bytes=cycle.store.read_bytes(wave['memory']).hex(),
                skills_bytes=cycle.store.read_bytes(wave['skills']).hex(), devices=self.policy['devices'],
                identities=identities, binding_contracts=binding_contracts, kdb_stat=kdb_stat, wall_deadline=wave['wall_deadline'])
            wave['admission_ref'] = cycle.store.put_json(admitted)
            state['active_wave'] = wave
            cycle.scheduler._save()
        return wave

    def _factory(self, wave):
        cycle = self.cycle
        admitted = cycle.store.read_json(wave['admission_ref'])
        resolver = cycle.services.executor_factory
        if admitted['identities']:
            resolver = FrozenExecutorFactory(resolver, json.dumps(admitted['identities'], sort_keys=True),
                json.dumps(admitted.get('binding_contracts', {}), sort_keys=True))
        return ProjectWorkerFactory(str(cycle.services.root), str(cycle.root),
            json.dumps(admitted), cycle.services.test, resolver)

    def _runner(self, operation):
        return self.cycle.open_project(operation)

    def _observe(self, pool, operation, row):
        path = pool._branch(operation) / 'worker.json'
        if not path.exists():
            return False  # The dispatch/identity gap is unknown, never absence.
        worker = json.loads(path.read_text())
        if worker['dispatch_id'] != row['dispatch_id'] or not worker.get('process') or _alive(worker['process']):
            return False
        admission = pool._branch(operation) / 'worker-admission.json'
        if admission.exists():
            record = json.loads(admission.read_text())
            if record['dispatch_id'] == row['dispatch_id']:
                reference = self.cycle.store.put_json(record)
                self.cycle.scheduler.state['active_wave'].setdefault('admission_receipts', {})[operation] = reference
                self.cycle.scheduler._save()
        if not (pool._branch(operation) / 'runner' / 'service-binding.json').exists():
            return self._observe_admission(pool, operation, row)
        runner = self._runner(operation)
        try:
            proof = runner.terminal_failure_proof()
        except (ValueError, RuntimeError):
            # A settled accepted receipt still requires matching canonical state.
            receipt = row.get('receipt')
            path = pool._branch(operation) / 'wave-result.json'
            if receipt is None and path.exists():
                receipt = json.loads(path.read_text())
            if not receipt or receipt['dispatch_id'] != row['dispatch_id'] or not receipt.get('settled'):
                return False
            result = receipt['result']
            if (runner.workflow.pending or runner.workflow.outcome != result['status']
                    or runner.backend.project_hash() != result['source_hash']):
                return False
            proof = None
        elapsed = runner.workflow.accounting.seconds()
        pool.reconcile(operation, lambda request, retained: dict(closed=True,
            observation='original controller process closed; canonical host effects verified',
            actor_seconds=elapsed, resume_ready=False))
        if proof and pool.status()['projects'][operation]['status'] == 'pending':
            event = runner.latest('project-run-outcome')
            result = runner.store.read_json(event['reference'])['result']
            if result['status'] == 'budget_exhausted' and elapsed >= row['allowance']:
                pool.settle_exhausted(operation, runner)
            else:
                pool.settle_failure(operation, runner)
        return True

    def _observe_admission(self, pool, operation, row):
        branch = pool._branch(operation)
        path = branch / 'worker-admission.json'
        if not path.exists():
            return False
        record = json.loads(path.read_text())
        if record['dispatch_id'] != row['dispatch_id'] or record['receipt'].get('admitted') is True:
            return False  # Binder/create failure is not proof of admission-only work.
        wave = self.cycle.scheduler.state['active_wave']
        admitted = self.cycle.store.read_json(wave['admission_ref'])
        request = pool._request(operation, row)
        receipt = record['receipt']
        unknown = ('probe_pending', 'probe_unknown', 'submission_unknown')
        def consumed():
            return (sum(r['elapsed'] for r in admission_intervals(branch))
                    + admitted['operations'][operation].get('source_interval', {}).get('elapsed', 0.))
        if receipt.get('failure', {}).get('type') in unknown:
            # Same dispatch/probe identity observes the retained handle; never
            # substitute a fresh dispatch while a health command may still run.
            started, monotonic_start = time.time(), time.monotonic()
            remaining = max(0., row['allowance'] - consumed())
            if admitted['wall_deadline'] is not None:
                remaining = min(remaining, max(0., admitted['wall_deadline'] - time.time()))
            if remaining <= 0:
                return False
            deadline = monotonic_start + min(45., remaining)
            try:
                receipt = self._factory(wave).executor_factory.guard(record['profile'], request,
                    phase='admission', remaining=lambda: max(0., deadline - time.monotonic()))
            except Exception as exc:
                receipt = getattr(exc, 'admission_receipt', dict(admitted=False,
                    failure=dict(type='probe_unknown'), error_type=type(exc).__name__))
            elapsed = time.monotonic() - monotonic_start
            observations = branch / ('admission-observations-' + row['dispatch_id'] + '.json')
            history = json.loads(observations.read_text()) if observations.exists() else []
            history.append(dict(interval_id='admission-observation:' + row['dispatch_id'] + ':' + str(len(history)),
                dispatch_id=row['dispatch_id'], receipt=receipt, started=started,
                elapsed=elapsed, ended=max(time.time(), started + elapsed)))
            atomic_json(observations, history)
        reference = self.cycle.store.put_json(dict(original=record, observed=receipt,
            intervals=admission_intervals(branch)))
        wave.setdefault('admission_receipts', {})[operation] = reference
        self.cycle.scheduler._save()
        elapsed = max(consumed(), row['actor_seconds'])
        pool.retain_admission_elapsed(operation, row['dispatch_id'], elapsed, reference)
        if receipt.get('failure', {}).get('type') in unknown:
            return False
        pool.reconcile(operation, lambda request, retained: dict(closed=True,
            observation='original pre-runner admission probe settled; no actor or target project command admitted',
            actor_seconds=max(elapsed, row['actor_seconds']), resume_ready=True, runner_exists=False,
            admission_ref=reference))
        return True

    def observe(self):
        wave = self.cycle.scheduler.state.get('active_wave')
        if not wave or 'admission_ref' not in wave:
            raise ValueError('wave admission incomplete; inspect original reservation before dispatch')
        pool = self._pool(wave)
        for operation, row in pool.status()['projects'].items():
            if row['status'] == 'pending':
                self._observe(pool, operation, row)
        state = pool.status()
        if all(row['status'] == 'completed' for row in state['projects'].values()):
            self._retain_setup(wave)
            pool.publish()
            self.cycle.scheduler.state['active_wave']['phase'] = 'settled'
            self.cycle.scheduler.state['active_wave']['retained_actor_seconds'] = {
                op: row['actor_seconds'] for op, row in state['projects'].items()}
            self.cycle.scheduler._save()
        return state

    def _retain_setup(self, wave):
        rows = []
        for op in wave['operations']:
            local = ContentAddressedArtifacts(self.cycle.root / 'waves' / wave['id'] /
                                              'projects' / op['id'] / 'service-artifacts')
            for path in local.objects.glob('*/*'):
                self.cycle.store.put_bytes(local.read_bytes(path.parent.name + path.name))
            path = local.root / 'setup.jsonl'
            if path.exists():
                rows.extend(dict(project_operation=op['id'], **json.loads(line))
                            for line in path.read_text().splitlines() if line)
        self.cycle.scheduler.state['active_wave']['setup_ref'] = self.cycle.store.put_json(rows)

    def run(self, spec, context):
        cycle = self.cycle
        wave = self._prepare(self.policy.get('wave_size', self.policy['max_parallel']))
        pool = self._pool(wave)
        self.active_pool = pool
        try:
            if wave['phase'] == 'executing':
                pool.run(self._factory(wave), should_stop=getattr(cycle, 'should_stop', None),
                         progress=getattr(cycle, 'wave_progress', None))
                self.observe()
            rows = pool.status()['projects']
            if any(row['status'] != 'completed' for row in rows.values()):
                raise RuntimeError('wave unsettled; observe retained controllers and target handles')
            operation = context['operation_id']
            runner = self._runner(operation)
            row = rows[operation]
            return cycle.finish_project(runner, operation, row['receipt']['result'],
                                        context['original_project_id'])
        finally:
            self.active_pool = None

    def stop(self):
        if self.active_pool:
            self.active_pool.stop()

    def resume(self):
        wave = self.cycle.scheduler.state.get('active_wave')
        if wave and 'admission_ref' in wave:
            self._pool(wave).resume()

    def continue_ready(self):
        """Explicit resume may admit only durable ready siblings, never pending work."""
        wave = self.cycle.scheduler.state.get('active_wave')
        if not wave:
            raise ValueError('retained wave admission required')
        if 'admission_ref' not in wave:
            # No pool/worker dispatch is reachable before this local CAS admission.
            wave = self._prepare(self.policy.get('wave_size', self.policy['max_parallel']))
        pool = self._pool(wave)
        self.observe()
        pool.resume()
        self.active_pool = pool
        try:
            return pool.run(self._factory(wave), should_stop=getattr(self.cycle, 'should_stop', None),
                            progress=getattr(self.cycle, 'wave_progress', None))
        finally:
            self.active_pool = None

    def recover(self, operation, authority):
        """Explicit harness continuation; semantic recovery stays in scheduler policy."""
        wave = self.cycle.scheduler.state.get('active_wave')
        if not wave or operation not in {op['id'] for op in wave['operations']}:
            raise ValueError('matching unfinished wave project required')
        pool = self._pool(wave)
        row = pool.status()['projects'][operation]
        if row['status'] == 'completed' and row['receipt']['result']['status'] == 'accepted':
            raise ValueError('accepted projects cannot be rerun')
        runner = self._runner(operation)
        if runner.workflow.pending:
            raise ValueError('canonical pending stage/command must be reconciled before continuation')
        if self.cycle.scheduler.remaining()['actor_seconds'] is not None:
            remaining = self.cycle.scheduler.remaining()['actor_seconds']
            reserved = sum(r['allowance'] for r in pool.status()['projects'].values())
            if remaining < reserved + 7200.:
                raise ValueError('parent elapsed allocation cannot reserve harness recovery')
        pool.reconcile(operation, lambda request, retained: dict(closed=True,
            observation=authority, actor_seconds=runner.workflow.accounting.seconds(),
            resume_ready=True, harness_failure=True))
        pool.grant_harness_recovery(operation, observation=authority)
        self.cycle.scheduler.state['active_wave']['phase'] = 'executing'
        self.cycle.scheduler._save()

    def apply_observation(self, operation, observation, *, continue_stage=False):
        """Explicit host evidence; never infer closure from an absent process alone."""
        wave = self.cycle.scheduler.state.get('active_wave')
        if not wave or operation not in {op['id'] for op in wave['operations']}:
            raise ValueError('matching retained wave admission required')
        pool = self._pool(wave)
        row = pool.status()['projects'][operation]
        worker_path = pool._branch(operation) / 'worker.json'
        worker = json.loads(worker_path.read_text())
        if (worker['dispatch_id'] != row['dispatch_id'] or not worker.get('process')
                or _alive(worker['process'])):
            raise ValueError('original controller must be observed closed first')
        if not (pool._branch(operation) / 'runner' / 'service-binding.json').exists():
            if not self._observe_admission(pool, operation, row):
                raise ValueError('original pre-runner admission remains unresolved; retain probe identity and lease')
            return None
        runner = self._runner(operation)
        for token, ended_at in observation.get('closed_actor_intervals', {}).items():
            if observation['controller_proof'].get('actor_closed') is not True:
                raise ValueError('observed actor closure required before closing actor intervals')
            runner.workflow.accounting.reconcile(token, ended_at)
        for command in observation.get('commands', []):
            runner.backend.reconcile_target(command['operation_id'], command['timing'],
                observation_seconds=command.get('observation_seconds', 45.))
        try:
            runner.reconcile()
        except RuntimeError:
            event = runner.latest('project-run-outcome')
            if not event:
                raise ValueError('retained run outcome required for terminal observation')
            result = runner.store.read_json(event['reference'])['result']
            runner.record_terminal_observation(result,
                controller_proof=observation['controller_proof'],
                observed_source_hash=runner.backend.project_hash())
        if continue_stage and runner.workflow.pending:
            pending = runner.workflow.pending
            ref = runner.record_admission_failure(pending['id'],
                controller_proof=observation['controller_proof'],
                observed_source_hash=runner.backend.project_hash())
            runner.workflow.reject_operation(pending['id'], ref, runner.backend.project_hash())
        return runner
