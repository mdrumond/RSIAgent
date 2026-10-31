"""Concrete CPU instance bindings. Controllers remain the sole execution loops.

Configuration uses project_configs[project_id] = ProjectConfig wire data and
kdb_collections = [collection names]. Paths are resolved from the instance.
ProjectSpec.command equals commands.test. Serial budgets cap per-project budgets.
Opt-in waves reserve the explicit7200-second actor allocation per child.
Starting folders are copied by ProjectRunner at admission; existing contracts
never re-read their original folders. New configuration revisions require an
explicit task-boundary adoption; active and historical contracts remain frozen.
Credentials are supplied by environment, never saved in contracts. Production
requires an existing pinned KnowledgeDB; tests must opt into TestBindings.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import time

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.ascend.subject_memory import SubjectMemory, digest as memory_digest
from rsi.configuration import _snapshot, show_configuration
from rsi.contracts import BudgetLimits, CycleSpec, Mode, ProjectSpec, Stage
from rsi.cycles import CycleScheduler, HostTaskResult
from rsi.learning import Learning, ProposalRejected, RejectedOperation, Validation
from rsi.project_runner import ProjectConfig, ProjectRunner
from rsi.project_composition import ProjectComposition
from rsi.python_execution import atomic_json
from rsi.recovery import CycleController, PairedRecovery, fork_inference
from rsi.runtime import WorkflowError
from rsi.roles import AdmittedEvidence, IsolatedRoles, _proposal
from rsi.skills import SkillBundle
from rsi import revisions
from rsi.setup_telemetry import SetupTelemetry
from rsi.service_tuning import TuningBindings
from rsi.observations import capture as capture_observations
from rsi.handoffs import capture_parent
from rsi.evaluation import SkillEvaluation, TuningController, case_identity


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('service identity requires a simple filename')
    return value


@dataclass(frozen=True)
class TestBindings:
    """Explicit test-only providers; production never silently falls back."""
    query: object
    session_factory: object
    roles_factory: object
    identity: str


class InstanceServices:
    def __init__(self, instance, *, environment=None, test_bindings=None, executor_factory=None):
        self.root = Path(instance).resolve(strict=True)
        self.environment = dict(os.environ if environment is None else environment)
        self.test = test_bindings
        self.executor_factory = executor_factory
        if test_bindings is not None and not isinstance(test_bindings, TestBindings):
            raise TypeError('explicit TestBindings required')
        self.store = ContentAddressedArtifacts(self.root / 'artifacts')
        self.setup = SetupTelemetry(self.store)

    def path(self, value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.root / path).resolve(strict=True)

    def executable(self, value):
        # Preserve venv interpreter symlink path: resolving changes sys.prefix.
        path = Path(value).expanduser()
        path = path if path.is_absolute() else self.root / path
        if not path.is_file():
            raise ValueError('configured executable does not exist')
        return path.absolute()

    def create_cycle(self, cycle_id, mode, *, projects=None, configuration=None, recover=None,
                     from_cycle=None, inheritance='auto', input_overrides=()):
        created = time.time()
        mode = Mode(mode)
        input_overrides = tuple(input_overrides)
        if recover is not None and type(recover) is not bool:
            raise ValueError('recovery override must be bool or null')
        root = self.root / 'runs' / _name(cycle_id)
        if root.exists():
            raise FileExistsError(root)
        snapshot = revisions.checked(configuration) if configuration is not None else show_configuration(self.root)
        config = snapshot['payload']
        projects = revisions.projects(config, mode, projects)
        if not projects:
            raise ValueError('cycle requires explicit curriculum projects')
        # Validate every configured host command before publishing the cycle.
        for spec in projects:
            self.project_config(config, spec)
        with self.setup.measure('accepted-input-capture', cycle_id):
            parent = capture_parent(self.root, mode, from_cycle, inheritance=inheritance,
                                    overrides=input_overrides, configuration=snapshot)
        requested_config = self.store.put_json(snapshot)
        # Explicit parent policy acknowledges inherited baseline settings. Keep new
        # curriculum/prompts/provider, but bind inherited KDB provenance consistently.
        if parent.parent_cycle is not None:
            source = self.store.read_json(parent.source_config_ref)['payload']
            keys = dict(memory=('memory_snapshot', 'memory_sha256'),
                skills=('baseline_skills', 'skills', 'skill_map_ref', 'skill_map'),
                kdb=('baseline_kdb', 'kdb', 'kdb_sha256', 'kdb_filename', 'kdb_collections', 'kdb_ingestion_refs'))
            config = json.loads(json.dumps(config))
            for component, names in keys.items():
                if component not in input_overrides:
                    for name in names:
                        # Collection selection is a current query input under auto;
                        # explicit parent policy also acknowledges its inherited selector.
                        if name == 'kdb_collections' and inheritance == 'auto':
                            continue
                        if name in source:
                            config[name] = source[name]
                        else:
                            config.pop(name, None)
            snapshot = _snapshot(config)
        bundle = SkillBundle.from_dict(self.store.read_json(parent.skills_ref))
        mapping = dict(bundle.mapping)
        skills = parent.skills_ref
        initial = self.store.read_bytes(parent.memory_ref)
        root.mkdir(parents=True)
        with self.setup.measure('baseline-kdb-copy', cycle_id):
            with (root / 'knowledge.sqlite3').open('xb') as stream:
                stream.write(self.store.read_bytes(parent.kdb_ref))
                stream.flush()
                os.fsync(stream.fileno())
        with self.setup.measure('accepted-memory-load', cycle_id):
            memory = self._memory(root / 'memory', initial)
        memory_ref = self.store.put_bytes(memory.snapshot())
        if memory_ref != parent.memory_ref:
            raise ValueError('copied accepted memory differs from captured parent')
        configuration_ref = self.store.put_json(snapshot)
        spec = CycleSpec(cycle_id, mode, projects, memory_ref, skills,
                         self.store.put_json({k.value: list(v) for k, v in mapping.items()}), configuration_ref)
        atomic_json(root / 'service.json', dict(spec=spec.to_dict(), config=snapshot,
            kdb=parent.kdb_ref, initial=initial.decode(), recover=recover, created=created,
            requested_configuration_ref=requested_config, input_handoff=asdict(parent),
            input_policy=dict(inheritance=inheritance, overrides=list(input_overrides), from_cycle=from_cycle),
            test_identity=self.test.identity if self.test else None))
        return self.open_cycle(cycle_id)

    @staticmethod
    def _memory(root, baseline):
        # A SubjectMemory snapshot contains accepted records, not a new text initial.
        snapshot = json.loads(baseline) if baseline else None
        if snapshot is not None and snapshot.get('schema') != 'subject-memory-snapshot-v1':
            raise ValueError('accepted SubjectMemory snapshot required')
        memory = SubjectMemory(root, (snapshot['initial_memory'].encode() if snapshot else b''))
        if snapshot:
            empty = not memory.records()
            with memory.transaction() as state:
                for record in snapshot['memories']:
                    version = memory_digest(record)
                    folder = memory.root / 'versions'
                    folder.mkdir(exist_ok=True)
                    path = folder / (version + '.json')
                    if path.exists() and json.loads(path.read_text()) != record:
                        raise ValueError('inherited accepted memory version differs')
                    if not path.exists():
                        atomic_json(path, record)
                    if empty:
                        state['active'][record['memory_id']] = version
                    # This is inheritance of an accepted snapshot, not a new task verdict.
                    identity = 'inherited-snapshot:' + hashlib.sha256(baseline).hexdigest() + ':' + version
                    state['admissions'].setdefault(identity,
                        dict(memory_id=record['memory_id'], version=version))
            if empty and json.loads(memory.snapshot()) != snapshot:
                raise ValueError('accepted baseline memory differs')
        return memory

    def project_config(self, config, spec):
        row = dict(config.get('project_configs', {})[spec.project_id])
        row['source'] = str(self.path(spec.starting_folder or row['source']))
        row['verifier'] = str(self.path(row['verifier']))
        result = ProjectConfig.from_dict(row)
        if tuple(result.commands['test']) != spec.command:
            raise ValueError('ProjectSpec.command must equal project_configs commands.test')
        return result

    def open_cycle(self, cycle_id):
        return BoundCycle(self, self.root / 'runs' / _name(cycle_id))

    def run_project(self, cycle_id, operation_id):
        """Continue the SAME admitted runner explicitly; never create on resume."""
        cycle = self.open_cycle(cycle_id)
        try:
            return cycle.run_project(operation_id)
        finally:
            cycle.close()

    def status(self, cycle_id):
        root = self.root / 'runs' / _name(cycle_id)
        result = {name: json.loads((root / name).read_text()) if (root / name).exists() else None
                for name in ('service.json', 'cycle.json', 'controller.json', 'learning.json', 'tuning.json', 'revision.pending.json')}
        from rsi.wave_inspection import inspect_waves
        result['waves'] = inspect_waves(root)
        return result


class BoundCycle(ProjectComposition, TuningBindings):
    def __init__(self, services, root):
        self.services, self.root, self.store = services, root, services.store
        revisions.recover(root, self.store)
        self.binding = json.loads((root / 'service.json').read_text())
        if self.binding['test_identity'] != (services.test.identity if services.test else None):
            raise ValueError('production/test backend binding differs')
        self.spec = CycleSpec.from_dict(self.binding['spec'])
        if self.store.read_json(self.spec.configuration_ref) != self.binding['config']:
            raise ValueError('frozen configuration differs from admitted revision')
        self.config = self.binding['config']['payload']
        self.role_operation = None
        self.roles_log = root / 'roles.json'
        self.role_events = json.loads(self.roles_log.read_text()) if self.roles_log.exists() else []
        self.database = None
        self.active_runner = None
        self._roles = None
        self.role_setup_started = None
        self.tuning_guidance = {}
        memory_root, learning_path = root / 'memory', root / 'learning.json'
        state_path = root / 'cycle.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get('isolation'):
            branch = Path(state['isolation']['branch_root'])
            memory_root, learning_path = branch / 'memory', branch / 'learning.json'
        policy = dict(self.config.get('cycle_limits', {}))
        if self.config.get('project_wave'):
            policy.setdefault('recovery_limit', None)
        first_clock = not state_path.exists()
        def cycle_clock():
            nonlocal first_clock
            if first_clock:
                first_clock = False
                return self.binding.get('created', time.time())
            return time.time()
        self.scheduler = CycleScheduler(state_path, self.spec, self.run,
            lambda *args: self.learning.memory_update(*args),
            skill_update=self.write_skills, isolate=self.isolate,
            recover=self.binding.get('recover') if self.binding.get('recover') is not None else
                self.config.get('recovery', {}).get('enabled', True),
            tuning_every=self.config.get('tuning', {}).get('every_projects', 5), clock=cycle_clock, **policy)
        try:
            with self.services.setup.measure('cycle-memory-learning-open', self.spec.cycle_id):
                self.memory = services._memory(memory_root, self.binding['initial'].encode())
                self.learning = self.make_learning(learning_path)
            if self.learning.active is None:
                self.learning.state['active'] = self.spec.baseline_skills
                self.learning._save()
        except BaseException:
            self.scheduler.close()
            raise
        self.controller = CycleController(root / 'controller.json', self.scheduler,
            self.learning, artifacts=self.store, pair_factory=self.pair)
        self.tuning = TuningController(root / 'tuning.json', self.controller,
                                       evaluation_factory=self.evaluation)
        self.waves = None
        if self.config.get('project_wave'):
            from rsi.service_wave import WaveBindings
            self.waves = WaveBindings(self)

    def make_learning(self, path):
        return Learning(path, self.store, self.memory, self, load_project=self.load_project,
            validate=self.validate, mode=self.spec.mode.value,
            role_elapsed=self.role_elapsed,
            unused_window=self.config.get('learning', {}).get('unused_window', 10))

    def observe_role(self, event, payload):
        if event == 'role_started' and self.role_setup_started is not None:
            self.services.setup.record('roles-worker-dependency-load', self.role_setup_started,
                success=True, scope=self.role_operation)
            self.role_setup_started = None
        self.role_events.append(dict(operation_id=self.role_operation, event=event, payload=payload))
        atomic_json(self.roles_log, self.role_events)

    def role_elapsed(self, operation):
        rows = [e['payload']['elapsed_seconds'] for e in self.role_events
                if e['operation_id'] == operation and e['event'] == 'role_finished']
        if len(rows) != 1:
            raise ValueError('one authoritative finished role receipt required')
        return rows[0]

    def roles(self):
        if self._roles is None:
            if self.services.test:
                self._roles = self.services.test.roles_factory(self.observe_role)
            else:
                from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
                from benchmarks.a5kernels.phase1_provider import direct_profile
                provider = self.config['provider']
                identity = next(i for i in MODEL_IDENTITIES if i.profile_id == provider['profile_id'])
                profile = direct_profile(identity)
                if profile.model != provider['model']:
                    raise ValueError('registered role model mismatch')
                self._roles = IsolatedRoles(str(self.services.executable(self.config['roles_python'])),
                    profile, observer=self.observe_role, environment=self.services.environment,
                    **self.config.get('roles', {}))
        return self._roles

    def skillwriter(self, prompt, evidence, *, deadline, wall_deadline=None):
        guidance = {key: self.config.get('prompts', {}).get(key, '')
                    for key in ('skillwriter', 'curriculum', 'waves')}
        allowance = deadline - time.monotonic()
        self.role_setup_started = time.monotonic()
        try:
            roles = self.roles()
            deadline = time.monotonic() + allowance
            return roles.skillwriter(prompt + '\nUser guidance:\n' + json.dumps(guidance)
                + '\nTuning steering:\n' + json.dumps(self.tuning_guidance), evidence, deadline=deadline,
                wall_deadline=wall_deadline)
        finally:
            self.finish_role_setup()

    def propose_curriculum(self, *, operation_id, deadline):
        """Typed, budgeted suggestions; commands still require host configuration."""
        operation = _name(operation_id)
        receipts = self.scheduler.state.get('extension_receipts', {})
        if operation in receipts:
            record = self.store.read_json(receipts[operation]['reference'])
            return _proposal('curriculum', record['proposal'], {record['configuration']})
        if any(e.get('id') == operation for e in self.scheduler.state['events']):
            raise ValueError('curriculum work pending; observe retained role receipts, never replay')
        if self.scheduler.state['pending'] or self.controller.state['pending']:
            raise ValueError('curriculum steering requires a task boundary')
        remaining = self.scheduler.remaining()
        for key in ('actor_seconds', 'wall_seconds'):
            if remaining[key] is not None:
                deadline = min(deadline, time.monotonic() + remaining[key])
        if deadline <= time.monotonic():
            raise TimeoutError('curriculum allocation exhausted')
        self.scheduler.state['events'].append(dict(kind='curriculum_requested', id=operation))
        self.scheduler._save()  # Durable before any role/provider setup or dispatch.
        self.role_operation = operation
        try:
            evidence = [AdmittedEvidence(self.spec.configuration_ref,
                dict(projects=[p.to_dict() for p in self.spec.projects],
                     memory=self.scheduler.state['memory']))]
            prompt = json.dumps({key: self.config.get('prompts', {}).get(key, '')
                                 for key in ('curriculum', 'waves')})
            allowance = deadline - time.monotonic()
            self.role_setup_started = time.monotonic()
            try:
                roles = self.roles()
                proposal = roles.curriculum(prompt, evidence, deadline=time.monotonic() + allowance)
            finally:
                self.finish_role_setup()
            reference = self.store.put_json(dict(proposal=asdict(proposal), operation_id=operation,
                                                configuration=self.spec.configuration_ref))
            self.scheduler.complete_extension(operation, reference,
                actor_seconds=self.role_elapsed(operation))
            return proposal
        finally:
            self.role_operation = None

    def finish_role_setup(self):
        if self.role_setup_started is not None:
            self.services.setup.record('roles-worker-setup-without-start-receipt',
                self.role_setup_started, success=False, scope=self.role_operation, actor_billable=None)
            self.role_setup_started = None

    def write_skills(self, skills, memory, result, context):
        self.role_operation = context['operation_id']
        try:
            return self.learning.skill_update(skills, memory, result, context)
        finally:
            self.role_operation = None

    def run(self, spec, memory_ref, skills_ref, context):
        if self.waves:
            return self.waves.run(spec, context)
        operation = _name(context['operation_id'])
        root = self.root / 'projects' / operation
        if root.exists():
            raise ValueError('existing admitted operation requires reconciliation')
        binding = dict(memory=memory_ref, skills=skills_ref, configuration=self.spec.configuration_ref,
                       kdb=self.binding['kdb'], provider=self.config['provider'],
                       test_identity=self.binding['test_identity'], actor_seconds_baseline=0.0)
        backend = self.store.put_json(binding)
        runner = self.runner(root, spec=spec, bundle=SkillBundle.from_dict(self.store.read_json(skills_ref)),
            config=self.services.project_config(self.config, spec), memory_ref=memory_ref, backend_identity=backend)
        atomic_json(root / 'training-identity.json', case_identity(runner))
        atomic_json(root / 'service-binding.json', binding)
        result = self.execute(runner)
        return self.finish_project(runner, operation, result, context['original_project_id'])

    def finish_project(self, runner, operation, result, original_id=None):
        pending = self.scheduler.state['pending']
        if not pending or pending['id'] != operation or pending['kind'] not in ('original', 'recovery'):
            raise ValueError('no matching scheduler admission')
        original_id = original_id or pending['original_id']
        outcome, failure = result['status'], None
        if outcome != 'accepted' and (runner.workflow.pending or outcome == 'suspended'):
            try:
                failure = runner.terminal_failure_proof(closure_ref=result.get('actor_closure_ref'))
                outcome = 'failed'
            except WorkflowError:
                pass  # Unknown effects remain suspended; no fabricated closure.
        self.retain(runner)
        baseline = json.loads((runner.root / 'service-binding.json').read_text()).get('actor_seconds_baseline', 0.0)
        total = runner.workflow.accounting.seconds()
        if type(baseline) not in (int, float) or not 0 <= baseline <= total:
            raise ValueError('invalid previously charged actor baseline')
        delta = total - baseline
        proof_ref = self.store.put_json(failure) if failure else None
        packet = dict(operation_id=operation, original_id=original_id, root=str(runner.root),
            result=result, workflow=runner.workflow.state, host_outcome=outcome,
            failure_proof_ref=proof_ref, actor_seconds_total=total,
            actor_seconds_baseline=baseline, actor_seconds_delta=delta,
            workflow_valid=False if failure else runner.workflow.outcome == 'accepted')
        ref = self.store.put_json(packet)
        atomic_json(runner.root / 'service-result.json', dict(reference=ref))
        if outcome not in ('accepted', 'rejected', 'budget_exhausted', 'failed') or (
                runner.workflow.pending and failure is None):
            raise RuntimeError('project suspended; reconcile same operation before cycle continuation')
        return HostTaskResult(outcome, ref, delta)

    def open_project(self, operation):
        relative = self.scheduler.state.get('project_roots', {}).get(operation)
        root = (self.root / relative).resolve() if relative else self.root / 'projects' / _name(operation)
        if not root.is_relative_to(self.root.resolve()):
            raise ValueError('project root escapes retained cycle')
        binding = json.loads((root / 'service-binding.json').read_text())
        if binding.get('executor_provenance') is not None:
            from rsi.executor_profiles import public_identity
            if self.services.executor_factory is None or public_identity(
                    self.services.executor_factory(binding['executor_identity'])) != binding['executor_provenance']:
                raise ValueError('host executor differs from frozen project device/runtime')
        if relative:
            wave_id = Path(relative).parts[1]
            waves = [*self.scheduler.state.get('wave_history', []), self.scheduler.state.get('active_wave')]
            wave = next(w for w in waves if w and w['id'] == wave_id)
            self.kdb_stat = self.store.read_json(wave['admission_ref'])['kdb_stat']
        return self.runner(root, memory_ref=binding['memory'],
                           backend_identity=self.store.put_json(binding), reopen=True,
                           configuration_ref=binding['configuration'])

    def retain(self, runner):
        for path in runner.store.objects.glob('*/*'):
            self.store.put_bytes(runner.store.read_bytes(path.parent.name + path.name))

    def load_project(self, row):
        operation = row['operation_id']
        runner = self.open_project(operation)
        self.retain(runner)
        packet = self.store.read_json(row['artifact_ref'])
        if packet['operation_id'] != operation or packet['original_id'] != row['original_id']:
            raise ValueError('project receipt scope differs')
        if (packet['result']['contract_ref'] != runner.contract_ref
                or packet['result']['execution_contract_ref'] != runner.backend.manifest['contract']):
            raise ValueError('project or execution contract differs')
        result = packet['result']
        validation = runner.latest('validation-results')
        host_outcome = packet.get('host_outcome', result['status'])
        proof = None
        if packet.get('failure_proof_ref'):
            proof = self.store.read_json(packet['failure_proof_ref'])
            if proof != runner.terminal_failure_proof(closure_ref=proof['closure_ref']):
                raise ValueError('terminal failure proof differs from authoritative checkpoint')
        if (packet['workflow'] != runner.workflow.state or row['outcome'] != host_outcome
                or runner.backend.project_hash() != result['source_hash']
                or (result['status'] == 'accepted' and (runner.workflow.outcome != 'accepted'
                    or runner.workflow.verified_source != result['source_hash']
                    or not validation or validation['source_hash'] != result['source_hash']
                    or validation['correctness'] is not True))):
            raise ValueError('project result differs from authoritative workflow/validation')
        queries, usage = [], []
        for event in runner.events:
            if event['type'] == 'project-tool' and event['name'] == 'query_kdb':
                receipt = runner.store.read_json(event['reference'])
                if receipt['context']['workflow'] != runner.contract_ref:
                    raise ValueError('query scope differs')
                queries.append(event['reference'])
            if event['type'] == 'skill-usage':
                usage.append({k: v for k, v in event.items() if k != 'type'})
        kind = 'correctness' if result['status'] == 'accepted' and result['correctness'] is True else 'failure'
        observations, summary = capture_observations(runner, self.store, host_outcome)
        observations_ref = self.store.put_json(observations)
        if proof:
            summary = str(proof['classification']) + ': ' + str(proof['error']) + '; ' + summary
        raw = list(result['events'])
        if proof:
            raw.extend([packet['failure_proof_ref'], proof['result_ref'], proof['accounting_snapshot']])
        return dict(project_id=row['original_id'], outcome=host_outcome,
            evidence={row['artifact_ref']: dict(kind=kind, summary=summary)},
            queries=queries, usage=usage, raw_refs=[*raw, observations_ref, *observations['artifact_refs']],
            observations=observations, observations_ref=observations_ref)

    def isolate(self, memory, skills, context):
        receipt = fork_inference(self.root / 'inference', self.memory, self.learning.path,
            artifacts=self.store, memory_ref=memory, skills_ref=skills, namespace=context['namespace'])
        self.memory = self.services._memory(Path(receipt.branch_root) / 'memory', self.binding['initial'].encode())
        self.learning = self.make_learning(Path(receipt.branch_root) / 'learning.json')
        self.controller.learning = self.learning
        return receipt

    def trial(self, request, *, reopen):
        runner = self.runner(request.root, spec=request.spec, bundle=request.bundle, config=request.config,
            memory_ref=request.memory_ref, backend_identity=request.backend_identity, reopen=reopen)
        canonical_run = runner.run
        def run_trial():
            self.active_runner = runner
            try:
                return canonical_run()
            except KeyboardInterrupt:
                runner.workflow.stop()
                self.scheduler.stop()
                self.retain(runner)
                raise
            finally:
                self.active_runner = None
        runner.run = run_trial
        return runner

    def pair(self, event, operation, remaining):
        path = self.root / 'pairs' / (_name(operation) + '.json')
        if path.exists():
            return PairedRecovery(path, artifacts=self.store, runner_factory=self.trial)
        checkpoint = self.open_project(event['original']['operation_id'])
        packet = self.store.read_json(event['original']['artifact_ref'])
        return PairedRecovery.create(path, checkpoint, artifacts=self.store, memory_ref=event['memory'],
            terminal_failure_ref=packet.get('failure_proof_ref'),
            kdb_ref=self.binding['kdb'], model_config=self.config['provider'], incumbent=event['incumbent'],
            candidate=event['candidate'], joint_limits=BudgetLimits.from_dict(
                self.config.get('recovery', {}).get('budgets', self.config['budgets'])),
            remaining_parent=remaining, runner_factory=self.trial)

    def validate(self, incumbent, candidate, cases, operation):
        suite = self.suite_validation(incumbent, candidate, cases, operation)
        if suite is not None:
            return suite
        admitted = {}
        for path in (self.root / 'pairs').glob('*.json'):
            pair = PairedRecovery(path, artifacts=self.store, runner_factory=self.trial)
            result = pair.result()
            if (result and pair.frozen['incumbent'] == incumbent
                    and pair.frozen['candidate'] == candidate):
                admitted[result['evidence_ref']] = result
        valid = bool(cases) and all(case in admitted and admitted[case]['decision'] == 'benefit'
            and admitted[case]['arms']['candidate']['workflow'] is True
            and admitted[case]['arms']['candidate']['correctness'] is True for case in cases)
        reference = self.store.put_json(dict(operation=operation, cases=cases, valid=valid))
        return Validation(True if valid else None, True if valid else None, reference)

    def reject_writer(self):
        """Consume only the producer's retained, observed semantic rejection."""
        pending = self.scheduler.state['pending']
        if not pending or pending['kind'] != 'skill':
            raise ValueError('matching pending writer operation required')
        operation = self.learning.state['operations'].get(pending['id'], {})
        if 'proposal' not in operation:
            raise RuntimeError('retained observed proposal required; unknown work stays pending')
        rejection = self.learning.reject_observed_proposal(pending['id'], operation['proposal'],
            namespace=self.scheduler.state['namespace'])
        self.scheduler.reconcile_rejected_skill(rejection)
        return rejection

    def reconcile(self, *, dispatch_ready=False):
        """Observe existing project completion; no new provider dispatch."""
        pending = self.scheduler.state['pending']
        if self.waves and self.scheduler.state.get('active_wave') and pending and pending['kind'] in ('original', 'recovery'):
            if dispatch_ready:
                self.waves.continue_ready()
            state = self.waves.observe()
            if any(row['status'] != 'completed' for row in state['projects'].values()):
                raise ValueError('wave requires observation of retained controller/target work')
            row = state['projects'][pending['id']]
            runner = self.open_project(pending['id'])
            result = self.finish_project(runner, pending['id'], row['receipt']['result'])
            self.scheduler.reconcile(pending['id'], result)
            return result
        if pending and pending['kind'] == 'memory':
            if pending['id'] in self.learning.state['operations']:
                result = self.learning.reconcile_memory(pending['id'])
            else:
                rows = self.scheduler.state['wave_results'] if self.spec.mode == Mode.PHASE1 else [self.scheduler.state['originals'][-1]]
                result = self.learning.memory_update(self.scheduler.state['memory'], rows,
                                                     self.scheduler._context(pending))
            self.scheduler.reconcile(pending['id'], result)
            return result
        if self.tuning.state['pending']:
            pending = self.tuning.state['pending']
            path = self.root / 'evaluations' / pending['operation_id'] / 'evaluation.json'
            if not path.exists():
                raise ValueError('tuning setup not complete; inspect retained inputs, never replay')
            evaluation = SkillEvaluation(path, artifacts=self.store, runner_factory=self.trial)
            if 'path' not in pending:
                self.tuning.bind_pending(evaluation)
            result = evaluation.result()
            if result is not None:
                return result
            if any(arm['status'] == 'pending'
                    for i in range(len(evaluation.plan['pairs']))
                    for arm in evaluation.pair(i).state['arms'].values()):
                return evaluation.reconcile()
            # A retained round can be between completed arms. Observation has
            # nothing to reconcile; the next normal controller step owns dispatch.
            return None
        if self.controller.state['pending']:
            return self.controller.reconcile()
        pending = self.scheduler.state['pending']
        if pending and pending['kind'] == 'skill':
            return self.reject_writer()
        if not pending or pending['kind'] not in ('original', 'recovery'):
            raise ValueError('explicit learning/absence reconciliation required')
        runner = self.open_project(pending['id'])
        path = runner.root / 'service-result.json'
        if path.exists():
            ref = json.loads(path.read_text())['reference']
            packet = self.store.read_json(ref)
            failure = packet.get('failure_proof_ref')
            if failure:
                proof = self.store.read_json(failure)
                if (packet['workflow'] != runner.workflow.state or
                        proof != runner.terminal_failure_proof(closure_ref=proof['closure_ref'])):
                    raise ValueError('failed task admission proof no longer matches')
                self.scheduler.reconcile(pending['id'], HostTaskResult('failed', ref,
                    packet['actor_seconds_delta']))
                return
        runner.reconcile()
        if runner.workflow.pending:
            raise ValueError('project is not terminal; continue same admitted runner explicitly')
        self.retain(runner)
        if not path.exists():
            raise ValueError('retained project result required')
        ref = json.loads(path.read_text())['reference']
        packet = self.store.read_json(ref)
        if packet['result']['status'] not in ('accepted', 'rejected', 'budget_exhausted'):
            raise ValueError('project is not terminal; continue same admitted runner explicitly')
        if packet['workflow'] != runner.workflow.state:
            raise ValueError('result is stale; retain current project completion first')
        self.scheduler.reconcile(pending['id'], HostTaskResult(packet['result']['status'], ref,
            packet.get('actor_seconds_delta', runner.workflow.accounting.seconds())))

    def adopt_revision(self, configuration=None):
        return revisions.adopt(self, configuration or show_configuration(self.services.root))

    def execute(self, runner):
        self.active_runner = runner
        try:
            return runner.run()
        except KeyboardInterrupt:
            # Runner/backend finally blocks have already reaped their owned children.
            runner.workflow.stop()
            self.scheduler.stop()
            self.retain(runner)
            raise
        finally:
            self.active_runner = None

    def step(self):
        """Delegate one dispatch to the shared controller; no service driver loop."""
        pending = self.scheduler.state['pending']
        if (pending and pending['kind'] == 'skill' and
                'rejection' in self.learning.state['operations'].get(pending['id'], {})):
            return self.reject_writer()  # Crash after Learning commit, before Cycle charge.
        events = self.scheduler.extension_events()
        due = next((e for e in events if e['kind'] in ('tuning_due', 'manual_tuning')), None)
        recovery_due = any(e['kind'] in ('paired_recovery_due', 'candidate_validation_due') for e in events)
        if due is not None and not self.tuning.state['pending'] and not recovery_due:
            if (due['kind'] == 'tuning_due' and due.get('reasons') == ['wave_end']
                    and not self.config.get('tuning', {}).get('at_wave_end', True)):
                ref = self.store.put_json(dict(event=due['id'], decision='skipped', reason='wave_end_disabled'))
                self.scheduler.complete_extension(due['id'], ref)
                self._close_wave_boundary()
                return dict(decision='skipped', evidence_ref=ref)
            self.tuning_settings()
            self.tuning_project(due)

        try:
            result = self.tuning.step()
            self._close_wave_boundary()
            return result
        except (ProposalRejected, RejectedOperation):
            # The producer revalidates exact scope, semantic invalidity and genuine
            # role timing. Provider/storage/unknown failures never enter this path.
            return self.reject_writer()

    def _close_wave_boundary(self):
        if self.waves and not self.controller.state['pending'] and not self.tuning.state['pending']:
            extensions = self.scheduler.extension_events()
            if not any(e['kind'] in ('paired_recovery_due', 'candidate_validation_due',
                                     'tuning_due', 'manual_tuning') for e in extensions):
                self.scheduler.close_original_wave()

    def run_project(self, operation_id):
        if self.waves and self.scheduler.state.get('active_wave'):
            self.waves.observe()
            raise ValueError('use wave resume after observing retained project handles')
        pending = self.scheduler.state['pending']
        if not pending or pending['id'] != operation_id or pending['kind'] not in ('original', 'recovery'):
            raise ValueError('explicit continuation requires matching unfinished scheduler admission')
        runner = self.open_project(operation_id)
        # Observation may finish a durable receipt, but must never replay unknown work.
        runner.reconcile()
        if runner.workflow.pending:
            raise ValueError('reconcile pending operation before explicit continuation')
        runner.workflow.resume()
        return self.finish_project(runner, operation_id, self.execute(runner))

    def close(self):
        self.scheduler.close()
        if self.database:
            self.database.close()
