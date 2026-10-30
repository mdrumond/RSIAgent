"""Concrete CPU instance bindings. Controllers remain the sole execution loops.

Configuration uses project_configs[project_id] = ProjectConfig wire data and
kdb_collections = [collection names]. Paths are resolved from the instance.
ProjectSpec.command equals commands.test. Global budgets cap per-project budgets.
Starting folders are copied by ProjectRunner at admission; existing contracts
never re-read their original folders. New configuration revisions require an
explicit new cycle (or a linked continuation at a task boundary).
Credentials are supplied by environment, never saved in contracts. Production
requires an existing pinned KnowledgeDB; tests must opt into TestBindings.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.ascend.subject_memory import SubjectMemory, digest as memory_digest
from rsi.actor import PiSessionAdapter
from rsi.configuration import show_configuration
from rsi.contracts import BudgetLimits, CycleSpec, Mode, ProjectSpec, Stage
from rsi.cycles import CycleScheduler, HostTaskResult
from rsi.learning import Learning, Validation
from rsi.project_runner import ProjectConfig, ProjectRunner
from rsi.python_execution import atomic_json
from rsi.recovery import CycleController, PairedRecovery, fork_inference
from rsi.roles import AdmittedEvidence, IsolatedRoles, _proposal
from rsi.skills import SkillBundle
from rsi.strategy import knowledge_db_query


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
    def __init__(self, instance, *, environment=None, test_bindings=None):
        self.root = Path(instance).resolve(strict=True)
        self.environment = dict(os.environ if environment is None else environment)
        self.test = test_bindings
        if test_bindings is not None and not isinstance(test_bindings, TestBindings):
            raise TypeError('explicit TestBindings required')
        self.store = ContentAddressedArtifacts(self.root / 'artifacts')

    def path(self, value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.root / path).resolve(strict=True)

    def create_cycle(self, cycle_id, mode, *, projects=None):
        mode = Mode(mode)
        root = self.root / 'runs' / _name(cycle_id)
        if root.exists():
            raise FileExistsError(root)
        snapshot = show_configuration(self.root)
        config = snapshot['payload']
        rows = projects if projects is not None else config[
            'phase1_projects' if mode == Mode.PHASE1 else 'phase2_projects']
        projects = tuple(p if isinstance(p, ProjectSpec) else ProjectSpec.from_dict(p) for p in rows)
        global_limits = BudgetLimits.from_dict(config['budgets']).to_dict()
        def limits(project):
            values = {key: min(v for v in (value, global_limits[key]) if v is not None)
                      if value is not None or global_limits[key] is not None else None
                      for key, value in project.budgets.to_dict().items()}
            return replace(project, budgets=BudgetLimits.from_dict(values))
        projects = tuple(limits(project) for project in projects)
        if not projects:
            raise ValueError('cycle requires explicit curriculum projects')
        # Validate every configured host command before publishing the cycle.
        for spec in projects:
            self.project_config(config, spec)
        mapping_path = self.path(config.get('skill_map_ref', config['skill_map']))
        mapping = {Stage(k): tuple(v) for k, v in json.loads(mapping_path.read_text()).items()}
        bundle = SkillBundle.capture(self.path(config.get('baseline_skills', config['skills'])), mapping)
        skills = self.store.put_json(bundle.to_dict())
        kdb = self.path(config.get('baseline_kdb', config['kdb']))
        if kdb.is_dir():
            kdb = kdb / config.get('kdb_filename', 'knowledge.sqlite3')
        if not kdb.is_file():
            raise ValueError('existing baseline knowledge.sqlite3 required')
        # Bootstrap inventory authenticates the copied baseline before opening SQLite.
        manifest = json.loads((self.root / 'instance.json').read_text())
        relative = str(kdb.relative_to(self.root)) if kdb.is_relative_to(self.root) else None
        expected = config.get('kdb_sha256') or manifest.get('files', {}).get(relative)
        if not expected or hashlib.sha256(kdb.read_bytes()).hexdigest() != expected:
            raise ValueError('verified baseline KDB checksum required')
        initial = b''
        if config.get('memory_snapshot'):
            initial = self.path(config['memory_snapshot']).read_bytes()
            if hashlib.sha256(initial).hexdigest() != config.get('memory_sha256'):
                raise ValueError('verified baseline memory checksum required')
        root.mkdir(parents=True)
        with sqlite3.connect(kdb.as_uri() + '?mode=ro', uri=True) as source:
            with sqlite3.connect(root / 'knowledge.sqlite3') as target:
                source.backup(target)
        kdb_ref = self.store.put_bytes((root / 'knowledge.sqlite3').read_bytes())
        memory = self._memory(root / 'memory', initial)
        memory_ref = self.store.put_bytes(memory.snapshot())
        configuration_ref = self.store.put_json(snapshot)
        spec = CycleSpec(cycle_id, mode, projects, memory_ref, skills,
                         self.store.put_json({k.value: list(v) for k, v in mapping.items()}), configuration_ref)
        atomic_json(root / 'service.json', dict(spec=spec.to_dict(), config=snapshot,
            kdb=kdb_ref, initial=initial.decode(), test_identity=self.test.identity if self.test else None))
        return self.open_cycle(cycle_id)

    @staticmethod
    def _memory(root, baseline):
        # A SubjectMemory snapshot contains accepted records, not a new text initial.
        snapshot = json.loads(baseline) if baseline else None
        if snapshot is not None and snapshot.get('schema') != 'subject-memory-snapshot-v1':
            raise ValueError('accepted SubjectMemory snapshot required')
        memory = SubjectMemory(root, (snapshot['initial_memory'].encode() if snapshot else b''))
        if snapshot and not memory.records():
            with memory.transaction() as state:
                for record in snapshot['memories']:
                    version = memory_digest(record)
                    folder = memory.root / 'versions'
                    folder.mkdir(exist_ok=True)
                    atomic_json(folder / (version + '.json'), record)
                    state['active'][record['memory_id']] = version
            if json.loads(memory.snapshot()) != snapshot:
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
        return {name: json.loads((root / name).read_text()) if (root / name).exists() else None
                for name in ('service.json', 'cycle.json', 'controller.json', 'learning.json')}


class BoundCycle:
    def __init__(self, services, root):
        self.services, self.root, self.store = services, root, services.store
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
        self._roles = None
        memory_root, learning_path = root / 'memory', root / 'learning.json'
        state_path = root / 'cycle.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get('isolation'):
            branch = Path(state['isolation']['branch_root'])
            memory_root, learning_path = branch / 'memory', branch / 'learning.json'
        policy = self.config.get('cycle_limits', {})
        self.scheduler = CycleScheduler(state_path, self.spec, self.run,
            lambda *args: self.learning.memory_update(*args),
            skill_update=self.write_skills, isolate=self.isolate,
            recover=self.config.get('recovery', {}).get('enabled', True),
            tuning_every=self.config.get('tuning', {}).get('every_projects', 5), **policy)
        try:
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

    def make_learning(self, path):
        return Learning(path, self.store, self.memory, self, load_project=self.load_project,
            validate=self.validate, mode=self.spec.mode.value,
            role_elapsed=self.role_elapsed,
            unused_window=self.config.get('learning', {}).get('unused_window', 10))

    def observe_role(self, event, payload):
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
                self._roles = IsolatedRoles(str(self.services.path(self.config['roles_python'])),
                    profile, observer=self.observe_role, environment=self.services.environment,
                    **self.config.get('roles', {}))
        return self._roles

    def skillwriter(self, prompt, evidence, *, deadline):
        guidance = {key: self.config.get('prompts', {}).get(key, '')
                    for key in ('skillwriter', 'curriculum', 'waves')}
        return self.roles().skillwriter(prompt + '\nUser guidance:\n' + json.dumps(guidance),
                                        evidence, deadline=deadline)

    def propose_curriculum(self, *, operation_id, deadline):
        """Typed, budgeted suggestions; commands still require host configuration."""
        operation = _name(operation_id)
        receipts = self.scheduler.state.get('extension_receipts', {})
        if operation in receipts:
            record = self.store.read_json(receipts[operation]['reference'])
            return _proposal('curriculum', record['proposal'], {self.spec.configuration_ref})
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
            proposal = self.roles().curriculum(prompt, evidence, deadline=deadline)
            reference = self.store.put_json(dict(proposal=asdict(proposal), operation_id=operation))
            self.scheduler.complete_extension(operation, reference,
                actor_seconds=self.role_elapsed(operation))
            return proposal
        finally:
            self.role_operation = None

    def write_skills(self, skills, memory, result, context):
        self.role_operation = context['operation_id']
        try:
            return self.learning.skill_update(skills, memory, result, context)
        finally:
            self.role_operation = None

    def query(self, memory_ref):
        snapshot = self.store.read_json(memory_ref)
        if snapshot.get('schema') != 'subject-memory-snapshot-v1':
            raise ValueError('accepted memory snapshot required')
        if self.services.test:
            return self.services.test.query
        if self.database is None:
            from benchmarks.a5kernels.embeddings import PinnedBGEEmbeddings
            from benchmarks.a5kernels.knowledge import KnowledgeDB
            path = self.root / 'knowledge.sqlite3'
            if hashlib.sha256(path.read_bytes()).hexdigest() != self.binding['kdb']:
                raise ValueError('frozen KDB changed')
            settings = dict(self.config.get('embeddings', {}))
            if settings.get('cache_dir'):
                settings['cache_dir'] = self.services.path(settings['cache_dir'])
            embeddings = PinnedBGEEmbeddings(**settings)
            embeddings._load()  # Actor setup, not a first-query model-loading charge.
            self.database = KnowledgeDB.open_read_only(path, embeddings)
        records = snapshot['memories']
        adapter = knowledge_db_query(self.database, tuple(self.config['kdb_collections']),
            negative_query=lambda _: [r for r in records if r['proposal']['kind'] != 'correctness'])
        def query(payload):
            result = adapter(payload)
            result['results'].extend(dict(memory=r, provenance={'snapshot': memory_ref})
                for r in records if r['proposal']['kind'] == 'correctness')
            result['provenance']['memory_snapshot'] = memory_ref
            return result
        return query

    def runner(self, root, *, spec=None, bundle=None, config=None, memory_ref,
               backend_identity, reopen=False):
        admitted_at = time.time()
        provider = self.config['provider']
        if not self.services.test:
            from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
            from benchmarks.a5kernels.phase1_provider import direct_profile
            profiles = [direct_profile(i) for i in MODEL_IDENTITIES if i.profile_id == provider['profile_id']]
            if len(profiles) != 1 or profiles[0].model != provider['model']:
                raise ValueError('registered actor model/profile mismatch')
        session = self.services.test.session_factory if self.services.test else (
            lambda: PiSessionAdapter(model=provider['model'], environ=self.services.environment))
        injections = dict(query=self.query(memory_ref), backend_identity=backend_identity,
                          session_factory=session, test_backend=self.services.test is not None)
        if not reopen:
            first = True
            def clock():
                nonlocal first
                if first:
                    first = False
                    return admitted_at
                return time.time()
            injections['clock'] = clock
        runner = ProjectRunner(root, **injections) if reopen else ProjectRunner.create(
            root, spec, bundle, config, **injections)
        context = runner.context
        runner.context = lambda: dict(context(), user_guidance=self.config.get('prompts', {}),
                                      configuration_ref=self.spec.configuration_ref)
        return runner

    def run(self, spec, memory_ref, skills_ref, context):
        operation = _name(context['operation_id'])
        root = self.root / 'projects' / operation
        if root.exists():
            raise ValueError('existing admitted operation requires reconciliation')
        binding = dict(memory=memory_ref, skills=skills_ref, configuration=self.spec.configuration_ref,
                       kdb=self.binding['kdb'], provider=self.config['provider'],
                       test_identity=self.binding['test_identity'])
        backend = self.store.put_json(binding)
        runner = self.runner(root, spec=spec, bundle=SkillBundle.from_dict(self.store.read_json(skills_ref)),
            config=self.services.project_config(self.config, spec), memory_ref=memory_ref, backend_identity=backend)
        atomic_json(root / 'service-binding.json', binding)
        result = runner.run()
        return self.finish_project(runner, operation, result, context['original_project_id'])

    def finish_project(self, runner, operation, result, original_id=None):
        pending = self.scheduler.state['pending']
        if original_id is None:
            if not pending or pending['id'] != operation:
                raise ValueError('no matching scheduler admission')
            original_id = pending['original_id']
        self.retain(runner)
        ref = self.store.put_json(dict(operation_id=operation, original_id=original_id,
            root=str(runner.root), result=result, workflow=runner.workflow.state))
        atomic_json(runner.root / 'service-result.json', dict(reference=ref))
        if result['status'] not in ('accepted', 'rejected', 'budget_exhausted') or runner.workflow.pending:
            raise RuntimeError('project suspended; reconcile same operation before cycle continuation')
        return HostTaskResult(result['status'], ref, runner.workflow.accounting.seconds())

    def open_project(self, operation):
        root = self.root / 'projects' / _name(operation)
        binding = json.loads((root / 'service-binding.json').read_text())
        return self.runner(root, memory_ref=binding['memory'],
                           backend_identity=self.store.put_json(binding), reopen=True)

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
        if packet['result']['contract_ref'] != runner.contract_ref:
            raise ValueError('project contract differs')
        result = packet['result']
        validation = runner.latest('validation-results')
        if (packet['workflow'] != runner.workflow.state or row['outcome'] != result['status']
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
        return dict(project_id=row['original_id'], outcome=result['status'],
            evidence={row['artifact_ref']: dict(kind=kind, summary=result['status'])},
            queries=queries, usage=usage, raw_refs=list(result['events']))

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
        return runner

    def pair(self, event, operation, remaining):
        path = self.root / 'pairs' / (_name(operation) + '.json')
        if path.exists():
            return PairedRecovery(path, artifacts=self.store, runner_factory=self.trial)
        checkpoint = self.open_project(event['original']['operation_id'])
        return PairedRecovery.create(path, checkpoint, artifacts=self.store, memory_ref=event['memory'],
            kdb_ref=self.binding['kdb'], model_config=self.config['provider'], incumbent=event['incumbent'],
            candidate=event['candidate'], joint_limits=BudgetLimits.from_dict(
                self.config.get('recovery', {}).get('budgets', self.config['budgets'])),
            remaining_parent=remaining, runner_factory=self.trial)

    def validate(self, incumbent, candidate, cases, operation):
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

    def reconcile(self):
        """Observe existing project completion; no new provider dispatch."""
        if self.controller.state['pending']:
            return self.controller.reconcile()
        pending = self.scheduler.state['pending']
        if not pending or pending['kind'] not in ('original', 'recovery'):
            raise ValueError('explicit learning/absence reconciliation required')
        runner = self.open_project(pending['id'])
        runner.reconcile()
        if runner.workflow.pending:
            raise ValueError('project is not terminal; continue same admitted runner explicitly')
        self.retain(runner)
        path = runner.root / 'service-result.json'
        if not path.exists():
            raise ValueError('retained project result required')
        ref = json.loads(path.read_text())['reference']
        packet = self.store.read_json(ref)
        if packet['result']['status'] not in ('accepted', 'rejected', 'budget_exhausted'):
            raise ValueError('project is not terminal; continue same admitted runner explicitly')
        if packet['workflow'] != runner.workflow.state:
            raise ValueError('result is stale; retain current project completion first')
        self.scheduler.reconcile(pending['id'], HostTaskResult(packet['result']['status'], ref,
            runner.workflow.accounting.seconds()))

    def step(self):
        """Delegate one dispatch to the shared controller; no service driver loop."""
        return self.controller.step()

    def run_project(self, operation_id):
        runner = self.open_project(operation_id)
        return self.finish_project(runner, operation_id, runner.run())

    def close(self):
        self.scheduler.close()
        if self.database:
            self.database.close()
