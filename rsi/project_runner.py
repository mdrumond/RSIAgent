"""Single-project production composition; no scheduler or alternate agent loop.

Inject a registered Pi session factory and a configured KDB callback (normally
strategy.knowledge_db_query with KnowledgeDB/PinnedBGE). No retrieval fallback.
JSON configuration contains trusted host commands, never actor shell commands.
One controller owns the task directory; receipts are authoritative only through
its durable CAS-backed event index. Unknown pending work is never replayed.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from coding.pi.schema import INTEGER, STRING, obj
from rsi.actor import EvidenceTool, SourceSnapshot, StageResult, WorkflowActor
from rsi.contracts import ProjectSpec, Stage, StepEvidence
from rsi.python_execution import PythonExecution, atomic_json, digest
from rsi.runtime import Workflow, WorkflowError
from rsi.skills import SkillBundle
from rsi.strategy import QueryUnavailable, StrategyEvidence


@dataclass(frozen=True)
class ProjectConfig:
    source: str
    verifier: str
    commands: dict
    editable_files: tuple[str, ...] = ('solution.py',)
    performance_seconds: float | None = None

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value['editable_files'] = tuple(value.get('editable_files', ('solution.py',)))
        result = cls(**value)
        if not {'compile', 'test', 'validation'} <= result.commands.keys():
            raise ValueError('compile, test and validation commands required')
        if result.performance_seconds is not None and 'timing' not in result.commands:
            raise ValueError('performance goal requires timing command')
        if not result.editable_files:
            raise ValueError('at least one editable file required')
        return result


class ProjectRunner:
    """create(root, spec, bundle, config, **injections); reopen with same injections.

    session_factory must return PiSessionAdapter for production. Tests explicitly
    supply scripted sessions and test_backend=True. Backend identity is frozen;
    credentials remain with the injected factory/backend, never in configuration.
    """
    @classmethod
    def create(cls, root, spec, bundle, config, **injections):
        config = ProjectConfig.from_dict(asdict(config) if isinstance(config, ProjectConfig) else config)
        backend = PythonExecution.snapshot(root, config.source, config.verifier,
            commands=config.commands, editable_files=config.editable_files,
            performance_seconds=config.performance_seconds)
        frozen = dict(spec=spec.to_dict(), skills=bundle.to_dict(), config=asdict(config),
                      backend_identity=injections['backend_identity'])
        atomic_json(backend.root / 'runner.json', {'contract': backend.store.put_json(frozen)})
        return cls(root, **injections)

    def __init__(self, root, *, query, backend_identity, session_factory, test_backend=False):
        self.backend = PythonExecution(root)
        self.root, self.store = self.backend.root, self.backend.store
        self.contract_ref = json.loads((self.root / 'runner.json').read_text())['contract']
        frozen = self.store.read_json(self.contract_ref)
        if backend_identity != frozen['backend_identity']:
            raise ValueError('configured backend identity differs from frozen task')
        self.spec = ProjectSpec.from_dict(frozen['spec'])
        self.bundle = SkillBundle.from_dict(frozen['skills'])
        self.config = ProjectConfig.from_dict(frozen['config'])
        self.event_path = self.root / 'events.json'
        self.event_refs = json.loads(self.event_path.read_text()) if self.event_path.exists() else []
        self.events = [self.store.read_json(ref) for ref in self.event_refs]
        state_path = self.root / 'workflow.json'
        persisted = json.loads(state_path.read_text()) if state_path.exists() else None
        self.workflow = Workflow(state_path, self.resolve,
            source_hash=persisted['source'] if persisted else self.backend.project_hash(),
            limits=self.spec.budgets)
        self.backend.workflow = self.workflow
        self.service = StrategyEvidence(self.store, self.append, lambda: self.workflow.state,
            self.bundle, query, workflow_id=self.contract_ref, backend_identity=backend_identity,
            events=self.events, test_backend=test_backend)
        self.selected_file = self.config.editable_files[0]
        self.actor = WorkflowActor(self.workflow, self.bundle, session_factory,
            perform=self.perform, preflight=self.preflight, emit=self.actor_event,
            source_snapshot=self.snapshot, host_category=None,
            evidence_tools=self.tools(), skill_reference=self.service.read_receipt)
        # Always use the durable authority, including skill reads on the live path.
        self.workflow.resolver = self.resolve

    def append(self, event):
        ref = self.store.put_json(event)
        atomic_json(self.event_path, [*self.event_refs, ref])
        self.event_refs.append(ref)
        self.events.append(event)

    def actor_event(self, kind, data):
        if kind == 'skill-usage':
            self.service.record_skill_event(data)
        self.append(dict(type=kind, **data))

    def resolve(self, kind, reference, stage, operation):
        return self.service.resolve(kind, reference, stage, operation)

    def snapshot(self):
        row = self.backend.read(self.selected_file)
        return SourceSnapshot(self.selected_file, row['content'], row['project_hash'])

    def tools(self):
        decision = obj(dict(group={'type': 'string', 'enum': ['results', 'negative_records']},
            index=INTEGER, action={'type': 'string', 'enum': ['used', 'rejected']},
            reason=STRING, redirection=STRING), ['group', 'index', 'action', 'reason'])
        definitions = {
            'query_kdb': obj({'query': STRING}),
            'retrospect': obj({'supplement': STRING}),
            'select_strategy': obj(dict(proposal=STRING, query_ref=STRING,
                retrospective_ref=STRING, skill_refs={'type': 'array', 'items': STRING},
                decisions={'type': 'array', 'items': decision})),
            'read_project_file': obj({'path': STRING}),
            'optimisation_decision': obj(dict(action={'type': 'string',
                'enum': ['noop', 'edit', 'profile']}, reason=STRING)),
        }
        return {name: EvidenceTool(schema, lambda stage, args, op, seconds, name=name:
            self.tool(name, stage, args, op)) for name, schema in definitions.items()}

    def tool(self, name, stage, args, operation):
        if name == 'query_kdb':
            ref = self.service.query(args)
        elif name == 'retrospect':
            ref = self.service.retrospect(**args)
        elif name == 'select_strategy':
            ref = self.service.select(**args)
        elif name == 'read_project_file':
            row = self.backend.read(args['path'])
            self.selected_file = args['path']
            ref = self.store.put_json(dict(path=self.selected_file, **row))
        else:
            if stage != Stage.OPTIMISATION or not args['reason'].strip():
                raise WorkflowError('reasoned optimisation decision requires optimisation stage')
            ref = self.store.put_json(args)
        self.append(dict(type='project-tool', operation_id=operation, stage=stage.value,
                         name=name, reference=ref))
        return dict(reference=ref, evidence=self.store.read_json(ref))

    def latest(self, kind, operation=None):
        return next((e for e in reversed(self.events) if e['type'] == kind and
                     (operation is None or e.get('operation_id') == operation)), None)

    def preflight(self, operation, context):
        if digest(self.backend.verifier.read_bytes()) != self.backend.contract['verifier']:
            raise WorkflowError('frozen verifier changed')
        for name in self.config.editable_files:
            self.backend.read(name)
        return self.result(operation, StageResult(artifact_refs=(self.contract_ref,)))

    def result(self, operation, result):
        self.append(dict(type='host-result', operation_id=operation, result=asdict(result)))
        return result

    def selection(self, operation, proposal):
        rows = [self.store.read_json(e['reference']) for e in self.events
                if e['type'] == 'project-tool' and e['operation_id'] == operation
                and e['name'] == 'select_strategy']
        if not rows or rows[-1]['payload']['proposal'] != proposal:
            raise WorkflowError('perform requires matching host-verified strategy selection')
        return rows[-1]['payload']

    @contextmanager
    def charged_host(self):
        token = self.workflow.accounting.start('implementation')
        try:
            yield
        finally:
            self.workflow.accounting.end(token)

    def perform(self, stage, args, operation, seconds):
        with self.charged_host():
            refs, selection, changed = [], None, None
            if stage in (Stage.DESIGN, Stage.OPTIMISATION):
                selection = self.selection(operation, args['proposal'])
            decision = None
            if stage == Stage.OPTIMISATION:
                decisions = [e for e in self.events if e['type'] == 'project-tool'
                             and e['operation_id'] == operation and e['name'] == 'optimisation_decision']
                if not decisions:
                    raise WorkflowError('explicit optimisation decision required')
                decision = self.store.read_json(decisions[-1]['reference'])['action']
                if (decision == 'edit') != ('edit' in args):
                    raise WorkflowError('optimisation decision does not match edit')
            if 'edit' in args:
                edit = args['edit']
                changed = self.backend.edit(edit['path'], edit['source'],
                    expected_file_hash=edit['sha256'], expected_project_hash=self.workflow.source_hash)
        passed = True
        receipts = []
        commands = (['compile', 'test'] if stage == Stage.TEST else
                    ['validation'] if stage == Stage.VALIDATION else
                    ['profile'] if decision == 'profile' else [])
        if stage == Stage.VALIDATION and self.config.performance_seconds is not None:
            commands.append('timing')
        for command in commands:
            if command not in self.backend.contract['commands']:
                raise WorkflowError('requested command is not in frozen task configuration')
            row = self.backend.run(operation + ':' + command, command,
                numeric=command in ('test', 'validation'), profile=command == 'profile')
            receipts.append(row)
            refs.append(self.store.put_json(row))
            if row['classification'] not in ('passed', 'performance_shortfall'):
                passed = False
                break
        with self.charged_host():
            if stage == Stage.TASK_CONSTRUCTION:
                refs.append(self.contract_ref)  # Already frozen before any model actions.
            if stage == Stage.ACCEPTANCE:
                validation = self.latest('validation-results')
                if not validation or validation['source_hash'] != self.backend.project_hash():
                    raise WorkflowError('acceptance requires fresh validation evidence')
                passed = validation['correctness'] and validation['performance'] != 'shortfall'
            if stage == Stage.VALIDATION:
                self.append(dict(type='validation-results', operation_id=operation,
                    source_hash=self.backend.project_hash(), correctness=passed,
                    performance='shortfall' if any(r['classification'] == 'performance_shortfall'
                        for r in receipts) else ('met' if any(r['command'] == 'timing' and r['classification'] == 'passed'
                        for r in receipts) else 'not_measured') if self.config.performance_seconds
                        else 'not_requested'))
            if receipts:
                self.append(dict(type='command-results', operation_id=operation, receipts=receipts))
            return self.result(operation, StageResult(passed=passed, artifact_refs=tuple(refs),
                retrieval_refs=(selection['query_ref'],) if selection else (),
                retrospective_ref=selection['retrospective_ref'] if selection else None,
                new_source_hash=changed))

    def context(self):
        last = self.latest('command-results')
        feedback = []
        for row in last['receipts'] if last else []:
            feedback.append(dict(classification=row['classification'],
                stdout=self.store.read_bytes(row['stdout']).decode(errors='replace'),
                stderr=self.store.read_bytes(row['stderr']).decode(errors='replace')))
        return dict(project=self.spec.to_dict(), contract=self.contract_ref,
            editable_files=list(self.config.editable_files), feedback=feedback,
            strategy_protocol='Read mapped skills; query_kdb and retrospect; select_strategy '
                'with every result decision; perform matching proposal. Optimisation also '
                'requires optimisation_decision. Use read_project_file to select editable source.')

    def reconcile(self):
        flow = self.workflow
        actual = self.backend.project_hash()
        if flow.pending:
            event = self.latest('stage-result', flow.pending['id'])
            if not event:
                raise WorkflowError('pending operation has no stage receipt; observe, never resubmit')
            result = StageResult(**event['result'])
            if actual != (result.new_source_hash or flow.source_hash):
                raise WorkflowError('source differs from pending result; explicit reconciliation required')
            self.actor.reconcile_pending(StepEvidence.from_dict(event['evidence']), result)
        elif actual != flow.source_hash:
            flow.source_changed(actual)

    def run(self):
        status, error = None, None
        try:
            self.reconcile()
            while not self.workflow.outcome:
                self.actor.step(self.context())
            status = self.workflow.outcome
        except QueryUnavailable as exc:
            status, error = 'suspended', str(exc)
        except (WorkflowError, ValueError, OSError) as exc:
            error = str(exc)
            status = 'budget_exhausted' if 'exhausted' in error else 'suspended'
        finally:
            self.actor.close()
        validation = self.latest('validation-results')
        if validation and validation['source_hash'] != self.backend.project_hash():
            validation = None
        return dict(status=status, error=error, stage=self.workflow.stage.value,
            correctness=validation['correctness'] if validation else None,
            performance=validation['performance'] if validation else 'not_measured',
            history=self.workflow.state['history'], pending=self.workflow.pending,
            events=tuple(self.event_refs), source_hash=self.backend.project_hash())
