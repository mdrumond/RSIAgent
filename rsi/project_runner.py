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
import time
from threading import RLock

from coding.pi.rpc import PiError, PiProviderError, PiDeadlineError, PiCancelled
from coding.pi.session import ActionError
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
        created = injections.get('clock', time.time)()
        config = ProjectConfig.from_dict(asdict(config) if isinstance(config, ProjectConfig) else config)
        backend = PythonExecution.snapshot(root, config.source, config.verifier,
            commands=config.commands, editable_files=config.editable_files,
            performance_seconds=config.performance_seconds)
        frozen = dict(created=created, spec=spec.to_dict(), skills=bundle.to_dict(), config=asdict(config),
                      backend_identity=injections['backend_identity'])
        atomic_json(backend.root / 'runner.json', {'contract': backend.store.put_json(frozen)})
        return cls(root, **injections)

    def __init__(self, root, *, query, backend_identity, session_factory, test_backend=False, clock=time.time):
        self.backend = PythonExecution(root)
        self.root, self.store = self.backend.root, self.backend.store
        self.contract_ref = json.loads((self.root / 'runner.json').read_text())['contract']
        frozen = self.store.read_json(self.contract_ref)
        if backend_identity != frozen['backend_identity']:
            raise ValueError('configured backend identity differs from frozen task')
        self.spec = ProjectSpec.from_dict(frozen['spec'])
        self.bundle = SkillBundle.from_dict(frozen['skills'])
        self.config = ProjectConfig.from_dict(frozen['config'])
        self.event_lock = RLock()
        self.event_path = self.root / 'events.json'
        self.event_refs = json.loads(self.event_path.read_text()) if self.event_path.exists() else []
        self.events = [self.store.read_json(ref) for ref in self.event_refs]
        state_path = self.root / 'workflow.json'
        persisted = json.loads(state_path.read_text()) if state_path.exists() else None
        self.workflow = Workflow(state_path, self.resolve,
            source_hash=persisted['source'] if persisted else self.backend.project_hash(),
            limits=self.spec.budgets, clock=clock)
        if 'created' in frozen and self.workflow.state['created'] != frozen['created']:
            self.workflow.state['created'] = frozen['created']
            self.workflow._save()
        self.backend.workflow = self.workflow
        self.service = StrategyEvidence(self.store, self.append, lambda: self.workflow.state,
            self.bundle, query, workflow_id=self.contract_ref, backend_identity=backend_identity,
            events=self.events, test_backend=test_backend)
        self.selected_file = self.config.editable_files[0]
        self.actor = WorkflowActor(self.workflow, self.bundle, session_factory,
            perform=self.perform, preflight=self.preflight, emit=self.actor_event,
            source_snapshot=self.snapshot, host_category=None,
            evidence_tools=self.tools(), skill_reference=self.service.read_receipt,
            managed_executions=True, perform_ready=self.perform_ready)
        # Always use the durable authority, including skill reads on the live path.
        self.workflow.resolver = self.resolve

    def append(self, event):
        with self.event_lock:
            ref = self.store.put_json(event)
            atomic_json(self.event_path, [*self.event_refs, ref])
            self.event_refs.append(ref)
            self.events.append(event)

    def actor_event(self, kind, data):
        if kind == 'skill-usage':
            self.service.record_skill_event(data)
        self.append(dict(type=kind, **data))

    def resolve(self, kind, reference, stage, operation):
        if kind == 'failed-operation':
            with self.event_lock:
                if not any(e['type'] == 'failed-operation-closure' and e['reference'] == reference
                           for e in self.events):
                    return False
                row = self.store.read_json(reference)
                pending = self.workflow.pending
                proof = row['controller_proof']
                return bool(proof.get('actor_closed') is True and proof.get('no_host_effects') is True
                    and isinstance(proof.get('observation'), str) and proof['observation'].strip()
                    and pending and proof.get('handle') == pending['handle']
                    and pending['id'] == operation and row == dict(
                    operation_id=operation, stage=stage.value, source_hash=self.workflow.source_hash,
                    contract_ref=self.contract_ref, handle=pending['handle'],
                    controller_proof=row['controller_proof'])
                    and self.backend.project_hash() == self.workflow.source_hash
                    and self.no_host_effects(operation))
        return self.service.resolve(kind, reference, stage, operation)

    def no_host_effects(self, operation):
        unsafe = {'host-dispatch', 'host-result', 'stage-result', 'command-results'}
        return not (any(e.get('operation_id') == operation and e['type'] in unsafe
                        for e in self.events) or any(
            row.get('stage_operation') == operation for row in
            self.workflow.state.get('execution_reservations', {}).values()) or any(
            e['type'] == 'strategy-evidence' and e['kind'] in ('retrieval', 'query-unavailable')
            and self.store.read_json(e['reference'])['context']['operation'] == operation
            for e in self.events))

    def record_admission_failure(self, operation, *, controller_proof, observed_source_hash):
        """Host-only API: attest observed terminal Pi closure, never an actor tool.

        controller_proof must retain handle, actor_closed=True, no_host_effects=True,
        and a nonempty observation describing the controller's closure evidence.
        Local receipts can veto that observation. This records proof only; call
        workflow.reject_operation(operation, returned_ref, observed_source_hash)
        explicitly to release pending state without refunding thinking/counts.
        """
        with self.event_lock:
            pending = self.workflow.pending
            if (not pending or pending['id'] != operation or not pending['handle']
                    or self.actor.session is not None
                    or observed_source_hash != self.workflow.source_hash
                    or observed_source_hash != self.backend.project_hash()
                    or controller_proof.get('handle') != pending['handle']
                    or controller_proof.get('actor_closed') is not True
                    or controller_proof.get('no_host_effects') is not True
                    or not isinstance(controller_proof.get('observation'), str)
                    or not controller_proof['observation'].strip()
                    or not self.no_host_effects(operation)):
                raise WorkflowError('admission failure requires observed closure and no host effects')
            ref = self.store.put_json(dict(operation_id=operation, stage=self.workflow.stage.value,
                source_hash=observed_source_hash, contract_ref=self.contract_ref,
                handle=pending['handle'], controller_proof=controller_proof))
            self.append(dict(type='failed-operation-closure', operation_id=operation, reference=ref))
            return ref

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
            'request_redesign': obj({'reason': STRING}),
            'optimisation_decision': obj(dict(action={'type': 'string',
                'enum': ['noop', 'edit', 'profile']}, reason=STRING)),
        }
        scopes = {'select_strategy': (Stage.DESIGN, Stage.OPTIMISATION),
                  'optimisation_decision': (Stage.OPTIMISATION,),
                  'request_redesign': (Stage.DEBUG, Stage.VALIDATION)}
        return {name: EvidenceTool(schema, lambda stage, args, op, seconds, name=name:
            self.tool(name, stage, args, op), stages=scopes.get(name))
            for name, schema in definitions.items()}

    def tool(self, name, stage, args, operation):
        read_error = False
        if name == 'query_kdb':
            ref = self.service.query(args)
        elif name == 'retrospect':
            ref = self.service.retrospect(**args)
        elif name == 'select_strategy':
            ref = self.service.select(**args)
        elif name == 'request_redesign':
            if (stage not in (Stage.DEBUG, Stage.VALIDATION) or stage != self.workflow.stage
                    or not self.workflow.pending or self.workflow.pending['id'] != operation
                    or not args['reason'].strip()):
                raise WorkflowError('redesign requires current debug/validation operation and reason')
            ref = self.store.put_json(dict(reason=args['reason'], operation_id=operation,
                stage=stage.value, source_hash=self.workflow.source_hash, contract_ref=self.contract_ref))
        elif name == 'read_project_file':
            try:
                row = self.backend.read(args['path'])
            except (FileNotFoundError, ValueError) as exc:
                read_error = True
                ref = self.store.put_json(dict(status='error', path=args['path'],
                    error_type=type(exc).__name__, message=str(exc),
                    editable_files=list(self.config.editable_files),
                    hint='Choose a project path; KDB citation paths are not project files.'))
            else:
                self.selected_file = args['path']
                ref = self.store.put_json(dict(path=self.selected_file, **row))
        else:
            if stage != Stage.OPTIMISATION or not args['reason'].strip():
                raise WorkflowError('reasoned optimisation decision requires optimisation stage')
            ref = self.store.put_json(args)
        self.append(dict(type='project-tool', operation_id=operation, stage=stage.value,
                         name=name, reference=ref, status='error' if read_error else 'ok'))
        return dict(reference=ref, evidence=self.store.read_json(ref),
                    status='error' if read_error else 'ok')

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

    def selection(self, operation):
        rows = [self.store.read_json(e['reference']) for e in self.events
                if e['type'] == 'project-tool' and e['operation_id'] == operation
                and e['name'] == 'select_strategy']
        if not rows:
            raise WorkflowError('perform requires host-verified strategy selection')
        # Recheck scope and authority; perform.proposal is only a display label.
        payload = rows[-1]['payload']
        if not all(self.resolve(kind, payload[field], self.workflow.stage, operation)
                   for kind, field in (('retrieval', 'query_ref'), ('retrospective', 'retrospective_ref'))):
            raise WorkflowError('strategy selection is no longer valid for this operation')
        return rows[-1]['payload']

    def perform_ready(self, stage, operation):
        if stage not in (Stage.DESIGN, Stage.OPTIMISATION):
            return True
        try:
            self.selection(operation)
        except WorkflowError:
            return False
        return stage != Stage.OPTIMISATION or any(
            e['type'] == 'project-tool' and e['operation_id'] == operation
            and e['name'] == 'optimisation_decision' for e in self.events)

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
                selection = self.selection(operation)
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
            redesign = False
            cap = self.workflow.limits.implementation_iterations
            design_cap = self.workflow.limits.design_iterations
            if (stage == Stage.DEBUG and self.workflow.state['tested'] != self.workflow.source_hash
                    and cap is not None and self.workflow.state['implementation'] >= cap
                    and (design_cap is None or self.workflow.state['design'] < design_cap)):
                self.tool('request_redesign', stage, dict(reason=
                    'Implementation iteration cap reached; revise design using retained failures'), operation)
            request = next((e for e in reversed(self.events) if e['type'] == 'project-tool'
                and e['operation_id'] == operation and e['name'] == 'request_redesign'), None)
            if request:
                proof = self.store.read_json(request['reference'])
                if (stage not in (Stage.DEBUG, Stage.VALIDATION) or proof['stage'] != stage.value
                        or proof['operation_id'] != operation or proof['source_hash'] != self.workflow.source_hash
                        or proof['contract_ref'] != self.contract_ref or not proof['reason'].strip()):
                    raise WorkflowError('redesign request scope mismatch')
                refs.append(request['reference'])
                redesign = True
            if stage == Stage.TASK_CONSTRUCTION:
                refs.append(self.contract_ref)  # Already frozen before any model actions.
            if stage == Stage.ACCEPTANCE:
                validation = self.latest('validation-results')
                if not validation or validation['source_hash'] != self.backend.project_hash():
                    raise WorkflowError('acceptance requires fresh validation evidence')
                passed = validation['correctness'] and validation['performance'] != 'shortfall'
            if stage == Stage.VALIDATION:
                correctness = any(r['command'] == 'validation' and r['classification'] == 'passed'
                                  for r in receipts)
                performance = ('shortfall' if any(r['classification'] == 'performance_shortfall'
                    for r in receipts) else ('met' if any(r['command'] == 'timing'
                    and r['classification'] == 'passed' for r in receipts) else 'not_measured')
                    if self.config.performance_seconds else 'not_requested')
                passed = passed and correctness and performance in ('met', 'not_requested')
                self.append(dict(type='validation-results', operation_id=operation,
                    source_hash=self.backend.project_hash(), correctness=correctness,
                    performance=performance))
            if receipts:
                self.append(dict(type='command-results', operation_id=operation, receipts=receipts))
            return self.result(operation, StageResult(passed=passed, artifact_refs=tuple(refs),
                retrieval_refs=(selection['query_ref'],) if selection else (),
                retrospective_ref=selection['retrospective_ref'] if selection else None,
                new_source_hash=changed, redesign=redesign))

    def context(self):
        last = self.latest('command-results')
        feedback = []
        for row in last['receipts'] if last else []:
            feedback.append(dict(classification=row['classification'],
                stdout=self.store.read_bytes(row['stdout']).decode(errors='replace'),
                stderr=self.store.read_bytes(row['stderr']).decode(errors='replace')))
        return dict(project=self.spec.to_dict(), contract=self.contract_ref,
            editable_files=list(self.config.editable_files), feedback=feedback,
            iterations={k: self.workflow.state[k] for k in ('design', 'implementation')},
            last_validation=self.latest('validation-results'),
            strategy_protocol='Read mapped skills; query_kdb and retrospect; select_strategy '
                'with every result decision; perform matching proposal. Optimisation also '
                'requires optimisation_decision. Use read_project_file to select editable source. '
                'In debug/validation request_redesign(reason) revisits design with fresh evidence.')

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
        status, error, classification = None, None, None
        try:
            self.reconcile()
            while not self.workflow.outcome:
                self.actor.step(self.context())
            status = self.workflow.outcome
        except PiError as exc:
            status, error = 'suspended', str(exc)
            classification = ('malformed_action' if isinstance(exc, ActionError) else
                'provider_timeout' if isinstance(exc, PiDeadlineError) else
                'provider_failure' if isinstance(exc, PiProviderError) else
                'actor_cancelled' if isinstance(exc, PiCancelled) else 'actor_failure')
            self.append(dict(type='project-failure', classification=classification,
                error_type=type(exc).__name__, pending=self.workflow.pending))
        except QueryUnavailable as exc:
            status, error = 'suspended', str(exc)
        except (WorkflowError, ValueError, OSError) as exc:
            error = str(exc)
            status = 'budget_exhausted' if 'exhausted' in error else 'suspended'
        finally:
            try:
                self.actor.close()
            except PiError as exc:
                self.append(dict(type='project-failure', classification='actor_cleanup_failure',
                    error_type=type(exc).__name__, pending=self.workflow.pending))
                if error is None:
                    status, error, classification = 'suspended', str(exc), 'actor_cleanup_failure'
        validation = self.latest('validation-results')
        if validation and validation['source_hash'] != self.backend.project_hash():
            validation = None
        return dict(status=status, error=error, classification=classification or status,
            stage=self.workflow.stage.value,
            correctness=validation['correctness'] if validation else None,
            performance=validation['performance'] if validation else 'not_measured',
            history=self.workflow.state['history'], pending=self.workflow.pending,
            events=tuple(self.event_refs), source_hash=self.backend.project_hash(),
            contract_ref=self.contract_ref, execution_contract_ref=self.backend.manifest['contract'],
            skill_bundle_hash=self.bundle.digest)
