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
from uuid import uuid4

from coding.pi.rpc import PiError, PiProviderError, PiDeadlineError, PiCancelled
from coding.pi.session import ActionError
from coding.pi.schema import INTEGER, STRING, obj
from rsi.actor import EvidenceTool, PiSessionAdapter, SourceSnapshot, StageResult, WorkflowActor
from rsi.contracts import ProjectSpec, Stage, StepEvidence
from rsi.python_execution import PythonExecution, atomic_json, digest
from rsi.event_journal import EventJournal
from rsi.runtime import Workflow, WorkflowError
from rsi.skills import SkillBundle
from rsi.strategy import QueryUnavailable, StrategyEvidence
from benchmarks.ascend.agent_context import ContextBudgetError


@dataclass(frozen=True)
class ProjectConfig:
    source: str
    verifier: str
    commands: dict
    editable_files: tuple[str, ...] = ('solution.py',)
    performance_seconds: float | None = None
    generated_paths: tuple[str, ...] = ()
    executor_identity: str | None = None
    retained_paths: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value['editable_files'] = tuple(value.get('editable_files', ('solution.py',)))
        generated = value.get('generated_paths', ())
        if not isinstance(generated, (list, tuple)) or any(not isinstance(p, str) for p in generated):
            raise ValueError('generated_paths must be an array of project-relative paths')
        value['generated_paths'] = tuple(generated)
        retained = value.get('retained_paths', ())
        if not isinstance(retained, (list, tuple)) or any(not isinstance(p, str) for p in retained):
            raise ValueError('retained_paths must be an array of project-relative output paths')
        value['retained_paths'] = tuple(retained)
        result = cls(**value)
        if result.executor_identity is not None and (
                not isinstance(result.executor_identity, str) or not result.executor_identity.strip()):
            raise ValueError('executor_identity must be a nonblank trusted binding')
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
        if config.executor_identity is not None and injections.get('executor_factory') is None:
            raise ValueError('frozen executor binding requires a trusted executor factory')
        backend = PythonExecution.snapshot(root, config.source, config.verifier,
            commands=config.commands, editable_files=config.editable_files,
            performance_seconds=config.performance_seconds, generated_paths=config.generated_paths,
            executor_identity=config.executor_identity, retained_paths=config.retained_paths)
        frozen = dict(created=created, spec=spec.to_dict(), skills=bundle.to_dict(), config=asdict(config),
                      backend_identity=injections['backend_identity'])
        atomic_json(backend.root / 'runner.json', {'contract': backend.store.put_json(frozen)})
        return cls(root, **injections)

    def __init__(self, root, *, query, backend_identity, session_factory, test_backend=False, clock=time.time,
                 executor_factory=None, command_guard=None):
        self.backend = PythonExecution(root, executor_factory=executor_factory, command_guard=command_guard)
        self.root, self.store = self.backend.root, self.backend.store
        self.contract_ref = json.loads((self.root / 'runner.json').read_text())['contract']
        frozen = self.store.read_json(self.contract_ref)
        if backend_identity != frozen['backend_identity']:
            raise ValueError('configured backend identity differs from frozen task')
        self.spec = ProjectSpec.from_dict(frozen['spec'])
        self.bundle = SkillBundle.from_dict(frozen['skills'])
        self.config = ProjectConfig.from_dict(frozen['config'])
        if self.config.executor_identity != self.backend.contract.get('executor_identity'):
            raise ValueError('executor identity differs between frozen contracts')
        self.event_lock = RLock()
        self.event_path = self.root / 'events.json'
        self.event_journal = EventJournal(self.root)
        self.event_refs = self.event_journal.refs
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
        self._factory_attempted = False
        self._session_closure_ref = None
        def tracked_factory():
            self._factory_attempted = True
            self._session_closure_ref = None
            session = session_factory()  # A raising factory may have unknown partial effects.
            original_close = getattr(session.close, '_rsi_original_close', session.close)
            def observed_close():
                original_close()
                self._session_closure_ref = self._observe_session_close(session)
            observed_close._rsi_original_close = original_close
            session.close = observed_close
            return session
        self.actor = WorkflowActor(self.workflow, self.bundle, tracked_factory,
            perform=self.perform, preflight=self.preflight, emit=self.actor_event,
            source_snapshot=self.snapshot, host_category=None,
            evidence_tools=self.tools(), skill_reference=self.service.read_receipt,
            managed_executions=True, perform_ready=self.perform_ready, recover_invalid_actions=True)
        # Always use the durable authority, including skill reads on the live path.
        self.workflow.resolver = self.resolve

    def append(self, event):
        with self.event_lock:
            ref = self.store.put_json(event)
            self.event_journal.append(ref)
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
                return bool(self._valid_controller_closure(pending, proof) and proof.get('no_host_effects') is True
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
        unsafe = {'host-dispatch', 'host-result', 'stage-result', 'command-results', 'query-dispatch'}
        return not (any(e.get('operation_id') == operation and e['type'] in unsafe
                        for e in self.events) or any(
            row.get('stage_operation') == operation for row in
            self.workflow.state.get('execution_reservations', {}).values()) or any(
            e['type'] == 'strategy-evidence' and e['kind'] in ('retrieval', 'query-unavailable')
            and self.store.read_json(e['reference'])['context']['operation'] == operation
            for e in self.events))

    def _valid_controller_closure(self, pending, proof):
        if not pending or proof.get('handle') != pending['handle']:
            return False
        if pending['handle'] is None:
            if (proof.get('operation_id') != pending['id']
                    or proof.get('no_actor_dispatched') is not True
                    or any(e['type'] == 'actor-dispatch' and e.get('operation_id') == pending['id']
                           for e in self.events)):
                return False
            return proof.get('actor_closed') is True or proof.get('actor_absent') is True
        return proof.get('actor_closed') is True

    def record_admission_failure(self, operation, *, controller_proof, observed_source_hash):
        """Host-only API: attest observed terminal Pi closure, never an actor tool.

        controller_proof must retain handle, observed closure, no_host_effects=True,
        and a nonempty observation. Before dispatch, a None handle additionally
        requires operation_id, no_actor_dispatched=True and actor_closed/actor_absent.
        Local receipts can veto that observation. This records proof only; call
        workflow.reject_operation(operation, returned_ref, observed_source_hash)
        explicitly to release pending state without refunding thinking/counts.
        """
        with self.event_lock:
            pending = self.workflow.pending
            if (not pending or pending['id'] != operation
                    or self.actor.session is not None
                    or observed_source_hash != self.workflow.source_hash
                    or observed_source_hash != self.backend.project_hash()
                    or not self._valid_controller_closure(pending, controller_proof)
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
            call = uuid4().hex
            self.append(dict(type='query-dispatch', operation_id=operation, call_id=call,
                             request_ref=self.store.put_json(args)))
            try:
                ref = self.service.query(args)
            except QueryUnavailable:
                self.append(dict(type='query-finished', operation_id=operation, call_id=call,
                                 status='unavailable'))
                raise
            self.append(dict(type='query-finished', operation_id=operation, call_id=call,
                             status='complete', reference=ref))
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
        evidence = self.store.read_json(ref)
        return dict(reference=ref, evidence=evidence, summary=self.evidence_summary(name, ref, evidence),
                    status='error' if read_error else 'ok')

    def evidence_summary(self, name, reference, record):
        """Small actual inventory for the actor's always-visible receipt index.

        Full CAS records stay in result.evidence for existing context paging;
        these summaries never stand in for considering the retained row content.
        """
        pending, stage = self.workflow.pending, self.workflow.stage
        summary = dict(tool=name, reference=reference, stage=stage.value,
            operation_id=pending['id'] if pending else None,
            host_perform_ready=bool(pending and self.perform_ready(stage, pending['id'])),
            mapped_skill_reads_required=list(self.bundle.skills_for(stage)))
        if name == 'query_kdb':
            payload = record['payload']
            response = payload['response']
            groups = ('results', 'negative_records')
            summary.update(query_ref=reference, query=payload['request']['query'],
                counts={group: len(response[group]) for group in groups},
                decision_indexes={group: list(range(len(response[group]))) for group in groups},
                requirements='Select with query_ref, retrospective_ref, mapped skill_refs; '
                    'every group/index needs used/rejected and reason; negative used needs redirection.')
        elif name == 'retrospect':
            payload = record['payload']
            summary.update(retrospective_ref=reference, history_count=len(payload['history']),
                failure_indexes=payload['failures'], redirection_indexes=payload['redirections'],
                requirements='Use this operation-scoped retrospective_ref in select_strategy.')
        elif name == 'select_strategy':
            payload = record['payload']
            summary.update(selection_ref=reference, query_ref=payload['query_ref'],
                retrospective_ref=payload['retrospective_ref'], decision_count=len(payload['decisions']),
                requirements='Selection recorded; optimisation also needs optimisation_decision before perform.')
        return summary

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
                self.append(dict(type='source-edit', operation_id=operation,
                    before_source=self.workflow.source_hash, after_source=changed,
                    path=edit['path'], source_ref=self.store.put_bytes(edit['source'].encode())))
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

    def _observe_session_close(self, session):
        """Called only after this actual session's close returned successfully.

        Wrapping the returned session preserves observations when actor setup
        closes/clears its slot before propagating a readiness error to run().
        A factory which raises before returning any session produces no proof.
        """
        known_pi = isinstance(session, PiSessionAdapter)
        rpc = session.session.rpc if known_pi else None
        absent = known_pi and rpc is None
        closed = (rpc is not None and rpc.process.poll() is not None) if known_pi else self.service.test_backend
        settled = session.session.gateway.finished.is_set() if known_pi else self.service.test_backend
        pending = self.workflow.pending
        return self._closure(dict(method='observed-session-close', actor_closed=closed,
            actor_absent=absent, provider_settled=settled,
            operation_id=pending['id'] if pending else None,
            no_actor_dispatched=bool(pending and pending['handle'] is None),
            observation='Owned session close returned; actual RPC exit/absence and provider settlement observed'))

    def _closure(self, observation):
        pending = self.workflow.pending
        dispatch = self.latest('actor-dispatch')
        row = dict(kind='rsi-actor-closure-v1', contract_ref=self.contract_ref,
            backend_identity=self.service.backend_identity, test_backend=self.service.test_backend,
            pending=pending, handle=pending['handle'] if pending else dispatch.get('handle') if dispatch else None,
            stage=self.workflow.stage.value, stage_initial_source=self.workflow.source_hash,
            source_hash=self.backend.project_hash(), observation=observation)
        ref = self.store.put_json(row)
        self.append(dict(type='actor-closure', reference=ref))
        return ref

    def _outcome(self, result, closure_ref):
        ref = self.store.put_json(dict(result=dict(result), closure_ref=closure_ref,
            workflow_state_ref=self.store.put_json(self.workflow.state)))
        self.append(dict(type='project-run-outcome', reference=ref))
        return ref

    def _completed_effects(self, observed_queries=()):
        """Read-only checks. Completed reads are evidence, not unknown effects."""
        flow = self.workflow
        if self.actor.session is not None or any(row[2] is None for row in flow.accounting.records.values()):
            raise WorkflowError('active actor or clock interval prevents failure proof')
        seconds = flow.accounting.seconds()  # Also rejects unresolved timing placements.
        if flow.limits.to_dict() != flow.state['limits']:
            raise WorkflowError('unpersisted budget change prevents failure proof')
        if (json.loads((self.root / 'manifest.json').read_text()) != self.backend.manifest
                or json.loads((self.root / 'runner.json').read_text())['contract'] != self.contract_ref):
            raise WorkflowError('frozen manifest pointer differs')
        contract = self.store.read_json(self.backend.manifest['contract'])
        frozen = self.store.read_json(self.contract_ref)
        config = ProjectConfig.from_dict(frozen['config'])
        if (contract['commands'] != {k: list(v) for k, v in config.commands.items()}
                or contract['editable_files'] != sorted(set(config.editable_files))
                or contract.get('generated_paths', []) != sorted(set(config.generated_paths))
                or contract['performance_seconds'] != config.performance_seconds
                or digest(self.backend.verifier.read_bytes()) != contract['verifier']):
            raise WorkflowError('frozen execution contract or verifier differs')
        commands, known_sources = [], {flow.source_hash}
        reservations = flow.state.get('execution_reservations', {})
        expected_paths = set()
        for operation, reservation in reservations.items():
            path = self.backend._receipt_path(operation)
            expected_paths.add(path)
            try:
                receipt = self.backend.receipt(operation)
                index = json.loads(path.read_text())
            except (OSError, ValueError, WorkflowError) as exc:
                raise WorkflowError('execution reservation has no complete receipt') from exc
            argv = [arg.replace('{verifier}', str(self.backend.verifier))
                    for arg in contract['commands'].get(receipt['command'], ())]
            if receipt['profiling']:
                profile = self.root / 'receipts' / (digest(operation.encode()) + '.profile')
                argv = [argv[0], '-m', 'cProfile', '-o', str(profile), *argv[1:]] if argv else []
            if (receipt['status'] != 'complete' or receipt['operation_id'] != operation
                    or receipt['stage_operation'] != reservation['stage_operation'] or receipt['argv'] != argv):
                raise WorkflowError('command receipt differs from frozen dispatch')
            commands.append(index['artifact'])
            known_sources.add(receipt['current_hash'])
        if set((self.root / 'receipts').glob('*.json')) != expected_paths:
            raise WorkflowError('unreserved command receipt prevents failure proof')
        actual = self.backend.project_hash()
        for event in self.events:
            if event['type'] == 'source-edit':
                source = self.store.read_bytes(event['source_ref'])
                if event['after_source'] == actual and self.backend._file(event['path']).read_bytes() == source:
                    known_sources.add(actual)
        if actual not in known_sources:
            raise WorkflowError('unobserved source change prevents failure proof')
        dispatched = {e['call_id']: e for e in self.events if e['type'] == 'query-dispatch'}
        finished = {e['call_id']: e for e in self.events if e['type'] == 'query-finished'}
        if set(dispatched) != set(finished) or any(e['status'] != 'complete' for e in finished.values()):
            raise WorkflowError('unknown or unavailable query prevents failure proof')
        retrievals = set(observed_queries)
        for call, event in finished.items():
            record = self.store.read_json(event['reference'])
            if (event['operation_id'] != dispatched[call]['operation_id']
                    or record['context']['operation'] != event['operation_id']
                    or record['payload']['request'] != self.store.read_json(dispatched[call]['request_ref'])):
                raise WorkflowError('query receipt scope mismatch')
            retrievals.add(event['reference'])
        retained = {e['reference'] for e in self.events if e['type'] == 'strategy-evidence'
                    and e['kind'] == 'retrieval'}
        if retrievals != retained or any(e['type'] == 'strategy-evidence' and e['kind'] == 'query-unavailable'
                                        for e in self.events):
            raise WorkflowError('query completion coverage is not established')
        for ref in retrievals:
            row = self.store.read_json(ref)
            if (row['kind'] != 'retrieval' or row['context']['workflow'] != self.contract_ref
                    or row['payload']['backend'] != self.service.backend_identity
                    or row['payload']['test_backend'] != self.service.test_backend):
                raise WorkflowError('untrusted retrieval receipt')
        return dict(command_receipts=commands, retrieval_receipts=sorted(retrievals),
                    actor_seconds_total=seconds, source_hash=actual)

    def record_terminal_observation(self, result, *, controller_proof, observed_source_hash):
        """Explicit trusted host observation for old runs; never an actor tool.

        The host must actually observe terminal Pi/provider closure and all query
        callbacks; merely reopening a runner is insufficient. Pending is retained.
        A pre-dispatch None handle requires the explicit operation_id and
        no_actor_dispatched=True plus observed actor closure or absence.
        """
        with self.event_lock:
            pending = self.workflow.pending
            if (not pending or self.actor.session is not None
                    or not self._valid_controller_closure(pending, controller_proof)
                    or controller_proof.get('provider_settled') is not True
                    or not isinstance(controller_proof.get('observation'), str)
                    or not controller_proof['observation'].strip()
                    or observed_source_hash != self.backend.project_hash()
                    or result['pending'] != pending or result['source_hash'] != observed_source_hash
                    or result['stage'] != self.workflow.stage.value
                    or result.get('contract_ref') != self.contract_ref
                    or result['status'] not in ('suspended', 'budget_exhausted', 'rejected')):
                raise WorkflowError('explicit terminal closure/source/result observation required')
            self._completed_effects(controller_proof.get('completed_query_refs', ()))
            closure = self._closure(dict(method='trusted-controller', **controller_proof))
            return self._outcome(result, closure)

    def terminal_failure_proof(self, *, closure_ref=None):
        """Read-only failure admission evidence; never complete or replay a stage.

        Service persists the returned packet and admits failure once. Missing
        observations remain suspended, including restarted unobserved actors.
        """
        with self.event_lock:
            outcomes = [e for e in reversed(self.events) if e['type'] == 'project-run-outcome']
            event = next((e for e in outcomes if self.store.read_json(e['reference'])['closure_ref']
                          and (closure_ref is None or self.store.read_json(e['reference'])['closure_ref'] == closure_ref)),
                         outcomes[0] if outcomes else None)
            if not event or self.workflow.state['stopped']:
                raise WorkflowError('no observed terminal failed run')
            outcome = self.store.read_json(event['reference'])
            result, ref = outcome['result'], outcome['closure_ref']
            if not ref or closure_ref is not None and closure_ref != ref or not any(
                    e['type'] == 'actor-closure' and e['reference'] == ref for e in self.events):
                raise WorkflowError('actual actor closure receipt required')
            closure = self.store.read_json(ref)
            if (result['status'] not in ('suspended', 'budget_exhausted', 'rejected')
                    or result['classification'] == 'actor_cleanup_failure'
                    or result.get('contract_ref') != self.contract_ref
                    or result['source_hash'] != closure['source_hash']
                    or result['stage'] != self.workflow.stage.value
                    or closure['contract_ref'] != self.contract_ref
                    or closure['backend_identity'] != self.service.backend_identity
                    or closure['test_backend'] != self.service.test_backend
                    or closure['pending'] != self.workflow.pending
                    or closure['stage_initial_source'] != self.workflow.source_hash
                    or closure['source_hash'] != self.backend.project_hash()
                    or self.store.read_json(outcome['workflow_state_ref']) != self.workflow.state):
                raise WorkflowError('failed run/closure no longer matches current checkpoint')
            observed = closure['observation']
            pending = closure['pending']
            absent = (observed.get('method') in ('host-no-actor-dispatch', 'observed-session-close', 'trusted-controller')
                and observed.get('actor_absent') is True
                and not any(e['type'] == 'actor-dispatch' and
                    (pending is None or e.get('operation_id') == pending['id']) for e in self.events))
            if (not absent and observed.get('actor_closed') is not True
                    or observed.get('provider_settled') is not True):
                raise WorkflowError('actor/provider closure was not observed')
            effects = self._completed_effects(observed.get('completed_query_refs', ()))
            packet = dict(kind='rsi-terminal-failure-v1', classification=result['classification'],
                outcome='failed', error=result['error'], failed_stage=result['stage'], pending=self.workflow.pending,
                correctness=result.get('correctness'), performance=result.get('performance'),
                stage_initial_source=self.workflow.source_hash, **effects,
                project_contract_ref=self.contract_ref, execution_contract_ref=self.backend.manifest['contract'],
                frozen_skill_hash=self.bundle.digest, result_ref=event['reference'], closure_ref=ref,
                accounting_snapshot=outcome['workflow_state_ref'], created=self.workflow.state['created'],
                limits=self.workflow.state['limits'], stage_completion_claim=False, replay_allowed=False)
            return json.loads(json.dumps(packet, allow_nan=False))  # No mutable state aliases.

    def run(self):
        status, error, classification, closure_ref = None, None, None, None
        typed_error = None
        entry_pending = self.workflow.pending
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
        except ContextBudgetError as exc:
            typed_error = exc
            status, error, classification = 'suspended', str(exc), 'context_budget_exhausted'
            self.append(dict(type='project-failure', classification=classification,
                error_type=type(exc).__name__, pending=self.workflow.pending))
        except (WorkflowError, ValueError, OSError) as exc:
            error = str(exc)
            status = 'budget_exhausted' if 'exhausted' in error else 'suspended'
        finally:
            try:
                self.actor.close()
                if self._session_closure_ref is not None:
                    row = self.store.read_json(self._session_closure_ref)
                    if row['pending'] == self.workflow.pending and row['stage'] == self.workflow.stage.value:
                        closure_ref = self._session_closure_ref
                elif (entry_pending is None and not self._factory_attempted
                        and not any(e['type'] == 'actor-dispatch' for e in self.events)
                        and not self.workflow.state.get('execution_reservations')
                        and not any(e['type'] == 'query-dispatch' for e in self.events)):
                    closure_ref = self._closure(dict(method='host-no-actor-dispatch',
                        actor_absent=True, actor_closed=False, provider_settled=True,
                        observation='This live controller reached failure without calling the actor factory or dispatching backend work'))
            except PiError as exc:
                self.append(dict(type='project-failure', classification='actor_cleanup_failure',
                    error_type=type(exc).__name__, pending=self.workflow.pending))
                if error is None:
                    status, error, classification = 'suspended', str(exc), 'actor_cleanup_failure'
        validation = self.latest('validation-results')
        if validation and validation['source_hash'] != self.backend.project_hash():
            validation = None
        result = dict(status=status, error=error, classification=classification or status,
            stage=self.workflow.stage.value,
            correctness=validation['correctness'] if validation else None,
            performance=validation['performance'] if validation else 'not_measured',
            history=self.workflow.state['history'], pending=self.workflow.pending,
            events=tuple(self.event_refs), source_hash=self.backend.project_hash(),
            contract_ref=self.contract_ref, execution_contract_ref=self.backend.manifest['contract'],
            skill_bundle_hash=self.bundle.digest, actor_closure_ref=closure_ref)
        result['outcome_ref'] = self._outcome(result, closure_ref)
        result['events'] = tuple(self.event_refs)
        with self.event_lock:
            self.event_journal.checkpoint()
        if typed_error is not None:
            # Preserve the original typed admission failure for domain callers'
            # abort classification, after retaining outcome and actor closure.
            raise typed_error
        return result
