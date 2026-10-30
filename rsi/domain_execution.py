"""Canonical workflow composition for existing non-Python execution domains.

Target bindings retain their compiler, oracle, measurements, memory and history.
They supply stage operations, not an actor/search loop. WorkflowActor is the
only actor loop; ProjectRunner owns run/reconciliation and evidence tools.
No target commands, remote transports or KDB implementation are introduced.
"""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from threading import RLock
import time
import uuid
from types import SimpleNamespace
from typing import Protocol

from rsi.actor import StageResult, WorkflowActor
from rsi.contracts import Stage
from rsi.project_runner import ProjectRunner
from rsi.python_execution import atomic_json
from rsi.runtime import Workflow, WorkflowError
from rsi.strategy import StrategyEvidence


class DomainExecutionProtocol(Protocol):
    identity: dict
    editable_files: tuple[str, ...]
    store: object
    def project_hash(self) -> str: ...
    def read(self, path: str) -> dict: ...
    def edit(self, path: str, text: str, expected_file_hash: str, expected_project_hash: str) -> str: ...
    def preflight(self, context: dict): ...
    def execute(self, stage: Stage, args: dict, operation: str, activity) -> 'DomainObservation': ...
    def context(self) -> dict: ...


class CanonicalConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class DomainObservation:
    """Trusted host outcome. Numerical success must come from the frozen oracle."""
    passed: bool
    artifact_refs: tuple[str, ...] = ()
    correctness: bool | None = None
    performance: str = 'not_measured'
    redesign: bool = False

    def __post_init__(self):
        if type(self.passed) is not bool or type(self.redesign) is not bool:
            raise TypeError('host outcome flags must be boolean')
        if self.correctness is not None and type(self.correctness) is not bool:
            raise TypeError('correctness must be boolean or unknown')
        if self.performance not in ('not_measured', 'not_requested', 'met', 'shortfall'):
            raise ValueError('unknown host performance result')


class DomainActivity:
    """Target operations use the existing durable clock and command admission.

    Call run once per real compilation/launch/profile, including failed work.
    Setup and actual data copying may use excluded run categories. Remote
    adapters must split those operations; enclosing a transfer in an execution
    interval still charges overlapping real work. Unknown interrupted operations
    stay pending and are never automatically replayed.
    """
    def __init__(self, runner, operation):
        self.runner, self.operation = runner, operation
        self.executions = []

    def run(self, name, category, callback, *, execution=False):
        flow = self.runner.workflow
        if category not in flow.accounting.BILLABLE | flow.accounting.EXCLUDED:
            raise ValueError('unknown domain activity category')
        if category in ('compilation', 'execution', 'profiling') and not execution:
            raise ValueError('compilation/execution/profiling require an execution reservation')
        seconds = self.runner.actor._available()
        if execution:
            key = self.operation + ':' + name
            if not flow.reserve_execution(key):
                raise WorkflowError('domain operation already reserved; observe before resuming')
        activity = uuid.uuid4().hex
        dispatch = dict(kind='rsi-domain-dispatch-v1', activity_id=activity,
            operation_id=self.operation, stage=flow.stage.value, name=name, category=category,
            execution=execution, reservation=self.operation+':'+name if execution else None,
            contract_ref=self.runner.contract_ref, identity=self.runner.checked_identity(self.runner.backend),
            backend_identity=self.runner.service.backend_identity, test_backend=self.runner.service.test_backend,
            source_hash=self.runner.backend.project_hash(), remaining=flow.remaining())
        dispatch_ref = self.runner.store.put_json(dispatch)
        self.runner.append(dict(type='domain-dispatch', operation_id=self.operation,
            activity_id=activity, reference=dispatch_ref))
        from rsi.domain_timing import CURRENT
        prior_intervals = set(flow.accounting.records)
        scope = dict(flow=flow, category=category, token=flow.accounting.start(category))
        context = CURRENT.set(scope)
        from rsi.domain_checkpoint import encode, has_pending_result
        def finish(value, status, terminal):
            flow.accounting.end(scope['token'])
            receipt = dict(kind='rsi-domain-result-v1', activity_id=activity, dispatch_ref=dispatch_ref,
                operation_id=self.operation, stage=flow.stage.value, contract_ref=self.runner.contract_ref,
                source_hash=self.runner.backend.project_hash(), observed_terminal=terminal, status=status,
                result=encode(value), accounting_ref=self.runner.store.put_json({key:record for key,record in flow.state['intervals'].items()
                    if key not in prior_intervals and record[2] is not None}))
            self.runner.append(dict(type='domain-result', operation_id=self.operation,
                activity_id=activity, reference=self.runner.store.put_json(receipt)))
            self.runner.capture_checkpoint(self.operation, scope='activity')
        try:
            value = callback(seconds)
        except BaseException as error:
            terminal = isinstance(error, Exception) and getattr(self.runner.backend,
                'activity_exception_terminal', lambda *_: False)(name, error) is True
            if terminal:
                finish(dict(error_type=type(error).__name__, message=str(error)), 'failed', True)
            else:
                flow.accounting.unresolved.add(scope['token'])
                self.runner.append(dict(type='domain-interrupted', operation_id=self.operation,
                    name=name, activity_id=activity, interval=scope['token']))
            raise
        finally:
            CURRENT.reset(context)
        terminal = not has_pending_result(value) and (not execution or getattr(self.runner.backend,
            'activity_observed_terminal', lambda *_: False)(name, value) is True)
        finish(value, 'complete' if terminal else 'unobserved', terminal)
        if not terminal:
            flow.state['timing_uncertainties'][activity] = dict(
                reason='native activity has no authoritative terminal observation', dispatch_ref=dispatch_ref)
            flow._save()
            raise WorkflowError('native activity remains unobserved; observe original dispatch before recovery')
        if execution:
            self.executions.append(name)
        self.runner.actor._available()
        return value


class _HostView:
    def __init__(self, host, contract):
        self.host, self.manifest = host, dict(contract=contract)

    def __getattr__(self, name):
        return getattr(self.host, name)


class DomainRunner(ProjectRunner):
    """Use ProjectRunner's control loop with an existing target host adapter.

    host exposes store, identity, editable_files, project_hash(), read(path),
    edit(path,text,expected_file_hash,expected_project_hash), preflight(context),
    execute(stage,args,operation,activity), context(). read returns content and
    project_hash. edit returns the new whole-project hash. identity freezes
    target/provider/KDB/memory/verifier, never credentials. Host operations must
    use DomainActivity for compilation/execution/profiling and remote copying.
    """
    @classmethod
    def create(cls, root, spec, bundle, host: DomainExecutionProtocol, **injections):
        root = Path(root)
        if (root / 'domain.json').exists():
            raise FileExistsError('domain task already exists; reopen the same task')
        identity = cls.checked_identity(host)
        frozen = dict(kind='rsi-domain-contract-v2', test_backend=injections.get('test_backend', False),
            spec=spec.to_dict(), skills=bundle.to_dict(), identity=identity,
            backend_identity=injections['backend_identity'], created=injections.get('clock', time.time)())
        reference = host.store.put_json(frozen)
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(root / 'domain.json', dict(contract=reference, store_root=os.path.relpath(host.store.root.resolve(), root.resolve())))
        return cls(root, host, **injections)

    @staticmethod
    def checked_identity(host):
        identity = json.loads(json.dumps(host.identity))
        if not isinstance(identity, dict) or any(not identity.get(key) for key in
                ('target', 'provider', 'kdb', 'memory', 'verifier')):
            raise CanonicalConfigurationError(
                'canonical domain binding requires frozen target, provider, KDB, memory and verifier identities')
        if not host.editable_files:
            raise CanonicalConfigurationError('canonical domain binding requires editable source files')
        return identity

    def __init__(self, root, host, *, query, backend_identity, session_factory,
                 test_backend=False, clock=time.time):
        from rsi.contracts import ProjectSpec
        from rsi.skills import SkillBundle
        self.root, self.backend, self.store = Path(root), host, host.store
        self.contract_ref = json.loads((self.root / 'domain.json').read_text())['contract']
        frozen = self.store.read_json(self.contract_ref)
        if (self.checked_identity(host) != frozen['identity'] or
                backend_identity != frozen['backend_identity']):
            raise CanonicalConfigurationError('domain target/provider/KDB/memory/verifier changed on resume')
        self.spec = ProjectSpec.from_dict(frozen['spec'])
        self.bundle = SkillBundle.from_dict(frozen['skills'])
        self.config = SimpleNamespace(editable_files=tuple(host.editable_files))
        self.selected_file = self.config.editable_files[0]
        self.event_lock, self.event_path = RLock(), self.root / 'events.json'
        self.event_refs = json.loads(self.event_path.read_text()) if self.event_path.exists() else []
        self.events = [self.store.read_json(ref) for ref in self.event_refs]
        if frozen.get('kind') == 'rsi-domain-contract-v2' and frozen['test_backend'] != test_backend:
            raise CanonicalConfigurationError('domain test/production backend changed on resume')
        state_path = self.root / 'workflow.json'
        persisted = json.loads(state_path.read_text()) if state_path.exists() else None
        if persisted and persisted['limits'] != frozen['spec']['budgets']:
            raise WorkflowError('native persisted budget differs from frozen allocation')
        self.restore_checkpoint(host, persisted)
        self.workflow = Workflow(state_path, self.resolve,
            source_hash=persisted['source'] if persisted else host.project_hash(), limits=self.spec.budgets, clock=clock)
        self.workflow.state['created'] = frozen['created']
        self.workflow._save()
        self.service = StrategyEvidence(self.store, self.append, lambda: self.workflow.state,
            self.bundle, query, workflow_id=self.contract_ref, backend_identity=backend_identity,
            events=self.events, test_backend=test_backend)
        self._factory_attempted = False
        self._session_closure_ref = None
        def tracked_factory():
            self._factory_attempted = True
            self._session_closure_ref = None
            session = session_factory()
            original_close = getattr(session.close, '_rsi_original_close', session.close)
            def observed_close():
                original_close()
                self._session_closure_ref = self._observe_session_close(session)
            observed_close._rsi_original_close = original_close
            session.close = observed_close
            return session
        host.runner = self
        self.actor = WorkflowActor(self.workflow, self.bundle, tracked_factory,
            perform=self.perform, preflight=self.preflight, emit=self.actor_event,
            source_snapshot=self.snapshot, host_category=None,
            evidence_tools=self.tools(), skill_reference=self.service.read_receipt,
            managed_executions=True, perform_ready=self.perform_ready, recover_invalid_actions=True)
        self.workflow.resolver = self.resolve
        host.workflow = self.workflow
        # Public result/reconciliation fields use the same frozen domain contract.
        self.backend = _HostView(host, self.contract_ref)
        if not persisted:
            self.capture_checkpoint(None, scope='initial')

    def actor_event(self, kind, data):
        super().actor_event(kind, data)
        observer = getattr(self.backend, 'on_actor_event', None)
        if observer is not None:
            observer(kind, data)

    def tools(self):
        tools = super().tools()
        extra = getattr(self.backend, 'evidence_tools', lambda: {})()
        if tools.keys() & extra.keys():
            raise ValueError('domain evidence tool shadows canonical action')
        return dict(tools, **extra)

    def tool(self, name, stage, args, operation):
        if name != 'query_kdb':
            return super().tool(name, stage, args, operation)
        call = uuid.uuid4().hex
        self.append(dict(type='query-dispatch', call_id=call, operation_id=operation,
            request_ref=self.store.put_json(args)))
        try:
            result = super().tool(name, stage, args, operation)
        except BaseException:
            self.append(dict(type='query-finished', call_id=call, operation_id=operation, status='unavailable'))
            raise
        self.append(dict(type='query-finished', call_id=call, operation_id=operation,
            status='complete', reference=result['reference']))
        return result

    def preflight(self, operation, context):
        if self.checked_identity(self.backend) != self.store.read_json(self.contract_ref)['identity']:
            raise CanonicalConfigurationError('domain baseline identity changed')
        value = self.backend.preflight(context)
        if isinstance(value, StageResult):
            return self.result(operation, value)
        if isinstance(value, DomainObservation):
            return self.result(operation, StageResult(passed=value.passed,
                artifact_refs=(*value.artifact_refs, self.contract_ref)))
        if value is not None and type(value) is not bool:
            raise TypeError('host preflight must return bool, None or trusted observation')
        ref = self.store.put_json(dict(operation_id=operation, stage=Stage.PREFLIGHT.value,
            contract_ref=self.contract_ref, passed=value is not False,
            reason='host preflight returned False' if value is False else 'host preflight completed'))
        return self.result(operation, StageResult(passed=value is not False,
            artifact_refs=(self.contract_ref, ref)))

    def perform(self, stage, args, operation, seconds):
        if self.checked_identity(self.backend) != self.store.read_json(self.contract_ref)['identity']:
            raise CanonicalConfigurationError('domain baseline changed during workflow')
        selection = self.selection(operation) if stage in (Stage.DESIGN, Stage.OPTIMISATION) else None
        activity = DomainActivity(self, operation)
        if stage == Stage.OPTIMISATION:
            decision = self.store.read_json(next(e['reference'] for e in reversed(self.events)
                if e['type'] == 'project-tool' and e['operation_id'] == operation
                and e['name'] == 'optimisation_decision'))
            if (decision['action'] == 'edit') != ('edit' in args):
                raise WorkflowError('optimisation decision does not match edit')
            args = dict(args, optimisation_decision=decision)
        old_hash = self.backend.project_hash()
        if old_hash != self.workflow.source_hash:
            raise WorkflowError('domain source changed outside current stage')
        if 'edit' in args:
            edit = args['edit']
            activity.run('edit', 'implementation', lambda _: self.backend.edit(edit['path'], edit['source'],
                expected_file_hash=edit['sha256'], expected_project_hash=old_hash))
        observation = self.backend.execute(stage, args, operation, activity)
        if not isinstance(observation, DomainObservation):
            raise TypeError('domain host must return DomainObservation')
        if stage in (Stage.TEST, Stage.VALIDATION) and not activity.executions:
            raise WorkflowError('domain test/validation requires a real admitted execution')
        for reference in observation.artifact_refs:
            self.store.read_bytes(reference)
        actual = self.backend.project_hash()
        if stage == Stage.VALIDATION:
            if observation.correctness is None:
                raise WorkflowError('domain validation requires frozen oracle evidence')
            self.append(dict(type='validation-results', operation_id=operation, source_hash=actual,
                correctness=observation.correctness, performance=observation.performance))
        if stage == Stage.ACCEPTANCE:
            validation = self.latest('validation-results')
            if not validation or validation['source_hash'] != actual:
                raise WorkflowError('domain acceptance requires current validation')
            if observation.passed and (validation['correctness'] is not True or
                                      validation['performance'] == 'shortfall' or
                                      self.spec.performance_goal is not None and validation['performance'] != 'met'):
                raise WorkflowError('domain host cannot accept a failed validation')
        redesign = observation.redesign
        implementation_cap = self.workflow.limits.implementation_iterations
        design_cap = self.workflow.limits.design_iterations
        if (stage == Stage.DEBUG and self.workflow.state['tested'] != old_hash
                and implementation_cap is not None
                and self.workflow.state['implementation'] >= implementation_cap
                and (design_cap is None or self.workflow.state['design'] < design_cap)):
            redesign = True
            self.append(dict(type='domain-redesign', operation_id=operation,
                reason='Implementation allocation reached; retain failures for next design', source_hash=old_hash))
        request = next((e for e in reversed(self.events) if e['type'] == 'project-tool'
            and e['operation_id'] == operation and e['name'] == 'request_redesign'), None)
        extra_refs = []
        if request:
            proof = self.store.read_json(request['reference'])
            if (stage not in (Stage.DEBUG, Stage.VALIDATION) or proof['stage'] != stage.value
                    or proof['operation_id'] != operation or proof['source_hash'] != old_hash
                    or proof['contract_ref'] != self.contract_ref or not proof['reason'].strip()):
                raise WorkflowError('domain redesign request scope mismatch')
            extra_refs.append(request['reference'])
            redesign = True
        ref = self.store.put_json(dict(operation_id=operation, stage=stage.value,
            source_before=old_hash, source_after=actual, observation=asdict(observation)))
        passed = observation.passed and (stage != Stage.VALIDATION or
            observation.correctness is True and observation.performance != 'shortfall' and
            (self.spec.performance_goal is None or observation.performance == 'met'))
        checkpoint = self.capture_checkpoint(operation, scope='stage')
        return self.result(operation, StageResult(passed=passed,
            artifact_refs=(*observation.artifact_refs, *extra_refs, ref, checkpoint),
            retrieval_refs=(selection['query_ref'],) if selection else (),
            retrospective_ref=selection['retrospective_ref'] if selection else None,
            new_source_hash=actual if actual != old_hash else None, redesign=redesign))

    def capture_checkpoint(self, operation, *, scope):
        from rsi.domain_checkpoint import encode
        host = getattr(self.backend, 'host', self.backend)
        sources = {path:self.store.put_bytes(host.read(path)['content'].encode()) for path in self.config.editable_files}
        snapshot = getattr(host, 'checkpoint', lambda: None)()
        row = dict(kind='rsi-domain-checkpoint-v1', contract_ref=self.contract_ref,
            identity=self.checked_identity(host), source_hash=host.project_hash(), sources=sources,
            operation_id=operation, scope=scope, host_state=encode(snapshot))
        ref = self.store.put_json(row)
        self.append(dict(type='domain-checkpoint', operation_id=operation, reference=ref))
        return ref

    def restore_checkpoint(self, host, persisted):
        if not persisted:
            return
        from rsi.domain_checkpoint import decode
        admitted = {ref for row in persisted['history'] for ref in row.get('evidence', {}).get('artifact_refs', ())}
        observed = set()
        for event in self.events:
            if event['type'] == 'domain-result':
                receipt = self.store.read_json(event['reference'])
                if receipt['observed_terminal'] is True:
                    observed.add((receipt['operation_id'], receipt['source_hash']))
            if event['type'] == 'domain-checkpoint':
                row = self.store.read_json(event['reference'])
                if row['scope'] == 'initial' or row['scope'] == 'activity' and (row['operation_id'],row['source_hash']) in observed:
                    admitted.add(event['reference'])
            if event['type'] == 'stage-result' and persisted.get('pending') and event['operation_id'] == persisted['pending']['id']:
                admitted.update(event['evidence']['artifact_refs'])
        event = next((e for e in reversed(self.events) if e['type'] == 'domain-checkpoint' and e['reference'] in admitted), None)
        if event is None:
            if persisted.get('outcome') == 'accepted':
                raise WorkflowError('accepted native task lacks retained typed checkpoint; explicit migration required')
            return
        row = self.store.read_json(event['reference'])
        if row['contract_ref'] != self.contract_ref or row['identity'] != self.checked_identity(host):
            raise WorkflowError('native checkpoint contract/target identity differs')
        snapshot = decode(row['host_state'])
        restore = getattr(host, 'restore_checkpoint', None)
        if restore is None:
            if host.project_hash() != row['source_hash']:
                raise WorkflowError('host cannot rehydrate its retained source checkpoint')
        else:
            sources = {path:self.store.read_bytes(ref).decode() for path,ref in row['sources'].items()}
            restore(snapshot, sources)
        if host.project_hash() != row['source_hash']:
            raise WorkflowError('restored native source differs from retained checkpoint')
        if persisted.get('outcome') == 'accepted':
            last = persisted['history'][-1] if persisted['history'] else {}
            if (persisted['pending'] is not None or last.get('evidence', {}).get('stage') != Stage.ACCEPTANCE.value
                    or last.get('passed') is not True or row['source_hash'] != persisted['source'] or persisted['verified'] != persisted['source']
                    or getattr(host, 'accepted_checkpoint_valid', lambda: False)() is not True):
                raise WorkflowError('accepted native checkpoint lacks current verified outcome')

    def _completed_effects(self, observed_queries=()):
        from rsi.domain_evidence import completed_effects
        self.workflow.accounting.seconds()
        if self.workflow.limits.to_dict() != self.workflow.state['limits']:
            raise WorkflowError('native budget changed without persistence')
        try:
            return completed_effects(self.root, self.store, self.contract_ref, self.workflow.state,
                self.events, self.service.backend_identity, self.service.test_backend,
                self.checked_identity(self.backend), self.backend.project_hash(),
                actor_active=self.actor.session is not None)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise WorkflowError('native completed-effect evidence is unavailable or corrupt') from error

    def export_task(self, destination):
        self._completed_effects()
        from rsi.domain_evidence import export_domain_task
        return export_domain_task(self.root, destination)

    def reconcile(self):
        if not self.workflow.pending and self.backend.project_hash() != self.workflow.source_hash:
            raise WorkflowError('domain source changed outside workflow; explicit reconciliation required')
        return super().reconcile()

    def run(self):
        boundary = getattr(self.backend, 'boundary_stage', None)
        if boundary is None:
            return super().run()
        actor = self.actor
        runner = self
        class BoundaryActor:
            def __getattr__(self, name):
                return getattr(actor, name)
            def step(self, context):
                stage = runner.workflow.stage
                result = actor.step(context)
                reached = getattr(runner.backend, 'boundary_reached', lambda stage: True)
                if stage == boundary and reached(stage):
                    runner.workflow.state['domain_boundary'] = boundary.value
                    runner.workflow.stop()
                    raise WorkflowError('requested domain workload boundary reached')
                return result
        self.actor = BoundaryActor()
        try:
            return super().run()
        finally:
            self.actor = actor

    def context(self):
        if self.checked_identity(self.backend) != self.store.read_json(self.contract_ref)['identity']:
            raise CanonicalConfigurationError('domain baseline changed during workflow')
        return dict(project=self.spec.to_dict(), contract=self.contract_ref,
            editable_files=list(self.config.editable_files), domain=self.backend.context(),
            strategy_protocol='Read mapped skills; query_kdb and retrospect; select_strategy '
                'with every result decision. Optimisation also requires optimisation_decision.')


@dataclass(frozen=True)
class CanonicalAdoption:
    """Target-specific factories translate contracts, never run a legacy loop.

    build(owner, **inputs) returns DomainRunner; finish(owner,runner,result,inputs)
    returns the existing public domain result and preserves its history/memory
    contract. distill, when needed, remains a supplied target noncoding operation.
    """
    build: object
    finish: object
    distill: object = None

    def run(self, owner, **inputs):
        runner = self.build(owner, **inputs)
        if not isinstance(runner, DomainRunner):
            raise TypeError('canonical adoption factory must return DomainRunner')
        result = runner.run()
        from rsi.domain_checkpoint import encode, decode
        event = runner.latest('domain-public-result')
        if event and runner.workflow.outcome == 'accepted':
            row = runner.store.read_json(event['reference'])
            if row['contract_ref'] != runner.contract_ref or row['source_hash'] != runner.backend.project_hash():
                raise WorkflowError('retained native public result scope differs')
            value = decode(row['value'])
            refresh = getattr(runner.backend, 'refresh_public_result', None)
            return refresh(value) if refresh is not None else value
        value = self.finish(owner, runner, result, inputs)
        if runner.workflow.outcome == 'accepted' and not getattr(value,'continuation',None):
            ref = runner.store.put_json(dict(contract_ref=runner.contract_ref,
                source_hash=runner.backend.project_hash(), value=encode(value)))
            runner.append(dict(type='domain-public-result', reference=ref))
        return value


def require_adoption(value):
    if not isinstance(value, CanonicalAdoption):
        raise CanonicalConfigurationError(
            'This programming entrypoint requires CanonicalAdoption with a frozen skill bundle, '
            'baseline KDB/memory, provider and target verifier binding. Configure the canonical '
            'domain factory; an old actor/search loop is not a valid fallback.')
    return value
