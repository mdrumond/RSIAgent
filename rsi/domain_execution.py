"""Canonical workflow composition for existing non-Python execution domains.

Target bindings retain their compiler, oracle, measurements, memory and history.
They supply stage operations, not an actor/search loop. WorkflowActor is the
only actor loop; ProjectRunner owns run/reconciliation and evidence tools.
No target commands, remote transports or KDB implementation are introduced.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from threading import RLock
import time
from types import SimpleNamespace

from rsi.actor import StageResult, WorkflowActor
from rsi.contracts import Stage
from rsi.project_runner import ProjectRunner
from rsi.python_execution import atomic_json
from rsi.runtime import Workflow, WorkflowError
from rsi.strategy import StrategyEvidence


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
        self.runner.append(dict(type='domain-dispatch', operation_id=self.operation,
            name=name, category=category, execution=execution))
        token = flow.accounting.start(category)
        try:
            value = callback(seconds)
        except BaseException:
            # An exception does not prove a remote operation stopped. Keep the
            # interval unresolved until this host observes authoritative closure.
            self.runner.append(dict(type='domain-interrupted', operation_id=self.operation,
                name=name, interval=token))
            raise
        flow.accounting.end(token)
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
    def create(cls, root, spec, bundle, host, **injections):
        root = Path(root)
        if (root / 'domain.json').exists():
            raise FileExistsError('domain task already exists; reopen the same task')
        identity = cls.checked_identity(host)
        frozen = dict(spec=spec.to_dict(), skills=bundle.to_dict(), identity=identity,
            backend_identity=injections['backend_identity'], created=injections.get('clock', time.time)())
        reference = host.store.put_json(frozen)
        atomic_json(root / 'domain.json', dict(contract=reference))
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
        self.workflow = Workflow(self.root / 'workflow.json', self.resolve,
            source_hash=host.project_hash(), limits=self.spec.budgets, clock=clock)
        self.workflow.state['created'] = frozen['created']
        self.workflow._save()
        self.service = StrategyEvidence(self.store, self.append, lambda: self.workflow.state,
            self.bundle, query, workflow_id=self.contract_ref, backend_identity=backend_identity,
            events=self.events, test_backend=test_backend)
        self.actor = WorkflowActor(self.workflow, self.bundle, session_factory,
            perform=self.perform, preflight=self.preflight, emit=self.actor_event,
            source_snapshot=self.snapshot, host_category=None,
            evidence_tools=self.tools(), skill_reference=self.service.read_receipt,
            managed_executions=True, perform_ready=self.perform_ready)
        self.workflow.resolver = self.resolve
        # Public result/reconciliation fields use the same frozen domain contract.
        self.backend = _HostView(host, self.contract_ref)

    def preflight(self, operation, context):
        if self.checked_identity(self.backend) != self.store.read_json(self.contract_ref)['identity']:
            raise CanonicalConfigurationError('domain baseline identity changed')
        self.backend.preflight(context)
        return self.result(operation, StageResult(artifact_refs=(self.contract_ref,)))

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
        return self.result(operation, StageResult(passed=passed,
            artifact_refs=(*observation.artifact_refs, *extra_refs, ref),
            retrieval_refs=(selection['query_ref'],) if selection else (),
            retrospective_ref=selection['retrospective_ref'] if selection else None,
            new_source_hash=actual if actual != old_hash else None, redesign=redesign))

    def reconcile(self):
        if not self.workflow.pending and self.backend.project_hash() != self.workflow.source_hash:
            raise WorkflowError('domain source changed outside workflow; explicit reconciliation required')
        return super().reconcile()

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
        return self.finish(owner, runner, result, inputs)


def require_adoption(value):
    if not isinstance(value, CanonicalAdoption):
        raise CanonicalConfigurationError(
            'This programming entrypoint requires CanonicalAdoption with a frozen skill bundle, '
            'baseline KDB/memory, provider and target verifier binding. Configure the canonical '
            'domain factory; an old actor/search loop is not a valid fallback.')
    return value
