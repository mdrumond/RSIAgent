"""Host paired trials around ProjectRunner; no alternate actor or learning loop.

runner_factory(request, *, reopen=False) must bind its query/provider to the
request's immutable memory/KDB/model configuration and return ProjectRunner.
Its backend_identity is a digest of that complete binding. Unknown dispatches
require explicit observe/reconcile; neither arm is automatically replayed.
"""
from dataclasses import asdict, dataclass, replace
import copy
import json
import os
import shutil
from pathlib import Path
import uuid
import time
import math

from benchmarks.ascend.subject_memory import SubjectMemory, digest as memory_digest
from rsi.contracts import BudgetLimits, Mode, ProjectSpec, Stage
from rsi.cycles import IsolationSnapshot
from rsi.project_runner import ProjectConfig, ProjectRunner
from rsi.python_execution import atomic_json, digest
from rsi.runtime import WorkflowError
from rsi.skills import SkillBundle


def _files(root, store):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob('*')):
        name = path.relative_to(root)
        if any(p in ('.git', '__pycache__', '.pytest_cache') for p in name.parts):
            continue
        if path.is_symlink():
            raise ValueError('checkpoint symlinks unsupported')
        if path.is_file():
            result[name.as_posix()] = store.put_bytes(path.read_bytes())
    return result


def _half_limits(limits, remaining):
    values = limits.to_dict()
    for key in ('actor_seconds', 'wall_seconds'):
        bounds = [v for v in (values[key], remaining.get(key),
                  remaining.get('recovery_seconds') if key == 'actor_seconds' else None) if v is not None]
        values[key] = min(bounds) / 2 if bounds else None
    for key in ('design_iterations', 'implementation_iterations', 'execution_tries'):
        if values[key] is not None:
            values[key] //= 2
    return BudgetLimits.from_dict(values)  # Cannot admit two nonzero arms from an exhausted allocation.


@dataclass(frozen=True)
class TrialRequest:
    operation_id: str
    arm: str
    root: Path
    spec: ProjectSpec
    bundle: SkillBundle
    config: ProjectConfig
    memory_ref: str
    kdb_ref: str
    model_config: dict
    backend_identity: str
    frozen_ref: str


class PairedRecovery:
    """One pair per original by default; each trial starts at preflight.

    joint_limits is a NEW allocation bounded by remaining_parent, not a reset
    of the original workflow. Integer joint caps split evenly (odd remainder
    unused); None disables a cap. Original consumption is retained as provenance.
    terminal_failure_ref may name an exact public terminal_failure_proof packet
    retained in artifacts. It admits a known-closed failed checkpoint without
    clearing pending or claiming stage completion. Unknown work remains blocked.
    defer_wall_start freezes inputs now but starts the relative pair window at
    its first dispatch; an absolute parent bound still applies during staging.
    One writer owns path, as for the canonical runner and Learning.
    """
    @classmethod
    def create(cls, path, checkpoint, *, artifacts, memory_ref, kdb_ref,
               model_config, incumbent, candidate, joint_limits,
               remaining_parent, runner_factory, terminal_failure_ref=None, clock=None, defer_wall_start=False):
        if not isinstance(checkpoint, ProjectRunner):
            raise ValueError('canonical checkpoint required')
        if type(defer_wall_start) is not bool:
            raise ValueError('defer_wall_start must be boolean')
        clock = clock or checkpoint.workflow.clock
        created = clock()
        walls = [v for v in (joint_limits.wall_seconds, remaining_parent.get('wall_seconds')) if v is not None]
        window = min(walls) if walls else None
        deadline = created + window if window is not None and not defer_wall_start else None
        path = Path(path)
        if path.exists():
            raise ValueError('pair already exists; reopen it')
        failure = None
        if terminal_failure_ref is not None:
            retained = artifacts.read_json(terminal_failure_ref)
            failure = checkpoint.terminal_failure_proof(closure_ref=retained['closure_ref'])
            if retained != failure:
                raise ValueError('terminal failure reference differs from authoritative checkpoint')
            for obj in checkpoint.store.objects.glob('*/*'):
                artifacts.put_bytes(checkpoint.store.read_bytes(obj.parent.name + obj.name))
        if checkpoint.workflow.pending and failure is None:
            raise ValueError('reconciled checkpoint or observed terminal failure required')
        if checkpoint.backend.project_hash() != checkpoint.workflow.source_hash and failure is None:
            raise ValueError('checkpoint source requires reconciliation')
        for ref in (memory_ref, kdb_ref):
            artifacts.read_bytes(ref)
        for ref in (incumbent, candidate):
            SkillBundle.from_dict(artifacts.read_json(ref))
        config = asdict(checkpoint.config)
        config.pop('source')
        config.pop('verifier')
        original = checkpoint.store.read_json(checkpoint.contract_ref)
        if digest(checkpoint.backend.verifier.read_bytes()) != checkpoint.backend.contract['verifier']:
            raise ValueError('checkpoint verifier changed')
        frozen = dict(spec=checkpoint.spec.to_dict(), config=config,
            files=_files(checkpoint.backend.project, artifacts),
            verifier=artifacts.put_bytes(checkpoint.backend.verifier.read_bytes()),
            memory=memory_ref, kdb=kdb_ref, model_config=copy.deepcopy(model_config),
            backend=original['backend_identity'], incumbent=incumbent, candidate=candidate,
            allocation=_half_limits(joint_limits, remaining_parent).to_dict(),
            original_contract=artifacts.put_json(original),
            original_spent=checkpoint.workflow.accounting.seconds(),
            original_source=checkpoint.backend.project_hash(),
            stage_initial_source=checkpoint.workflow.source_hash,
            terminal_failure_ref=terminal_failure_ref,
            original_workflow_ref=artifacts.put_json(checkpoint.workflow.state))
        frozen_ref = artifacts.put_json(frozen)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, dict(frozen=frozen_ref, created=created, wall_deadline=deadline,
            wall_window_seconds=window if defer_wall_start else None,
            wall_started=None if defer_wall_start else created, last_clock=created, arms={name: dict(
            operation_id=uuid.uuid4().hex, status='ready') for name in ('incumbent', 'candidate')}))
        return cls(path, artifacts=artifacts, runner_factory=runner_factory, clock=clock)

    def __init__(self, path, *, artifacts, runner_factory, clock=time.time):
        self.clock = clock
        self.path, self.store, self.factory = Path(path), artifacts, runner_factory
        self.state = json.loads(self.path.read_text())
        self.frozen = self.store.read_json(self.state['frozen'])

    def _save(self):
        atomic_json(self.path, self.state)

    def bind_parent_wall_deadline(self, deadline):
        """Only tighten a retained pair deadline; observation remains possible."""
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError('finite parent deadline required')
        prior = self.state.get('wall_deadline')
        self.state['wall_deadline'] = min(prior, deadline) if prior is not None else deadline
        self._save()

    def _admit_wall(self):
        now = self.clock()
        if not math.isfinite(now) or now < self.state.get('last_clock', now):
            raise WorkflowError('pair clock moved backwards')
        deadline = self.state.get('wall_deadline')
        window = self.state.get('wall_window_seconds')
        start_window = window is not None and self.state.get('wall_started') is None
        if start_window:
            deadline = min(deadline, now + window) if deadline is not None else now + window
        allowance = self.frozen['allocation']['wall_seconds']
        if allowance is not None and 'wall_deadline' not in self.state:
            raise WorkflowError('legacy pair requires authoritative parent wall deadline')
        if deadline is not None and (deadline <= now or allowance is not None and deadline - now < allowance):
            raise WorkflowError('insufficient parent wall allocation for frozen arm')
        if start_window:
            self.state.update(wall_started=now, wall_deadline=deadline)
        self.state['last_clock'] = now
        self._save()  # Persist the first dispatch origin before staging/factory.
        return now

    def request(self, arm):
        frozen, row = self.frozen, self.state['arms'][arm]
        base = self.path.parent / (self.path.stem + '-trials') / row['operation_id']
        inputs = base / 'inputs'
        # Independent mutable copies; no hardlinks and no input from the other arm.
        if not inputs.exists():
            inputs.mkdir(parents=True)
            for name, ref in frozen['files'].items():
                target = inputs / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(self.store.read_bytes(ref))
            (base / 'verify.py').write_bytes(self.store.read_bytes(frozen['verifier']))
        if _files(inputs, self.store) != frozen['files'] or digest((base / 'verify.py').read_bytes()) != frozen['verifier']:
            raise ValueError('frozen trial inputs changed')
        binding = dict(memory=frozen['memory'], kdb=frozen['kdb'],
                       model_config=frozen['model_config'], backend=frozen['backend'])
        return TrialRequest(row['operation_id'], arm, base / 'run',
            replace(ProjectSpec.from_dict(frozen['spec']),
                    budgets=BudgetLimits.from_dict(frozen['allocation'])),
            SkillBundle.from_dict(self.store.read_json(frozen[arm])),
            ProjectConfig.from_dict(dict(frozen['config'], source=str(inputs), verifier=str(base / 'verify.py'))),
            frozen['memory'], frozen['kdb'], copy.deepcopy(frozen['model_config']),
            self.store.put_json(binding), self.state['frozen'])

    def _check(self, runner, request, *, fresh=False):
        if not isinstance(runner, ProjectRunner) or runner.root.resolve() != request.root.resolve():
            raise ValueError('factory must return canonical ProjectRunner in its assigned directory')
        contract = runner.store.read_json(runner.contract_ref)
        if (runner.spec != request.spec or runner.bundle.digest != request.bundle.digest
                or contract['backend_identity'] != request.backend_identity
                or asdict(runner.config) != asdict(request.config)
                or runner.backend.contract['commands'] != self.frozen['config']['commands']
                or runner.backend.contract['performance_seconds'] != self.frozen['config']['performance_seconds']
                or digest(runner.backend.verifier.read_bytes()) != self.frozen['verifier']):
            raise ValueError('unequal trial input, model, verifier, bundle or budget')
        if fresh and (runner.workflow.stage != Stage.PREFLIGHT or runner.workflow.pending
                      or runner.workflow.state['history'] or runner.workflow.accounting.seconds()
                      or _files(runner.backend.project, self.store) != self.frozen['files']):
            raise ValueError('new trial must start at preflight from the frozen checkpoint')

    def step(self):
        if any(row['status'] == 'pending' for row in self.state['arms'].values()):
            raise WorkflowError('observe pending arm; never resubmit')
        arm = next((name for name, row in self.state['arms'].items() if row['status'] == 'ready'), None)
        if arm is None:
            return self.result()
        started = self._admit_wall()
        request = self.request(arm)
        row = self.state['arms'][arm]
        bounds = [v for v in (self.state.get('wall_deadline'),
            started + request.spec.budgets.wall_seconds if request.spec.budgets.wall_seconds is not None else None)
            if v is not None]
        row['wall_deadline'] = min(bounds) if bounds else None
        row['status'] = 'pending'
        self._save()  # Includes factory/setup failure: absence must be confirmed explicitly.
        runner = self.factory(request, reopen=False)
        self._check(runner, request, fresh=True)
        if row['wall_deadline'] is not None:
            runner.workflow.constrain_wall_deadline('paired:' + request.operation_id, row['wall_deadline'])
        row['run_result'] = runner.run()
        self._save()
        return self.reconcile(arm, runner)

    def reconcile(self, arm, runner=None, *, not_dispatched=False):
        row = self.state['arms'][arm]
        if row['status'] == 'complete':
            return self.store.read_json(row['receipt'])
        if row['status'] != 'pending':
            raise WorkflowError('arm is not pending')
        request = self.request(arm)
        if not_dispatched:
            if runner is not None or request.root.exists():
                raise WorkflowError('cannot confirm absence of retained trial work')
            row['status'] = 'ready'
            self._save()
            return None
        if runner is None:
            runner = self.factory(request, reopen=True)  # Observe only; NEVER run().
        self._check(runner, request)
        flow = runner.workflow
        terminal = row.get('run_result', {}).get('status')
        if terminal is None and flow.outcome:
            terminal = flow.outcome
        failure = None
        if flow.outcome not in ('accepted', 'rejected'):
            failure = runner.terminal_failure_proof()
            terminal = 'failed'
            for obj in runner.store.objects.glob('*/*'):
                self.store.put_bytes(runner.store.read_bytes(obj.parent.name + obj.name))
        if terminal not in ('accepted', 'rejected', 'budget_exhausted', 'failed'):
            raise WorkflowError('arm not terminal; observe original runner, never start a new trial')
        if (flow.accounting.unresolved or flow.accounting.uncertainties or (
                flow.source_hash != runner.backend.project_hash() and failure is None)):
            raise WorkflowError('arm accounting/source unreconciled')
        validation = runner.latest('validation-results')
        if validation and validation['source_hash'] != runner.backend.project_hash():
            validation = None
        # Admission-closure entries retain failed setup/thinking consumption,
        # but do not establish completed stages or substitute for skill reads.
        stages = {h['evidence']['stage'] for h in flow.state['history'] if 'evidence' in h}
        complete = {s.value for s in Stage} <= stages
        correctness = failure['correctness'] if failure else validation['correctness'] if validation else None
        workflow = complete and flow.verified_source == flow.source_hash if flow.outcome == 'accepted' else bool(stages)
        seconds = flow.accounting.seconds()
        cap = request.spec.budgets.actor_seconds
        if failure is not None or cap is not None and seconds > cap:
            workflow = False
        classification = failure['classification'] if failure else terminal
        # Provider/setup-only differences do not demonstrate better programming
        # skills. A current-source failed command is independent task evidence.
        task_failure = classification in ('budget_exhausted', 'rejected') or bool(failure and any(
            (command := self.store.read_json(ref))['current_hash'] == runner.backend.project_hash()
            and command['classification'] in ('numeric_failure', 'compile_failure', 'execution_failure')
            for ref in failure['command_receipts']))
        receipt = dict(operation_id=request.operation_id, frozen=request.frozen_ref,
            arm=arm, root=str(request.root), skill_bundle_hash=runner.bundle.digest,
            workflow=workflow, correctness=correctness,
            performance=failure['performance'] if failure else validation['performance'] if validation else 'not_measured',
            outcome=terminal, classification=classification, task_failure=task_failure, actor_seconds=seconds,
            wall_deadline=row.get('wall_deadline'),
            wall_limited=row.get('wall_deadline') is not None and self.clock() >= row['wall_deadline'],
            source_hash=runner.backend.project_hash(), source_files=_files(runner.backend.project, self.store),
            terminal_failure_ref=self.store.put_json(failure) if failure else None,
            contract=self.store.put_json(runner.store.read_json(runner.contract_ref)),
            execution_contract=self.store.put_json(runner.backend.contract),
            workflow_receipt=self.store.put_json(flow.state))
        row.update(status='complete', receipt=self.store.put_json(receipt))
        self._save()
        return receipt

    def result(self):
        if any(row['status'] != 'complete' for row in self.state['arms'].values()):
            return None
        rows = {name: self.store.read_json(row['receipt']) for name, row in self.state['arms'].items()}
        success = {name: r['workflow'] is True and r['correctness'] is True
                   and r['outcome'] == 'accepted' and r['performance'] not in ('shortfall', 'not_measured')
                   for name, r in rows.items()}
        decision = ('benefit' if success['candidate'] and not success['incumbent']
                    and rows['incumbent'].get('task_failure', rows['incumbent']['outcome'] in ('budget_exhausted', 'rejected')) else
                    'regression' if success['incumbent'] and not success['candidate']
                    and rows['candidate'].get('task_failure', rows['candidate']['outcome'] in ('budget_exhausted', 'rejected'))
                    else 'inconclusive')
        if any(r.get('wall_limited') for r in rows.values()):
            decision = 'inconclusive'  # Retain raw outcomes; truncated comparisons cannot promote.
        return dict(decision=decision, frozen=self.state['frozen'], arms=rows,
                    actor_seconds=sum(r['actor_seconds'] for r in rows.values()),
                    evidence_ref=self.store.put_json(dict(frozen=self.state['frozen'], arms=rows, decision=decision)))


def fork_inference(root, parent_memory, learning_path, *, artifacts, memory_ref, skills_ref, namespace):
    """Fork accepted SubjectMemory heads and Learning state; no second memory DB.

    CAS may be shared. Callers construct the inference Learning against the
    returned branch paths before dispatch. There is no implicit parent promotion.
    """
    root = Path(root).resolve()
    if root.exists():
        proof = json.loads((root / 'isolation.json').read_text())
        receipt = IsolationSnapshot(namespace, memory_ref, skills_ref, str(root),
                                    digest((root / 'isolation.json').read_bytes()))
        receipt.verify()
        if proof['parent'] != str(parent_memory.root.resolve()):
            raise ValueError('inference fork parent differs')
        return receipt
    snapshot = parent_memory.snapshot()
    if artifacts.read_bytes(memory_ref) != snapshot:
        raise ValueError('baseline does not match accepted parent memory')
    state = json.loads(Path(learning_path).read_text())
    if any('result' not in op for op in state['operations'].values()):
        raise WorkflowError('reconcile parent learning before fork')
    if state['active'] not in (None, skills_ref):
        raise ValueError('parent active skill differs')
    destination = root
    root = root.with_name(root.name + '.fork-' + uuid.uuid4().hex)
    root.mkdir(parents=True)
    memory = SubjectMemory(root / 'memory', parent_memory.initial.encode())
    # Snapshot supplies accepted ledger heads too; independent transaction publishes them.
    with memory.transaction() as branch:
        for record in json.loads(snapshot)['memories']:
            version = memory_digest(record)
            versions = memory.root / 'versions'
            versions.mkdir(exist_ok=True)
            atomic_json(versions / (version + '.json'), record)
            branch['active'][record['memory_id']] = version
    state['policy']['mode'] = Mode.INFERENCE.value
    state['stopped'] = False
    atomic_json(root / 'learning.json', state)
    atomic_json(root / 'isolation.json', dict(namespace=namespace, memory=memory_ref,
        skills=skills_ref, parent=str(parent_memory.root.resolve()), snapshot=digest(snapshot)))
    if memory.snapshot() != snapshot:
        raise ValueError('forked accepted memory differs')
    # Only a fully built branch is visible to the scheduler; temporary state is
    # never used as an inference namespace after a disconnect.
    try:
        os.rename(root, destination)
    except BaseException:
        shutil.rmtree(root)
        raise
    return IsolationSnapshot(namespace, memory_ref, skills_ref, str(destination),
                             digest((destination / 'isolation.json').read_bytes()))


class CycleController:
    """Drain PR10 extensions before dispatching another original project.

    pair_factory(event, operation_id, remaining) returns a persistent
    PairedRecovery. Learning.validate must consume paired evidence case IDs;
    only beneficial pairs enter its promotion gate. success_cases(event) may
    supply frozen independent validation cases for successful originals.
    Tuning events remain unacknowledged for PR11.
    """
    def __init__(self, path, scheduler, learning, *, artifacts, pair_factory, success_cases=None):
        self.path, self.scheduler, self.learning = Path(path), scheduler, learning
        self.store, self.pair_factory, self.success_cases = artifacts, pair_factory, success_cases
        self.state = json.loads(self.path.read_text()) if self.path.exists() else dict(pending=None, completed={})

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.path, self.state)

    def step(self):
        if self.state['pending']:
            raise WorkflowError('extension pending; observe and reconcile before dispatch')
        if self.scheduler.state['stopped']:
            return None
        event = next((e for e in self.scheduler.extension_events()
                      if e['kind'] in ('paired_recovery_due', 'candidate_validation_due')), None)
        if event is None:
            return self.scheduler.step()
        operation_id = uuid.uuid4().hex
        self.state['pending'] = dict(event=event, operation_id=operation_id)
        self._save()
        return self.reconcile(dispatch=True)

    def reconcile(self, *, dispatch=False):
        pending = self.state['pending']
        if not pending:
            raise WorkflowError('no pending extension')
        event, operation = pending['event'], pending['operation_id']
        result = pending.get('result')
        if result is None:
            if event['kind'] == 'paired_recovery_due':
                if 'wall_deadline' not in pending:
                    cap = self.scheduler.config['wall_seconds']
                    pending['wall_deadline'] = self.scheduler.state['created'] + cap if cap is not None else None
                    self._save()
                if dispatch and pending['wall_deadline'] is not None and self.scheduler.clock() >= pending['wall_deadline']:
                    raise WorkflowError('parent wall allocation exhausted; observe retained arms only')
                pair = self.pair_factory(copy.deepcopy(event), operation, self.scheduler.remaining())
                if pending['wall_deadline'] is not None:
                    pair.bind_parent_wall_deadline(pending['wall_deadline'])
                if dispatch:
                    while pair.result() is None:
                        if self.scheduler.state['stopped']:
                            raise WorkflowError('cycle stopped between paired arms')
                        pair.step()
                result = pair.result()
                if result is None:
                    raise WorkflowError('observe/reconcile retained paired arms before continuing')
            else:
                cases = self.success_cases(event) if self.success_cases else ()
                result = dict(decision='validation' if cases else 'pending', cases=list(cases), actor_seconds=0.)
            pending['result'] = result
            self._save()
        if result['decision'] in ('benefit', 'validation') and 'active' not in pending:
            if self.scheduler.state['stopped']:
                raise WorkflowError('cycle stopped before validation')
            wall = self.scheduler.remaining()['wall_seconds']
            if wall is not None and wall <= 0:
                raise WorkflowError('parent wall allocation exhausted before validation')
            cases = result.get('cases', [result['evidence_ref']] if 'evidence_ref' in result else [])
            # Learning owns durable validation identity; an interrupted callback
            # must be reconciled there before this coordinator can continue.
            pending['active'] = self.learning.evaluate(event['candidate'], cases, operation + ':validation')
            self._save()
        if pending.get('active') == event['candidate'] and self.learning.active == event['candidate']:
            self.scheduler.publish_validated_skills(event['incumbent'], self.learning.active)
        receipt = self.store.put_json(dict(event=event['id'], operation_id=operation, result=result,
                                          active=self.learning.active))
        self.scheduler.complete_extension(event['id'], receipt,
            actor_seconds=result['actor_seconds'], recovery=event['kind'] == 'paired_recovery_due')
        self.state['completed'][event['id']] = receipt
        self.state['pending'] = None
        self._save()
        return self.store.read_json(receipt)
