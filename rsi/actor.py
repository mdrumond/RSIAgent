"""Pi action selection for the host-owned canonical workflow.

Host callbacks own execution and verification; a model proposal never proves
success. PR7 supplies project snapshots and execution receipts, PR6 supplies
retrieval/retrospective authority through Workflow.resolver. Pending work must
be reconciled by its owner, never redispatched by constructing another actor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import queue
import threading
import time
import uuid

from coding.pi.schema import STRING, obj, validate
from coding.pi.session import PiCodingSession
from rsi.contracts import Stage, StepEvidence
from rsi.runtime import WorkflowError
from rsi.skills import SkillUsage


SCHEMAS = {
    'read_skill': obj({'name': STRING, 'section': STRING}, ['name']),
    'perform': obj({'proposal': STRING, 'edit': obj({
        'path': STRING, 'sha256': STRING, 'source': STRING})}, ['proposal']),
}
SYSTEM = (
    'You select actions for a host-controlled programming workflow. '
    'Read every mapped skill using read_skill before perform. Only the host '
    'executes proposals and determines outcomes or stage transitions. '
    'Use the current stage and observation; never skip stages or claim success '
    'from a queued action. Source edits require the exact editable path/hash. '
    'Design and optimisation require real retrospective and KDB evidence. '
    'Select exactly one typed host action per turn.'
)


@dataclass(frozen=True)
class SourceSnapshot:
    """Host snapshot: editable file text plus independent whole-project hash."""
    path: str
    text: str
    project_hash: str

    @property
    def sha256(self):
        return sha256(self.text.encode()).hexdigest()


@dataclass(frozen=True)
class StageResult:
    """Trusted host results, never accepted directly from model arguments."""
    passed: bool = True
    artifact_refs: tuple[str, ...] = ()
    retrieval_refs: tuple[str, ...] = ()
    retrospective_ref: str | None = None
    new_source_hash: str | None = None
    redesign: bool = False


@dataclass(frozen=True)
class EvidenceTool:
    """Host-owned PR6 tool; invoke(stage, args, operation, seconds) returns data."""
    schema: dict
    invoke: object


class PiSessionAdapter:
    """Reuse Pi's schemas, loopback gateway, context paging and exact edits.

    Eager startup + get_state handshake exclude process readiness from charged
    next_action time. Setup has its own RPC timeout; model deadlines are assigned
    only afterward from the remaining charged allocation.
    """
    def __init__(self, **kwargs):
        self.session = PiCodingSession(schemas=SCHEMAS, system_prompt=SYSTEM, **kwargs)

    def prepare(self, timeout, *, schemas):
        session = self.session
        session.schemas.update(schemas)
        session._start()
        request = uuid.uuid4().hex
        session.rpc.send({'type': 'get_state', 'id': request})
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Pi readiness timeout')
            try:
                kind, line = session.rpc.records.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError('Pi readiness timeout') from None
            if kind == 'eof':
                raise WorkflowError('Pi exited during setup')
            event = json.loads(line)
            session.rpc.events.append(event)
            if event.get('id') == request and event.get('type') == 'response':
                if not event.get('success'):
                    raise WorkflowError('Pi readiness rejected')
                return

    def next_action(self, context, allowed, *, seconds, record, source):
        session = self.session
        # threading's maximum is a technical RPC bound, not a workflow budget.
        session.deadline = time.monotonic() + (threading.TIMEOUT_MAX if seconds is None else seconds)
        session.gateway.deadline = session.deadline
        write = None
        if source is not None and Stage(context['stage']) in (Stage.IMPLEMENTATION, Stage.DEBUG, Stage.OPTIMISATION):
            write = lambda text: ('perform', {'proposal': 'Apply exact source edit',
                'edit': {'path': source.path, 'sha256': source.sha256, 'source': text}})
        return session.next_action(context, allowed, record=record,
            source=source.text if source else None, write_action=write)

    def close(self):
        self.session.close()


class WorkflowActor:
    """Single-controller stage driver; call step() once per workflow stage.

    perform(stage, arguments, operation_id, remaining_seconds) returns StageResult
    from host verification. It must enforce the supplied duration while working;
    the controller checks consumption again before releasing the result. Its
    default interval covers host work; an executor owning transfer/compile/run
    intervals must set host_category=None and instrument every interval itself.
    emit must durably retain events. Exceptions preserve pending operations for
    explicit host reconciliation; no receipt is reconstructed or invented.
    """
    def __init__(self, workflow, bundle, session_factory, *, perform, preflight,
                 emit, source_snapshot=None, host_category='implementation', setup_timeout=60,
                 evidence_tools=None, skill_reference=None):
        self.workflow, self.bundle = workflow, bundle
        self.session_factory, self.session = session_factory, None
        self.perform, self.preflight, self.emit = perform, preflight, emit
        self.source_snapshot, self.host_category = source_snapshot, host_category
        self.setup_timeout = setup_timeout
        self.skill_reference = skill_reference or (lambda reference: reference)
        self.evidence_tools = dict(evidence_tools or {})
        if self.evidence_tools.keys() & (SCHEMAS.keys() | {'read_context', 'edit_source', 'snapshot'}):
            raise ValueError('evidence tool shadows a core action')
        self.schemas = dict(SCHEMAS, **{name: tool.schema for name, tool in self.evidence_tools.items()})
        self._usage = None
        self._reads = {}
        self._upstream_resolver = workflow.resolver
        workflow.resolver = self._resolve

    def _resolve(self, kind, reference, stage, operation):
        usage = self._usage
        if kind != 'skill':
            return self._upstream_resolver(kind, reference, stage, operation)
        if usage is None or usage.operation_id != operation or usage.stage != stage:
            return self._upstream_resolver(kind, reference, stage, operation)
        record = self._reads.get(reference)
        if record is None:
            return self._upstream_resolver(kind, reference, stage, operation)
        try:
            usage.resolve_read(record['receipt_id'], stage=stage, skill=record['skill'],
                               section=record['section'])
            return True
        except ValueError:
            return False

    def _read_reference(self, usage):
        receipt = usage.last_read_receipt
        reference = self.skill_reference(receipt['receipt_id'])
        if not isinstance(reference, str) or not reference.strip():
            raise WorkflowError('host skill reference mapping failed')
        self._reads[reference] = receipt
        return reference

    def reconcile_pending(self, evidence, result):
        """Replay trusted persisted results using the upstream receipt authority.

        Caller loads evidence/result from its host log, never from actor claims.
        The upstream resolver must verify persisted scoped skill reads (including
        mapped coverage), retrievals and retrospectives. No new reads occur here.
        """
        self._available()
        self.workflow.reconcile(evidence.operation_id, evidence, passed=result.passed,
            new_source_hash=result.new_source_hash, redesign=result.redesign)

    def _available(self):
        remaining = self.workflow.remaining()
        values = [v for v in remaining.values() if v is not None]
        seconds = min(values) if values else None
        if seconds is not None and seconds <= 0:
            raise WorkflowError('time budget exhausted')
        if self.workflow.state['stopped']:
            raise WorkflowError('workflow stopped')
        return seconds

    def _timed(self, category, callback):
        token = self.workflow.accounting.start(category) if category else None
        try:
            return callback()
        finally:
            if token:
                self.workflow.accounting.end(token)

    def _setup(self):
        if self.session is not None:
            return
        def prepare():
            self.session = self.session_factory()
            wall = self.workflow.remaining()['wall_seconds']
            self.session.prepare(min(self.setup_timeout, wall) if wall is not None else self.setup_timeout,
                                 schemas=self.schemas)
        try:
            self._timed('setup', prepare)
        except BaseException:
            self.close()
            raise

    def step(self, context):
        flow = self.workflow
        operation = flow.begin()
        stage = flow.stage
        usage = self._usage = SkillUsage(self.bundle, operation, stage,
            lambda event: self.emit('skill-usage', event), clock=flow.clock)
        names, refs, read_names = usage.expose(), [], set()
        self._reads = {}
        read_material, evidence_material = [], []
        scope = dict(operation_id=operation, stage=stage.value)
        record = lambda kind, data: self.emit(kind, dict(data, **scope))
        observation = None
        if stage == Stage.PREFLIGHT:
            # Preflight belongs to the host, including its actual skill accesses.
            for name in names:
                usage.read(name)
                refs.append(self._read_reference(usage))
            flow.dispatched(operation, 'host:'+operation)
            result = self._timed('setup', lambda: self.preflight(operation, context))
        else:
            self._setup()
            self._available()
            scope['handle'] = 'pi:'+operation+':'+uuid.uuid4().hex
            record('actor-dispatch', {'timestamp': flow.clock()})
            flow.dispatched(operation, scope['handle'])
            while True:
                seconds = self._available()
                source = self._timed('implementation', self.source_snapshot) if self.source_snapshot else None
                seconds = self._available()
                if source and source.project_hash != flow.source_hash:
                    raise WorkflowError('source changed outside current operation')
                payload = dict(task=context, stage=stage.value, operation_id=operation,
                    skills=list(names), read_skills=list(read_material), evidence=list(evidence_material),
                    observation=observation,
                    source={'path': source.path, 'text': source.text, 'sha256': source.sha256,
                            'project_hash': source.project_hash} if source else None)
                scope['request_id'] = uuid.uuid4().hex
                record('actor-request', {'timestamp': flow.clock()})
                name, args = self._timed('thinking', lambda: self.session.next_action(
                    payload, list(self.schemas), seconds=seconds, record=record, source=source))
                self._available()
                if name not in self.schemas:
                    raise WorkflowError('unknown actor action')
                validate(args, self.schemas[name])
                if name == 'read_skill':
                    text = self._timed('retrieval', lambda: usage.read(args['name'], args.get('section')))
                    receipt = usage.last_read_receipt
                    # A section read is genuine evidence, but does not replace the
                    # required full step contract. Return it without inventing reads.
                    reference = self._read_reference(usage)
                    refs.append(reference)
                    read_material.append(dict(skill=args['name'], section=args.get('section'),
                        content=text, receipt_id=reference, raw_receipt_id=receipt['receipt_id']))
                    if args.get('section') is None:
                        read_names.add(args['name'])
                    observation = dict(content=text, receipt_id=reference, raw_receipt_id=receipt['receipt_id'])
                    continue
                if name in self.evidence_tools:
                    seconds = self._available()
                    observation = self._timed('retrieval', lambda: self.evidence_tools[name].invoke(
                        stage, args, operation, seconds))
                    material = {'name': name, 'result': observation}
                    evidence_material.append(material)
                    record('evidence-tool-result', material)
                    continue
                if read_names != set(names):
                    raise WorkflowError('read every mapped skill before performing stage')
                if 'edit' in args:
                    edit = args['edit']
                    if (stage not in (Stage.IMPLEMENTATION, Stage.DEBUG, Stage.OPTIMISATION)
                            or source is None or edit['path'] != source.path
                            or edit['sha256'] != source.sha256):
                        raise WorkflowError('source edit is unavailable or stale')
                seconds = self._available()
                record('host-dispatch', {'timestamp': flow.clock()})
                result = self._timed(self.host_category,
                    lambda: self.perform(stage, args, operation, seconds))
                break
        if not isinstance(result, StageResult):
            raise TypeError('host must return StageResult')
        evidence = StepEvidence(stage=stage, operation_id=operation, source_hash=flow.source_hash,
            skill_refs=tuple(refs), artifact_refs=result.artifact_refs,
            retrieval_refs=result.retrieval_refs, retrospective_ref=result.retrospective_ref)
        record('stage-result', dict(result=asdict(result), evidence=evidence.to_dict()))
        self._available()
        flow.complete(evidence, passed=result.passed, new_source_hash=result.new_source_hash,
                      redesign=result.redesign)
        return result

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None
