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

from coding.pi.context import ContextRecords, encoded
from coding.pi.schema import STRING, obj, validate
from coding.pi.session import ActionError, PiCodingSession
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
    'Read results appear in the next host snapshot as executed observations. '
    'Use current_skill_content for already-read skills; use read_context when '
    'content is paged. Do not repeat a completed read because its native receipt was queued. '
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
class _StageProposal:
    result: StageResult
    evidence: StepEvidence


@dataclass(frozen=True)
class EvidenceTool:
    """Host-owned PR6 tool; invoke(stage, args, operation, seconds) returns data."""
    schema: dict
    invoke: object
    stages: tuple[Stage, ...] | None = None

    def __post_init__(self):
        if self.stages is not None:
            object.__setattr__(self, 'stages', tuple(Stage(stage) for stage in self.stages))


class WorkflowContextRecords(ContextRecords):
    """Canonical projection: current completed reads are usable without paging.

    Histories retain the existing record/page mechanism. Only actual host-returned
    text is inlined, within fixed per-result and complete-snapshot byte bounds.
    """
    def snapshot(self, context, *, target=24576):
        result = super().snapshot(context, target=target)
        original, projected = context.get('host', {}), result.get('host', {})
        observation = original.get('observation')
        if isinstance(observation, dict) and observation.get('executed') is True:
            for key, limit in (('content', 4096), ('summary', 2048)):
                value = observation.get(key)
                if value is None or len(encoded(value)) > limit:
                    continue
                old = projected['observation'].get(key)
                projected['observation'][key] = value
                if len(encoded(result)) > target:
                    projected['observation'][key] = old
        latest = {}
        for row in original.get('read_skills', []):
            if row.get('section') is None:
                latest[row['skill']] = dict(receipt_id=row['receipt_id'], content=row['content'])
        visible = {}
        projected['current_skill_content'] = visible
        for name, row in reversed(list(latest.items())):
            if len(encoded(row)) > 4096:
                continue
            visible[name] = row
            if len(encoded(visible)) > 4096 or len(encoded(result)) > target:
                del visible[name]
        if len(encoded(result)) > target:
            projected.pop('current_skill_content', None)
        return result


class SkillDelivery:
    """Evidence that actual forwarded requests contained complete skill bytes."""
    def __init__(self):
        self.operation = None
        self.expected, self.delivered, self.pages = {}, set(), {}

    def begin(self, operation):
        if operation != self.operation:
            self.operation = operation
            self.expected, self.delivered, self.pages = {}, set(), {}

    def expect(self, reference, content):
        self.expected[reference] = content

    def observe(self, request, record):
        strings, pages = [], []
        def visit(value):
            if isinstance(value, str):
                strings.append(value)
                if value.startswith(('{', '[')):
                    try:
                        visit(json.loads(value))
                    except ValueError:
                        pass
            elif isinstance(value, list):
                for item in value:
                    visit(item)
            elif isinstance(value, dict):
                if {'record', 'offset', 'text'} <= value.keys():
                    pages.append(value)
                for item in value.values():
                    visit(item)
        visit(request.get('messages', request.get('input', [])))
        for reference, content in self.expected.items():
            if reference in self.delivered:
                continue
            raw, digest = content.encode(), sha256(content.encode()).hexdigest()
            spans = self.pages.setdefault(digest, [])
            for page in pages:
                offset, text = page['offset'], page['text']
                if page['record'] != digest or type(offset) is not int or offset < 0 or not isinstance(text, str):
                    continue
                fragment = text.encode()
                if raw[offset:offset+len(fragment)] == fragment:
                    spans.append((offset, offset+len(fragment)))
            boundary = 0
            for start, end in sorted(spans):
                if start > boundary:
                    break
                boundary = max(boundary, end)
            if any(content in text for text in strings) or boundary >= len(raw):
                self.delivered.add(reference)
                record('skill-delivered', dict(receipt_id=reference, content_hash=digest))

    def missing(self, references):
        return [dict(receipt_id=ref, content=self.expected[ref])
                for ref in dict.fromkeys(references) if ref not in self.delivered]


class SnapshotPiSessionAdapter:
    """Reuse Pi's schemas, loopback gateway, context paging and exact edits.

    Eager startup + get_state handshake exclude process readiness from charged
    next_action time. Setup has its own RPC timeout; model deadlines are assigned
    only afterward from the remaining charged allocation.
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('context_factory', WorkflowContextRecords)
        self.delivery = SkillDelivery()
        self.session = PiCodingSession(schemas=SCHEMAS, system_prompt=SYSTEM,
                                       first_action_only=True, **kwargs)
        self.session.gateway.request_projector = self._project_tools

    def _project_tools(self, request, **_kwargs):
        # Keep the provider-visible and host-validated perform schema identical.
        for tool in request.get('tools', []):
            function = tool.get('function', tool)
            if function.get('name') == 'perform':
                function['parameters'] = self.session.schemas['perform']
        return request

    def expect_read(self, reference, content):
        self.delivery.expect(reference, content)

    def missing_reads(self, references):
        return self.delivery.missing(references)

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
        self.delivery.begin(context.get('operation_id'))
        mutable = Stage(context['stage']) in (Stage.IMPLEMENTATION, Stage.DEBUG, Stage.OPTIMISATION)
        session.schemas['perform'] = SCHEMAS['perform'] if mutable else obj({'proposal': STRING})
        def observed(kind, data):
            if kind == 'pi-provider-request':
                self.delivery.observe(data['request'], record)
            record(kind, data)
        # threading's maximum is a technical RPC bound, not a workflow budget.
        session.deadline = time.monotonic() + (threading.TIMEOUT_MAX if seconds is None else seconds)
        session.gateway.deadline = session.deadline
        write = None
        if ('perform' in allowed and source is not None
                and Stage(context['stage']) in (Stage.IMPLEMENTATION, Stage.DEBUG, Stage.OPTIMISATION)):
            write = lambda text: ('perform', {'proposal': 'Apply exact source edit',
                'edit': {'path': source.path, 'sha256': source.sha256, 'source': text}})
        return session.next_action(context, allowed, record=observed,
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
    TEST/VALIDATION callbacks reserve one execution by default. Multi-command
    callbacks reserve additional command IDs with Workflow.reserve_execution.
    managed_executions=True delegates all command reservations to the callback;
    production bindings must use the guarded backend for every actual command.
    emit must durably retain events. Exceptions preserve pending operations for
    explicit host reconciliation; no receipt is reconstructed or invented.
    """
    def __init__(self, workflow, bundle, session_factory, *, perform, preflight,
                 emit, source_snapshot=None, host_category='implementation', setup_timeout=60,
                 evidence_tools=None, skill_reference=None, managed_executions=False, perform_ready=None, recover_invalid_actions=False):
        self.workflow, self.bundle = workflow, bundle
        self.session_factory, self.session = session_factory, None
        self.perform, self.preflight, self.emit = perform, preflight, emit
        self.source_snapshot, self.host_category = source_snapshot, host_category
        self.setup_timeout = setup_timeout
        self.managed_executions = managed_executions
        self.recover_invalid_actions = recover_invalid_actions
        self.perform_ready = perform_ready or (lambda stage, operation: True)
        self.skill_reference = skill_reference or (lambda reference: reference)
        self.evidence_tools = dict(evidence_tools or {})
        if self.evidence_tools.keys() & (SCHEMAS.keys() | {'read_context', 'edit_source', 'snapshot', 'propose_batch', 'execute_call'}):
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
        dispatcher = self._stage(context)
        try:
            turn = next(dispatcher)
        except StopIteration as done:
            return self._commit(done.value)
        if hasattr(self.session, 'run_stage'):
            return self._commit(self.session.run_stage(dispatcher, turn, self))
        while True:
            payload, allowed, source, record = turn
            try:
                action = self._timed('thinking', lambda: self.session.next_action(
                    payload, allowed, seconds=self._available(), record=record, source=source))
            except ActionError as error:
                try:
                    turn = dispatcher.throw(error)
                except StopIteration as done:
                    return self._commit(done.value)
                continue
            try:
                turn = dispatcher.send(action)
            except StopIteration as done:
                return self._commit(done.value)

    def _commit(self, proposal):
        self._available()
        result, evidence = proposal.result, proposal.evidence
        self.emit('stage-result', dict(result=asdict(result), evidence=evidence.to_dict(),
            stage=evidence.stage.value, operation_id=evidence.operation_id,
            settlement='host' if evidence.stage == Stage.PREFLIGHT else 'transport'))
        self.workflow.complete(evidence, passed=result.passed, new_source_hash=result.new_source_hash,
                               redesign=result.redesign)
        return result

    def _stage(self, context):
        flow = self.workflow
        operation = flow.begin()
        stage = flow.stage
        usage = self._usage = SkillUsage(self.bundle, operation, stage,
            lambda event: self.emit('skill-usage', event), clock=flow.clock)
        names, refs, read_names = usage.expose(), [], set()
        self._reads = {}
        read_material, evidence_material = [], []
        skill_inventory, evidence_inventory, evidence_summaries = {}, {}, {}
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
                allowed = ['read_skill'] + [name for name, tool in self.evidence_tools.items()
                    if tool.stages is None or stage in tool.stages]
                if read_names == set(names) and self._timed('retrieval',
                        lambda: self.perform_ready(stage, operation)):
                    allowed.append('perform')
                seconds = self._available()
                inventory = dict(skill_receipts=list(skill_inventory.values()),
                                 evidence_receipts=list(evidence_inventory.values()))
                if any(len(json.dumps(rows).encode()) > 3500 for rows in inventory.values()):
                    raise WorkflowError('receipt inventory exceeds visible context capacity')
                payload = dict(task=context, stage=stage.value, operation_id=operation,
                    skills=list(names), read_skill_names=[name for name in names if name in read_names],
                    unread_skills=[name for name in names if name not in read_names],
                    read_skills=list(read_material), evidence=list(evidence_material),
                    **inventory, evidence_summaries=dict(evidence_summaries),
                    observation=observation,
                    source={'path': source.path, 'text': source.text, 'sha256': source.sha256,
                            'project_hash': source.project_hash} if source else None)
                scope['request_id'] = uuid.uuid4().hex
                record('actor-request', {'timestamp': flow.clock()})
                try:
                    name, args = yield payload, allowed, source, record
                except ActionError as error:
                    if not self.recover_invalid_actions:
                        raise
                    self._available()
                    observation = dict(event='invalid-action', error=str(error)[:500],
                                       accepted=False, allowed=list(allowed))
                    record('actor-invalid-action', observation)
                    continue
                self._available()
                if name == 'perform' and read_names != set(names):
                    raise WorkflowError('read every mapped skill before performing stage')
                if name not in allowed:
                    raise WorkflowError('actor action unavailable in this stage or readiness state')
                validate(args, self.schemas[name])
                if name == 'read_skill':
                    # Optional empty sections from tool callers mean a full read.
                    # Check model arguments before emitting any usage receipt;
                    # storage, delivery and timing failures must still propagate.
                    args = dict(args)
                    if not args.get('section', '').strip():
                        args.pop('section', None)
                    error = None
                    if args['name'] not in names:
                        error = 'skill must be exposed before read'
                    elif ('section' in args and args['section'] not in
                          self.bundle.document(args['name']).sections):
                        error = 'unknown skill section'
                    if error is not None:
                        if not self.recover_invalid_actions:
                            raise ValueError(error)
                        observation = dict(event='invalid-action', action=name,
                                           error=error, accepted=False,
                                           allowed=list(allowed))
                        record('actor-invalid-action', observation)
                        continue
                    text = self._timed('retrieval', lambda: usage.read(args['name'], args.get('section')))
                    receipt = usage.last_read_receipt
                    # A section read is genuine evidence, but does not replace the
                    # required full step contract. Return it without inventing reads.
                    reference = self._read_reference(usage)
                    if hasattr(self.session, 'expect_read'):
                        self.session.expect_read(reference, text)
                    refs.append(reference)
                    read_material.append(dict(skill=args['name'], section=args.get('section'),
                        content=text, receipt_id=reference, raw_receipt_id=receipt['receipt_id']))
                    if (args.get('section') is None or args['name'] not in skill_inventory
                            or skill_inventory[args['name']]['section'] is not None):
                        skill_inventory[args['name']] = dict(skill=args['name'],
                            section=args.get('section'), receipt_id=reference)
                    if args.get('section') is None:
                        read_names.add(args['name'])
                    observation = dict(action='read_skill', executed=True, status='completed',
                        skill=args['name'], section=args.get('section'), content=text,
                        receipt_id=reference, raw_receipt_id=receipt['receipt_id'])
                    continue
                if name in self.evidence_tools:
                    if ('skill_refs' in args and
                            any(ref not in self._reads for ref in args['skill_refs'])):
                        error = 'skill_refs must contain current host read receipt IDs, not skill names'
                        if not self.recover_invalid_actions:
                            raise WorkflowError(error)
                        observation = dict(event='invalid-action', action=name, error=error,
                                           accepted=False, skill_receipts=inventory['skill_receipts'])
                        record('actor-invalid-action', observation)
                        continue
                    seconds = self._available()
                    observation = self._timed('retrieval', lambda: self.evidence_tools[name].invoke(
                        stage, args, operation, seconds))
                    observation = (dict(observation, action=name, executed=True) if isinstance(observation, dict)
                                   else dict(result=observation, action=name, executed=True))
                    material = {'name': name, 'result': observation}
                    evidence_material.append(material)
                    if isinstance(observation, dict) and isinstance(observation.get('reference'), str):
                        evidence_inventory[name] = dict(name=name, reference=observation['reference'])
                        if 'summary' in observation:
                            evidence_summaries[name] = observation['summary']
                    record('evidence-tool-result', material)
                    continue
                if hasattr(self.session, 'missing_reads'):
                    missing = self.session.missing_reads(refs)
                    if missing:
                        observation = dict(event='skill-delivery-required', accepted=False,
                            reads={row['receipt_id']:dict(content=row['content']) for row in missing})
                        record('skill-delivery-required', observation)
                        continue
                if 'edit' in args:
                    edit = args['edit']
                    if (stage not in (Stage.IMPLEMENTATION, Stage.DEBUG, Stage.OPTIMISATION)
                            or source is None or edit['path'] != source.path
                            or edit['sha256'] != source.sha256):
                        error = 'source edit is unavailable or stale'
                        if not self.recover_invalid_actions:
                            raise WorkflowError(error)
                        observation = dict(event='invalid-action', error=error, accepted=False, allowed=list(allowed))
                        record('actor-invalid-action', observation)
                        continue
                if stage == Stage.OPTIMISATION:
                    decisions = [row['result'].get('evidence', {}) for row in evidence_material
                                 if row['name'] == 'optimisation_decision']
                    if decisions and (decisions[-1].get('action') == 'edit') != ('edit' in args):
                        error = 'optimisation decision does not match edit; noop/profile must omit edit'
                        if not self.recover_invalid_actions:
                            raise WorkflowError(error)
                        observation = dict(event='invalid-action', error=error, accepted=False,
                                           optimisation_decision=decisions[-1])
                        record('actor-invalid-action', observation)
                        continue
                seconds = self._available()
                if stage in (Stage.TEST, Stage.VALIDATION):
                    if self.managed_executions:
                        cap = flow.limits.execution_tries
                        if cap is not None and flow.state['executions'] >= cap:
                            raise WorkflowError('execution cap exhausted')
                    elif not flow.reserve_execution(operation):
                        raise WorkflowError('stage execution already reserved; reconcile before dispatch')
                record('host-dispatch', {'timestamp': flow.clock()})
                result = self._timed(self.host_category,
                    lambda: self.perform(stage, args, operation, seconds))
                break
        if not isinstance(result, StageResult):
            raise TypeError('host must return StageResult')
        evidence = StepEvidence(stage=stage, operation_id=operation, source_hash=flow.source_hash,
            skill_refs=tuple(refs), artifact_refs=result.artifact_refs,
            retrieval_refs=result.retrieval_refs, retrospective_ref=result.retrospective_ref)
        record('stage-result-proposed', dict(result=asdict(result), evidence=evidence.to_dict()))
        self._available()
        return _StageProposal(result, evidence)

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None


class PiSessionAdapter(SnapshotPiSessionAdapter):
    """Canonical Pi-native controller; Pi owns conversation and compaction."""
    def __init__(self, **kwargs):
        from coding.pi.controller import ControllerSession
        self.delivery = SkillDelivery()
        self.session = ControllerSession(schemas=SCHEMAS, system_prompt=(
            'You execute the canonical RSI workflow through completed host tools. '
            'Read all mapped skills, query real evidence, and follow the authoritative stage. '
            'Tool results are actual host observations. Read paged resources with read_context. '
            'Only perform completes a stage; model assertions do not establish success. '
            'Pi manages conversation and compaction. Preserve receipt IDs and pending work.'), **kwargs)
        self.session.gateway.request_projector = self._project_tools

    def run_stage(self, dispatcher, turn, actor):
        return self.session.run_stage(dispatcher, turn, actor, self.delivery)
