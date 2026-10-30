"""Pi-native completed tools, dispatched on the controller's main thread.

HTTP handlers only queue requests. The RPC pump owns host callbacks, original
argument validation, timing transitions, and durable records. Pi alone owns its
conversation, tool exchanges, and automatic compaction.
"""
from contextlib import ExitStack
from dataclasses import asdict
import json
import hashlib
import queue
import threading
import time

from .context import encoded
from .rpc import PiError, PiDeadlineError
from .schema import obj, STRING, validate
from .session import ActionError, PiCodingSession, ROOT


class ControllerSession(PiCodingSession):
    def __init__(self, *, wall_deadline=None, **kwargs):
        super().__init__(**kwargs)
        self.request_deadline_provenance = dict(session=self.deadline, gateway=self.gateway.deadline)
        self.enclosing_deadline = wall_deadline
        self._configure_controller()

    def _configure_controller(self):
        self.extension_file = ROOT / 'integrations/pi/rsi-controller.ts'
        self.settings_override = {'compaction': dict(enabled=True, reserveTokens=8192, keepRecentTokens=1024)}
        self.context_window = 16384
        self.requests = queue.Queue()
        self.thinking = None
        self.control = {}
        self.batches = {}
        self.record = lambda kind, data: self._enqueue(('record', kind, data))
        self.gateway.on_record = lambda kind,data:self.record(kind,data)

    @classmethod
    def from_existing_session(cls, session, *, wall_deadline=None):
        """Transfer one unstarted registered route; never reopen a live session.

        No provider client is constructed and no process starts here. Closing the
        old raw owner becomes a no-op; this controller exclusively owns cleanup.
        wall_deadline is an explicitly supplied monotonic parent-wall deadline;
        legacy request deadlines are never inferred to be wall constraints.
        """
        if not isinstance(session, PiCodingSession):
            raise TypeError('expected a registered PiCodingSession')
        if session.closed or session.rpc is not None or hasattr(session.gateway,'server'):
            raise ValueError('only unstarted open Pi sessions can be adopted')
        controller = cls.__new__(cls)
        controller.__dict__ = session.__dict__.copy()
        controller.schemas = dict(session.schemas)
        # Legacy A3 deadlines represent actor allocation, not a parent wall.
        # Preserve them as provenance only; explicit wall limits are separate.
        controller.request_deadline_provenance = dict(session=session.deadline,gateway=session.gateway.deadline)
        controller.enclosing_deadline = wall_deadline
        controller._configure_controller()
        session.closed = True
        session.stack = ExitStack()
        return controller

    def _enqueue(self, request):
        done, response = threading.Event(), {}
        self.requests.put((request, done, response))
        while not done.wait(.05):
            if self.closed:
                raise ValueError('controller transport closed; pending work requires reconciliation')
        if 'error' in response:
            raise ValueError(response['error'])
        return response.get('value')

    def _operation(self, name, args):
        return self._enqueue(('tool', name, args))

    def run_stage(self, dispatcher, turn, actor, delivery):
        current, completed = turn, []
        delivery.begin(turn[0]['operation_id'])
        record = turn[3]
        def remaining():
            seconds = actor._available()
            if self.enclosing_deadline is not None:
                inherited = self.enclosing_deadline-time.monotonic()
                if inherited <= 0:
                    raise PiDeadlineError('enclosing Pi deadline exhausted')
                seconds = inherited if seconds is None else min(seconds,inherited)
            return seconds
        def deadlines():
            seconds = remaining()
            self.deadline = time.monotonic() + (threading.TIMEOUT_MAX if seconds is None else seconds)
            self.gateway.deadline = self.deadline
        def pause():
            if self.thinking is not None:
                actor.workflow.accounting.end(self.thinking)
                self.thinking = None
        def resume():
            deadlines()
            self.thinking = actor.workflow.accounting.start('thinking')
        def status():
            payload, allowed, source, _ = current
            mutable = payload['stage'] in ('implementation', 'debug', 'optimisation')
            self.schemas['perform'] = actor.schemas['perform'] if mutable else obj({'proposal': STRING})
            self.allowed = set(allowed) | {'read_context'}
            if mutable and source is not None and 'perform' in self.allowed:
                self.allowed.add('edit_source')
            if completed:
                return dict(done=True, allowed=[], operation_id=payload['operation_id'])
            # State, not history: results stay in Pi's native tool conversation.
            return dict(done=False, allowed=sorted(self.allowed), stage=payload['stage'],
                operation_id=payload['operation_id'], skill_receipts=payload['skill_receipts'],
                unread_skills=payload['unread_skills'], read_skill_names=payload['read_skill_names'],
                source=resource(payload['source']), remaining_seconds=remaining())
        def resource(value):
            compact = self.context.compact(value, threshold=3072)
            if len(encoded(compact)) <= 6144:
                return compact
            # Page large raw results, retaining authoritative IDs/summaries.
            result = {'full_observation': self.context.admit(value)}
            if isinstance(value, dict):
                for key in ('reference', 'receipt_id', 'action', 'executed', 'status',
                            'path', 'sha256', 'project_hash', 'project_id', 'goal', 'contract'):
                    if key in value:
                        result[key] = value[key] if len(encoded(value[key])) <= 512 else self.context.admit(value[key])
                if 'summary' in value:
                    summary = value['summary']
                    result['summary'] = summary if len(encoded(summary)) <= 2048 else self.context.admit(summary)
            return result
        def execute(name, args):
            nonlocal current
            if name == 'read_context':
                try:
                    validate(args, self.schemas[name])
                    page = actor._timed('retrieval', lambda:self.context.read(
                        args['record'], args.get('offset',0), min(args.get('limit',3072),3072)))
                except ValueError as error:
                    return dict(observation=dict(error=str(error),executed=False),isError=True,state=status())
                record('pi-context-read', dict(record=page['record'], offset=page['offset'], bytes=len(page['text'].encode())))
                return dict(observation=page, terminate=False, state=status())
            if name not in self.allowed:
                return dict(observation=dict(error='action unavailable', executed=False), isError=True, state=status())
            try:
                validate(args, self.schemas[name])
            except ValueError as error:
                record('pi-action-rejected', dict(error=str(error), name=name))
                return dict(observation=dict(error=str(error), executed=False), isError=True, state=status())
            record('pi-action', dict(name=name, arguments=args, executed=False))
            if name == 'edit_source':
                source = current[2]
                if source is None or args['sha256'] != source.sha256 or not args['old_text'] or source.text.count(args['old_text']) != 1:
                    return dict(observation=dict(error='source edit is stale or ambiguous', executed=False),isError=True,state=status())
                candidate = source.text.replace(args['old_text'],args['new_text'],1)
                record('pi-edit-mapped', dict(source_sha256=source.sha256,
                    candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest()))
                name,args = 'perform', dict(proposal='Apply exact source edit',
                    edit=dict(path=source.path,sha256=source.sha256,source=candidate))
            try:
                current = dispatcher.send((name,args))
            except StopIteration as result:
                completed.append(result.value)
                return dict(observation=dict(host_completed=True, workflow_committed=False, result=asdict(result.value.result)), terminate=True, state={'allowed':[]})
            return dict(observation=resource(current[0]['observation']), terminate=False, state=status())
        def handle(name, args):
            if name == 'snapshot':
                return status()
            if name == 'propose_batch':
                calls = args['calls']
                ids = [call['id'] for call in calls]
                if len(set(ids)) != len(ids):
                    raise ValueError('duplicate model call identity')
                # Save original arguments before Pi's coercion; execute in model order.
                self.batches[args['batch']] = dict(calls=json.loads(json.dumps(calls)), next=0,
                    truncated=args.get('truncated',False))
                record('pi-batch', args)
                return {}
            if name == 'tool_result':
                batch = self.batches.get(args['batch'])
                if batch and batch['next'] < len(batch['calls']) and batch['calls'][batch['next']]['id'] == args['id']:
                    # Pi may reject malformed args before invoking our tool.
                    # This is feedback, never evidence of host execution.
                    batch['next'] += 1
                    record('pi-sdk-feedback', args)
                return {}
            if name != 'execute_call':
                raise ValueError('unknown native controller operation')
            batch = self.batches[args['batch']]
            index = batch['next']
            if index >= len(batch['calls']) or batch['calls'][index]['id'] != args['id']:
                raise ValueError('tool call is out of model order or already handled')
            batch['next'] += 1
            call = batch['calls'][index]
            if batch['truncated'] or completed:
                return dict(observation=dict(error='batch truncated or stage already completed', executed=False),
                            isError=True, terminate=True, state={'allowed':[]})
            return execute(call['name'],call['arguments'])
        def pump():
            try:
                request, done, response = self.requests.get_nowait()
            except queue.Empty:
                return
            try:
                kind, name, data = request
                if kind == 'record':
                    if name == 'pi-provider-request':
                        delivery.observe(data['request'],record)
                    record(name,data)
                    response['value'] = None
                else:
                    pause()
                    response['value'] = handle(name,data)
                    resume()
            except BaseException as error:
                response['error'] = str(error)
                raise
            finally:
                done.set()
        self.rpc.timeout = threading.TIMEOUT_MAX
        resume()
        try:
            prompt = dict(task=resource(turn[0]['task']), **status())
            self.rpc.prompt(json.dumps(prompt), pump=pump, remaining_seconds=remaining,
                on_event=lambda event:record('pi-event',dict(event=event)))
            while not self.gateway.finished.is_set() or not self.requests.empty():
                pump()
                remaining()
                self.gateway.finished.wait(timeout=.01)
            if not completed:
                raise ActionError('Pi settled without completing the host stage; pending operation retained')
            return completed[0]
        finally:
            pause()

    def close(self):
        # Wake blocked HTTP handlers before the inherited RPC/server cleanup.
        while not self.requests.empty():
            _, done, response = self.requests.get_nowait()
            response['error'] = 'controller closed; reconcile pending work'
            done.set()
        super().close()
