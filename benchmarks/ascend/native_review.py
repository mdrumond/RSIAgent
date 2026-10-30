"""Pi-native sequential review reads, with host-owned admission and elapsed limits."""
import json
from collections import Counter

from coding.pi.context import encoded
from coding.pi.schema import obj, STRING, INTEGER, validate
from coding.pi.session import PiCodingSession, ROOT
from .review_context import ReviewContextRecords


def original_arguments(value, schema):
    """Validate the original JSON, never Pi's coerced replacement."""
    if 'anyOf' in schema:
        for alternative in schema['anyOf']:
            try:
                original_arguments(value, alternative)
                return
            except (ValueError, TypeError):
                pass
        raise ValueError('argument does not match its declared alternatives')
    if schema.get('type') == 'null':
        if value is not None:
            raise ValueError('expected null')
        return
    validate(value, schema)
    if isinstance(value, dict):
        for key, item in value.items():
            original_arguments(item, schema.get('properties', {}).get(key, {}))
    elif isinstance(value, list):
        if len(value) < schema.get('minItems', 0):
            raise ValueError('argument list is empty')
        for item in value:
            original_arguments(item, schema.get('items', {}))
    if isinstance(value, str) and schema.get('pattern'):
        import re
        if not re.fullmatch(schema['pattern'], value):
            raise ValueError('expected an exact bare SHA256 artifact identity')


class RecordedReviewContext(ReviewContextRecords):
    def __init__(self, record):
        super().__init__()
        self.record = record

    def admit(self, value):
        raw = value.encode() if isinstance(value, str) else encoded(value)
        import hashlib
        digest = hashlib.sha256(raw).hexdigest()
        fresh = digest not in self.records
        reference = super().admit(value)
        if fresh:
            self.record('pi-context-record', dict(**reference, text=raw.decode('utf-8')))
        return reference


class PiReviewSession(PiCodingSession):
    extension_file = ROOT / 'integrations/pi/verifier-tools.ts'

    def __init__(self, **options):
        super().__init__(**options)
        self.context = RecordedReviewContext(lambda kind, payload: self.record(kind, payload))
        self.calls, self.context_reads, self.failure = {}, Counter(), None
        self.state = None
        self.schemas['read_artifact'] = dict(self.schemas['read_artifact'], properties={
            **self.schemas['read_artifact']['properties'],
            'limit': {'type': 'integer', 'minimum': 1, 'maximum': 65536}})
        self.schemas['read_context'] = obj({'record': STRING, 'offset': INTEGER,
            'limit': {'type': 'integer', 'minimum': 1, 'maximum': 2048}}, ['record'])

    def _operation(self, name, args):
        if name == 'compaction':
            result = super()._operation(name, args)
            self.failure = self.compaction_failure
            return result
        with self.lock:
            if name == 'snapshot':
                return dict(allowed=self.state.allowed() + ['read_context'],
                    review_context=self.context.snapshot(self.view(), target=12000))
            if name == 'transcript':
                message = args['message']
                self.record('pi-transcript', dict(batch=args['batch'], message=message))
                key = (args['batch'], message.get('toolCallId'))
                if message.get('role') == 'toolResult' and key in self.calls and not self.calls[key]['used']:
                    self.calls[key]['used'] = True
                    self.record('pi-call-result', dict(batch=args['batch'], id=message['toolCallId'],
                        name=self.calls[key]['name'], disposition='sdk-feedback', isError=message.get('isError')))
                return {}
            if name == 'propose_batch':
                calls = args['calls']
                duplicates = Counter(call['id'] for call in calls)
                for call in calls:
                    error = None
                    try:
                        if args.get('truncated') or duplicates[call['id']] != 1:
                            raise ValueError('truncated or duplicate tool call identity')
                        if call['name'] not in self.state.allowed() + ['read_context']:
                            raise ValueError('tool is not allowed')
                        original_arguments(call['arguments'], self.schemas[call['name']])
                        if call['name'] in {'submit_review', 'request_check'} and len(calls) != 1:
                            raise ValueError('submit_review and request_check must be standalone assistant calls')
                    except (ValueError, TypeError, KeyError) as exc:
                        error = str(exc)
                    saved = dict(call, error=error, used=False)
                    self.calls[(args['batch'], call['id'])] = saved
                    self.record('pi-call-proposed', dict(batch=args['batch'], **call,
                        disposition='rejected' if error else 'admitted-for-execution', error=error))
                return {}
            if name != 'execute_call':
                raise ValueError('unknown native review operation')
            call = self.calls[(args['batch'], args['id'])]
            if call['used']:
                raise ValueError('tool call identity was already executed')
            call['used'] = True
            try:
                self.state.remaining()
                if self.failure is not None:
                    raise self.failure
                if call['error']:
                    observation = dict(error=call['error'], review_state=self.state.summary())
                elif call['name'] == 'read_context':
                    payload = call['arguments']
                    limit = min(2048, payload.get('limit', 2048))
                    observation = self.context.read(payload['record'], payload.get('offset', 0), limit)
                    while len(encoded(observation)) > 3500:
                        limit //= 2
                        observation = self.context.read(payload['record'], payload.get('offset', 0), limit)
                    key = (payload['record'], observation['offset'], len(observation['text'].encode()))
                    self.context_reads[key] += 1
                    self.record('pi-context-read', dict(record=payload['record'], offset=observation['offset'],
                        bytes=len(observation['text'].encode()), repeat=self.context_reads[key] > 1))
                else:
                    arguments = json.loads(encoded(call['arguments']))
                    value = arguments if call['name'] == 'submit_review' else {call['name']: arguments}
                    observation = self.state.execute(value)
                failed = 'error' in observation
                if call['name'] == 'knowledge_query' and not failed:
                    full = self.context.admit(self.state.raw_observation)
                    if len(encoded(observation)) <= 2800:
                        observation = dict(observation, full_observation=full)
                    else:
                        observation = dict(full_observation=full, ready_for_decision=self.state.summary()['ready_for_decision'])
                self.record('pi-call-result', dict(batch=args['batch'], id=args['id'], name=call['name'],
                    disposition='feedback' if failed else 'executed', terminal=self.state.decision is not None))
                if len(encoded(observation)) > 4096:
                    observation = dict(full_observation=self.context.admit(observation),
                        ready_for_decision=self.state.summary()['ready_for_decision'])
                return dict(observation=observation, isError=failed, terminate=self.state.decision is not None)
            except (ValueError, TypeError, KeyError) as error:
                self.record('pi-call-result', dict(batch=args['batch'], id=args['id'], name=call['name'],
                    disposition='feedback', error=str(error)))
                return dict(observation=dict(error=str(error), review_state=self.state.summary()), isError=True, terminate=False)
            except Exception as error:
                # Propagate typed provider/deadline failures on the host thread,
                # after Pi settles this tool batch; never turn them into a verdict.
                self.failure = error
                self.record('pi-call-result', dict(batch=args['batch'], id=args['id'], name=call['name'],
                    disposition='interrupted', error_type=type(error).__name__))
                return dict(observation=dict(error=type(error).__name__), isError=True, terminate=True)

    def view(self):
        return dict(host=dict(project=self.state.context, review_state=self.state.summary(),
            validation_feedback=self.state.feedback), remaining_seconds=self.state.remaining(), source_sha256=None)

    def review(self, state):
        self.state, state.native = state, True
        self.record = state.agent._record
        self._start()
        while state.decision is None:
            self.rpc.timeout = state.remaining()
            try:
                self.rpc.prompt('Review the project using the current host review_context and native tools. '
                                'Finish with a standalone justified submit_review when ready.')
            except Exception:
                state.remaining()
                if self.failure is not None:
                    raise self.failure
                raise
            if self.failure is not None:
                raise self.failure
            state.remaining()
            if state.decision is None:
                self.record('pi-review-continuation', state.summary())
        return state.decision
