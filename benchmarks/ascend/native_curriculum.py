"""Typed native role proposals for existing subject curriculum admission."""
import hashlib
import json
from pathlib import Path
import time

from coding.pi.filesystem import NativeFiles
from coding.pi.native import NativeSession, NativeTool, NativeError
from coding.pi.session import ActionError
from core.artifacts import ContentAddressedArtifacts, canonical_bytes
from rsi.domain_timing import excluded_phase
from rsi.python_execution import atomic_json


class NativeSubjectRole:
    def __init__(self, root, profile, environ, *, validator=None, artifacts=None, setup_scope=None):
        self.root, self.profile, self.environ = Path(root), profile, environ
        self.model = profile.identity.backend_model.value
        self.validator, self.artifacts, self.setup_scope = validator, dict(artifacts or {}), setup_scope
        self.last_receipt = None

    def __call__(self, messages, *, remaining_seconds):
        return self._call(messages, remaining_seconds, wave=False)

    def wave(self, messages, *, remaining_seconds):
        return self._call(messages, remaining_seconds, wave=True)

    def _call(self, messages, seconds, *, wave):
        if seconds <= 0:
            raise TimeoutError('native curriculum allocation exhausted before dispatch')
        identity = hashlib.sha256(canonical_bytes(dict(messages=messages, wave=wave,
            model=self.model))).hexdigest()
        root = self.root/identity
        root.mkdir(parents=True, exist_ok=True)
        state_path = root/'operation.json'
        store = ContentAddressedArtifacts(root/'artifacts')
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['status'] == 'completed':
                self.last_receipt = store.read_json(state['result_ref'])
                return self.last_receipt['response']
            raise NativeError('retained native curriculum operation requires observation; never replay')
        state = dict(status='pending', operation_id=identity, input_ref=store.put_json(messages))
        atomic_json(state_path, state)
        context = root/'context.json'
        context.write_text(json.dumps(messages, ensure_ascii=False, indent=2))
        files = NativeFiles(root/'files', store=store, read_paths=[context])
        files.read_paths.update({Path(path).resolve():reference for path, reference in self.artifacts.items()})
        chosen = []
        def submit(args, scope):
            if chosen:
                raise ValueError('one role proposal already retained')
            if wave:
                decision, payload = args['decision'], args['payload']
                if decision in ('knowledge_query', 'read_artifact'):
                    response = json.dumps({decision:payload})
                else:
                    response = 'DECISION: '+decision+'\n'+json.dumps(payload)
            else:
                response = json.dumps(args['value'])
            if self.validator:
                self.validator(json.loads(response))
            chosen.append(response)
            return dict(content=[dict(type='text', text='Typed proposal retained; domain admission remains host-owned.')], terminate=True)
        shape = (dict(type='object', properties=dict(
                    decision=dict(type='string', enum=['WAVE', 'SATURATED', 'STALLED', 'knowledge_query', 'read_artifact']),
                    payload=dict(type='object', additionalProperties=True)), required=['decision', 'payload'], additionalProperties=False)
                 if wave else dict(type='object', properties=dict(value=dict(anyOf=[
                     dict(type='object', additionalProperties=True), dict(type='array', items={})])),
                     required=['value'], additionalProperties=False))
        tool = NativeTool('submit_curriculum_proposal', 'Submit the typed domain proposal; host admission is authoritative.', shape, submit)
        session = NativeSession(root/'session', tools=[tool], environ=self.environ,
            profile_id=self.profile.profile_id, **files.session_options(),
            system_prompt='Read the retained role context with native read. Use the typed proposal tool. '
                'The model does not supply prose wire headers. Domain constraints and actual retained evidence remain authoritative.')
        actor_start = None
        try:
            with self.setup_scope() if self.setup_scope else excluded_phase('setup'):
                session.prepare()
            actor_start = time.monotonic()
            deadline = actor_start+seconds
            result = session.prompt(json.dumps(dict(context_path=str(context.resolve()),
                proposal_kind='wave or read-only retrieval request' if wave else 'domain JSON proposal')),
                operation_id=identity, remaining=lambda: deadline-time.monotonic())
            if not chosen or not result.provider_settled or result.pending_tools:
                raise ActionError('native curriculum settled without a known typed proposal')
            self.last_receipt = dict(response=chosen[0], actor_seconds=time.monotonic()-actor_start, usage=result.usage)
            state.update(status='completed', result_ref=store.put_json(self.last_receipt))
            atomic_json(state_path, state)
            return chosen[0]
        finally:
            session.close()
            references = [store.put_bytes(path.read_bytes()) for path in session.root.rglob('*.jsonl')]
            atomic_json(root/'closure.json', dict(provider_settled=session.provider_settled,
                pending_tools=sorted(session.pending_tools), process_exit=session.process.poll() if session.process else None,
                actor_seconds=time.monotonic()-actor_start if actor_start is not None else None, artifact_refs=references))
