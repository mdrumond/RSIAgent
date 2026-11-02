"""Admitted filesystem operations for native Pi tools; no tool or model loop."""
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile

from rsi.contracts import Stage
from rsi.python_execution import atomic_json, digest


class NativeFiles:
    def __init__(self, root, *, store, read_paths=(), backend=None, record=None):
        self.root, self.store, self.backend = Path(root), store, backend
        self.root.mkdir(parents=True, exist_ok=True)
        self.record = record or (lambda kind, payload: None)
        self.read_paths = {}
        for value in read_paths:
            path = Path(value).resolve(strict=True)
            if not path.is_file():
                raise ValueError('admitted read resource must be a file')
            self.read_paths[path] = digest(path.read_bytes())

    def session_options(self):
        return dict(tools_factory=str(Path(__file__).resolve().parents[2] / 'integrations/pi/native-files.mjs'),
            tools_options=dict(writable=self.backend is not None), callbacks={
                'rsi_fs_access': self.access, 'rsi_fs_read': self.read,
                'rsi_fs_write': self.write, 'rsi_fs_mkdir': self.mkdir})

    @staticmethod
    def _result(details):
        return dict(content=[], details=details)

    def _path(self, value, *, write=False):
        path = Path(value)
        if not path.is_absolute():
            raise PermissionError('native filesystem requires an absolute admitted path')
        resolved = path.resolve()
        if not write and resolved in self.read_paths:
            if digest(resolved.read_bytes()) != self.read_paths[resolved]:
                raise ValueError('admitted read resource changed')
            return resolved
        if self.backend is None or not resolved.is_relative_to(self.backend.project):
            raise PermissionError('path is outside admitted project and resources')
        checked = self.backend._file(path.relative_to(self.backend.project).as_posix())
        if write and checked.relative_to(self.backend.project).as_posix() not in self.backend.contract['editable_files']:
            raise PermissionError('path is not in the frozen editable manifest')
        return checked

    def _scope(self, context):
        left = context['remaining']() if context.get('remaining') else None
        if left is not None and left <= 0:
            raise PermissionError('native filesystem deadline exhausted')
        if not context.get('operation_id') or not context.get('tool_call_id'):
            raise ValueError('native operation and tool call identities required')

    def access(self, args, context):
        self._scope(context)
        if not self._path(args['path'], write=args.get('write', False)).is_file():
            raise ValueError('admitted file does not exist')
        return self._result({})

    def read(self, args, context):
        self._scope(context)
        path = self._path(args['path'])
        data = path.read_bytes()
        reference = self.store.put_json(dict(schema='rsi-native-file-read-v1',
            operation_id=context['operation_id'], tool_call_id=context['tool_call_id'],
            path=str(path), artifact_ref=self.store.put_bytes(data)))
        self.record('native-file-read', dict(reference=reference))
        return self._result(dict(data=base64.b64encode(data).decode(), reference=reference))

    def _mutation(self, context):
        self._scope(context)
        flow = self.backend.workflow if self.backend else None
        if (flow is None or flow.outcome or flow.state['stopped'] or not flow.pending
                or flow.pending['id'] != context['operation_id']
                or flow.stage not in (Stage.IMPLEMENTATION, Stage.DEBUG, Stage.OPTIMISATION)):
            raise PermissionError('pending mutation stage ownership required')

    def _settle(self, path, state, status):
        reference = self.store.put_json(dict(state['intent'], status=status))
        state['transitions'].append(reference)
        state['intent'] = None
        atomic_json(path, state)
        self.record('native-source-transition', dict(reference=reference))
        return reference

    def _state(self, operation_id):
        path = self.root / (hashlib.sha256(operation_id.encode()).hexdigest() + '.json')
        flow = self.backend.workflow
        if path.exists():
            state = json.loads(path.read_text())
            if state['operation_id'] != operation_id or state['base'] != flow.source_hash:
                raise ValueError('pending mutation differs from original workflow source')
        else:
            if self.backend.project_hash() != flow.source_hash:
                raise ValueError('unreceipted source change before native mutation')
            state = dict(schema='rsi-native-source-chain-v1', operation_id=operation_id,
                         base=flow.source_hash, head=flow.source_hash, transitions=[], intent=None)
        if state['intent']:
            current = self.backend.project_hash()
            if current == state['intent']['after']:
                state['head'], status = current, 'applied'
            elif current == state['intent']['before']:
                status = 'not_applied'
            else:
                raise ValueError('native mutation closure unknown; inspect retained intent')
            self._settle(path, state, status)
        if self.backend.project_hash() != state['head']:
            raise ValueError('source differs from authorized pending head')
        return path, state

    def write(self, args, context):
        self._mutation(context)
        target = self._path(args['path'], write=True)
        path, state = self._state(context['operation_id'])
        for reference in state['transitions']:
            if self.store.read_json(reference)['tool_call_id'] == context['tool_call_id']:
                raise ValueError('native tool write already attempted; inspect its retained receipt')
        data = args['content'].encode('utf-8')
        name = target.relative_to(self.backend.project).as_posix()
        after = self.backend.project_hash(replacements={name: data})
        staging = self.root/'staging'
        if staging.resolve().is_relative_to(self.backend.project.resolve()):
            raise PermissionError('native edit staging must be outside hashed project source')
        staging.mkdir(mode=0o700, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        if staging.stat().st_dev != target.parent.stat().st_dev:
            raise PermissionError('atomic native edit requires same-filesystem staging')
        state['intent'] = dict(schema='rsi-native-source-transition-v1', operation_id=context['operation_id'],
            tool_call_id=context['tool_call_id'], base=state['base'], before=state['head'], after=after,
            path=name, artifact_ref=self.store.put_bytes(data))
        atomic_json(path, state)  # Intent and content precede source mutation.
        descriptor, temporary = tempfile.mkstemp(dir=staging, prefix='native-edit-')
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            if target.exists():
                os.chmod(temporary, target.stat().st_mode & 0o777)
            os.replace(temporary, target)
            fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        if self.backend.project_hash() != after:
            raise ValueError('native mutation closure unknown; inspect retained intent')
        state['head'] = after
        reference = self._settle(path, state, 'applied')
        return self._result(dict(reference=reference, source_hash=after))

    def mkdir(self, args, context):
        self._mutation(context)
        requested = Path(args['path'])
        allowed = [self.backend._file(name).parent for name in self.backend.contract['editable_files']]
        if not requested.is_absolute() or not any(parent == requested for parent in allowed):
            raise PermissionError('directory is not an admitted editable parent')
        return self._result({})  # write() creates the checked parent after retaining intent.

    def pending_source(self, operation_id):
        """Retained final head for host Workflow.complete; never commit a stage here."""
        flow = self.backend.workflow
        if not flow.pending or flow.pending['id'] != operation_id:
            raise ValueError('pending source requires the original operation')
        _, state = self._state(operation_id)
        reference = self.store.put_json(dict(state, artifact_refs=state['transitions']))
        return dict(source_hash=state['head'], reference=reference)
