"""Native filesystem view of a trusted single-source domain host."""
from pathlib import Path
from types import SimpleNamespace

from coding.pi.filesystem import NativeFiles
from rsi.domain_execution import DomainActivity
from rsi.python_execution import atomic_json, digest


class DomainFiles(NativeFiles):
    def __init__(self, root, *, runner, read_paths=(), record=None):
        self.runner = runner
        self.host = runner.backend
        names = tuple(runner.config.editable_files)
        if len(names) != 1:
            raise ValueError('single-source domain host required')
        self.name = names[0]
        project = Path(root)/'project'
        project.mkdir(parents=True, exist_ok=True)
        def file(name):
            if name != self.name:
                raise PermissionError('file is outside the frozen domain source')
            path = project/name
            if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink():
                raise PermissionError('domain source path is not admitted')
            return path
        def project_hash(*, replacements=None):
            if replacements is not None:
                if set(replacements) != {self.name}:
                    raise PermissionError('domain replacement differs from frozen source')
                return digest(replacements[self.name])
            return self.host.project_hash()
        backend = SimpleNamespace(project=project, workflow=runner.workflow,
            contract=dict(editable_files=names), _file=file, project_hash=project_hash)
        super().__init__(Path(root)/'transitions', store=runner.store, backend=backend,
                         read_paths=read_paths, record=record)
        self._refresh()

    def _refresh(self):
        content = self.host.read(self.name)['content']
        if digest(content.encode()) != self.host.project_hash():
            raise ValueError('domain source bytes differ from authoritative host identity')
        path = self.backend._file(self.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def read(self, args, context):
        self._refresh()
        return super().read(args, context)

    def write(self, args, context):
        self._mutation(context)
        target = self._path(args['path'], write=True)
        path, state = self._state(context['operation_id'])
        if any(self.store.read_json(ref)['tool_call_id'] == context['tool_call_id']
               for ref in state['transitions']):
            raise ValueError('domain tool write already attempted; inspect retained receipt')
        data = args['content'].encode()
        state['intent'] = dict(schema='rsi-native-source-transition-v1',
            operation_id=context['operation_id'], tool_call_id=context['tool_call_id'],
            base=state['base'], before=state['head'], after=digest(data),
            path=self.name, artifact_ref=self.store.put_bytes(data))
        atomic_json(path, state)
        stage, operation = self.runner.workflow.stage, context['operation_id']
        activity = DomainActivity(self.runner, operation)
        edit = dict(path=self.name, source=args['content'], sha256=state['head'])
        # Preserve domain-specific guards, including pre-edit comparison measurements.
        admission = getattr(self.host, 'admit_proposal', None)
        if admission:
            admission(stage, dict(proposal='Native source edit', edit=edit), operation)
        before_edit = getattr(self.host, 'before_edit', None)
        if before_edit:
            before_edit(stage, edit, operation, activity)
        activity.run('edit', 'implementation', lambda _: self.host.edit(
            self.name, args['content'], expected_file_hash=state['head'], expected_project_hash=state['head']))
        if self.host.project_hash() != state['intent']['after']:
            raise ValueError('native domain edit closure unknown; inspect original activity')
        self._refresh()
        state['head'] = self.host.project_hash()
        reference = self._settle(path, state, 'applied')
        return self._result(dict(reference=reference, source_hash=state['head']))
