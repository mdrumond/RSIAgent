"""Native SDK tools driving the existing host Workflow dispatcher."""
from dataclasses import asdict
import json
from pathlib import Path
import time

from .filesystem import NativeFiles
from .native import NativeSession, NativeTool, NativeError, model_profile
from .session import ActionError
from rsi.python_execution import digest


class NativeWorkflowSession:
    def __init__(self, *, model='deepseek-flash', environ=None, wall_deadline=None, **options):
        self.model, self.environ, self.enclosing_deadline = model, environ, wall_deadline
        self.options = options
        self.session = self.files = self.runner = None
        self.current = self.dispatcher = self.actor = self.thinking = None
        self.completed = []
        self.skill_paths = {}
        self.schemas = {}
        self.archived = False

    def configure(self, runner):
        if self.session is not None:
            raise ValueError('native workflow session is already configured')
        self.runner = runner
        root = runner.root / 'native'
        resources = root / 'skills'
        for document in runner.bundle.documents:
            path = resources/document.name/'SKILL.md'
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_text() != document.content:
                raise ValueError('frozen native skill changed')
            path.write_text(document.content)
            self.skill_paths[path.resolve()] = document.name
        if hasattr(runner.backend, 'project'):
            self.files = NativeFiles(root/'source-transitions', store=runner.store,
                read_paths=self.skill_paths, backend=runner.backend, record=self._file_event)
        else:
            from rsi.native_source import DomainFiles
            self.files = DomainFiles(root/'source-transitions', runner=runner,
                                     read_paths=self.skill_paths, record=self._file_event)

    def _file_event(self, kind, payload):
        self.runner.actor_event(kind, dict(payload, operation_id=self.runner.workflow.pending['id']))
        receipt = self.runner.store.read_json(payload['reference'])
        if kind == 'native-file-read':
            name = self.skill_paths.get(Path(receipt['path']))
            if name and self.current is not None and name in self.current[0]['skills']:
                self._dispatch('read_skill', dict(name=name))
        elif receipt.get('status') == 'applied':
            self.runner.append(dict(type='source-edit', operation_id=receipt['operation_id'],
                before_source=receipt['before'], after_source=receipt['after'],
                path=receipt['path'], source_ref=receipt['artifact_ref']))

    def _remaining(self):
        bounds = [value for value in self.runner.workflow.remaining().values() if value is not None]
        remaining = min(bounds) if bounds else None
        if self.runner.workflow.state['stopped'] and not self.session.cancelled.is_set():
            self.session.abort()
        if self.enclosing_deadline is not None:
            wall = self.enclosing_deadline-time.monotonic()
            remaining = wall if remaining is None else min(remaining, wall)
        return remaining

    def _pause(self):
        if self.thinking is not None:
            self.runner.workflow.accounting.end(self.thinking)
            self.thinking = None

    def _resume(self):
        if self.actor is not None:
            self.thinking = self.runner.workflow.accounting.start('thinking')

    def _observe(self, event):
        self.runner.actor_event('pi-native-event', dict(event=event))
        if event['type'] == 'host_callback_start':
            self._pause()
        elif event['type'] == 'host_callback_end':
            self._resume()

    def _retain_text(self, value, *, suffix='.json'):
        data = value.encode() if isinstance(value, str) else json.dumps(value, ensure_ascii=False).encode()
        reference = self.runner.store.put_bytes(data)
        path = self.runner.root/'native'/'evidence'/(reference+suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        self.files.read_paths[path.resolve()] = reference
        return dict(path=str(path.resolve()), bytes=len(data), artifact_ref=reference)

    def _dispatch(self, name, args):
        if self.completed:
            raise ValueError('this host stage already completed')
        if name not in self.current[1]:
            raise ValueError('tool is unavailable in the current host stage')
        try:
            self.current = self.dispatcher.send((name, args))
        except StopIteration as result:
            self.completed.append(result.value)
            return dict(host_stage_completed=True, result=asdict(result.value.result))
        return self.current[0]['observation']

    def _state(self):
        if self.current is None:
            raise ValueError('no native workflow stage is active')
        payload = self.current[0]
        return dict(stage=payload['stage'], operation_id=payload['operation_id'],
                    done=bool(self.completed), allowed=[] if self.completed else list(self.current[1]),
                    skill_receipts=payload['skill_receipts'], evidence_receipts=payload['evidence_receipts'])

    def _state_tool(self, args, context):
        return dict(content=[dict(type='text', text=json.dumps(dict(workflow_state=self._state())))])

    def _tool(self, name):
        def invoke(args, context):
            if context['operation_id'] != self.runner.workflow.pending['id']:
                raise ValueError('native tool differs from pending workflow operation')
            observation = self._dispatch(name, args)
            text = json.dumps(observation, ensure_ascii=False)
            if len(text.encode()) > 8192:
                observation = dict(retained_observation=self._retain_text(text),
                                   stage=self.runner.workflow.stage.value)
            return dict(content=[dict(type='text', text=json.dumps(dict(observation=observation, workflow_state=self._state())))],
                        terminate=bool(self.completed))
        return invoke

    def prepare(self, timeout, *, schemas):
        if self.runner is None:
            raise ValueError('native workflow session needs its trusted runner binding')
        self.schemas = dict(schemas)
        self.schemas['perform'] = dict(type='object', properties=dict(proposal=dict(type='string')),
                                      required=['proposal'], additionalProperties=False)
        identities = ('deepseek-flash-native-v1', 'openai-gpt-5.6-sol-v1')
        profile = next((identity for identity in identities
            if self.model in (identity, model_profile(identity)['model'],
                model_profile(identity)['provider']+'/'+model_profile(identity)['model'])), None)
        if profile is None:
            raise ValueError('registered native programming profile required')
        tools = [NativeTool(name, 'Execute the host '+name+' operation.', schema, self._tool(name))
                 for name, schema in self.schemas.items() if name not in ('read_skill', 'read_project_file')]
        options = self.files.session_options()
        def charged(name, callback):
            def invoke(args, context):
                category = 'implementation' if name in ('rsi_fs_write', 'rsi_fs_mkdir') else 'retrieval'
                return self.actor._timed(category, lambda: callback(args, context))
            return invoke
        options['callbacks'] = {name: charged(name, callback)
                                for name, callback in options['callbacks'].items()}
        options['callbacks']['rsi_workflow_state'] = self._state_tool
        options['tools_factory'] = str(Path(__file__).resolve().parents[2]/'integrations/pi/workflow-tools.mjs')
        options['tools_options']['skills'] = [dict(name=name, description='Frozen workflow guidance: '+name,
            filePath=str(path), baseDir=str(path.parent), disableModelInvocation=False,
            sourceInfo=dict(path=str(path), source='rsi-frozen-bundle', scope='project', origin='top-level'))
            for path, name in self.skill_paths.items()]
        self.session = NativeSession(self.runner.root/'native'/'session', cwd=self.files.backend.project,
            profile_id=profile, tools=tools, environ=self.environ, observer=self._observe,
            system_prompt='Use native read/edit/write and the host workflow tools. Read mapped skill files. '
                'Design and optimisation require actual KDB and retrospective evidence. '
                'Only a successful host perform completes a stage; ordinary tool calls continue normally. '
                'Native edits are retained while the current stage is pending. No arbitrary shell is available.',
            **options)
        self.session.prepare(timeout)

    def pending_source(self, operation_id):
        return self.files.pending_source(operation_id)

    def run_stage(self, dispatcher, turn, actor):
        self.current, self.dispatcher, self.actor, self.completed = turn, dispatcher, actor, []
        operation = turn[0]['operation_id']
        task = self._retain_text(turn[0]['task'])
        prompt = dict(task=task, stage=turn[0]['stage'], operation_id=operation,
            user_guidance=turn[0]['task'].get('user_guidance', {}),
            mapped_skills=[dict(name=name, path=str(path)) for path, name in self.skill_paths.items()
                           if name in turn[0]['skills']],
            editable_files=[str(self.files.backend.project/name) for name in self.runner.config.editable_files])
        self._resume()
        try:
            self.session.prompt(json.dumps(prompt), operation_id=operation, remaining=self._remaining)
            if not self.completed:
                raise ActionError('Native Pi settled without completing the host stage')
            return self.completed[0]
        finally:
            self._pause()
            self.current = self.dispatcher = self.actor = None

    @property
    def completion_tokens(self):
        return self.session.completion_tokens if self.session else 0

    def close(self):
        self._pause()
        if self.session:
            self.session.close()
            if not self.archived:
                paths = [self.session.root/'native-events.jsonl', self.session.root/'native-worker.stderr',
                         *sorted((self.session.root/'sessions').glob('*.jsonl'))]
                references = [self.runner.store.put_bytes(path.read_bytes()) for path in paths if path.is_file()]
                self.runner.append(dict(type='native-session-artifacts', artifact_refs=references,
                    provider_settled=self.session.provider_settled,
                    pending_tools=sorted(self.session.pending_tools)))
                self.archived = True
