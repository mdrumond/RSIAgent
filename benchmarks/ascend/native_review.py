"""Native SDK review tools with host-owned evidence and decision admission."""
import json
from pathlib import Path
import tempfile
from uuid import uuid4

from coding.pi.filesystem import NativeFiles
from coding.pi.native import NativeSession, NativeTool
from coding.pi.session import ActionError
from core.artifacts import ContentAddressedArtifacts


class PiReviewSession:
    def __init__(self, *, model, schemas, environ=None, system_prompt='', root=None, **options):
        from benchmarks.a3_experiments import BackendModel
        from benchmarks.a3_model_profiles import load_a3_model_profile
        self.profile = load_a3_model_profile(BackendModel(model)).profile_id
        self.schemas, self.environ, self.system_prompt = schemas, environ, system_prompt
        self.root = (Path(root)/uuid4().hex if root else Path(tempfile.mkdtemp(prefix='rsi-native-review-')))
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = ContentAddressedArtifacts(self.root/'artifacts')
        self.session = None
        self.paths = {}

    def _retain(self, value):
        data = json.dumps(value, ensure_ascii=False, indent=2).encode()
        reference = self.store.put_bytes(data)
        path = self.root/'context'/(reference+'.json')
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        self.files.read_paths[path.resolve()] = reference
        return dict(path=str(path.resolve()), artifact_ref=reference)

    def _materialize(self, args, context):
        path = Path(args['path']).resolve()
        if path in self.paths and not path.exists():
            reference = self.paths[path]
            data = self.state.agent.read_artifact(reference, remaining_seconds=self.state.remaining())
            if not isinstance(data, bytes) or self.store.put_bytes(data) != reference:
                raise ValueError('admitted review artifact bytes differ')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self.files.read_paths[path] = reference

    def _paths(self):
        for reference in self.state.known:
            self.paths[(self.root/'evidence'/(reference+'.txt')).resolve()] = reference
        return [dict(sha256=reference, path=str(path)) for path, reference in self.paths.items()]

    def _read_observed(self, args, context):
        path = Path(args['path']).resolve()
        reference = self.paths.get(path)
        if reference is not None:
            data = self.store.read_bytes(reference)
            lines = data.decode('utf-8').splitlines(keepends=True)
            start = max(0, args.get('offset', 1)-1)
            truncation = (args.get('details') or {}).get('truncation', {})
            count = (truncation['outputLines'] if truncation.get('truncated') else
                     args.get('limit', len(lines)))
            if not truncation.get('firstLineExceedsLimit'):
                first = len(''.join(lines[:start]).encode())
                last = len(''.join(lines[:start+count]).encode())
                self.state.read.add(reference)
                self.state.observed.setdefault(reference, []).append((first, last))
                self.state.sizes[reference] = len(data)
                self.state.cache[reference] = data
                self.state.agent._record('native-review-read', dict(sha256=reference,
                    tool_call_id=context['tool_call_id'], start_line=start+1, lines=count,
                    truncation=truncation, artifact_ref=self.store.put_json(args)))
        return dict(content=[], details={})

    def _tool(self, name):
        def invoke(args, context):
            if name not in self.state.allowed():
                raise ValueError('review tool is not currently admitted')
            value = args if name == 'submit_review' else {name:args}
            observation = self.state.execute(value)
            retained = self._retain(observation)
            summary = dict(observation=observation if len(json.dumps(observation)) <= 8192 else retained,
                           review_state=self.state.summary(), artifact_files=self._paths())
            self.state.agent._record('native-review-tool', dict(name=name,
                tool_call_id=context['tool_call_id'], reference=retained['artifact_ref']))
            return dict(content=[dict(type='text', text=json.dumps(summary))],
                        isError='error' in observation, terminate=self.state.decision is not None)
        return invoke

    def review(self, state):
        self.state, state.native = state, True
        self.files = NativeFiles(self.root/'files', store=self.store,
            record=lambda kind, payload: state.agent._record(kind, payload))
        options = self.files.session_options()
        for name in ('rsi_fs_access', 'rsi_fs_read'):
            callback = options['callbacks'][name]
            def materialized(args, context, callback=callback):
                self._materialize(args, context)
                return callback(args, context)
            options['callbacks'][name] = materialized
        options['callbacks']['rsi_review_read_observed'] = self._read_observed
        options['tools_factory'] = str(Path(__file__).resolve().parents[2]/'integrations/pi/review-tools.mjs')
        tools = [NativeTool(name, 'Host-admitted review '+name, schema, self._tool(name))
                 for name, schema in self.schemas.items() if name != 'read_artifact']
        context = self._retain(dict(project=state.context, review_state=state.summary(),
                                    artifact_files=self._paths(), feedback=state.feedback))
        self.session = NativeSession(self.root/'session', tools=tools, profile_id=self.profile,
            system_prompt=self.system_prompt, environ=self.environ,
            observer=lambda event:state.agent._record('pi-native-event', event), **options)
        self.session.prepare()
        self.session.prompt(json.dumps(dict(context=context,
            instruction='Read the retained context and admitted artifact files with native read, then use submit_review.')),
            operation_id='review-'+self.root.name, remaining=state.remaining)
        if state.decision is None:
            raise ActionError('Native review settled without a host-admitted decision')
        return state.decision

    def close(self):
        if self.session:
            self.session.close()
            refs = [self.store.put_bytes(path.read_bytes()) for path in self.session.root.rglob('*.jsonl')]
            self.state.agent._record('native-review-session', dict(artifact_refs=refs, store=str(self.store.root)))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
