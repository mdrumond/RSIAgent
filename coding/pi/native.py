"""Thin synchronous host boundary to the installed Pi SDK; Pi owns the agent loop."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from uuid import uuid4


class NativeError(RuntimeError):
    def __init__(self, message, *, settlement=None):
        super().__init__(message)
        self.settlement = settlement


class ProviderFailure(NativeError):
    pass


class DeadlineExceeded(NativeError, TimeoutError):
    pass


class Cancelled(NativeError):
    pass


@dataclass(frozen=True)
class NativeTool:
    name: str
    description: str
    parameters: dict
    callback: object


@dataclass(frozen=True)
class NativeSettlement:
    status: str
    provider_settled: bool
    pending_tools: tuple[str, ...]
    session_file: str | None
    usage: tuple[dict, ...]


def model_profile(identity):
    from benchmarks.a3_experiments import BackendModel
    from benchmarks.a3_model_profiles import load_a3_model_profile
    for model in (BackendModel.DEEPSEEK_FLASH, BackendModel.GPT_5_6_SOL):
        profile = load_a3_model_profile(model)
        if profile.profile_id == identity:
            return dict(provider=profile.provider.lower(), model=profile.model,
                base_url=profile.base_url, credential_env=profile.credential_env,
                max_tokens=profile.generation.max_tokens, reasoning=profile.generation.thinking)
    raise ValueError('unregistered native model profile')


def runtime_preflight(node_modules=None, node_executable=None):
    root = Path(__file__).resolve().parents[2]
    modules = Path(node_modules or os.environ.get('RSI_PI_NODE_MODULES', root/'integrations/pi/node_modules')).resolve()
    manifest = modules/'@earendil-works/pi-coding-agent/package.json'
    node = node_executable or os.environ.get('RSI_PI_NODE') or shutil.which('node')
    if not node or not manifest.is_file() or json.loads(manifest.read_text())['version'] != '1.3.0':
        raise NativeError('Node and installed Pi SDK 1.3.0 are required')
    version = subprocess.check_output([node, '--version'], text=True).strip()
    if int(version.lstrip('v').split('.')[0]) < 22:
        raise NativeError('Pi SDK requires Node >=22')
    return node, modules


class NativeSession:
    def __init__(self, root, *, tools=(), profile_id='deepseek-flash-native-v1',
                 system_prompt='', node_modules=None, tools_factory=None,
                 resource_options=None, tools_options=None, cwd=None,
                 callbacks=None, observer=None, environ=None, node_executable=None,
                 model_metadata=None):
        self.root = Path(root).resolve()
        self.cwd = Path(cwd or root).resolve()
        self.tools, self.profile = tuple(tools), model_profile(profile_id)
        self.callbacks = dict(callbacks or {})
        for tool in self.tools:
            if tool.name in self.callbacks:
                raise ValueError('duplicate native callback')
            self.callbacks[tool.name] = tool.callback
        self.node_modules, self.tools_factory = node_modules, tools_factory
        self.system_prompt, self.resource_options = system_prompt, resource_options or {}
        self.tools_options = tools_options or {}
        self.node_executable, self.model_metadata = node_executable, model_metadata
        self.environment = os.environ if environ is None else environ
        self.observer = observer or (lambda event: None)
        self.events, self.usage = queue.Queue(), []
        self.process = self.reader = None
        self.session_file, self.operation_id = None, None
        self.pending_tools = set()
        self.provider_settled, self.running, self.closed = True, False, False
        self.cancelled = threading.Event()
        self.write_lock = threading.Lock()

    @property
    def completion_tokens(self):
        return None if any('output' not in usage for usage in self.usage) else sum(u['output'] for u in self.usage)

    def _send(self, value):
        with self.write_lock:
            try:
                self.process.stdin.write(json.dumps(value, allow_nan=False)+'\n')
                self.process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as error:
                raise NativeError('native transport closed', settlement=self._settlement('unknown')) from error

    def _record(self, event):
        secret = self.environment.get(self.profile['credential_env'], '')
        raw = json.dumps(dict(event, operation_id=self.operation_id), ensure_ascii=False, allow_nan=False)
        if secret:
            raw = raw.replace(secret, '[redacted]')
        with (self.root/'native-events.jsonl').open('a') as stream:
            stream.write(raw+'\n'); stream.flush(); os.fsync(stream.fileno())
        self.observer(json.loads(raw))

    def prepare(self, timeout=30.):
        if self.closed:
            raise NativeError('native session is closed')
        if self.process:
            return self
        node, modules = runtime_preflight(self.node_modules, self.node_executable)
        secret = self.environment.get(self.profile['credential_env'])
        if not secret:
            raise NativeError('selected native provider credential is missing')
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root/'native-events.jsonl').touch(mode=0o600, exist_ok=True)
        worker = Path(__file__).resolve().parents[2]/'integrations/pi/native-session.mjs'
        env = dict(PATH=os.environ.get('PATH', os.defpath), RSI_NATIVE_API_KEY=secret)
        stderr = (self.root/'native-worker.stderr').open('w')
        try:
            self.process = subprocess.Popen([node, str(worker)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=stderr, text=True, env=env)
        finally:
            stderr.close()
        def read():
            try:
                for line in self.process.stdout:
                    self.events.put(json.loads(line))
            except (ValueError, OSError) as error:
                self.events.put(dict(type='protocol_error', message=type(error).__name__))
            finally:
                self.events.put(dict(type='eof'))
        self.reader = threading.Thread(target=read, daemon=True); self.reader.start()
        self._send(dict(type='start', modules=str(modules), root=str(self.root), cwd=str(self.cwd), profile=self.profile,
            system_prompt=self.system_prompt, resource_options=self.resource_options,
            tools_options=self.tools_options, model_metadata=self.model_metadata,
            tools_factory=str(Path(self.tools_factory).resolve()) if self.tools_factory else None,
            tools=[dict(name=t.name, description=t.description, parameters=t.parameters) for t in self.tools]))
        end = time.monotonic()+timeout
        try:
            self._wait('ready', lambda: end-time.monotonic())
        except BaseException:
            self.close()
            raise
        return self

    def _settlement(self, status):
        return NativeSettlement(status, self.provider_settled, tuple(sorted(self.pending_tools)),
                                self.session_file, tuple(self.usage))

    def _wait(self, terminal, remaining):
        expired, grace = False, None
        while True:
            left = remaining() if remaining else None
            if left is not None and left <= 0 and not expired:
                expired, grace = True, time.monotonic()+3
                self.abort()
            if expired and time.monotonic() >= grace:
                raise DeadlineExceeded('native deadline exhausted; closure unknown', settlement=self._settlement('unknown'))
            try:
                event = self.events.get(timeout=.02)
            except queue.Empty:
                continue
            kind = event['type']
            if kind in ('eof', 'protocol_error', 'fatal'):
                raise NativeError(event.get('message', 'native worker exited'), settlement=self._settlement('unknown'))
            if kind == 'tool':
                identity, name = event['tool_call_id'], event['name']
                self.pending_tools.add(identity)
                self._record(dict(type='host_callback_start', tool_call_id=identity, name=name))
                try:
                    if expired or self.cancelled.is_set():
                        raise ValueError('host action not admitted after cancellation/deadline')
                    context = dict(tool_call_id=identity, operation_id=self.operation_id, remaining=remaining)
                    result = self.callbacks[name](event['arguments'], context)
                except Exception as error:
                    origin = 'domain' if isinstance(error, (ValueError, PermissionError)) else 'host'
                    self._record(dict(type='host_callback_error', tool_call_id=identity, origin=origin,
                                      error_type=type(error).__name__, message=str(error)))
                    result = dict(content=[dict(type='text', text=str(error))], isError=True)
                self._send(dict(type='tool_result', id=event['id'], result=result))
                self.pending_tools.discard(identity)
                self._record(dict(type='host_callback_end', tool_call_id=identity, name=name))
                continue
            self._record(event)
            if kind == 'agent_settled':
                self.provider_settled = True
            if kind == 'message_end' and event.get('message', {}).get('role') == 'assistant':
                usage = event['message'].get('usage')
                self.usage.append(usage if usage is not None else {})
            if kind == terminal:
                self.session_file = event.get('session_file', self.session_file)
                result = self._settlement(event.get('status', 'settled'))
                if expired:
                    raise DeadlineExceeded('native deadline exhausted', settlement=result)
                if self.cancelled.is_set():
                    raise Cancelled('native operation cancelled', settlement=result)
                if event.get('error'):
                    raise ProviderFailure(event['error'], settlement=result)
                return result

    def prompt(self, text, *, operation_id, remaining=None):
        self.prepare()
        if self.running or self.closed:
            raise NativeError('native session is busy or closed')
        self.operation_id, self.running, self.provider_settled = operation_id, True, False
        self.cancelled.clear()
        self._send(dict(type='prompt', text=text))
        try:
            return self._wait('settled', remaining)
        except KeyboardInterrupt:
            self.abort()
            end = time.monotonic()+3
            try:
                self._wait('settled', lambda: end-time.monotonic())
            except NativeError:
                pass
            raise
        finally:
            self.running = False

    def abort(self):
        self.cancelled.set()
        if self.process and self.process.poll() is None:
            self._send(dict(type='abort'))

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process:
            if self.process.poll() is None:
                self._send(dict(type='close'))
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.terminate(); self.process.wait(timeout=3)
            self.reader.join(timeout=3)
            for stream in (self.process.stdin, self.process.stdout):
                try:
                    stream.close()
                except OSError:
                    pass  # A lost transport does not establish provider/tool closure.

    def __enter__(self):
        return self.prepare()

    def __exit__(self, *args):
        self.close()
