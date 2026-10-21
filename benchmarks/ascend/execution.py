"""Execution selection and retained local subprocesses, independent of remote access."""
import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
import json
import math
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time
import uuid

from optimizer.control import ObservationUnavailable, WorkerStopped
from optimizer.jobs import atomic, canonical

_deadline = ContextVar('local_kernel_deadline', default=float('inf'))


@contextmanager
def process_deadline(deadline):
    token = _deadline.set(min(_deadline.get(), deadline))
    try:
        yield
    finally:
        _deadline.reset(token)


def execution_mode(config):
    mode = config.get('execution', 'remote')
    if mode not in ('local', 'remote'):
        raise ValueError('execution must be local or remote')
    return mode


def apply_worker_config(config, worker_config_path=None):
    settings = dict(config)
    if worker_config_path is not None:
        worker = json.loads(Path(worker_config_path).read_text())
        settings.update(worker)
        if 'target' in config and 'target' in worker:
            settings['target'] = config['target']
            settings['transport_target'] = worker['target']
        if execution_mode(worker) == 'remote' and 'profile' in worker and 'execution_profile' not in worker:
            settings['execution_profile'] = worker['profile']
    if execution_mode(settings) == 'local':
        workspace = Path(settings.get('workspace', ''))
        if not workspace.is_absolute():
            raise ValueError('local execution requires an absolute workspace')
        if type(settings.get('physical_device')) is not int or settings['physical_device'] < 0:
            raise ValueError('local execution requires a nonnegative physical_device')
        if not isinstance(settings.get('arch'), str) or not settings['arch'].strip():
            raise ValueError('local execution requires a compiler arch')
        seconds = settings.get('max_seconds', 600)
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('max_seconds must be finite positive seconds')
        settings['workspace'] = str(workspace.resolve())
        settings['remote_workspace'] = settings['workspace']
        settings.setdefault('profile', 'local')
        settings['execution_profile'] = 'local'
        settings.setdefault('target', 'local')
        settings.setdefault('runtime', 'current-python')
        settings.setdefault('lock_directory', str(Path(os.environ.get('XDG_STATE_HOME', Path.home() / '.local/state')) / 'rsi/device-locks'))
    return settings


def local_environment(config):
    environment = dict(os.environ)
    device = str(config['physical_device'])
    visible = environment.get('ASCEND_RT_VISIBLE_DEVICES')
    if visible and device not in [item.strip() for item in visible.split(',')]:
        raise ValueError('physical_device is excluded by ASCEND_RT_VISIBLE_DEVICES')
    environment['ASCEND_RT_VISIBLE_DEVICES'] = device
    environment['RSI_LOCAL_PHYSICAL_DEVICE'] = device
    # Kernel helpers must remain importable when their working directory is a job.
    root = str(Path(__file__).resolve().parents[2])
    environment['PYTHONPATH'] = os.pathsep.join(filter(None, (root, environment.get('PYTHONPATH'))))
    return environment


def local_identity(config):
    compiler = shutil.which('bisheng')
    fingerprint = None
    if compiler:
        path = Path(compiler).resolve()
        stat = path.stat()
        fingerprint = [str(path), stat.st_size, stat.st_mtime_ns]
    return {'execution': 'local', 'host': socket.gethostname(),
            'workspace': str(Path(config['workspace']).resolve()),
            'physical_device': config['physical_device'], 'arch': config['arch'],
            'python': sys.executable, 'compiler': fingerprint,
            'cann_version': os.environ.get('ASCEND_CANN_VERSION', ''),
            'catlass_revision': config.get('catlass_revision', config.get('runtime_catlass_revision'))}


def bind_local_identity(root, config):
    path = Path(root) / 'local-execution-identity.json'
    identity = local_identity(config)
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('local execution identity changed on resume')
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic(path, canonical(identity))
    return identity


def preflight_local(config, *, catlass=False):
    """Probe the actual activated environment before starting a model session."""
    settings = apply_worker_config(config)
    probe = '''import json,os,shutil,subprocess
from pathlib import Path
import torch,torch_npu
torch.npu.set_device(0)
torch.empty(1,device='npu:0');torch.npu.synchronize()
compiler=shutil.which('bisheng')
if compiler is None: raise RuntimeError('bisheng compiler not found on PATH')
version=subprocess.run([compiler,'--version'],capture_output=True,text=True,timeout=10,check=True)
cann=os.environ.get('ASCEND_CANN_VERSION') or str(getattr(getattr(torch_npu,'version',None),'cann',''))
home=os.environ.get('ASCEND_HOME_PATH') or os.environ.get('ASCEND_TOOLKIT_HOME')
if not cann and home:
 for name in ['version.cfg','version.info']:
  p=Path(home)/name
  if p.is_file(): cann=p.read_text().strip();break
profiler=shutil.which('msprof')
print(json.dumps(dict(cann_version=cann or 'unreported',compiler_version=version.stdout.strip(),
 profiler_version=profiler or 'unavailable',torch_version=torch.__version__,
 product=torch.npu.get_device_name(0))))
'''
    timeout = min(settings.get('preflight_timeout_seconds', 120), settings.get('max_seconds', 600))
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('preflight_timeout_seconds must be finite positive seconds')
    try:
        result = subprocess.run([sys.executable, '-c', probe], env=local_environment(settings),
                                capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f'local runtime preflight exceeded {timeout:g} seconds; check the activated runtime and selected device') from error
    if result.returncode:
        raise RuntimeError('local kernel runtime preflight failed: ' + result.stderr[-4000:])
    identity = {**local_identity(settings), **json.loads(result.stdout.splitlines()[-1])}
    if catlass:
        source = Path(settings.get('catlass_source', ''))
        revision = settings.get('catlass_revision') or settings.get('runtime_catlass_revision')
        if not source.is_absolute() or not revision:
            raise ValueError('local Catlass execution requires catlass_source and catlass_revision')
        actual = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        if actual != revision:
            raise ValueError('local Catlass revision differs from the configured revision')
        identity['catlass_revision'] = actual
    return identity


def _start_ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


class LocalProcessRunner:
    """A journaled process group. Repeated observation never repeats execution."""
    def __init__(self, config, *, deadline=float('inf'), stopped=lambda: False, retain=lambda h: None):
        self.config, self.stopped, self.retain = config, stopped, retain
        self.deadline = deadline

    def cancel_pending(self, directory):
        directory = Path(directory)
        path = directory / 'transport.json'
        if not path.exists():
            return
        state = json.loads(path.read_text())
        if state['state'] in ('completed', 'failed', 'cancelled'):
            return
        if not state.get('pid'):
            raise ObservationUnavailable('local dispatch has no process identity; inspect its journal')
        current = _start_ticks(state['pid'])
        if current is not None and current != state['start_ticks']:
            raise ObservationUnavailable('local process identity changed; refusing to cancel a different process')
        try:
            os.killpg(state['pid'], signal.SIGTERM)
            time.sleep(.2)
            os.killpg(state['pid'], signal.SIGKILL)
        except ProcessLookupError:
            pass
        state.update(state='cancelled', returncode=-signal.SIGTERM)
        atomic(path, canonical(state))

    def run(self, argv, directory, *, cwd=None, env=None, stdin=None):
        process = None
        deadline = min(self.deadline, _deadline.get(), time.monotonic() + self.config.get('max_seconds', 600))
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'transport.json'
        payload = {'argv': list(map(str, argv)), 'cwd': str(Path(cwd or directory).resolve()),
                   'stdin': stdin, 'deadline': deadline}
        identity = hashlib.sha256(canonical({k: v for k, v in payload.items() if k != 'deadline'})).hexdigest()
        if path.exists():
            state = json.loads(path.read_text())
            if state.get('execution_identity') != local_identity(self.config):
                raise ValueError('local execution identity changed on resume')
            if state.get('request_sha256') != identity:
                raise ValueError('local operation must resume with its original command')
            if state['state'] == 'dispatching':
                raise ObservationUnavailable('local dispatch interrupted before process identity was retained')
            if state['state'] == 'cancelled':
                return self.run(argv, directory / 'retry', cwd=cwd or directory, env=env, stdin=stdin)
        else:
            if self.stopped() or time.monotonic() >= deadline:
                raise WorkerStopped('local operation stopped before dispatch')
            state = {'state': 'dispatching', 'handle': 'local:' + uuid.uuid4().hex,
                     'request_sha256': identity, 'execution_identity': local_identity(self.config)}
            atomic(path, canonical(state))
            self.retain(state['handle'])
            locks = Path(apply_worker_config(self.config)['lock_directory'])
            atomic(directory / 'process.json', canonical({**payload,
                'device_lock': str(locks / f"kernel-{self.config['physical_device']}.lock")}))
            environment = local_environment(self.config)
            if env:
                for key, value in env.items():
                    if value is None:
                        environment.pop(key, None)
                    else:
                        environment[key] = value
            process = subprocess.Popen([sys.executable, '-m', 'benchmarks.ascend.execution', str(directory)],
                                       env=environment, start_new_session=True,
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # The supervisor publishes its identity and terminal result independently.
            while json.loads(path.read_text())['state'] == 'dispatching':
                if process.poll() is not None:
                    raise ObservationUnavailable('local supervisor exited before recording its process identity')
                time.sleep(.01)
        while True:
            state = json.loads(path.read_text())
            if state['state'] in ('completed', 'failed', 'cancelled'):
                break
            if self.stopped() or time.monotonic() >= deadline:
                self.cancel_pending(directory)
                if process is not None:
                    process.wait(timeout=2)
                raise WorkerStopped('local operation stopped or time budget exhausted')
            if _start_ticks(state['pid']) != state['start_ticks']:
                raise ObservationUnavailable('local operation lost its process without a terminal receipt')
            time.sleep(.05)
        stdout = (directory / 'process.stdout').read_text(errors='replace')
        stderr = (directory / 'process.stderr').read_text(errors='replace')
        atomic(directory / 'transport.log', (stdout + '\n' + stderr).encode())
        if state['state'] == 'cancelled':
            if process is not None:
                process.wait(timeout=2)
            raise WorkerStopped('local operation was cancelled')
        if process is not None:
            process.wait(timeout=2)
        result = subprocess.CompletedProcess(argv, state['returncode'], stdout, stderr)
        result.local_handle = state['handle']
        result.operation_directory = directory
        return result


def _supervise(directory):
    from optimizer.transport import exclusive
    directory = Path(directory)
    path = directory / 'transport.json'
    state = json.loads(path.read_text())
    state.update(state='running', pid=os.getpid(), start_ticks=_start_ticks(os.getpid()))
    atomic(path, canonical(state))
    payload = json.loads((directory / 'process.json').read_text())
    with (directory / 'process.stdout').open('wb') as stdout, (directory / 'process.stderr').open('wb') as stderr:
        try:
            remaining = max(.001, payload['deadline'] - time.monotonic()) if payload['deadline'] else None
            with exclusive(payload['device_lock'], wait=True,
                           stopped=lambda: payload['deadline'] is not None and time.monotonic() >= payload['deadline']):
                remaining = max(.001, payload['deadline'] - time.monotonic()) if payload['deadline'] else None
                result = subprocess.run(payload['argv'], cwd=payload['cwd'], stdout=stdout, stderr=stderr,
                                        input=payload['stdin'].encode() if payload['stdin'] is not None else None,
                                        timeout=remaining)
            state.update(state='completed' if result.returncode == 0 else 'failed', returncode=result.returncode)
        except subprocess.TimeoutExpired:
            # Kill the complete group, including grandchildren and this supervisor.
            state.update(state='cancelled', returncode=124)
            atomic(path, canonical(state))
            os.killpg(os.getpid(), signal.SIGKILL)
        except Exception as error:
            stderr.write(str(error).encode())
            state.update(state='failed', returncode=1)
        atomic(path, canonical(state))


if __name__ == '__main__':
    _supervise(sys.argv[1])
