"""Clock-domain-safe native command supervisors and their terminal proofs."""
import base64
import hashlib
import inspect
import json
import math


LAUNCHER = 'elapsed-receipt-v1'


def payload_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _supervise(payload):
    import ctypes
    import hashlib
    import json
    import os
    from pathlib import Path
    import signal
    import subprocess
    import threading
    import time

    body = {key: value for key, value in payload.items() if key != 'payload_sha256'}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != payload['payload_sha256'] or payload['argv'] != payload['binding']['argv']:
        return 125
    root = Path(payload['result_directory'])
    root.mkdir(parents=True, exist_ok=True)
    try:
        claim = os.open(root / 'started', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(claim)
    except FileExistsError:
        return 125  # An observable command is never executed a second time.
    proof = dict(schema='rsi-command-result-v1', binding=payload['binding'],
                 payload_sha256=payload['payload_sha256'])
    outputs = {name: (root / (name+'.bin')).open('wb') for name in ('stdout', 'stderr')}
    errors, dropped, threads = [], [], []
    child, termination = None, 'exit'
    # Adopt and reap descendants instead of publishing a receipt while a daemon
    # still owns the payload's process group or output descriptors.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'cannot own native descendants')

    def stop(signum, frame):
        nonlocal termination
        termination = 'cancelled'
        if child is not None:
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass

    for signum in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)

    def drain(name, pipe, fd):
        try:
            os.set_blocking(fd, False)
            while True:
                data = pipe.read1(65536)
                if not data: break
                outputs[name].write(data)
                try:
                    while data:
                        size = os.write(fd, data)
                        data = data[size:]
                except (BrokenPipeError, BlockingIOError):
                    dropped.append(name)  # The durable stream remains complete.
        except BaseException as error:
            errors.append(type(error).__name__)
        finally:
            pipe.close()

    def group_alive():
        while True:
            try:
                if os.waitpid(-child.pid, os.WNOHANG)[0] == 0: break
            except ChildProcessError: break
        try: os.killpg(child.pid, 0)
        except ProcessLookupError: return False
        return True

    current_boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if current_boot != payload['remote_boot_id']:
        code, termination = 124, 'boot-changed'
    elif termination == 'cancelled' or time.monotonic() >= payload['remote_deadline']:
        code, termination = 124, 'deadline'
    else:
        parent = os.getpid()
        def own_parent():
            if libc.prctl(1, signal.SIGKILL, 0, 0, 0): os._exit(125)
            if os.getppid() != parent: os._exit(125)
            if time.monotonic() >= payload['remote_deadline']: os._exit(124)
        child = subprocess.Popen(payload['argv'], start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=own_parent)
        if termination == 'cancelled': os.killpg(child.pid, signal.SIGKILL)
        for name, pipe, fd in [('stdout', child.stdout, 1), ('stderr', child.stderr, 2)]:
            thread = threading.Thread(target=drain, args=(name, pipe, fd), daemon=True)
            thread.start(); threads.append(thread)
        try:
            code = child.wait(timeout=max(0, payload['remote_deadline']-time.monotonic()))
        except subprocess.TimeoutExpired:
            termination = 'deadline'
            try: os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            code = None
        # Even a successful leader may have left descendants holding stdout.
        end = time.monotonic()+10
        while group_alive() and time.monotonic() < end:
            try: os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            time.sleep(.02)
        if group_alive():
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        try: code = child.wait(timeout=2)
        except subprocess.TimeoutExpired: code = None
        end = time.monotonic()+2
        while group_alive() and time.monotonic() < end: time.sleep(.01)
        for thread in threads: thread.join(timeout=2)
        if code is not None: code = code if code >= 0 else 128-code
        if termination in ('deadline', 'cancelled'): code = 124
    settled = child is None or code is not None and not group_alive()
    complete = not errors and not any(thread.is_alive() for thread in threads)
    for name, stream in outputs.items():
        if complete:
            stream.flush(); os.fsync(stream.fileno()); stream.close()
            raw = (root / (name+'.bin')).read_bytes()
            proof[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    proof.update(returncode=code, termination=termination,
                 process_group_reaped=settled, output_complete=complete)
    temporary = root / 'terminal.tmp'
    with temporary.open('w') as out:
        json.dump(proof, out, sort_keys=True); out.flush(); os.fsync(out.fileno())
    temporary.replace(root / 'terminal.json')
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(directory)
    finally: os.close(directory)
    return code if settled and complete and not dropped and code is not None else 125


def launcher_source(payload):
    body = inspect.getsource(_supervise)
    return ("set -eu\nexport ASCEND_RT_VISIBLE_DEVICES="+str(payload['binding']['route']['physical_device'])+
        "\nexport ASCEND_VISIBLE_DEVICES="+str(payload['binding']['route']['physical_device'])+
        "\nexec python - <<'RSI_COMMAND'\nimport json,sys\n"+body+
        '\nsys.exit(_supervise(json.loads('+repr(json.dumps(payload))+')))\nRSI_COMMAND\n')


def clock_source():
    return ("python - <<'RSI_CLOCK'\nimport json,time\nfrom pathlib import Path\n"
        "print(json.dumps(dict(monotonic=time.monotonic(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())))\nRSI_CLOCK\n")


def receipt_reader_source(directory):
    return ("python - <<'RSI_RECEIPT'\nimport base64,json\nfrom pathlib import Path\n"
        'root=Path('+repr(directory)+')\n'
        "if not (root/'terminal.json').is_file(): print(json.dumps(dict(available=False)))\n"
        "else:\n result={'available':True,'receipt':json.loads((root/'terminal.json').read_bytes())}\n"
        " for name in ('stdout','stderr'): result[name]=base64.b64encode((root/(name+'.bin')).read_bytes()).decode()\n"
        " print(json.dumps(result))\nRSI_RECEIPT\n")


def validate_receipt(value, payload):
    if not isinstance(value, dict) or value.get('available') is not True:
        return None
    proof = value.get('receipt', {})
    body = {key: item for key, item in payload.items() if key != 'payload_sha256'}
    if payload_digest(body) != payload['payload_sha256']:
        raise ValueError('retained command payload digest changed')
    if (not isinstance(proof, dict) or proof.get('schema') != 'rsi-command-result-v1'
            or proof.get('binding') != payload['binding']
            or proof.get('payload_sha256') != payload['payload_sha256']
            or proof.get('process_group_reaped') is not True or proof.get('output_complete') is not True
            or type(proof.get('returncode')) is not int or not 0 <= proof['returncode'] <= 255):
        raise ValueError('remote terminal receipt identity or settlement is invalid')
    streams = {}
    for name in ('stdout', 'stderr'):
        raw = base64.b64decode(value[name], validate=True)
        if proof.get(name) != dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()):
            raise ValueError('remote terminal receipt output is incomplete')
        streams[name] = raw.decode('utf-8', errors='replace')
    return dict(returncode=proof['returncode'], **streams)


def validate_clock(sample):
    if (not isinstance(sample, dict) or not isinstance(sample.get('boot_id'), str) or not sample['boot_id']
            or isinstance(sample.get('monotonic'), bool) or not isinstance(sample.get('monotonic'), (int, float))
            or not math.isfinite(sample['monotonic']) or sample['monotonic'] < 0):
        raise ValueError('remote monotonic clock sample is invalid')
    return sample
