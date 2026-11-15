"""Trusted experiment checker. Never imported from an editable task directory."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback

torch = None

ROOT = Path(__file__).parent.resolve()


class HarnessRefusal(Exception):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def health():
    physical = os.environ.get('ASCEND_PHYSICAL_DEVICE')
    if physical is None:
        raise HarnessRefusal('physical device identity not declared')
    receipts = {}
    for kind in ['board', 'health', 'proc-mem', 'usages']:
        try:
            p = subprocess.run(['npu-smi', 'info', '-t', kind, '-i', physical, '-c', '0'], capture_output=True, text=True, timeout=5)
        except (subprocess.TimeoutExpired, OSError) as error:
            raise HarnessRefusal('bounded health query timed out: ' + kind) from error
        if p.returncode:
            raise HarnessRefusal('health query failed: ' + kind)
        receipts[kind] = p.stdout
    if not re.search(r'Chip Name\s*:\s*' + re.escape(EXPECTED['product']) + r'(?:\s|$)', receipts['board']):
        raise HarnessRefusal('qualified A3 product identity changed')
    if not re.search(r'Health Status\s*:\s*OK', receipts['health']) or not re.search(r'Error Code\s*:\s*NA', receipts['health']):
        raise HarnessRefusal('device fault/health refusal')
    processes = [int(pid) for pid in re.findall(r'Process id:\s*(\d+)', receipts['proc-mem'])]
    if any(pid != os.getpid() for pid in processes):
        raise HarnessRefusal('foreign device process: ' + repr(processes))
    if not processes:
        utilization = [int(n) for n in re.findall(r'Ai(?:core|vector) Usage Rate\(%\)\s*:\s*(\d+)', receipts['usages'])]
        if not utilization or any(utilization):
            raise HarnessRefusal('device not idle before initialization')
    return {'physicalDevice': int(physical), 'logicalDevice': 0, 'queries': receipts}


def runtime(expected):
    global EXPECTED
    EXPECTED = expected
    actual_identity = load(ROOT / 'runtime-identity.py', 'trusted_runtime_identity').identity()
    if actual_identity != expected:
        raise HarnessRefusal('qualified runtime identity changed')
    global torch
    receipt = health()
    try:
        import torch as torch_module
        import torch_npu
        import triton
    except (ImportError, OSError) as error:
        raise HarnessRefusal('required A3 runtime unavailable') from error
    torch = torch_module
    actual = {'torch': torch.__version__, 'torch_npu': torch_npu.__version__, 'triton': triton.__version__}
    try:
        torch.npu.set_device(0)
    except Exception as error:
        raise HarnessRefusal('NPU initialization failed before candidate import: ' + str(error)) from error
    receipt['runtime'] = actual
    return receipt


def cpu_copy(value):
    return value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value


def outputs(value):
    return value if isinstance(value, (tuple, list)) else (value,)


def numeric_check(spec, family, case, original, host, expected):
    if family == 'splitk':
        excess = spec.check(*original, *host, expected)
        assert excess <= 0, f'splitk tolerance exceeded: {excess}'
    elif family == 'gated':
        spec.check(*original, case, host[0])
    else:
        spec.check(*original, host[0])


def validate(spec, family, case, original, device_args, result, expected, previous=None):
    result = outputs(result)
    assert result and all(isinstance(x, torch.Tensor) and x.device.type == 'npu' for x in result), 'actual NPU outputs required'
    for x in result:
        assert x.is_contiguous(), 'output must be contiguous'
        assert torch.isfinite(x).all().item(), 'nonfinite output'
        assert all(x.untyped_storage().data_ptr() != a.untyped_storage().data_ptr() for a in device_args if isinstance(a, torch.Tensor)), 'output aliases input'
    if len(result) > 1:
        assert len({x.untyped_storage().data_ptr() for x in result}) == len(result), 'outputs alias each other'
    for before, after in zip(original, device_args):
        if isinstance(before, torch.Tensor):
            assert torch.equal(before, after.detach().cpu()), 'input mutation'
    host = tuple(x.detach().cpu() for x in result)
    numeric_check(spec, family, case, original, host, expected)
    for old, old_host in previous or []:
        for old_tensor, snapshot, new_tensor in zip(old, old_host, result):
            assert torch.equal(old_tensor.detach().cpu(), snapshot), 'later call changed prior output'
            assert old_tensor.untyped_storage().data_ptr() != new_tensor.untyped_storage().data_ptr(), 'output storage reused across calls'
    return [*(previous or []), (result, tuple(x.clone() for x in host))]


def fixtures(family, cases):
    spec = load(ROOT / 'fixtures/trusted' / family / 'spec.py', 'trusted_' + family)
    records = []
    for case in cases:
        args = spec.make_inputs(case)
        expected = spec.evaluate(*args) if family != 'gated' else spec.evaluate(args[0], args[1], case)
        records.append((case, args, expected))
    return spec, records


def first_call(module, args, poison=False):
    if not poison:
        return module.run(*args)
    # An uninitialized result must not pass by inheriting an allocator's prior
    # contents. Poison only the GEMM output allocation for initial correctness
    # checks; warm and timed calls use the untouched allocation implementation.
    original_empty = torch.empty
    shape = (args[0].shape[0], args[1].shape[1])
    def empty(*positional, **keywords):
        result = original_empty(*positional, **keywords)
        if result.device.type == 'npu' and result.dtype == torch.float32 and tuple(result.shape) == shape:
            result.fill_(float('nan'))
        return result
    torch.empty = empty
    try:
        return module.run(*args)
    finally:
        torch.empty = original_empty


def check_module(path, family, cases, timed=True):
    module = load(path, 'candidate_' + sha(path)[:12] + '_' + str(time.monotonic_ns()))
    spec, records = fixtures(family, cases)
    raw = []
    for case, original, expected in records:
        device_args = tuple(x.to('npu:0') if isinstance(x, torch.Tensor) else x for x in original)
        previous = None
        # First-call correctness is checked before any warming; second call also
        # verifies output ownership/cache behavior. Compilation is never timed.
        for _ in range(2):
            value = first_call(module, device_args, poison=family == 'gemm')
            torch.npu.synchronize()
            previous = validate(spec, family, case, original, device_args, value, expected, previous)
        if timed and not raw:
            for _ in range(3):
                value = module.run(*device_args)
                torch.npu.synchronize()
                previous = validate(spec, family, case, original, device_args, value, expected, previous)
            for _ in range(7):
                torch.npu.synchronize()
                start = time.perf_counter()
                value = module.run(*device_args)
                torch.npu.synchronize()
                raw.append(time.perf_counter() - start)
                previous = validate(spec, family, case, original, device_args, value, expected, previous)
    return {'latencySeconds': statistics.median(raw) if raw else None, 'rawSeconds': raw, 'checkedCases': [x['id'] for x in cases]}


def main(work, config):
    path = Path(work) / 'kernel.py'
    identity = sha(path)
    try:
        binding = runtime(config['runtimeIdentity'])
        measured = check_module(path, config['family'], config['cases'])
        assert sha(path) == identity, 'kernel changed during verification'
        print(json.dumps({'correct': True, 'kernelSha256': identity, 'latencySeconds': measured['latencySeconds'], 'evidence': {**measured, **binding, 'protocol': 'first-call-check-then-3warmup-7-synchronized-callable-median'}}))
    except HarnessRefusal as error:
        print(json.dumps({'correct': None, 'bucket': 'harness', 'kernelSha256': identity, 'evidence': {'reason': str(error)}}))
        raise SystemExit(125)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({'correct': False, 'kernelSha256': identity, 'evidence': {'reason': str(error), 'type': type(error).__name__}}))
        raise SystemExit(1)


def pair(active, candidate, config):
    paths = [Path(active) / 'kernel.py', Path(candidate) / 'kernel.py']
    identities = [sha(p) for p in paths]
    binding = runtime(config['runtimeIdentity'])
    family, cases = config['family'], config['cases']
    for path in paths:
        check_module(path, family, cases, timed=False)
    spec, records = fixtures(family, cases)
    case, original, expected = records[0]
    args = tuple(x.to('npu:0') if isinstance(x, torch.Tensor) else x for x in original)
    modules = [load(p, 'pair_' + str(i) + '_' + str(time.monotonic_ns())) for i, p in enumerate(paths)]
    previous = [None, None]
    for _ in range(5):
        for i, module in enumerate(modules):
            value = module.run(*args)
            torch.npu.synchronize()
            previous[i] = validate(spec, family, case, original, args, value, expected, previous[i])
    raw = [[], []]
    order = []
    for turn in range(20):
        sequence = [0, 1] if turn % 2 == 0 else [1, 0]
        order.append(sequence)
        for i in sequence:
            torch.npu.synchronize()
            start = time.perf_counter()
            value = modules[i].run(*args)
            torch.npu.synchronize()
            raw[i].append(time.perf_counter() - start)
            previous[i] = validate(spec, family, case, original, args, value, expected, previous[i])
    # Admission was checked before this worker initialized its NPU context.
    # A fresh idle probe belongs after this numeric worker exits, not while its
    # own device context is necessarily live (PID namespaces can differ).
    assert identities == [sha(p) for p in paths], 'kernel changed during paired timing'
    print(json.dumps({'correct': True, 'activeLatencySeconds': statistics.median(raw[0]), 'candidateLatencySeconds': statistics.median(raw[1]),
                      'activeKernelSha256': identities[0], 'candidateKernelSha256': identities[1],
                      'evidence': {**binding, 'rawSeconds': {'active': raw[0], 'candidate': raw[1]}, 'order': order,
                                   'warmup': 5, 'rounds': 20, 'checkedCases': [x['id'] for x in cases], 'protocol': 'alternating-synchronized-callable-median'}}))
