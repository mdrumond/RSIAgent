"""Standalone frozen Triton compile/launch verifier for admitted Ascend runtimes.

Copy this file outside the actor-editable snapshot as its trusted verifier.
The host freezes cases and runtime identity hashes in argv. No imports of the
controller package are needed in the target Python environment.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_json(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Frozen host input hash mismatch')
    return json.loads(data)


def load_module(name, path):
    """Register modules before executing decorators that inspect their module."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def record(directory, name, value):
    """Append evidence before later work can fail."""
    with (directory / name).open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True) + '\n')
        stream.flush()


def snapshot_inputs(args, directory):
    inputs = directory / 'inputs'
    inputs.mkdir()
    for field in ('candidate', 'cases', 'runtime_manifest', 'exercise_profile', 'exercise_producer', 'health_guard'):
        source = getattr(args, field, None)
        if source and Path(source).is_file():
            shutil.copyfile(source, inputs / (field + Path(source).suffix))


def run_driver(args):
    """Retain Python/native output, inputs and partial artifacts on every outcome."""
    started = time.monotonic()
    directory = Path(args.artifacts) / (args.mode + '-' + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    args.artifacts = str(directory)
    outcome = {'status': 'failed', 'mode': args.mode}
    try:
        with (directory/'stdout.log').open('w') as out, (directory/'stderr.log').open('w') as err:
            sys.stdout.flush()
            sys.stderr.flush()
            descriptors = [os.dup(1), os.dup(2)]
            try:
                os.dup2(out.fileno(), 1)
                os.dup2(err.fileno(), 2)
                with redirect_stdout(out), redirect_stderr(err):
                    try:
                        snapshot_inputs(args, directory)
                        outcome = execute(args)
                    except BaseException as exc:
                        outcome.update(error_type=type(exc).__name__, error=str(exc))
                        outcome['status'] = getattr(exc, 'status', 'failed')
                        if getattr(exc, 'facts', None):
                            outcome['exercise_failure'] = exc.facts
                        traceback.print_exc(file=err)
                        raise
            finally:
                out.flush()
                err.flush()
                ctypes.CDLL(None).fflush(None)
                for descriptor, original in zip((1, 2), descriptors):
                    os.dup2(original, descriptor)
                    os.close(original)
    finally:
        outcome['elapsed_seconds'] = time.monotonic() - started
        (directory/'result.json').write_text(json.dumps(outcome, sort_keys=True, allow_nan=False))
        artifacts = [{'path': path.relative_to(directory).as_posix(), 'sha256': sha256(path),
                      'bytes': path.stat().st_size} for path in sorted(directory.rglob('*')) if path.is_file()]
        (directory/'manifest.json').write_text(json.dumps({'artifacts': artifacts}, sort_keys=True))
        print(json.dumps({'status': outcome['status'], 'result': str(directory/'result.json'),
                          'manifest': str(directory/'manifest.json'),
                          'manifest_sha256': sha256(directory/'manifest.json')}))
    return outcome


def audit_candidate(path):
    """Reject ordinary framework fallback before importing a candidate.

    This is a behavioral admission check, not a process security sandbox.
    Kernel dispatch observation and independent numeric checks are also required.
    """
    tree = ast.parse(Path(path).read_text())
    torch_names = {'torch'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            torch_names.update(alias.asname or alias.name for alias in node.names if alias.name == 'torch')
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('torch'):
            if node.module != 'torch' or any(alias.name not in {'Tensor', 'nn'} for alias in node.names):
                raise ValueError('Importing framework compute functions is not permitted')
    allocation = {'empty', 'empty_like', 'zeros', 'zeros_like', 'ones', 'ones_like'}
    metadata = {'numel', 'size', 'stride', 'dim', 'ndimension', 'view', 'reshape', 'permute', 'transpose'}
    forbidden = {'sum', 'mean', 'matmul', 'mm', 'bmm', 'add', 'mul', 'relu', 'softmax',
                 'cpu', 'numpy', 'tolist', 'item', 'contiguous', 'float', 'half', 'to'}
    kernel_names = set()
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    definitions = {node.name: node for node in functions}
    model = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ModelNew'), None)
    if model is None:
        raise ValueError('Candidate requires ModelNew.forward')
    roots = [node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == 'forward']
    if not roots:
        raise ValueError('Candidate requires ModelNew.forward')
    reachable, pending = set(), roots[:]
    while pending:
        function = pending.pop()
        if id(function) in reachable:
            continue
        reachable.add(id(function))
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
                if name in definitions and id(definitions[name]) not in reachable:
                    pending.append(definitions[name])
    # Follow actual helper arguments: shape arithmetic is scalar, while wrapper
    # inputs and their aliases remain tensors. A helper may be reached with both.
    taints = {id(function): set() for function in functions}
    returns_tensor = {name: False for name in definitions}
    for root in roots:
        taints[id(root)].update(arg.arg for arg in root.args.args if arg.arg != 'self')

    def tensor(expression, names):
        if expression is None:
            return False
        if isinstance(expression, ast.Name):
            return expression.id in names
        if isinstance(expression, ast.Attribute):
            return False if expression.attr in {'shape', 'dtype', 'device', 'ndim'} else tensor(expression.value, names)
        if isinstance(expression, ast.Call):
            if isinstance(expression.func, ast.Attribute):
                if expression.func.attr in {'numel', 'size', 'stride', 'dim', 'ndimension'}:
                    return False
                if expression.func.attr in allocation:
                    return True
            name = expression.func.id if isinstance(expression.func, ast.Name) else getattr(expression.func, 'attr', None)
            if name in definitions:
                return returns_tensor[name]
            return any(tensor(child, names) for child in ast.iter_child_nodes(expression))
        return any(tensor(child, names) for child in ast.iter_child_nodes(expression))

    changed = True
    while changed:
        changed = False
        for function in functions:
            if id(function) not in reachable:
                continue
            names = taints[id(function)]
            before = len(names)
            for node in ast.walk(function):
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and tensor(node.value, names):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    names.update(child.id for target in targets for child in ast.walk(target) if isinstance(child, ast.Name))
                if isinstance(node, ast.Call):
                    name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', None)
                    if name in definitions:
                        callee = definitions[name]
                        params = [arg.arg for arg in callee.args.args if arg.arg != 'self']
                        bindings = dict(zip(params, node.args))
                        bindings.update((kw.arg, kw.value) for kw in node.keywords if kw.arg is not None)
                        prior = len(taints[id(callee)])
                        taints[id(callee)].update(key for key, value in bindings.items() if tensor(value, names))
                        changed |= len(taints[id(callee)]) != prior
                if isinstance(node, ast.Return) and node.value is not None and tensor(node.value, names):
                    if not returns_tensor[function.name]:
                        returns_tensor[function.name] = True
                        changed = True
            changed |= len(names) != before
    for function in functions:
        is_kernel = any(isinstance(decorator, ast.Attribute) and decorator.attr == 'jit'
                        or isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                        and decorator.func.attr == 'jit' for decorator in function.decorator_list)
        if is_kernel:
            kernel_names.add(function.name)
            continue
        if id(function) not in reachable:
            continue
        parameters = taints[id(function)]
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner, method = node.func.value, node.func.attr
                if isinstance(owner, ast.Name) and owner.id in torch_names and method not in allocation:
                    raise ValueError('Framework compute is not permitted in a candidate wrapper')
                if method in forbidden and method not in metadata:
                    raise ValueError('Tensor compute or host data reads are not permitted in wrappers')
            if isinstance(node, ast.BinOp) and tensor(node, parameters):
                raise ValueError('Direct tensor arithmetic belongs in a Triton kernel')
    if not kernel_names:
        raise ValueError('Candidate has no Triton JIT kernel')
    return sorted(kernel_names)


@contextmanager
def launch_capture(jit_type, directory):
    """Observe actual JIT run returns, archive compiler outputs, restore the hook."""
    original = jit_type.run
    launches = []

    def run(kernel, *args, **kwargs):
        record(directory, 'launches.jsonl', {'phase': 'dispatch', 'warmup': bool(kwargs.get('warmup', False))})
        try:
            compiled = original(kernel, *args, **kwargs)
        except BaseException as exc:
            record(directory, 'launches.jsonl', {'phase': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
            raise
        if not kwargs.get('warmup', False):
            event = {'kernel': getattr(kernel, '__name__', getattr(kernel, 'name', type(kernel).__name__)),
                     'artifacts': {}}
            for name, content in getattr(compiled, 'asm', {}).items():
                if not isinstance(content, (str, bytes)):
                    continue
                data = content.encode() if isinstance(content, str) else content
                digest = hashlib.sha256(data).hexdigest()
                suffix = re.sub(r'[^a-zA-Z0-9_-]', '_', str(name))
                dest = directory / (digest + '.' + suffix)
                dest.write_bytes(data)
                event['artifacts'][str(name)] = {'file': dest.name, 'sha256': digest}
            launches.append(event)
            record(directory, 'launches.jsonl', dict(event, phase='returned'))
        return compiled

    jit_type.run = run
    try:
        yield launches
    finally:
        jit_type.run = original


def make_case(torch, operation, case):
    dtype = getattr(torch, case['dtype'])
    generator = torch.Generator(device='cpu').manual_seed(case['seed'])
    shape = case['shape']
    if case['layout'] == 'transpose':
        x = torch.randn((shape[1], shape[0]), generator=generator, dtype=dtype).transpose(0, 1)
    else:
        x = torch.randn(shape, generator=generator, dtype=dtype)
    if operation == 'add':
        y = torch.randn(shape, generator=generator, dtype=dtype)
        return [x, y], (x.float() + y.float()).to(dtype)
    if operation == 'row_sum':
        return [x], x.float().sum(dim=1).to(dtype)
    if operation == 'strided_copy':
        return [x], x.contiguous().clone()
    raise ValueError('Unsupported frozen operation')


def check_output_layout(output, operation):
    if operation == 'strided_copy' and not output.is_contiguous():
        raise ValueError('Strided copy must return contiguous output')


def exercise_evidence(args, contract, context):
    """Produce fresh evidence, using a frozen producer and candidate-agnostic argv."""
    if not args.exercise_producer or sha256(args.exercise_producer) != args.exercise_producer_sha256:
        raise ValueError('Validation requires its frozen exercise producer')
    producer = load_module('_rsi_triton_exercises', args.exercise_producer)
    args.verifier_path = __file__
    args.native_command = lambda directory, candidate=None: native_command(args, directory, candidate=candidate)
    return producer.produce(args, contract, context)


def remaining():
    deadline = os.environ.get('RSI_COMMAND_DEADLINE_MONOTONIC')
    return max(0.0, float(deadline)-time.monotonic()) if deadline else None


def fresh_device_health(args, identity, directory, *, require_idle=False):
    """Consume the common standalone admission probe; never trust cached health."""
    if not args.health_guard or sha256(args.health_guard) != args.health_guard_sha256:
        raise ValueError('Fresh device admission requires the frozen common probe')
    profile = identity.get('device_health')
    if not profile:
        raise ValueError('Runtime identity requires an actual per-device health profile')
    namespace = {'__name__': '_rsi_device_health', '__file__': str(Path(args.health_guard).resolve())}
    exec(compile(Path(args.health_guard).read_text(), args.health_guard, 'exec'), namespace)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    configuration = dict(profile, require_idle=bool(require_idle or profile.get('require_idle', False)))
    receipt = namespace['local_probe'](physical_device=int(args.physical_device),
                                       configuration=configuration, remaining=remaining)
    (directory/'guard.json').write_text(json.dumps(receipt, sort_keys=True))
    for query, result in receipt.get('raw_measurements', {}).get('queries', {}).items():
        for stream in ('stdout', 'stderr'):
            (directory/(query+'.'+stream)).write_text(result.get(stream, ''))
    if (not isinstance(receipt, dict) or receipt.get('admitted') is not True
            or str(receipt.get('physical_device')) != args.physical_device
            or not receipt.get('observation') or not receipt.get('raw_measurements')):
        raise ValueError('Fresh bounded device probe did not admit the physical device')
    return receipt


def native_command(args, directory, *, candidate=None):
    """One fresh native child, inheriting the outer CPL command's owned group."""
    return [sys.executable, str(Path(__file__).resolve()), '--mode', 'test',
            '--candidate', str(Path(candidate or args.candidate).resolve()),
            '--cases', str(Path(args.cases).resolve()), '--cases-sha256', args.cases_sha256,
            '--runtime-manifest', str(Path(args.runtime_manifest).resolve()),
            '--runtime-sha256', args.runtime_sha256, '--physical-device', args.physical_device,
            '--artifacts', str(directory/'application'), '--cache',
            str(Path(args.cache).resolve()/'studies'/hashlib.sha256(str(directory).encode()).hexdigest()),
            '--health-guard', str(Path(args.health_guard).resolve()),
            '--health-guard-sha256', args.health_guard_sha256]


def execute(args):
    if args.mode != 'validation':
        return execute_native(args)
    contract = frozen_json(args.cases, args.cases_sha256)
    identity = frozen_json(args.runtime_manifest, args.runtime_sha256)
    directory = Path(args.artifacts)
    source_hash = sha256(args.candidate)
    native = directory/'native'
    native.mkdir()
    with (native/'stdout.log').open('wb') as out, (native/'stderr.log').open('wb') as err:
        result = subprocess.run(native_command(args, native), stdout=out, stderr=err, timeout=remaining())
    if result.returncode:
        raise ValueError('Fresh native correctness child failed; raw diagnostics retained')
    context = None
    for path in (native/'application').rglob('result.json'):
        value = json.loads(path.read_text())
        if (value.get('status') == 'passed' and value.get('source_sha256') == source_hash
                and value.get('cases_sha256') == args.cases_sha256
                and value.get('runtime_manifest_sha256') == args.runtime_sha256):
            context = value
            for case in context['cases']:
                for launch in case['launches']:
                    for artifact in launch['artifacts'].values():
                        artifact['file'] = str((path.parent/artifact['file']).relative_to(directory))
            break
    if context is None:
        raise ValueError('Native child did not retain a matching verified result')
    (directory/'native-result.json').write_text(json.dumps(context, sort_keys=True, allow_nan=False))
    args.health_probe = lambda path, require_idle=False: fresh_device_health(args, identity, path, require_idle=require_idle)
    context['exercise_evidence'] = exercise_evidence(args, contract, context)
    if sha256(args.candidate) != source_hash:
        raise ValueError('Candidate source changed during exercise production')
    frozen_json(args.cases, args.cases_sha256)
    frozen_json(args.runtime_manifest, args.runtime_sha256)
    context.update(mode='validation', scope='native launch/numerics plus current-candidate trusted exercise production')
    return context


def execute_native(args):
    contract = frozen_json(args.cases, args.cases_sha256)
    identity = frozen_json(args.runtime_manifest, args.runtime_sha256)
    if str(identity['physical_device']) != args.physical_device:
        raise ValueError('Runtime manifest and selected physical device differ')
    mask = os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    if mask is not None and mask != args.physical_device:
        raise ValueError('Existing physical-device mask differs from the frozen binding')
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.physical_device
    os.environ['TRITON_CACHE_DIR'] = str(Path(args.cache).resolve())
    kernels = audit_candidate(args.candidate)
    directory = Path(args.artifacts)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'candidate.ast.txt').write_text(ast.dump(ast.parse(Path(args.candidate).read_text()),
                                                      include_attributes=True, indent=2))
    fresh_device_health(args, identity, directory/'health-before-runtime')
    consumers = identity.get('compiler_consumer_files')
    if not consumers:
        raise ValueError('Actual preflight compiler consumer fingerprints are required')
    for consumer in consumers:
        if sha256(consumer['path']) != consumer['sha256']:
            raise ValueError('Installed compiler consumer differs from frozen runtime preflight: ' + consumer['path'])
    for metadata in identity.get('cann_metadata', []):
        if sha256(metadata['path']) != metadata['sha256']:
            raise ValueError('CANN metadata differs from frozen runtime preflight')
    # Runtime imports occur only in the admitted target process, after mask setup.
    import torch
    import torch_npu
    import triton
    from triton.runtime.jit import JITFunction

    observed = {'torch': str(torch.__version__), 'torch_npu': str(torch_npu.__version__),
                'triton': str(triton.__version__)}
    for key, value in observed.items():
        if identity['expected_versions'].get(key) != value:
            raise ValueError('Admitted runtime version mismatch: ' + key)
    torch.npu.set_device('npu:0')
    target = triton.runtime.driver.active.get_current_target()
    compiler_target = {name: getattr(target, name) for name in ('backend', 'arch', 'warp_size')
                       if hasattr(target, name)}
    if not compiler_target:
        raise ValueError('The active compiler did not expose its supported target')
    if identity.get('expected_compiler_target') is not None and compiler_target != identity['expected_compiler_target']:
        raise ValueError('Active compiler target differs from the admitted target')
    device_name = torch.npu.get_device_name(0)
    candidate_path = Path(args.candidate).resolve()
    source_hash = sha256(candidate_path)
    module = load_module('_rsi_triton_candidate', candidate_path)
    candidate = module.ModelNew().eval()
    results = []
    cases = contract['cases'][:1] if args.mode == 'compile' else contract['cases']
    if not cases:
        raise ValueError('Frozen correctness cases must not be empty')
    with torch.no_grad(), launch_capture(JITFunction, directory) as launches:
        for case in cases:
            record(directory, 'cases.jsonl', {'case_id': case['case_id'], 'status': 'started'})
            inputs, golden = make_case(torch, contract['operation'], case)
            case_file = directory / ('case-' + hashlib.sha256(case['case_id'].encode()).hexdigest() + '.pt')
            raw_case = {'inputs': inputs, 'expected': golden}
            torch.save(raw_case, case_file)
            device_inputs = [value.to('npu:0', memory_format=torch.preserve_format) for value in inputs]
            snapshots = [value.clone() for value in device_inputs]
            torch.npu.synchronize()
            before = len(launches)
            output = candidate(*device_inputs)
            torch.npu.synchronize()
            if len(launches) == before:
                raise ValueError('No actual Triton JIT launch was observed for the case')
            if isinstance(output, torch.Tensor):
                raw_case['actual'] = output.detach().cpu()
                torch.save(raw_case, case_file)
            if not isinstance(output, torch.Tensor) or output.device.type != 'npu':
                raise ValueError('Candidate must return an NPU tensor')
            check_output_layout(output, contract['operation'])
            if output.shape != golden.shape or output.dtype != golden.dtype:
                raise ValueError('Output shape or dtype differs from the frozen oracle')
            if any(not torch.equal(a, b) for a, b in zip(snapshots, device_inputs)):
                raise ValueError('Candidate mutated an input')
            actual = output.detach().cpu()
            torch.testing.assert_close(actual, golden, atol=case['atol'], rtol=case['rtol'], equal_nan=True)
            results.append({'case_id': case['case_id'], 'status': 'passed',
                            'launches': launches[before:], 'shape': list(actual.shape),
                            'dtype': str(actual.dtype),
                            'actual_sample': actual.reshape(-1)[:8].tolist(),
                            'expected_sample': golden.reshape(-1)[:8].tolist(),
                            'max_absolute_error': float((actual.float()-golden.float()).abs().max())})
            record(directory, 'cases.jsonl', results[-1])
    if sha256(candidate_path) != source_hash:
        raise ValueError('Candidate source changed during execution')
    frozen_json(args.cases, args.cases_sha256)
    frozen_json(args.runtime_manifest, args.runtime_sha256)
    import triton.language as language
    result = {'schema_version': 1, 'mode': args.mode, 'status': 'passed',
            'project_id': contract['project_id'], 'source_sha256': source_hash,
            'verifier_sha256': sha256(__file__), 'cases_sha256': args.cases_sha256,
            'runtime_manifest_sha256': args.runtime_sha256, 'observed_versions': observed,
            'compiler_target': compiler_target, 'runtime_device_name': device_name,
            'admitted_soc': identity.get('soc'),
            'declared_runtime_identity': identity, 'physical_device': args.physical_device,
            'logical_device': 0, 'kernel_names': kernels, 'cases': results,
            'observed_api': {name: callable(getattr(language, name, None))
                             for name in ('load', 'store', 'sum', 'arange', 'device_print',
                                          'gather', 'histogram', 'associative_scan', 'sort')},
            'expected_project_hash': os.environ.get('RSI_EXPECTED_SOURCE_HASH'),
            'scope': 'native launch/numerics'}
    (directory/'native-result.json').write_text(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('compile', 'test', 'validation'), required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--cases-sha256', required=True)
    parser.add_argument('--runtime-manifest', required=True)
    parser.add_argument('--runtime-sha256', required=True)
    parser.add_argument('--physical-device', required=True)
    parser.add_argument('--artifacts', default='outputs/triton')
    parser.add_argument('--cache', default='caches/triton')
    parser.add_argument('--exercise-profile')
    parser.add_argument('--exercise-profile-sha256')
    parser.add_argument('--exercise-producer')
    parser.add_argument('--exercise-producer-sha256')
    parser.add_argument('--health-guard', required=True)
    parser.add_argument('--health-guard-sha256', required=True)
    args = parser.parse_args()
    run_driver(args)


if __name__ == '__main__':
    main()
