"""Standalone frozen Triton compile/launch verifier for admitted Ascend runtimes.

Copy this file outside the actor-editable snapshot as its trusted verifier.
The host freezes cases and runtime identity hashes in argv. No imports of the
controller package are needed in the target Python environment.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_json(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Frozen host input hash mismatch')
    return json.loads(data)


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
    for function in functions:
        is_kernel = any(isinstance(decorator, ast.Attribute) and decorator.attr == 'jit'
                        or isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                        and decorator.func.attr == 'jit' for decorator in function.decorator_list)
        if is_kernel:
            kernel_names.add(function.name)
            continue
        if id(function) not in reachable:
            continue
        parameters = {arg.arg for arg in function.args.args} - {'self'}
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner, method = node.func.value, node.func.attr
                if isinstance(owner, ast.Name) and owner.id in torch_names and method not in allocation:
                    raise ValueError('Framework compute is not permitted in a candidate wrapper')
                if method in forbidden and method not in metadata:
                    raise ValueError('Tensor compute or host data reads are not permitted in wrappers')
            if isinstance(node, ast.BinOp) and any(isinstance(value, ast.Name) and value.id in parameters
                                                  for value in (node.left, node.right)):
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
        compiled = original(kernel, *args, **kwargs)
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


def exercise_evidence(args, contract, source_hash):
    """Check host-staged exercise artifacts, never candidate-written attestations."""
    required = set(contract.get('required_evidence', ()))
    generated = {'successful_compile', 'kernel_launch', 'numeric_verification', 'ast_audit'}
    missing = required - generated
    if not missing:
        return {}
    if not args.exercise_evidence or not args.exercise_evidence_sha256:
        raise ValueError('Validation requires host-owned exercise evidence: ' + ', '.join(sorted(missing)))
    manifest = frozen_json(args.exercise_evidence, args.exercise_evidence_sha256)
    for name, expected in (('source_sha256', source_hash), ('cases_sha256', args.cases_sha256),
                           ('runtime_manifest_sha256', args.runtime_sha256)):
        if manifest.get(name) != expected:
            raise ValueError('Exercise evidence identity mismatch: ' + name)
    root = Path(args.exercise_evidence).resolve().parent
    retained = {}
    for name in sorted(missing):
        artifact = manifest['artifacts'].get(name)
        if not artifact or not artifact.get('producer_operation_id'):
            raise ValueError('Missing host producer receipt for exercise evidence: ' + name)
        path = (root / artifact['path']).resolve()
        if not path.is_relative_to(root) or sha256(path) != artifact['sha256']:
            raise ValueError('Exercise artifact is missing or changed: ' + name)
        retained[name] = artifact
    return retained


def execute(args):
    contract = frozen_json(args.cases, args.cases_sha256)
    identity = frozen_json(args.runtime_manifest, args.runtime_sha256)
    if str(identity['physical_device']) != args.physical_device:
        raise ValueError('Runtime manifest and selected physical device differ')
    mask = os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    if mask is not None and mask != args.physical_device:
        raise ValueError('Existing physical-device mask differs from the frozen binding')
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.physical_device
    os.environ['TRITON_CACHE_DIR'] = str(Path(args.artifacts).resolve() / 'jit-cache')
    kernels = audit_candidate(args.candidate)
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
    module_spec = importlib.util.spec_from_file_location('_rsi_triton_candidate', candidate_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    candidate = module.ModelNew().eval()
    directory = Path(args.artifacts)
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    cases = contract['cases'][:1] if args.mode == 'compile' else contract['cases']
    if not cases:
        raise ValueError('Frozen correctness cases must not be empty')
    with torch.no_grad(), launch_capture(JITFunction, directory) as launches:
        for case in cases:
            inputs, golden = make_case(torch, contract['operation'], case)
            device_inputs = [value.to('npu:0', memory_format=torch.preserve_format) for value in inputs]
            snapshots = [value.clone() for value in device_inputs]
            torch.npu.synchronize()
            before = len(launches)
            output = candidate(*device_inputs)
            torch.npu.synchronize()
            if len(launches) == before:
                raise ValueError('No actual Triton JIT launch was observed for the case')
            if not isinstance(output, torch.Tensor) or output.device.type != 'npu':
                raise ValueError('Candidate must return an NPU tensor')
            if output.shape != golden.shape or output.dtype != golden.dtype:
                raise ValueError('Output shape or dtype differs from the frozen oracle')
            if any(not torch.equal(a, b) for a, b in zip(snapshots, device_inputs)):
                raise ValueError('Candidate mutated an input')
            actual = output.detach().cpu()
            torch.testing.assert_close(actual, golden, atol=case['atol'], rtol=case['rtol'], equal_nan=True)
            results.append({'case_id': case['case_id'], 'status': 'passed',
                            'launches': launches[before:], 'shape': list(actual.shape),
                            'dtype': str(actual.dtype),
                            'max_absolute_error': float((actual.float()-golden.float()).abs().max())})
    if sha256(candidate_path) != source_hash:
        raise ValueError('Candidate source changed during execution')
    frozen_json(args.cases, args.cases_sha256)
    frozen_json(args.runtime_manifest, args.runtime_sha256)
    evidence = exercise_evidence(args, contract, source_hash) if args.mode == 'validation' else {}
    return {'schema_version': 1, 'mode': args.mode, 'status': 'passed',
            'project_id': contract['project_id'], 'source_sha256': source_hash,
            'verifier_sha256': sha256(__file__), 'cases_sha256': args.cases_sha256,
            'runtime_manifest_sha256': args.runtime_sha256, 'observed_versions': observed,
            'compiler_target': compiler_target, 'runtime_device_name': device_name,
            'admitted_soc': identity.get('soc'),
            'declared_runtime_identity': identity, 'physical_device': args.physical_device,
            'logical_device': 0, 'kernel_names': kernels, 'cases': results,
            'exercise_evidence': evidence,
            'expected_project_hash': os.environ.get('RSI_EXPECTED_SOURCE_HASH'),
            'scope': 'native launch/numerics' if args.mode != 'validation' else 'native launch/numerics plus frozen host exercise artifacts'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('compile', 'test', 'validation'), required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--cases-sha256', required=True)
    parser.add_argument('--runtime-manifest', required=True)
    parser.add_argument('--runtime-sha256', required=True)
    parser.add_argument('--physical-device', required=True)
    parser.add_argument('--artifacts', default='artifacts/triton')
    parser.add_argument('--exercise-evidence')
    parser.add_argument('--exercise-evidence-sha256')
    args = parser.parse_args()
    started = time.monotonic()
    result = execute(args)
    result['elapsed_seconds'] = time.monotonic() - started
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
