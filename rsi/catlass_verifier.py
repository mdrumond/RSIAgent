"""Standalone trusted Catlass ABI v1 driver, copied into the frozen verifier.

This file uses stdlib until runtime identity is checked. It never invokes an
Actor or provider. A generated verifier embeds FROZEN_CONTRACT at its end.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import time

ABI_VERSION = 1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_git(repository, *arguments):
    return subprocess.run(['git', '-C', repository, *arguments], check=True,
                          capture_output=True, text=True).stdout.strip()


def project_hash(contract):
    entries = []
    names = set(contract['execution_inputs'])
    for path in Path.cwd().rglob('*'):
        name = path.relative_to(Path.cwd()).as_posix()
        if (any(part in {'__pycache__', '.pytest_cache'} for part in Path(name).parts)
                or any(name == item or name.startswith(item.rstrip('/') + '/')
                       for item in contract['generated_paths'])):
            continue
        if path.is_file() or path.is_symlink():
            names.add(name)
    for name in sorted(names):
        path = Path(name)
        if path.is_symlink():
            raise ValueError('candidate snapshot contains a symlink')
        entries.append([name, sha(path) if path.is_file() else None])
    return hashlib.sha256(json.dumps(entries, separators=(',', ':')).encode()).hexdigest()


def configure_target_python_paths(paths):
    """Optional explicit registry-derived paths in target Python3.11 only."""
    if not paths:
        return
    if sys.version_info[:2]!=(3,11):
        raise ValueError('target Python paths must not enter a controller interpreter')
    if not isinstance(paths,list) or any(not isinstance(path,str) or not Path(path).is_absolute()
                                        or not Path(path).is_dir() for path in paths):
        raise ValueError('target_python_paths must be explicit existing absolute registry paths')
    for path in reversed(paths):
        if path in sys.path:sys.path.remove(path)
        sys.path.insert(0,path)


def check_runtime(contract, capture_root=None):
    profile = contract['runtime_identity']
    if sys.version_info[:2] != (3, 11):
        raise ValueError('Catlass target requires Python 3.11')
    configure_target_python_paths(profile.get('target_python_paths',[]))
    if checked_git(profile['catlass_source'], 'rev-parse', 'HEAD') != profile['source_revision']:
        raise ValueError('Catlass source revision mismatch')
    checked_git(profile['catlass_source'], 'diff', '--exit-code', 'HEAD')
    link = checked_git(profile['catlass_source'], 'ls-tree', 'HEAD', profile['dependency_gitlink'])
    if len(link.split()) < 3 or link.split()[2] != profile['dependency_revision']:
        raise ValueError('AscendNPU-IR gitlink mismatch')
    for item in profile['consumer_files']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('runtime consumer/compiler identity mismatch')
    version = Path(profile['cann_version_file']).read_text()
    if version != profile['cann_version_contents'] or not profile['cann_version'].startswith('9.3.'):
        raise ValueError('exact project CANN 9.3 identity mismatch')
    # This manifest hash binds the installed consumer to main's discovered
    # .npuir-pipeline identity, not merely to a source checkout's gitlink.
    if sha(profile['pipeline_manifest']) != profile['pipeline_manifest_sha256']:
        raise ValueError('installed pipeline manifest mismatch')
    for name, env_name in [('TlaCompile', 'TLA_DSL_COMPILE'), ('hivmc-a5', 'TLA_DSL_HIVMC_A5')]:
        selected = os.environ.get(env_name) or shutil.which(name)
        if selected is None or str(Path(selected).resolve()) != str(Path(profile['compiler_paths'][name]).resolve()):
            raise ValueError('runtime compiler path mismatch')
        if capture_root is not None:
            root = Path(capture_root).resolve()
            owned = (Path.cwd() / 'outputs').resolve()
            if not root.is_relative_to(owned):
                raise ValueError('capture wrappers must be inside the current trusted outputs tree')
            wrapper = root / 'wrappers' / name
            if not wrapper.is_file():
                raise ValueError('trusted capture wrapper missing')
            selected = str(wrapper)
        os.environ[env_name] = selected
    import catlass
    from catlass._mlir import ir
    if sha(catlass.__file__) != profile['catlass_module_sha256']:
        raise ValueError('imported Catlass package identity mismatch')
    with ir.Context():
        pass
    return {'source_revision': profile['source_revision'],
            'dependency_revision': profile['dependency_revision'],
            'cann_version': profile['cann_version'],
            'python': sys.version.split()[0], 'catlass_module_sha256': sha(catlass.__file__)}


def case_inputs(case, operation):
    length = case['length']
    if operation == 'convert':
        probes = [1.0006, -1.0006, 1.00048828125, 1.00146484375,
                  .333251953125, .00000008940696716308594]
        return ([struct.unpack('f', struct.pack('f', probes[i % len(probes)]))[0]
                 for i in range(length)], [0.] * length)
    rng = random.Random(case['seed'])
    return ([rng.randint(-127, 127) / 16 for _ in range(length)],
            [rng.randint(-127, 127) / 16 for _ in range(length)])


def reference(operation, left, right):
    if operation == 'add':
        return [a + b for a, b in zip(left, right, strict=True)]
    if operation == 'multiply':
        return [a * b for a, b in zip(left, right, strict=True)]
    if operation == 'relu':
        return [max(a, 0.) for a in left]
    if operation == 'convert':
        return [struct.unpack('e', struct.pack('e', a))[0] for a in left]
    raise ValueError('unknown frozen operation')


def load_kernel(path, allowed_helpers=(), entrypoint='kernel'):
    tree = ast.parse(path.read_text())
    # ABI v1 accepts kernel definitions, documented Catlass imports and numeric
    # constants. It does not execute an Actor-owned host launch/fallback.
    kernels = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entrypoint]
    if len(kernels) != 1:
        raise ValueError('ABI v1 requires exactly one kernel definition')
    arguments = kernels[0].args
    if (len(arguments.args) != 3 or arguments.posonlyargs or arguments.kwonlyargs
            or arguments.vararg or arguments.kwarg or arguments.defaults):
        raise ValueError('ABI v1 requires exactly three tensor arguments')
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.Import) and all(a.name == 'catlass.tla' for a in node.names):
            continue
        if isinstance(node, ast.FunctionDef):
            if node.name == entrypoint:
                continue
            if node.name in allowed_helpers and not node.decorator_list:
                continue
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            value = node.value
            if isinstance(value, ast.Constant) and type(value.value) in (int, float):
                continue
            if (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)
                    and value.value.id == 'tla' and value.attr in ('Float16', 'Float32', 'BFloat16')):
                continue
        raise ValueError('candidate module violates frozen Catlass ABI v1')
    spec = importlib.util.spec_from_file_location('rsi_catlass_candidate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, entrypoint)


def run(contract, mode, case_index=None, capture_root=None, profile_capture=False):
    global capability_gate, compiler_repair_gate, launch_output, fresh_device_health, profile_validation
    if 'capability_gate' not in globals():
        from rsi.catlass_capabilities import capability_gate, compiler_repair_gate, launch_output, fresh_device_health, profile_validation
    started = time.monotonic()
    current_source = project_hash(contract)
    expected_source = os.environ.get('RSI_EXPECTED_SOURCE_HASH', current_source)
    if current_source != expected_source:
        raise ValueError('target snapshot source identity mismatch')
    # Default local PythonExecution binds this same digest in its normal receipt.
    # Set only this verifier process's child environment, never controller state.
    os.environ['RSI_EXPECTED_SOURCE_HASH'] = expected_source
    if (os.environ.get('RSI_OUTER_CPL_HANDLE') != contract['outer_cpl_handle']
            or os.environ.get('RSI_CPL_RUNTIME') != contract['execution_runtime']):
        raise ValueError('outer CPL admission/runtime identity mismatch')
    if capture_root is not None and mode != 'compile':
        raise ValueError('capture wrapper execution is compile-only')
    if contract['abi_version'] != ABI_VERSION:
        raise ValueError('unknown Catlass kernel ABI')
    physical = contract['physical_device']
    if profile_capture:
        if (mode!='test' or case_index!=0 or contract['exercise']['kind']!='msprof-basic'
                or not contract.get('profile_capture')):
            raise ValueError('direct-physical child requires frozen trusted msprof capture binding')
        if any(os.environ.get(name) for name in ('ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES')):
            raise ValueError('direct-physical profiler child must be unmasked')
    elif contract.get('direct_physical') or os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!=str(physical):
        raise ValueError('ordinary launch must mask the admitted physical device')
    is_profile_validation=mode=='validation' and contract['exercise']['kind']=='msprof-basic'
    fresh_device_health(contract,require_idle=profile_capture or is_profile_validation)
    if is_profile_validation:
        if case_index is not None:
            raise ValueError('validation must cover all frozen cases')
        result=profile_validation(contract,source_hash=expected_source,started=started)
        if project_hash(contract)!=expected_source:
            raise ValueError('source changed during profile validation')
        return result
    identity = check_runtime(contract, capture_root)
    import torch
    import torch_npu  # noqa: F401 - registers NPU device support
    import catlass.tla as tla
    from catlass.tla.runtime import from_dlpack
    if mode=='preflight':
        payload=dict(kind='target-runtime-preflight',python=sys.version.split()[0],
            identity=identity,source_hash=expected_source,physical_device=physical,
            outer_cpl_handle=contract['outer_cpl_handle'],correctness=False,device_launched=False,
            target_python_paths=contract['runtime_identity'].get('target_python_paths',[]),
            torch_module=str(torch.__file__),torch_npu_module=str(torch_npu.__file__))
        Path('outputs').mkdir(exist_ok=True)
        Path('outputs/runtime-preflight.json').write_text(json.dumps(payload,indent=2)+'\n')
        print('RSI_CATLASS_PREFLIGHT='+json.dumps(payload,sort_keys=True))
        return payload
    application_device = physical if profile_capture else 0
    torch.npu.set_device(application_device)
    candidate = Path('candidate.py')
    candidate_hash = sha(candidate)
    kernel = load_kernel(candidate, contract['allowed_helpers'], contract.get('entrypoint', 'kernel'))
    cache = Path(os.environ['CATLASS_DSL_CACHE_DIR'])
    cache.mkdir(parents=True, exist_ok=True)
    # The marker is an optimisation only: bind it to exact runtime identity.
    # BC preparation is here inside the billable command, never transfer/setup.
    marker = cache / 'rsi-bc-identity.json'
    bc_identity = json.dumps(identity, sort_keys=True)
    if not marker.exists() or marker.read_text() != bc_identity:
        subprocess.run([sys.executable, '-m', 'catlass.bc_compile'], check=True)
        marker.write_text(bc_identity)
    results = []
    repair_receipt = None
    if mode == 'validation' and case_index is not None:
        raise ValueError('validation must cover all frozen cases')
    cases = contract['cases'] if case_index is None else [contract['cases'][case_index]]
    for case in cases:
        left, right = case_inputs(case, contract['operation'])
        padded = (len(left) + 63) // 64 * 64
        if padded > 512:
            raise ValueError('case exceeds ABI allocation capacity')
        a = torch.tensor(left + [0.] * (padded - len(left)), dtype=torch.float32, device='npu')
        b = torch.tensor(right + [0.] * (padded - len(right)), dtype=torch.float32, device='npu')
        out = torch.tensor([float('nan')]*padded, dtype=torch.float32, device='npu')
        args = [from_dlpack(t, layout_tag=tla.arch.RowMajor).mark_compact_shape_dynamic(0)
                for t in (a, b, out)]
        if mode == 'validation' and contract['exercise']['kind'] == 'compiler-repair' and repair_receipt is None:
            def compile_failed_original(path):
                original = load_kernel(path, contract['allowed_helpers'], contract.get('entrypoint', 'kernel'))
                tla.compile(original, *args, options='--npu-arch 3510')
            repair_receipt = compiler_repair_gate(contract, compile_failed_original, candidate_hash)
        artifact = tla.compile(kernel, *args, options='--npu-arch 3510')
        if repair_receipt is not None:
            repair_receipt['corrected_compiled'] = True
        export = None
        attribute = contract['runtime_identity'].get('artifact_export_attribute')
        if attribute:
            export = artifact
            for component in attribute.split('.'):
                export = getattr(export,component)
            if not isinstance(export,str) or not export:
                raise ValueError('compiled artifact export metadata unavailable')
        result = dict(case=case, compiled=True, launched=False, numeric_match=False,
                      exported_kernel=export)
        if mode != 'compile':
            with launch_output(Path('outputs') / ('profile-cases' if profile_capture else mode+'-cases') / ('case-' + str(len(results))) / 'device-output') as device_output:
                artifact(*args, block_num=case['block_count'])
                torch.npu.synchronize()
            values = out[:len(left)].cpu().tolist()
            expected = reference(contract['operation'], left, right)
            for i, (actual, want) in enumerate(zip(values, expected, strict=True)):
                if not math.isfinite(actual) or actual != want:
                    raise ValueError(f'numeric mismatch case={case} index={i}: {actual!r} != {want!r}')
            result.update(launched=True, numeric_match=True, device_output=device_output)
        results.append(result)
    if sha(candidate) != candidate_hash or project_hash(contract) != expected_source:
        raise ValueError('candidate changed during command')
    capability = None
    if mode == 'validation':
        def compile_original(path):
            original = load_kernel(path, contract['allowed_helpers'], contract.get('entrypoint', 'kernel'))
            tla.compile(original, *args, options='--npu-arch 3510')
        capability = capability_gate(contract, compile_original=compile_original, candidate_hash=candidate_hash, repair_receipt=repair_receipt)
    if project_hash(contract) != expected_source:
        raise ValueError('source changed during capability validation')
    payload = dict(abi_version=ABI_VERSION, mode=mode, identity=identity,
                   source_hash=expected_source, candidate_sha256=candidate_hash,
                   outer_cpl_handle=contract['outer_cpl_handle'], physical_device=physical,
                   execution_runtime=contract['execution_runtime'], logical_device=0 if not profile_capture else None,
                   application_device=application_device,
                   device_addressing='direct-physical' if profile_capture else 'masked-logical-zero',
                   case_hash=contract['case_hash'],
                   cases=results, elapsed=time.monotonic() - started,
                   correctness=mode != 'compile', capability_evidence=capability,
                   exercise_validated=mode == 'validation')
    Path('outputs').mkdir(exist_ok=True)
    Path('outputs/catlass-result.json').write_text(json.dumps(payload, indent=2) + '\n')
    print('RSI_CATLASS_RESULT=' + json.dumps(payload, sort_keys=True))
    return payload


def main(contract):
    global CapabilityUnavailable
    if 'CapabilityUnavailable' not in globals():
        from rsi.catlass_capabilities import CapabilityUnavailable
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=('preflight','compile', 'test', 'validation'))
    parser.add_argument('--case-index', type=int)
    parser.add_argument('--capture-root')
    parser.add_argument('--profile-capture', action='store_true')
    args = parser.parse_args()
    try:
        run(contract, args.mode, args.case_index, args.capture_root, args.profile_capture)
    except CapabilityUnavailable as error:
        payload = dict(outcome='capability-study-unavailable', exercise=contract['exercise']['kind'],
                       reason=str(error), exercise_validated=False,
                       source_hash=os.environ.get('RSI_EXPECTED_SOURCE_HASH'), case_hash=contract['case_hash'])
        Path('outputs').mkdir(exist_ok=True)
        Path('outputs/catlass-capability-study.json').write_text(json.dumps(payload, indent=2) + '\n')
        print('RSI_CATLASS_CAPABILITY_STUDY=' + json.dumps(payload, sort_keys=True))
        raise SystemExit(78)
