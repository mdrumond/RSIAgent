"""A5 Catlass ABI v1 and numerical controls retained from preserved 2d23.

No orchestration imports. Compilation, launch, and comparison are separate facts.
"""
import ast
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys

from .runner import sha, compare, bind_device, load_candidate
from .health import local_probe

ABI_VERSION = 1

def checked_git(repository, *arguments):
    return subprocess.run(['git', '-C', repository, *arguments], check=True,
                          capture_output=True, text=True).stdout.strip()

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

def _definition_form(node, *, annotation=False, constants=()):
    if node is None:
        return True
    if isinstance(node, ast.Constant):
        if annotation and isinstance(node.value, str):
            try:
                return _definition_form(ast.parse(node.value, mode='eval').body, annotation=True, constants=constants)
            except SyntaxError:
                return False
        return not annotation or node.value is None or type(node.value) is int
    if isinstance(node, ast.Name):
        return node.id in constants or annotation and node.id in {'int', 'float', 'bool', 'str', 'tuple', 'list', 'dict'}
    if isinstance(node, ast.Attribute):
        return (isinstance(node.value, ast.Name) and node.value.id == 'tla' and
                node.attr in {'Tensor', 'Constexpr', 'Float16', 'Float32', 'BFloat16',
                              'Int8', 'Int16', 'Int32', 'Int64', 'UInt8', 'UInt16', 'UInt32', 'UInt64', 'Bool'})
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_definition_form(item, annotation=annotation, constants=constants) for item in node.elts)
    if isinstance(node, ast.Subscript):
        return annotation and _definition_form(node.value, annotation=True, constants=constants) and _definition_form(node.slice, annotation=True, constants=constants)
    return (not annotation and isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float))


def load_kernel(path, allowed_helpers=(), entrypoint='kernel', *, expected_sha256):
    tree = ast.parse(path.read_text())
    constants = {target.id for node in tree.body if isinstance(node, ast.Assign) and
                 ((isinstance(node.value, ast.Constant) and type(node.value.value) in (int, float)) or
                  isinstance(node.value, ast.Attribute) and _definition_form(node.value))
                 for target in node.targets if isinstance(target, ast.Name)}
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
            args = node.args
            annotations = [arg.annotation for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs,
                           *([args.vararg] if args.vararg else []), *([args.kwarg] if args.kwarg else [])]]
            definitions_safe = (not getattr(node, 'type_params', []) and
                all(_definition_form(value, annotation=True, constants=constants) for value in [*annotations, node.returns]) and
                all(_definition_form(value, constants=constants) for value in [*args.defaults, *args.kw_defaults]))
            approved_kernel = (len(node.decorator_list) == 1 and isinstance(node.decorator_list[0], ast.Attribute)
                and isinstance(node.decorator_list[0].value, ast.Name) and node.decorator_list[0].value.id == 'tla'
                and node.decorator_list[0].attr == 'kernel')
            if definitions_safe and ((node.name == entrypoint and approved_kernel) or
                    (node.name in allowed_helpers and node.name != entrypoint and not node.decorator_list)):
                continue
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            value = node.value
            if isinstance(value, ast.Constant) and type(value.value) in (int, float):
                continue
            if (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)
                    and value.value.id == 'tla' and value.attr in ('Float16', 'Float32', 'BFloat16')):
                continue
        raise ValueError('candidate module violates frozen Catlass ABI v1')
    module = load_candidate('rsi_catlass_candidate', path, expected_sha256)
    return getattr(module, entrypoint)


def check_runtime(profile):
    """Check selected source, installed consumer and actual imported bindings."""
    if sys.version_info[:2] != (3, 11):
        raise ValueError('Catlass target requires Python 3.11')
    configure_target_python_paths(profile.get('target_python_paths', []))
    if checked_git(profile['catlass_source'], 'rev-parse', 'HEAD') != profile['source_revision']:
        raise ValueError('Catlass source revision mismatch')
    checked_git(profile['catlass_source'], 'diff', '--exit-code', 'HEAD')
    link = checked_git(profile['catlass_source'], 'ls-tree', 'HEAD', profile['dependency_gitlink']).split()
    if len(link) < 3 or link[0] != '160000' or link[2] != profile['dependency_revision']:
        raise ValueError('AscendNPU-IR gitlink mismatch')
    version = Path(profile['cann_version_file']).read_text()
    if version != profile['cann_version_contents'] or not profile['cann_version'].startswith('9.3.'):
        raise ValueError('exact project CANN 9.3 identity mismatch')
    if sha(profile['pipeline_manifest']) != profile['pipeline_manifest_sha256']:
        raise ValueError('installed pipeline manifest mismatch')
    fingerprints = {str(Path(item['path']).resolve()): item['sha256']
                    for item in profile['compiler_consumer_files']}
    import catlass
    from catlass._mlir import ir
    from catlass.execution import resolve_bridge_extension_path, _find_hivmc_a5
    selected = {'compiler_bridge': resolve_bridge_extension_path(), 'hivmc-a5': _find_hivmc_a5()}
    for name, path in selected.items():
        expected = str(Path(profile['compiler_paths'][name]).resolve())
        if path is None or str(Path(path).resolve()) != expected:
            raise ValueError('runtime compiler path mismatch: '+name)
        if expected not in fingerprints or sha(expected) != fingerprints[expected]:
            raise ValueError('selected compiler fingerprint mismatch: '+name)
    if sha(catlass.__file__) != profile['module_sha256']['catlass']:
        raise ValueError('imported Catlass package identity mismatch')
    with ir.Context():
        pass
    return dict(source_revision=profile['source_revision'], dependency_revision=profile['dependency_revision'],
                cann_version=profile['cann_version'], catlass_module_sha256=sha(catlass.__file__))


def execute(args, contract, identity, directory, report, save):
    if contract['abi_version'] != ABI_VERSION:
        raise ValueError('unknown Catlass kernel ABI')
    physical = bind_device(args, identity, 'a5')
    if args.mode == 'performance':
        report['performance'] = 'unmeasured'
        raise ValueError('A5 performance requires a separately validated timing adapter')
    report['health'] = local_probe(physical_device=physical,
        configuration=identity['device_health'], remaining=lambda: None)
    save()
    if not report['health']['admitted']:
        raise ValueError('fresh bounded A5 health did not admit device')
    report['runtime_identity'] = check_runtime(identity)
    import torch
    import torch_npu
    import catlass.tla as tla
    from catlass.tla.runtime import from_dlpack
    for name, module in [('torch', torch), ('torch_npu', torch_npu)]:
        if str(module.__version__) != identity['expected_versions'][name] or sha(module.__file__) != identity['module_sha256'][name]:
            raise ValueError('imported runtime identity mismatch: '+name)
    torch.npu.set_device(0)
    kernel = load_kernel(Path(args.candidate), contract.get('allowed_helpers', ()), contract.get('entrypoint', 'kernel'), expected_sha256=args.source_sha256)
    os.environ['CATLASS_DSL_CACHE_DIR'] = str(directory/'cache')
    child_environment = os.environ.copy()
    paths = identity.get('target_python_paths', [])
    if paths:
        inherited = child_environment.get('PYTHONPATH')
        child_environment['PYTHONPATH'] = os.pathsep.join(paths + ([inherited] if inherited else []))
    subprocess.run([sys.executable, '-m', 'catlass.bc_compile'], check=True, env=child_environment)
    report.update(physical_device=physical, logical_device=0, cases=[])
    for index, case in enumerate(contract['cases']):
        if type(case['length']) is not int or not 1 <= case['length'] <= 512 or type(case['block_count']) is not int or case['block_count'] < 1:
            raise ValueError('case exceeds frozen ABI allocation or launch extent')
        left, right = case_inputs(case, contract['operation'])
        expected = reference(contract['operation'], left, right)
        padded = (len(left)+63)//64*64
        a = torch.tensor(left+[0.]*(padded-len(left)), dtype=torch.float32, device='npu')
        b = torch.tensor(right+[0.]*(padded-len(right)), dtype=torch.float32, device='npu')
        out = torch.tensor([float('nan')]*padded, dtype=torch.float32, device='npu')
        tensors = [from_dlpack(t, layout_tag=tla.arch.RowMajor).mark_compact_shape_dynamic(0) for t in (a, b, out)]
        report['compile'] = 'running'
        save()
        artifact = tla.compile(kernel, *tensors, options='--npu-arch 3510')
        report['compile'] = 'passed'
        observation = dict(case=case, compile='passed', launch='not-run', correctness='not-run')
        report['cases'].append(observation)
        save()
        if args.mode == 'compile':
            continue
        snapshots = [a.clone(), b.clone()]
        report['launch'] = observation['launch'] = 'running'
        save()
        artifact(*tensors, block_num=case['block_count'])
        torch.npu.synchronize()
        report.update(launch='passed', device_evidence=True, correctness='running')
        observation.update(launch='passed', correctness='running')
        values = out[:len(left)].cpu().tolist()
        observation.update(actual=[x if math.isfinite(x) else repr(x) for x in values], expected=expected)
        torch.save(dict(left=left, right=right, observed_left=a.cpu(), observed_right=b.cpu(), actual=out.cpu(), expected=expected), directory/('case-'+str(index)+'.pt'))
        save()
        try:
            if not torch.equal(a, snapshots[0]) or not torch.equal(b, snapshots[1]):
                raise ValueError('candidate mutated an input')
            compare(values, expected)
        except ValueError:
            observation['correctness'] = 'failed'
            save()
            raise
        observation['correctness'] = 'passed'
        save()
    if args.mode != 'compile':
        report['correctness'] = 'passed'
