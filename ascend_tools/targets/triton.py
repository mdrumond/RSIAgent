"""A3 oracle and observed launches extracted from 13b978b."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from contextlib import contextmanager

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


def execute(args, contract, identity, directory, report, save):
    """Observe real JIT returns; candidate scores and metadata are never consumed."""
    import os
    from .health import local_probe
    from .runner import sha
    physical = identity['physical_device']
    if type(physical) is not int or physical < 0 or identity['device_health']['product'] != 'a3':
        raise ValueError('explicit A3 physical-device identity required')
    existing = os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    if existing is not None and existing != str(physical):
        raise ValueError('existing physical-device mask mismatch')
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(physical)
    os.environ['TRITON_CACHE_DIR'] = str(directory/'cache')
    health = dict(identity['device_health'], require_idle=args.mode == 'performance')
    report['health'] = local_probe(physical_device=physical, configuration=health, remaining=lambda: None)
    save()
    if not report['health']['admitted']:
        raise ValueError('fresh bounded device health did not admit device')
    import torch
    import torch_npu
    import triton
    from triton.runtime.jit import JITFunction
    for name, module in [('torch', torch), ('torch_npu', torch_npu), ('triton', triton)]:
        if str(module.__version__) != identity['expected_versions'][name]:
            raise ValueError('imported runtime version mismatch: '+name)
        if sha(module.__file__) != identity['module_sha256'][name]:
            raise ValueError('imported runtime module mismatch: '+name)
    torch.npu.set_device('npu:0')
    active = triton.runtime.driver.active.get_current_target()
    observed = {name: getattr(active, name) for name in ('backend', 'arch', 'warp_size') if hasattr(active, name)}
    if not observed or observed != identity['expected_compiler_target']:
        raise ValueError('active compiler target mismatch')
    report.update(compiler_target=observed, runtime_device_name=torch.npu.get_device_name(0),
                  physical_device=physical, logical_device=0, cases=[])
    candidate = load_module('_trusted_triton_candidate', args.candidate).ModelNew().eval()
    with torch.no_grad(), launch_capture(JITFunction, directory) as launches:
        for index, case in enumerate(contract['cases']):
            inputs, expected = make_case(torch, contract['operation'], case)
            device_inputs = [value.to('npu:0', memory_format=torch.preserve_format) for value in inputs]
            snapshots = [value.clone() for value in device_inputs]
            torch.npu.synchronize()
            before = len(launches)
            report['compile'] = 'running'
            save()
            output = candidate(*device_inputs)
            # Compile receipt is recorded before synchronization or numerical failure.
            if len(launches) == before:
                raise ValueError('no actual Triton JIT dispatch observed')
            report['compile'] = 'passed'
            save()
            report['launch'] = 'running'
            save()
            torch.npu.synchronize()
            report.update(launch='passed', device_evidence=True)
            save()
            if not isinstance(output, torch.Tensor) or output.device.type != 'npu':
                raise ValueError('candidate must return NPU tensor')
            actual = output.detach().cpu()
            torch.save(dict(inputs=inputs, expected=expected, actual=actual), directory/('case-'+str(index)+'.pt'))
            if args.mode == 'compile':
                continue
            report['correctness'] = 'running'
            save()
            check_output_layout(output, contract['operation'])
            if any(not torch.equal(a, b) for a, b in zip(snapshots, device_inputs)):
                raise ValueError('candidate mutated input')
            torch.testing.assert_close(actual, expected, atol=case['atol'], rtol=case['rtol'], equal_nan=False)
            report['cases'].append(dict(case_id=case['case_id'], correctness='passed', launches=launches[before:]))
            save()
    if args.mode != 'compile':
        report['correctness'] = 'passed'
    if args.mode == 'performance':
        # Performance is deliberately not inferred from wall time including compilation.
        report['performance'] = 'unmeasured'
        raise ValueError('device performance measurement requires a separately validated timing adapter')
