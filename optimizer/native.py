"""Host-owned Ascend C evaluation; invoke only inside an isolated worker.

The caller selects the runtime/device and bounds this process. Source is trusted
worker code, not sandboxed. A candidate never chooses inputs or comparison policy.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import uuid

from .cases import digest
from .reference import (POLICY, PreparedReference, ReferenceError, _input_state,
                        _leaves, _model_state, _tensor_identity, compare_outputs,
                        prepare_reference)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def load_reference(directory, device):
    """Authenticate a trusted frozen bundle before torch deserialization."""
    import torch
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['policy'] != POLICY:
        raise ValueError('baseline comparison policy differs')
    if _sha((root / 'reference.py').read_bytes()) != manifest['source_sha256']:
        raise ValueError('baseline source digest differs')
    if digest(manifest['cases']) != manifest['suite_sha256']:
        raise ValueError('baseline suite digest differs')
    payload = (root / 'baseline.pt').read_bytes()
    if _sha(payload) != manifest['baseline_sha256']:
        raise ValueError('baseline tensor digest differs')
    import io
    frozen = torch.load(io.BytesIO(payload), map_location=device, weights_only=False)
    spec = manifest['submission']
    if _sha(spec['source'].encode()) != manifest['source_sha256']:
        raise ValueError('baseline submission source differs')
    namespace = {'__name__': 'optimizer_frozen_reference', 'torch': torch}
    # Initialization is reproducible; retained weights replace initialized state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(manifest['seed'])
        exec(compile(spec['source'], '<frozen-reference>', 'exec'), namespace)
        model = namespace[spec['entrypoint']]
        if isinstance(model, type):
            model = model(*spec.get('init_args', []), **spec.get('init_kwargs', {})).to(device).eval()
            model.load_state_dict(frozen['state_dict'], strict=True)
            model.requires_grad_(False)
    prepared = PreparedReference(spec, model, frozen['cases'], manifest)
    if _model_state(model) != manifest['state']:
        raise ValueError('baseline model state differs')
    if len(prepared.cases) != len(manifest['cases']):
        raise ValueError('baseline case count differs')
    for row, expected in zip(prepared.cases, manifest['cases']):
        if row['case'] != {k: expected[k] for k in row['case']}:
            raise ValueError('baseline case identity differs')
        if _input_state(row['args'], row['kwargs']) != expected['inputs']:
            raise ValueError('baseline inputs differ')
        outputs = {p: _tensor_identity(x) if x is not None else None for p, x in _leaves(row['output'])}
        if outputs != expected['outputs']:
            raise ValueError('baseline outputs differ')
    return prepared


def _inputs(row):
    return list(row['args']) + list(row['kwargs'].values())


def _state(prepared):
    import torch
    return list(prepared.model.state_dict().items()) if isinstance(prepared.model, torch.nn.Module) else []


def _copy_state(prepared):
    import torch
    # clone() compacts padded views, invalidating the advertised STATE_STRIDE ABI.
    return [torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device).copy_(x)
            for _, x in _state(prepared)]


def describe_case(prepared, row):
    """Ordered ABI: tensor inputs, module-state tensors, then tensor outputs.

    Scalars are compile-time constants ARG0... in positional/keyword order.
    Tensor indexes omit scalars. Outputs flatten depth-first in container order.
    """
    import torch
    values = _inputs(row)
    metadata = lambda x: {k: v for k, v in _tensor_identity(x).items() if k != 'sha256'}
    return {'inputs': [metadata(x) for x in values if isinstance(x, torch.Tensor)],
            'scalars': {f'arg{i}': x for i, x in enumerate(values) if not isinstance(x, torch.Tensor)},
            'argument_order': list(range(len(row['args']))) + list(row['kwargs']),
            'state': [{'name': name, **metadata(x)} for name, x in _state(prepared)],
            'outputs': [{'path': path, **metadata(x)} for path, x in _leaves(row['output']) if x is not None]}


def render_wrapper(candidate, contract, namespace, block_count):
    """Render case-specialized constants and a host-owned Torch dispatcher op."""
    if not namespace.isidentifier() or not isinstance(block_count, int) or block_count < 1:
        raise ValueError('invalid namespace/block count')
    constants = []
    for family in ('inputs', 'state', 'outputs'):
        prefix = {'inputs': 'INPUT', 'state': 'STATE', 'outputs': 'OUTPUT'}[family]
        for i, item in enumerate(contract[family]):
            name = f'{prefix}{i}'
            constants.append(f'constexpr int64_t {name}_NUMEL = {math.prod(item["shape"])};')
            constants.append(f'constexpr int64_t {name}_RANK = {len(item["shape"])};')
            for field in ('shape', 'strides'):
                for j, value in enumerate(item[field]):
                    constants.append(f'constexpr int64_t {name}_{"SIZE" if field == "shape" else "STRIDE"}{j} = {value};')
    for name, value in contract['scalars'].items():
        if value is None:
            constants.append(f'constexpr auto {name.upper()} = nullptr;')
        elif isinstance(value, (int, float, bool)) and (not isinstance(value, float) or math.isfinite(value)):
            constants.append(f'constexpr auto {name.upper()} = {json.dumps(value)};')
        else:
            raise ValueError('native scalar specialization supports finite numeric/bool/None values')
    dtype_names = {'float32': 'Float', 'float64': 'Double', 'float16': 'Half', 'bfloat16': 'BFloat16',
                   'int64': 'Long', 'int32': 'Int', 'int16': 'Short', 'int8': 'Char', 'uint8': 'Byte', 'bool': 'Bool'}
    allocations = []
    for i, item in enumerate(contract['outputs']):
        shape = '{' + ','.join(map(str, item['shape'])) + '}'
        strides = '{' + ','.join(map(str, item['strides'])) + '}'
        dtype = dtype_names[item['dtype'].removeprefix('torch.')]
        allocations.append(f'auto output{i} = torch::empty_strided({shape}, {strides}, options.dtype(torch::k{dtype}));')
    pointers = ([f'(GM_ADDR)owned_inputs[{i}].data_ptr()' for i in range(len(contract['inputs']))] +
                [f'(GM_ADDR)owned_state[{i}].data_ptr()' for i in range(len(contract['state']))] +
                [f'(GM_ADDR)output{i}.data_ptr()' for i in range(len(contract['outputs']))])
    outputs = ', '.join(f'output{i}' for i in range(len(contract['outputs'])))
    return '\n'.join(['#include "kernel_operator.h"', '#include <torch/all.h>', '#include <torch/library.h>',
        '#include "torch_npu/csrc/core/npu/NPUStream.h"', '#include "torch_npu/csrc/framework/OpCommand.h"',
        *constants, f'#define optimizer_kernel {namespace}_kernel', candidate,
        f'namespace {namespace}_host {{',
        'std::vector<torch::Tensor> launch(at::TensorList inputs, at::TensorList state) {',
        'std::vector<torch::Tensor> owned_inputs(inputs.begin(), inputs.end());',
        'std::vector<torch::Tensor> owned_state(state.begin(), state.end());',
        'auto options = torch::TensorOptions().device(c10::Device(c10::DeviceType::PrivateUse1, c10_npu::getCurrentNPUStream().device_index()));',
        *allocations, 'auto stream = c10_npu::getCurrentNPUStream().stream(false);',
        'auto run = [=]() -> int {', f'::optimizer_kernel<<<{block_count}, nullptr, stream>>>({", ".join(pointers)});',
        'return 0; };', f'at_npu::native::OpCommand::RunOpApi("{namespace}", run);',
        f'return {{{outputs}}};', '}', '}', f'TORCH_LIBRARY({namespace}, m) {{ m.def("launch(Tensor[] inputs, Tensor[] state) -> Tensor[]"); }}',
        f'TORCH_LIBRARY_IMPL({namespace}, PrivateUse1, m) {{ m.impl("launch", {namespace}_host::launch); }}'])


def _compile_case(source, contract, directory, *, arch, block_count):
    import torch
    import torch_npu
    import shutil
    import subprocess
    from benchmarks.a3kernels.ascendc_runtime.host_driver import _toolchain_paths
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    namespace = 'optimizer_' + uuid.uuid4().hex
    (root / 'kernel.cpp').write_text(render_wrapper(source, contract, namespace, block_count))
    spec = {'compiler': 'bisheng', 'arch': arch, 'source': 'kernel.cpp', 'output': 'kernel.so'}
    (root / 'build.json').write_text(json.dumps(spec))
    compiler = shutil.which('bisheng')
    if not compiler:
        raise RuntimeError('bisheng compiler not found on PATH')
    includes, libraries = _toolchain_paths(torch, torch_npu)
    library = root / 'kernel.so'
    argv = [compiler, '-x', 'asc', f'--npu-arch={arch}', '-shared', '-fPIC', '-std=c++17',
            f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}',
            '-ltorch_npu', '-ltorch', '-lc10', str((root / 'kernel.cpp').resolve()),
            '-o', str(library.resolve()), *(f'-I{x}' for x in includes), *(f'-L{x}' for x in libraries)]
    (root / 'command.json').write_text(json.dumps(argv))
    with (root / 'compile.log').open('w') as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not library.is_file():
        raise RuntimeError('compiler produced no library')
    torch.ops.load_library(str(library.resolve()))
    return getattr(torch.ops, namespace).launch, {'binary_sha256': _sha(library.read_bytes()),
            'wrapper_sha256': _sha((root / 'kernel.cpp').read_bytes()), 'build': spec, 'namespace': namespace}


def _rebuild(template, leaves):
    import torch
    if isinstance(template, torch.Tensor):
        return next(leaves)
    if template is None:
        return None
    if isinstance(template, dict):
        return {k: _rebuild(v, leaves) for k, v in template.items()}
    return type(template)(_rebuild(v, leaves) for v in template)


def _benchmark(*args, **kwargs):
    from .measurement import benchmark
    return benchmark(*args, **kwargs)


def evaluate(reference, candidate, directory, *, device='npu:0', arch, block_count=1,
             benchmark=False, profile=False):
    """Evaluate exactly a prepared/frozen suite; timing starts after all cases pass.

    One process per candidate is required: device crashes cannot be recovered here.
    No generation occurs in this function. Candidate is Ascend C device source.
    """
    import torch
    if device.startswith('npu'):
        import torch_npu  # noqa: F401; register NPU backend before deserialization
        torch.npu.set_device(device)
    prepared = load_reference(reference, device) if isinstance(reference, (str, Path)) else reference
    if not isinstance(prepared, PreparedReference) or not prepared.cases:
        raise ValueError('a nonempty prepared/frozen reference is required')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'candidate.cpp').write_text(candidate)
    result = {'status': 'verified', 'candidate_sha256': _sha(candidate.encode()),
              'suite_sha256': prepared.manifest['suite_sha256'], 'arch': arch,
              'device': device, 'block_count': block_count, 'cases': []}
    runners = []
    for i, row in enumerate(prepared.cases):
        record = {'case': row['case']['id'], 'contract': describe_case(prepared, row)}
        result['cases'].append(record)
        try:
            run, provenance = _compile_case(candidate, record['contract'], root / f'case-{i}', arch=arch, block_count=block_count)
            record['compilation'] = provenance
            def fresh(row=row, run=run):
                args, kwargs = copy.deepcopy((row['args'], row['kwargs']))
                tensors = [x for x in list(args) + list(kwargs.values()) if isinstance(x, torch.Tensor)]
                states = _copy_state(prepared)
                before = _input_state(args, kwargs), [_tensor_identity(x) for x in states]
                output = list(run(tensors, states))
                if device.startswith('npu'):
                    torch.npu.synchronize()
                if before != (_input_state(args, kwargs), [_tensor_identity(x) for x in states]):
                    raise ReferenceError('candidate mutates inputs or frozen state')
                if len(output) != len(record['contract']['outputs']):
                    raise ReferenceError('candidate output count differs')
                return _rebuild(row['output'], iter(output))
            actual = fresh()
            record['correctness'] = compare_outputs(row['output'], actual, inputs=_inputs(row))
            runners.append((row, run))
        except Exception as exc:
            result['status'] = 'incorrect'
            record['error'] = f'{type(exc).__name__}: {exc}'
            log = root / f'case-{i}' / 'compile.log'
            if log.is_file():
                # The complete log remains an artifact; include bounded diagnostics
                # in actor-visible results so compiler failures can be repaired.
                record['compiler_diagnostics'] = log.read_text(errors='replace')[-24000:]
    if result['status'] == 'verified':
        for record, (row, run) in zip(result['cases'], runners):
            # All input copies happen before capture; verified kernels cannot mutate them.
            args, kwargs = copy.deepcopy((row['args'], row['kwargs']))
            tensors = [x for x in list(args) + list(kwargs.values()) if isinstance(x, torch.Tensor)]
            states = _copy_state(prepared)
            functions = {'torch': lambda args=args, kwargs=kwargs: prepared.invoke(*args, **kwargs),
                         'ascendc': lambda run=run, tensors=tensors, states=states: run(tensors, states)}
            identity = {k: result[k] for k in ('candidate_sha256', 'suite_sha256', 'arch', 'device')}
            identity['case'] = record['case']
            for name, fn in functions.items():
                if benchmark:
                    try:
                        timing = _benchmark(fn, root / f'{record["case"]}-{name}-timing', identity=identity)
                    except Exception as exc:
                        timing = {'status': 'unavailable', 'latency_us': None,
                                  'error': f'{type(exc).__name__}: {exc}'}
                    record.setdefault('timing', {})[name] = timing
                if profile:
                    from .profiles import profile as capture
                    try:
                        diagnostic = capture(fn, root / f'{record["case"]}-{name}-profile', identity=identity)
                    except Exception as exc:
                        diagnostic = {'status': 'unavailable', 'error': f'{type(exc).__name__}: {exc}'}
                    record.setdefault('profiling', {})[name] = diagnostic
            if benchmark:
                a, b = (record['timing'][key]['latency_us'] for key in ('torch', 'ascendc'))
                record['speedup'] = a / b if a is not None and b is not None and b > 0 else None
    (root / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
