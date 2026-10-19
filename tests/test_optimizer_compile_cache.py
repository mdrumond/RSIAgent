"""Exercise real cache orchestration with a small fake compiler, not an NPU."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from optimizer import native


@pytest.fixture
def compiler(tmp_path, monkeypatch):
    import torch
    from benchmarks.a3kernels.ascendc_runtime import host_driver
    executable = tmp_path / 'bisheng'
    executable.write_text('fake compiler')
    monkeypatch.setattr(native.shutil, 'which', lambda name: str(executable))
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace(__version__='test-npu'))
    monkeypatch.setattr(host_driver, '_toolchain_paths', lambda *args: ([], []))
    monkeypatch.setattr(native.subprocess, 'check_output', lambda *args, **kwargs: 'bisheng test-v1')
    calls = []
    loads = []
    def run(argv, **kwargs):
        calls.append(argv)
        Path(argv[argv.index('-o') + 1]).write_bytes(b'compiled binary')
    monkeypatch.setattr(native.subprocess, 'run', run)
    monkeypatch.setattr(torch.ops, 'load_library', lambda path: loads.append(path))
    class Ops:
        def __getattr__(self, name):
            if name == 'load_library':
                return lambda path: loads.append(path)
            return SimpleNamespace(launch=lambda inputs, states: [inputs[0] + 1])
    monkeypatch.setattr(torch, 'ops', Ops())
    monkeypatch.setattr(native, '_LOADED', {})
    return calls, loads


def contract():
    tensor = {'shape': [4], 'strides': [1], 'dtype': 'torch.float32', 'device': 'cpu'}
    return {'inputs': [tensor], 'outputs': [{'path': 'output', **tensor}], 'state': [],
            'scalars': {'arg1': 2}, 'argument_order': [0, 1]}


def build(tmp_path, label, **kwargs):
    return native._compile_case(kwargs.pop('source', 'void optimizer_kernel() {}'),
        kwargs.pop('contract', contract()), tmp_path / label, arch=kwargs.pop('arch', 'dav-2201'),
        block_count=kwargs.pop('block_count', 1), cache_directory=tmp_path / 'cache', **kwargs)


def test_compile_evaluate_profile_reuse_one_binary_and_registration(tmp_path, compiler):
    records = [build(tmp_path, name)[1] for name in ('compile', 'evaluate', 'profile')]
    assert len(compiler[0]) == len(compiler[1]) == 1
    assert [row['cache_hit'] for row in records] == [False, True, True]
    assert len({row['binary_sha256'] for row in records}) == 1
    assert len({row['wrapper_sha256'] for row in records}) == 1
    assert len({row['namespace'] for row in records}) == 1
    assert (tmp_path / 'evaluate' / 'kernel.cpp').read_bytes() == (tmp_path / 'compile' / 'kernel.cpp').read_bytes()


def test_public_compile_evaluate_profile_share_cache(tmp_path, monkeypatch, compiler):
    from optimizer.reference import prepare_reference
    from optimizer import profiles
    prepared = prepare_reference({'source': 'def forward(x): return x + 1', 'entrypoint': 'forward',
        'signatures': [{'args': [{'shape': [4], 'dtype': 'float32'}]}]}, max_cases=1)
    monkeypatch.setattr(profiles, 'profile', lambda fn, *args, **kwargs: {'status': 'available'})
    common = {'device': 'cpu', 'arch': 'test', 'cache_directory': tmp_path / 'cache'}
    compiled = native.compile_candidate(prepared, 'source', tmp_path / 'compile', **common)
    evaluated = native.evaluate(prepared, 'source', tmp_path / 'evaluate', **common)
    profiled = native.evaluate(prepared, 'source', tmp_path / 'profile', profile=True, **common)
    assert compiled['status'] == 'compiled'
    assert evaluated['status'] == profiled['status'] == 'verified'
    assert [r['cases'][0]['compilation']['cache_hit'] for r in (compiled, evaluated, profiled)] == [False, True, True]
    assert len(compiler[0]) == len(compiler[1]) == 1


@pytest.mark.parametrize('field,value', [('source', 'different source'), ('arch', 'dav-3510'),
    ('block_count', 2), ('shape', [8]), ('strides', [2]), ('dtype', 'torch.float16'),
    ('scalars', {'arg1': 3}), ('state', [{'name': 'weight', 'shape': [4], 'strides': [1], 'dtype': 'torch.float32'}]),
    ('outputs', [{'path': 'output', 'shape': [8], 'strides': [1], 'dtype': 'torch.float32'}])])
def test_contract_source_arch_and_launch_invalidate(tmp_path, compiler, field, value):
    original = build(tmp_path, 'first')[1]
    args = {}
    spec = copy.deepcopy(contract())
    if field in ('source', 'arch', 'block_count'):
        args[field] = value
    elif field in ('shape', 'strides', 'dtype'):
        spec['inputs'][0][field] = value
    else:
        spec[field] = value
    changed = build(tmp_path, 'changed', contract=spec, **args)[1]
    assert len(compiler[0]) == 2
    assert original['cache_key'] != changed['cache_key']


def test_observed_runtime_invalidates(tmp_path, monkeypatch, compiler):
    first = build(tmp_path, 'first')[1]
    monkeypatch.setenv('ASCEND_HOME_PATH', '/different/runtime')
    changed = build(tmp_path, 'changed')[1]
    assert first['cache_key'] != changed['cache_key']
    assert len(compiler[0]) == 2


def test_observed_compiler_and_torch_versions_invalidate(tmp_path, monkeypatch, compiler):
    first = build(tmp_path, 'first')[1]
    monkeypatch.setattr(native.subprocess, 'check_output', lambda *args, **kwargs: 'bisheng test-v2')
    second = build(tmp_path, 'second')[1]
    monkeypatch.setattr(sys.modules['torch_npu'], '__version__', 'new-npu')
    third = build(tmp_path, 'third')[1]
    assert len({row['cache_key'] for row in (first, second, third)}) == 3
    assert len(compiler[0]) == 3


@pytest.mark.parametrize('name', ['kernel.cpp', 'kernel.so', 'compiled.json'])
def test_corrupt_cache_recompiles_before_loading(tmp_path, compiler, name):
    first = build(tmp_path, 'first')[1]
    (tmp_path / 'cache' / first['cache_key'] / name).write_bytes(b'corrupt')
    second = build(tmp_path, 'second')[1]
    assert not second['cache_hit']
    assert len(compiler[0]) == 2


def test_failed_compile_not_published(tmp_path, monkeypatch, compiler):
    original = native.subprocess.run
    def fail(*args, **kwargs):
        kwargs['stdout'].write('compiler diagnostic')
        raise RuntimeError('compiler failed')
    monkeypatch.setattr(native.subprocess, 'run', fail)
    with pytest.raises(RuntimeError, match='compiler failed'):
        build(tmp_path, 'failed')
    assert 'diagnostic' in (tmp_path / 'failed' / 'compile.log').read_text()
    for name in ('kernel.cpp', 'build.json', 'command.json'):
        assert (tmp_path / 'failed' / name).is_file()
    assert not (tmp_path / 'failed' / 'compiled.json').exists()
    assert not list((tmp_path / 'cache').glob('*/compiled.json'))
    monkeypatch.setattr(native.subprocess, 'run', original)
    assert not build(tmp_path, 'retry')[1]['cache_hit']


def test_bounded_eviction_keeps_current_entry(tmp_path, monkeypatch, compiler):
    monkeypatch.setattr(native, '_CACHE_BYTES', 1)
    first = build(tmp_path, 'first')[1]
    second = build(tmp_path, 'second', source='different source')[1]
    assert not (tmp_path / 'cache' / first['cache_key']).exists()
    assert (tmp_path / 'cache' / second['cache_key']).is_dir()
