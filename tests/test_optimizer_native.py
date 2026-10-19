import json

import pytest
torch = pytest.importorskip('torch')

from optimizer.reference import prepare_reference, freeze_reference
from optimizer import native


def reference():
    return prepare_reference({'source': 'def kernel(x, scale): return {"a": x * scale, "b": (x + 1, None)}',
                              'entrypoint': 'kernel', 'signatures': [{'args': [
                                  {'shape': [3, 2], 'dtype': 'float32', 'strides': [1, 3]},
                                  {'scalar': 2}]}]}, max_cases=1)


def test_wrapper_preserves_layout_scalars_nested_outputs_and_namespace():
    prepared = reference()
    contract = native.describe_case(prepared, prepared.cases[0])
    assert contract['inputs'][0]['strides'] == [1, 3]
    assert contract['scalars'] == {'arg1': 2}
    assert len(contract['outputs']) == 2
    source = native.render_wrapper('extern "C" __global__ __aicore__ void optimizer_kernel(GM_ADDR input0, GM_ADDR output0, GM_ADDR output1) {}', contract, 'optimizer_abc', 1)
    assert 'INPUT0_STRIDE0 = 1' in source
    assert 'ARG1 = 2' in source
    assert 'std::vector<torch::Tensor> owned_inputs(inputs.begin(), inputs.end())' in source
    assert 'owned_inputs[0].data_ptr()' in source
    assert 'TORCH_LIBRARY(optimizer_abc' in source
    assert '#define optimizer_kernel optimizer_abc_kernel' in source
    assert 'empty_strided' in source


def test_frozen_bundle_authenticates_and_never_regenerates(tmp_path, monkeypatch):
    prepared = reference()
    freeze_reference(prepared, tmp_path)
    monkeypatch.setattr(native, 'prepare_reference', lambda *a, **k: pytest.fail('regenerated'))
    loaded = native.load_reference(tmp_path, 'cpu')
    assert loaded.manifest['suite_sha256'] == prepared.manifest['suite_sha256']
    (tmp_path / 'baseline.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='baseline'):
        native.load_reference(tmp_path, 'cpu')


def test_all_cases_pass_before_any_timing(tmp_path, monkeypatch):
    prepared = reference()
    prepared = prepare_reference({**prepared.spec, 'signatures': prepared.spec['signatures'] * 2}, max_cases=2)
    events = []
    def compile_case(source, contract, directory, **kwargs):
        index = len(events)
        events.append('compile')
        def run(inputs, states):
            x = inputs[0]
            return [x * 2 if index == 0 else x * 0, x + 1]
        return run, {'binary_sha256': 'fake'}
    monkeypatch.setattr(native, '_compile_case', compile_case)
    monkeypatch.setattr(native, '_benchmark', lambda *a, **k: pytest.fail('timed failed suite'))
    result = native.evaluate(prepared, 'source', tmp_path, device='cpu', arch='test', benchmark=True)
    assert result['status'] == 'incorrect'
    assert len(result['cases']) == 2
    assert not any('timing' in row for row in result['cases'])


def test_module_state_and_fresh_inputs(tmp_path, monkeypatch):
    spec = {'source': 'class Model(torch.nn.Module):\n def __init__(self):\n  super().__init__()\n  self.register_buffer("weight", torch.ones(3))\n def forward(self, x): return x + self.weight',
            'entrypoint': 'Model', 'signatures': [{'args': [{'shape': [3], 'dtype': 'float32'}]}]}
    prepared = prepare_reference(spec, max_cases=1)
    assert native.describe_case(prepared, prepared.cases[0])['state'][0]['name'] == 'weight'
    def compile_case(source, contract, directory, **kwargs):
        def run(inputs, states):
            out = inputs[0] + states[0]
            inputs[0].zero_()
            return [out]
        return run, {}
    monkeypatch.setattr(native, '_compile_case', compile_case)
    result = native.evaluate(prepared, 'source', tmp_path, device='cpu', arch='test')
    assert result['status'] == 'incorrect'
    assert 'mutat' in result['cases'][0]['error']


def test_success_nested_outputs_timing_and_profile_failure(tmp_path, monkeypatch):
    prepared = reference()
    events = []
    def compile_case(source, contract, directory, **kwargs):
        def run(inputs, states):
            events.append('run')
            return [inputs[0] * 2, inputs[0] + 1]
        return run, {'binary_sha256': 'fake'}
    def benchmark(fn, directory, **kwargs):
        assert events == ['run']
        return {'latency_us': 2 if 'ascendc' in str(directory) else 4}
    monkeypatch.setattr(native, '_compile_case', compile_case)
    monkeypatch.setattr(native, '_benchmark', benchmark)
    from optimizer import profiles
    monkeypatch.setattr(profiles, 'profile', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('unsupported')))
    result = native.evaluate(prepared, 'source', tmp_path, device='cpu', arch='test', benchmark=True, profile=True)
    assert result['status'] == 'verified'
    assert result['cases'][0]['speedup'] == 2
    assert result['cases'][0]['profiling']['torch']['status'] == 'unavailable'
    assert json.loads((tmp_path / 'result.json').read_text()) == result


def test_compile_failure_retained_without_timing(tmp_path, monkeypatch):
    monkeypatch.setattr(native, '_compile_case', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('compile failed')))
    result = native.evaluate(reference(), 'bad', tmp_path, device='cpu', arch='test', benchmark=True)
    assert result['status'] == 'incorrect'
    assert 'compile failed' in result['cases'][0]['error']


def test_compiler_diagnostics_are_returned_for_agent_repair(tmp_path, monkeypatch):
    import subprocess
    def fail(source, contract, directory, **kwargs):
        directory.mkdir()
        (directory / 'compile.log').write_text('kernel.cpp:7: error: unknown identifier MissingApi\n')
        raise subprocess.CalledProcessError(1, ['bisheng', 'kernel.cpp'])
    monkeypatch.setattr(native, '_compile_case', fail)
    result = native.evaluate(reference(), 'bad', tmp_path, device='cpu', arch='test')
    assert 'unknown identifier MissingApi' in result['cases'][0]['compiler_diagnostics']


def test_padded_module_state_layout_survives_verification_and_timing(tmp_path, monkeypatch):
    spec = {'source': 'class Model(torch.nn.Module):\n def __init__(self):\n  super().__init__()\n  self.register_buffer("weight", torch.arange(8.).reshape(2, 4)[:, :3])\n def forward(self, x): return x + self.weight',
            'entrypoint': 'Model', 'signatures': [{'args': [{'shape': [2, 3], 'dtype': 'float32'}]}]}
    prepared = prepare_reference(spec, max_cases=1)
    calls = []

    def compile_case(source, contract, directory, **kwargs):
        def run(inputs, states):
            assert list(states[0].stride()) == contract['state'][0]['strides'] == [4, 1]
            calls.append('run')
            return [inputs[0] + states[0]]
        return run, {}

    def benchmark(fn, directory, **kwargs):
        fn()
        return {'latency_us': 1}

    monkeypatch.setattr(native, '_compile_case', compile_case)
    monkeypatch.setattr(native, '_benchmark', benchmark)
    result = native.evaluate(prepared, 'source', tmp_path, device='cpu', arch='test', benchmark=True)
    assert result['status'] == 'verified', result
    assert calls == ['run', 'run']  # verification and timed Ascend C callable


@pytest.mark.parametrize('mutation', ['case', 'input', 'output', 'manifest'])
def test_mutated_prepared_reference_is_rejected_before_compilation(tmp_path, monkeypatch, mutation):
    prepared = reference()
    if mutation == 'case':
        prepared.cases[0]['case']['seed'] += 1
    elif mutation == 'input':
        prepared.cases[0]['args'][0].add_(1)
    elif mutation == 'output':
        prepared.cases[0]['output']['a'].add_(1)
    else:
        prepared.manifest['cases'][0]['inputs']['args'][0]['sha256'] = 'modified'
    monkeypatch.setattr(native, '_compile_case', lambda *a, **k: pytest.fail('compiled mutated suite'))
    with pytest.raises(ValueError, match='baseline'):
        native.evaluate(prepared, 'source', tmp_path, device='cpu', arch='test')
