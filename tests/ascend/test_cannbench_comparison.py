import copy
from unittest.mock import Mock
import pytest

torch = pytest.importorskip('torch')
from ascend_tools.baselines import pack
from ascend_tools.baselines.comparison import finite_comparison, native_reference
from ascend_tools.baselines.inputs import call, inputs
from ascend_tools.baselines import qualification as q
from ascend_tools.baselines import project_verifier as verifier

WORKLOAD = 'lact_ttt_chunk'


def region_result(region, errors, cpu_errors, size=5):
    value = {'small': 1e-6, 'cancel': .05, 'normal': 1.}[region]
    oracle = torch.full((size,), value, dtype=torch.float64)
    actual = oracle.to(torch.bfloat16)
    native = actual.clone()
    bad = 4e-5 if region == 'small' else 0.
    actual[:errors] = bad
    native[:cpu_errors] = bad
    return finite_comparison(actual, oracle, .02, native)


@pytest.mark.parametrize('region', ['small', 'cancel'])
def test_special_region_zero_cpu_and_two_to_one_count_boundaries(region):
    assert region_result(region, 2, 0)['passed']
    assert not region_result(region, 3, 0)['passed']
    assert region_result(region, 4, 2)['passed']
    assert not region_result(region, 5, 2)['passed']


def test_clean_normal_reference_never_licenses_candidate_normal_errors():
    result = region_result('normal', 1, 0)
    assert not result['passed']
    assert result['normalErrors'] == 1 and result['normalCpuErrors'] == 0
    assert region_result('normal', 4, 2)['passed']
    assert not region_result('normal', 5, 2)['passed']


@pytest.mark.parametrize('golden,actual,region', [
    (2**-8, 0., 'cancel'), (.125, 0., 'normal'), (.01, .125, 'normal')])
def test_bfloat16_region_boundaries_are_strict(golden, actual, region):
    reference = torch.full((3,), golden, dtype=torch.float64)
    output = torch.full((3,), actual, dtype=torch.bfloat16)
    result = finite_comparison(output, reference, .02, reference.to(output.dtype))
    assert not result['passed']
    assert result[region+'Errors'] == 3
    assert result['smallErrors'] == 0


def test_oracle_is_rounded_to_output_dtype_before_relative_comparison():
    actual = torch.tensor([1.0078125], dtype=torch.bfloat16)
    oracle = torch.tensor([1.005], dtype=torch.float64)
    result = finite_comparison(actual, oracle, 1e-7)
    assert result['passed'] and result['mere'] == result['mare'] == 0.


def test_first_screen_and_mean_only_failure_follow_pinned_verdict_order():
    oracle = torch.full((3,), .003, dtype=torch.float64)
    actual = torch.full((3,), .00304, dtype=torch.bfloat16)
    # First-screen success precedes special-region error counting.
    assert finite_comparison(actual, oracle, .02, oracle.to(actual.dtype))['passed']
    result = finite_comparison(torch.tensor([1.005], dtype=torch.float32),
                               torch.tensor([1.], dtype=torch.float64), .001)
    # The pinned second stage uses error counts, even for mean-only failures.
    assert result['mere'] > .001 and result['normalErrors'] == 0 and result['passed']


@pytest.mark.parametrize('invalid', ['nan', 'shape'])
def test_invalid_native_reference_remains_a_hard_failure(invalid):
    value = torch.ones(3)
    native = torch.full((3,), float('nan')) if invalid == 'nan' else torch.ones(2)
    with pytest.raises(ValueError, match='Invalid finite comparison'):
        finite_comparison(value, value.double(), .001, native)


def small_inputs(_workload, case):
    _, _, cases = pack(WORKLOAD)
    small = copy.deepcopy(cases[0])
    small['input_shape'] = [[1, 3, 1, 2]]*3 + [[1, 1, 3, 2], [1, 1, 2, 3], [1, 1, 3, 2], [1, 3, 1]]
    small['dtype'] = ['float32']*7
    return inputs(WORKLOAD, small)


def test_full_frozen_lact_case3_qualifies_and_correct_candidate_output_is_accepted(monkeypatch):
    reference_calls = Mock(wraps=q.native_reference)
    monkeypatch.setattr(q, 'native_reference', reference_calls)
    result = q.execute_case(WORKLOAD, 3, 'cpu', warmup=0, repeats=1)
    assert result['passed'] and result['id'] == 3
    assert reference_calls.call_count == 1
    _, _, cases = pack(WORKLOAD)
    module, args, kwargs = inputs(WORKLOAD, cases[2])
    oracle = call(WORKLOAD, module, [a.double() for a in args], kwargs, oracle=True)
    native = native_reference(WORKLOAD, module, args, kwargs)
    assert verifier.output_check(WORKLOAD, tuple(native), oracle, args, 'cpu', native) >= 0
    with pytest.raises(ValueError, match='Numerical mismatch'):
        verifier.output_check(WORKLOAD, tuple(torch.zeros_like(t) for t in native), oracle, args, 'cpu', native)


def test_candidate_timing_reuses_one_cpu_reference_outside_every_measurement(monkeypatch):
    spec, directory, cases = pack(WORKLOAD)
    monkeypatch.setattr(verifier, 'pack', lambda _: (spec, directory, cases[:1]))
    calls, references = [], []
    def native(*args):
        calls.append('cpu-reference')
        value = native_reference(*args)
        references.append(value)
        return value
    monkeypatch.setattr(verifier, 'native_reference', native)
    checked = Mock(wraps=verifier.output_check)
    monkeypatch.setattr(verifier, 'output_check', checked)
    module, _, _ = small_inputs(WORKLOAD, cases[0])
    def measure(label, operation):
        calls.append(label)
        return operation(), 1. if label == 'candidate' else 2.
    result = verifier.evaluate(WORKLOAD, module.lact_ttt_chunk, build=small_inputs,
                               performance=True, measure=measure, warmup=0, repeats=2)
    assert result['passed'] and len(result['cases']) == 1
    assert calls == ['cpu-reference', 'baseline', 'candidate', 'candidate', 'baseline']
    assert checked.call_count == 3
    assert all(call.args[-1] is references[0] for call in checked.call_args_list)


@pytest.mark.parametrize('path', ['qualification', 'candidate'])
def test_native_cpu_reference_input_mutation_is_rejected_before_target_execution(monkeypatch, path):
    spec, directory, cases = pack(WORKLOAD)
    module, args, kwargs = small_inputs(WORKLOAD, cases[0])
    def mutating(_workload, _module, values, attrs):
        values[0].add_(1.)
        return native_reference(_workload, _module, values, attrs)
    if path == 'qualification':
        monkeypatch.setattr(q, 'inputs', lambda *_: (module, args, kwargs))
        monkeypatch.setattr(q, 'native_reference', mutating)
        with pytest.raises(ValueError, match='reference mutated input'):
            q.execute_case(WORKLOAD, 1, 'cpu', warmup=0, repeats=1)
    else:
        monkeypatch.setattr(verifier, 'pack', lambda _: (spec, directory, cases[:1]))
        monkeypatch.setattr(verifier, 'native_reference', mutating)
        candidate = Mock(side_effect=module.lact_ttt_chunk)
        with pytest.raises(ValueError, match='Reference mutated input'):
            verifier.evaluate(WORKLOAD, candidate, build=lambda *_: (module, args, kwargs))
        assert candidate.call_count == 0


@pytest.mark.parametrize('workload', ['fa', 'gdn', 'bsa', 'kimi_delta_attention'])
def test_existing_custom_workloads_do_not_execute_an_extra_cpu_benchmark(workload):
    assert native_reference(workload, object(), [], {}) is None
