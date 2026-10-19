import pytest

from optimizer.cases import generate_cases, materialize_case, validate_submission


def submission(**changes):
    return dict(source="def kernel(x): return x + 1", entrypoint="kernel",
                signatures=[{"args": [{"shape": [32], "dtype": "float32"}]}], **changes)


def test_cases_preserve_representative_and_are_deterministic():
    spec = submission()
    suite = generate_cases(spec)
    assert suite == generate_cases(spec)
    assert suite["cases"][0]["args"][0]["shape"] == [32]
    assert len(suite["cases"]) <= 24
    assert {case["args"][0]["shape"][0] for case in suite["cases"]} >= {1, 31, 32, 33}
    for case in suite["cases"]:
        a, _ = materialize_case(case)
        b, _ = materialize_case(case)
        assert a[0].equal(b[0])


def test_invalid_and_constrained_variants_are_explicit():
    spec = submission()
    spec["signatures"][0]["args"][0]["constraints"] = {"multiple_of": [8]}
    suite = generate_cases(spec)
    assert suite["rejected"]
    assert all(c["args"][0]["shape"][0] % 8 == 0 for c in suite["cases"])
    spec["signatures"][0]["args"][0]["dtype"] = "mystery"
    with pytest.raises(ValueError, match="dtype"):
        validate_submission(spec)


def test_scalar_kwargs_and_strided_inputs():
    spec = submission()
    spec["signatures"] = [{"args": [{"shape": [2, 3], "dtype": "float16", "strides": [1, 2]}],
                           "kwargs": {"scale": {"scalar": 2}}}]
    args, kwargs = materialize_case(generate_cases(spec)["cases"][0])
    assert args[0].stride() == (1, 2)
    assert kwargs == {"scale": 2}


def test_representatives_never_silently_truncated():
    spec = submission()
    spec["signatures"] *= 25
    with pytest.raises(ValueError, match="representative"):
        generate_cases(spec)


def test_padded_strides_survive_device_transfer():
    spec = submission()
    spec['signatures'] = [{'args': [{'shape': [2, 3], 'dtype': 'float32', 'strides': [4, 1]}]}]
    case = generate_cases(spec)['cases'][0]
    args, _ = materialize_case(case, device='meta')
    assert args[0].stride() == (4, 1)
