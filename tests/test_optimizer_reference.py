import json

import pytest
torch = pytest.importorskip("torch")

from optimizer.reference import ReferenceError, compare_outputs, freeze_reference, prepare_reference


def spec(source, entrypoint="kernel", **extra):
    return {"source": source, "entrypoint": entrypoint,
            "signatures": [{"args": [{"shape": [4], "dtype": "float32"}]}], **extra}


def test_function_nested_output_and_frozen_artifacts(tmp_path):
    prepared = prepare_reference(spec("def kernel(x): return {'a': x+1, 'b': (x*2, [x.sum()])}"))
    assert prepared.cases
    first = prepared.cases[0]
    compare_outputs(first["output"], prepared.invoke(*first["args"], **first["kwargs"]), inputs=first["args"])
    manifest = freeze_reference(prepared, tmp_path)
    assert manifest["source_sha256"] and manifest["policy"]["source_sha256"]
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    assert (tmp_path / "baseline.pt").exists()


def test_module_parameters_are_frozen_and_constructor_supported():
    prepared = prepare_reference(spec("import torch\nclass Model(torch.nn.Module):\n def __init__(self, scale):\n  super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(float(scale)))\n def forward(self, x): return x*self.weight", "Model", init_args=[3]))
    assert not prepared.model.training
    assert not prepared.model.weight.requires_grad
    assert prepared.cases[0]["output"].equal(prepared.cases[0]["args"][0] * 3)


@pytest.mark.parametrize("body,reason", [
    ("x.add_(1); return x", "mutat"),
    ("return x + torch.rand_like(x)", "determin"),
    ("x.sum().backward(); return x", "backward"),
])
def test_unsupported_reference_is_rejected_with_reason(body, reason):
    with pytest.raises(ReferenceError, match=reason):
        prepare_reference(spec("import torch\ndef kernel(x):\n " + body))


def test_stateful_module_is_rejected():
    with pytest.raises(ReferenceError, match="mutat"):
        prepare_reference(spec("import torch\nclass Model(torch.nn.Module):\n def __init__(self):\n  super().__init__(); self.register_buffer('count',torch.tensor(0))\n def forward(self,x):\n  self.count.add_(1); return x", "Model"))


def test_stateful_function_is_rejected_even_when_output_is_unchanged():
    with pytest.raises(ReferenceError, match="mutat"):
        prepare_reference(spec("count = 0\ndef kernel(x):\n global count\n count += 1\n return x"))


def test_comparison_checks_full_structure_shape_dtype_and_all_leaves():
    golden = {"a": torch.ones(2), "b": [torch.zeros(2)]}
    for actual in ({"a": torch.ones(2)},
                   {"a": torch.ones(1, 2), "b": [torch.zeros(2)]},
                   {"a": torch.ones(2).half(), "b": [torch.zeros(2)]},
                   {"a": torch.ones(2), "b": [torch.ones(2)]}):
        with pytest.raises(ReferenceError):
            compare_outputs(golden, actual)


def test_cannbot_three_checks_nan_inf_and_integer_dispatch():
    x = torch.ones(10)
    with pytest.raises(ReferenceError):
        compare_outputs(x, x + 1e-3)
    compare_outputs(torch.tensor([float('nan'), float('inf'), -float('inf')]),
                    torch.tensor([float('nan'), float('inf'), -float('inf')]))
    with pytest.raises(ReferenceError):
        compare_outputs(torch.tensor([float('inf')]), torch.tensor([-float('inf')]))
    compare_outputs(torch.tensor([4]), torch.tensor([5]), inputs=[torch.ones(1)])
    with pytest.raises(ReferenceError):
        compare_outputs(torch.tensor([4]), torch.tensor([5]), inputs=[torch.ones(1, dtype=torch.int64)])


def test_exactly_ninety_percent_match_passes_pinned_ratio():
    golden = torch.ones(10)
    candidate = golden.clone()
    candidate[-1] += 1e-3
    metrics = compare_outputs(golden, candidate)
    assert metrics["leaves"][0]["matched_ratio"] == 0.9


def test_scalar_inputs_and_keyword_tensor_are_supported():
    submission = spec("def kernel(x, scale, *, bias): return x*scale+bias")
    submission["signatures"][0]["args"].append({"scalar": 3})
    submission["signatures"][0]["kwargs"] = {"bias": {"shape": [4], "dtype": "float32"}}
    prepared = prepare_reference(submission)
    assert prepared.cases


def test_generated_invalid_shape_is_reported_and_representative_is_kept():
    submission = spec("def kernel(x): return x.reshape(2, 2)")
    prepared = prepare_reference(submission)
    assert prepared.cases[0]["case"]["representative"]
    assert any("reshape" in rejected["error"] or "shape" in rejected["error"] for rejected in prepared.manifest["rejected"])


def test_private_module_state_mutation_is_rejected():
    with pytest.raises(ReferenceError, match="mutat"):
        prepare_reference(spec("import torch\nclass Model(torch.nn.Module):\n def __init__(self):\n  super().__init__(); self._count = 0\n def forward(self, x):\n  self._count += 1\n  return x + 1", "Model"))


def test_quantized_int64_comparison_cannot_overflow():
    limits = torch.iinfo(torch.int64)
    for a, b in [(limits.min, 0), (limits.min, limits.max), (0, limits.min)]:
        with pytest.raises(ReferenceError):
            compare_outputs(torch.tensor([a]), torch.tensor([b]), inputs=[torch.ones(1)])
    for a, b in [(limits.min, limits.min + 1), (limits.max, limits.max - 1)]:
        compare_outputs(torch.tensor([a]), torch.tensor([b]), inputs=[torch.ones(1)])


def test_dict_output_paths_do_not_collide():
    prepared = prepare_reference(spec("def kernel(x): return {'a.b': x, 'a': {'b': x + 1}, 'a[0]': x + 2}"), max_cases=1)
    outputs = prepared.manifest['cases'][0]['outputs']
    assert len(outputs) == 3
    assert len({row['sha256'] for row in outputs.values()}) == 3


def test_frozen_module_load_preserves_state_layout_and_nested_outputs(tmp_path, monkeypatch):
    from optimizer import reference
    submission = spec(
        "class Model(torch.nn.Module):\n def __init__(self):\n  super().__init__(); self.weight = torch.nn.Parameter(torch.randn(2, 3))\n def forward(self, x): return {'a': x + self.weight, 'b': (x * 2, None)}",
        "Model")
    submission['signatures'][0]['args'][0].update(shape=[2, 3], strides=[1, 2])
    prepared = prepare_reference(submission, max_cases=1)
    freeze_reference(prepared, tmp_path)
    monkeypatch.setattr(reference, 'generate_cases', lambda *a, **k: pytest.fail('regenerated cases'))
    monkeypatch.setattr(reference, 'materialize_case', lambda *a, **k: pytest.fail('regenerated tensors'))
    loaded = reference.load_reference(tmp_path, 'cpu')
    reference.validate_reference(loaded)
    assert loaded.manifest['suite_sha256'] == prepared.manifest['suite_sha256']
    assert torch.equal(loaded.model.weight, prepared.model.weight)
    assert loaded.cases[0]['args'][0].stride() == (1, 2)
    assert torch.equal(loaded.cases[0]['args'][0], prepared.cases[0]['args'][0])
    compare_outputs(prepared.cases[0]['output'], loaded.cases[0]['output'], non_compute=True)
    compare_outputs(loaded.cases[0]['output'], loaded.invoke(*loaded.cases[0]['args']), non_compute=True)


@pytest.mark.parametrize('corruption', ['source', 'suite', 'tensors', 'policy'])
def test_frozen_load_rejects_corruption_before_deserialization(tmp_path, monkeypatch, corruption):
    from optimizer.reference import load_reference
    prepared = prepare_reference(spec('def kernel(x): return x + 1'), max_cases=1)
    freeze_reference(prepared, tmp_path)
    if corruption == 'source':
        (tmp_path / 'reference.py').write_text('def kernel(x): return x')
    elif corruption == 'tensors':
        (tmp_path / 'baseline.pt').write_bytes(b'corrupt')
    else:
        path = tmp_path / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['suite_sha256' if corruption == 'suite' else 'policy'] = 'changed'
        path.write_text(json.dumps(manifest))
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('deserialized corrupt bundle'))
    with pytest.raises(ValueError):
        load_reference(tmp_path, 'cpu')


@pytest.mark.parametrize('mutation', ['policy', 'source', 'state'])
def test_reference_validation_rejects_changed_workload(mutation):
    from optimizer.reference import validate_reference
    prepared = prepare_reference(spec(
        "class Model(torch.nn.Module):\n def __init__(self):\n  super().__init__(); self.register_buffer('weight', torch.ones(4))\n def forward(self, x): return x + self.weight",
        'Model'), max_cases=1)
    if mutation == 'policy':
        prepared.manifest['policy'] = {}
    elif mutation == 'source':
        prepared.spec['source'] += '\n'
    else:
        prepared.model.weight.zero_()
    with pytest.raises(ValueError):
        validate_reference(prepared)
