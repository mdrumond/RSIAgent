import json

import pytest
import torch

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
