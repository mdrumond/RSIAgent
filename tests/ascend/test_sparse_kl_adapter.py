"""Exercise native-branch precision on CPU; this is not NPU qualification."""
import ast
import types
import pytest
import torch
from ascend_tools.baselines import pack
from ascend_tools.baselines.inputs import _sparse_kl_fp32, load

WORKLOAD = 'sparse_lightning_indexer_grad_kl_loss_enhance'


def native_branch_on_cpu(adapted):
    spec, directory, _ = pack(WORKLOAD)
    tree = ast.parse((directory / spec['source']).read_text())
    if adapted:
        tree = _sparse_kl_fp32(tree)
    class Native(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'native' for t in node.targets):
                node.value = ast.Constant(True)
            return node
    module = types.ModuleType('native_cpu_probe')
    exec(compile(ast.fix_missing_locations(Native().visit(tree)), str(directory / spec['source']), 'exec'), module.__dict__)
    return module.sparse_lightning_indexer_grad_kl_loss_enhance


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('layout', ['TND', 'BSND'])
def test_native_fp32_adapter_matches_cpu_oracle_and_preserves_inputs(dtype, layout):
    args = [torch.tensor([[[40., 0.]]], dtype=dtype),
            torch.tensor([[[1., 0.]], [[-1., 0.]]], dtype=dtype),
            torch.tensor([[[2., 1.]]], dtype=dtype),
            torch.tensor([[[1., 0.]], [[0., 1.]]], dtype=dtype),
            torch.tensor([[.5]], dtype=dtype), torch.tensor([[[0, 1]]])]
    if layout == 'BSND':
        args = [value.unsqueeze(0) for value in args]
    kwargs = dict(layout=layout, actual_seq_lengths_query=[1], actual_seq_lengths_key=[2], sparse_mode=0)
    before = [x.clone() for x in args]
    _, module, _ = load(WORKLOAD)
    expected = module.sparse_lightning_indexer_grad_kl_loss_enhance(*args, **kwargs)
    actual = native_branch_on_cpu(True)(*args, **kwargs)
    assert len(actual) == 4
    for got, reference in zip(actual, expected):
        assert got.dtype == torch.float32 and got.device.type == 'cpu'
        assert torch.isfinite(got).all().item()
        torch.testing.assert_close(got, reference, atol=0, rtol=0)
    for original, value in zip(before, args):
        assert torch.equal(original, value)
    if dtype == torch.float16:
        broken = native_branch_on_cpu(False)(*args, **kwargs)
        assert any(not torch.isfinite(value).all().item() for value in broken)


def test_adapter_refuses_missing_or_duplicate_assignment_sites():
    for source in ('def unrelated(): pass',
                   'def _sparse_lightning_kl_grad_impl():\n native = True\n calc_dtype = None\n native = False'):
        with pytest.raises(ValueError, match='no longer matches'):
            _sparse_kl_fp32(ast.parse(source))
