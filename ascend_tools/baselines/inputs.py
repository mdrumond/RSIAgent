"""Generate exact frozen case shapes; only placement is adapted for supplied GDN."""
import ast
import json
import types
import torch
from . import pack


def _sparse_kl_fp32(tree):
    """Keep upstream KL math/device placement, repairing its low-precision path.

    FP16 clamp_min(1e-8) becomes zero before log, yielding nonfinite loss/gradients.
    CPU golden already computes and returns all four tensors in FP32. Match that
    contract on NPU, including FP32 inputs, without its CPU fallback. Vendored
    source bytes, cases and tolerances remain untouched.
    """
    sites = []
    class Precision(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name != '_sparse_lightning_kl_grad_impl':
                return node
            for statement in node.body:
                if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                    continue
                name = getattr(statement.targets[0], 'id', None)
                if name == 'native':
                    statement.value = ast.parse("query.device.type == 'npu'", mode='eval').body
                    sites.append(name)
                elif name == 'calc_dtype':
                    statement.value = ast.parse('torch.float32', mode='eval').body
                    sites.append(name)
            return node
    tree = Precision().visit(tree)
    if sorted(sites) != ['calc_dtype', 'native']:
        raise ValueError('Sparse KL compatibility adapter no longer matches frozen reference')
    return ast.fix_missing_locations(tree)


def load(workload, case=None):
    spec, directory, cases = pack(workload)
    module = types.ModuleType('baseline_' + workload)
    module.__file__ = str(directory / spec['source'])
    tree = ast.parse((directory / spec['source']).read_text())
    if workload == 'sparse_lightning_indexer_grad_kl_loss_enhance':
        tree = _sparse_kl_fp32(tree)
    if workload == 'gdn':
        # Its original generator hardcodes .npu(); CPU construction must not allocate an NPU.
        class Placement(ast.NodeTransformer):
            def visit_Call(self, node):
                node = self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'npu':
                    return ast.copy_location(ast.Call(ast.Attribute(node.func.value, 'to', ast.Load()),
                                                      [ast.Constant('cpu')], []), node)
                return node
        tree = ast.fix_missing_locations(Placement().visit(tree))
    if case is not None and workload in ('gdn', 'bsa'):
        class OneCase(ast.NodeTransformer):
            def visit_Assign(self, node):
                if workload == 'bsa' and any(isinstance(t, ast.Name) and t.id == 'cases' for t in node.targets):
                    return ast.copy_location(ast.Assign(node.targets, ast.List([ast.Name('_case', ast.Load())], ast.Load())), node)
                return self.generic_visit(node)
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load) and node.id in ('case_index', 'case_idx'):
                    return ast.copy_location(ast.BinOp(node, ast.Add(), ast.Constant(case['case_id'] - 1)), node)
                return node
        tree = ast.fix_missing_locations(OneCase().visit(tree))
        module._case = case
    exec(compile(tree, module.__file__, 'exec'), module.__dict__)
    if case is not None and workload == 'gdn':
        module._load_cases = lambda: [case]
    return spec, module, cases


def inputs(workload, case):
    spec, module, _ = load(workload, case)
    torch.manual_seed(case.get('seed', case['case_id']))
    if workload == 'fa':
        args = [(torch.randn(case.get('b',1), case.get('h',1), case['n'], case['d']) * .25).to(
            getattr(torch,case.get('dtype','float32'))) for _ in range(3)]
        return module, args, {'causal': case['causal']}
    if workload in ('gdn', 'bsa'):
        return module, module.get_input_groups()[0], {}
    proto = json.loads((__import__('pathlib').Path(module.__file__).parent / 'proto.json').read_text())
    args = []
    for shape, dtype, limits in zip(case['input_shape'], case['dtype'], case['value_range']):
        dt = getattr(torch, dtype)
        if dt.is_floating_point:
            value = torch.empty(shape, dtype=torch.float32).uniform_(*limits).to(dt)
        else:
            value = torch.randint(int(limits[0]), int(limits[1]) + 1, shape, dtype=dt)
        args.append(value)
    names = [p['name'] for p in proto['inputs']]
    kwargs = dict(case.get('attrs', {}))
    if hasattr(module, 'get_input'):
        # The upstream hook constructs legal sparse indices and optional interface inputs.
        args = module.get_input(**dict(zip(names, args)), **kwargs)
    return module, args, kwargs


def call(workload, module, args, kwargs, oracle=False):
    if workload in ('gdn', 'bsa'):
        return module.Model()(*args, **kwargs)
    spec, _, _ = pack(workload)
    name = spec['oracle'] if oracle else spec['function']
    function = getattr(module, name)
    return function(*args, **kwargs)


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for item in value for t in tensors(item)]
    if value is None:
        return []
    raise TypeError('Reference must return tensors')
