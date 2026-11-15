"""CPU comparator checks only; these are not NPU or latency evidence."""
import importlib.util
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeIdentityControls(unittest.TestCase):
    def test_semantic_identity_ignores_stamps_and_invalidates_real_artifacts(self):
        module = load(ROOT / 'cli/skill-suites/runtime-identity.py', 'runtime_identity')
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            compiler = home / 'compiler'
            compiler.write_text('compiler-v1')
            source = home / 'source.py'
            source.write_text('source-v1')
            (home / 'version.info').write_text('cann-v1')
            identity = {'binding': {'target': 'registered', 'runtime': 'profile', 'product': 'a3'},
                        'python': ['version', ['python-path', 1, 2]], 'machine': 'fixture',
                        'packages': {name: ['v1', str(home), 'record', [[str(source), 1, 2]]]
                                     for name in ('torch', 'torch-npu', 'triton')},
                        'compilers': {'compiler': [str(compiler), 1, 2]},
                        'cann': {'ASCEND_HOME_PATH': str(home)}}
            report = home / 'setup.json'
            report.write_text(json.dumps({'identity': identity}))
            board = SimpleNamespace(stdout='Chip Name : fixture-a3\n')
            source_reads = []
            original_open = Path.open
            def record_open(path, *args, **kwargs):
                if path.resolve() == source.resolve() and args and args[0] == 'rb':
                    source_reads.append(path)
                return original_open(path, *args, **kwargs)
            with patch.dict(os.environ, RSI_SETUP_REPORT=str(report), ASCEND_PHYSICAL_DEVICE='4'), \
                    patch.object(module.subprocess, 'run', return_value=board), \
                    patch.object(Path, 'open', new=record_open):
                first = module.identity()
                self.assertEqual(len(source_reads), 1)
                # The probe crosses JSON before a fresh checker recomputes identity.
                wire_identity = json.loads(json.dumps(first))
                self.assertEqual(wire_identity, module.identity())
                self.assertEqual(len(source_reads), 2)
                identity['python'][1] = ['other-python-path', 30, 40]
                for data in identity['packages'].values():
                    data[1] = 'relocated-installation'
                    data[3][0][1:] = [999, 1000]
                identity['compilers']['compiler'][1:] = [999, 1000]
                report.write_text(json.dumps({'identity': identity}))
                self.assertEqual(wire_identity, module.identity())
                compiler.write_text('compiler-v2')
                second = module.identity()
                self.assertNotEqual(first, second)
                source.write_text('source-v2')
                third = module.identity()
                self.assertNotEqual(second, third)
                (home / 'version.info').write_text('cann-v2')
                self.assertNotEqual(third, module.identity())


class CuratedNumericControls(unittest.TestCase):
    def test_output_container_and_arity_match_frozen_contracts(self):
        core = load(ROOT / 'cli/skill-suites/kernel-check-core.py', 'container_core')
        core.torch = torch
        tensor = torch.ones(2)
        for family in ('masked', 'gemm', 'gated'):
            with self.subTest(family=family, result='tensor'):
                self.assertIs(core.contract_outputs(tensor, family)[0], tensor)
            for wrong in ((tensor,), [tensor], (tensor, tensor.clone()), [tensor, tensor.clone()]):
                with self.subTest(family=family, result=type(wrong).__name__), self.assertRaises(AssertionError):
                    core.contract_outputs(wrong, family)
        for correct in ((tensor, tensor.clone()), [tensor, tensor.clone()]):
            self.assertEqual(len(core.contract_outputs(correct, 'splitk')), 2)
        for wrong in (tensor, (), (tensor,), (tensor, tensor.clone(), tensor.clone()), [tensor, 'not-a-tensor']):
            with self.subTest(splitk=type(wrong).__name__), self.assertRaises(AssertionError):
                core.contract_outputs(wrong, 'splitk')

    def test_references_and_independent_wrong_outputs(self):
        torch.set_num_threads(1)
        core = load(ROOT / 'cli/skill-suites/kernel-check-core.py', 'curated_core')
        packs = json.loads((ROOT / 'cli/skill-suites/a3-packs.json').read_text())['packs']
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            for pack in packs:
                for name, text in pack['trusted'].items():
                    path = home / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text)
                reference = home / 'reference.py'
                reference.write_text(pack['reference'])
                good = load(reference, 'reference_' + pack['id'])
                family = pack['family']
                spec = load(home / 'fixtures/trusted' / family / 'spec.py', 'spec_' + pack['id'])
                # One bounded existing boundary fixture per pack; full private cases run on the NPU.
                candidates = [case for entry in pack['cases'] for case in entry['variants']]
                size = lambda case: case.get('numel', math.prod(case.get('shape', case.get('mnk', [1]))))
                for case in [min(candidates, key=size)]:
                    args = spec.make_inputs(case)
                    expected = spec.evaluate(args[0], args[1], case) if family == 'gated' else spec.evaluate(*args)
                    result = tuple(x.clone() for x in core.outputs(good.run(*args)))
                    with self.subTest(pack=pack['id'], case=case['id'], control='reference'):
                        core.numeric_check(spec, family, case, args, result, expected)
                    for index in range(len(result)):
                        wrong = tuple(x + 37 if i == index else x.clone() for i, x in enumerate(result))
                        with self.subTest(pack=pack['id'], case=case['id'], control=index):
                            with self.assertRaises(AssertionError):
                                core.numeric_check(spec, family, case, args, wrong, expected)
        for excess in (float('nan'), .1):
            with self.subTest(invalid_score=excess), self.assertRaises(AssertionError):
                core.numeric_check(SimpleNamespace(check=lambda *args: excess), 'splitk', {}, (), (), None)


if __name__ == '__main__':
    unittest.main()
