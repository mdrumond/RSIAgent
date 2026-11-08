"""Frozen tiny arithmetic oracle: execution smoke, never an optimization score."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=('construction', 'baseline', 'correctness', 'performance'), required=True)
parser.add_argument('--device', default='cpu')
parser.add_argument('--single', action='store_true')
args = parser.parse_args()
import torch

if args.device != 'cpu':
    import torch_npu  # Registers the actual Ascend device backend.
    assert args.device == 'npu:0', 'Smokes use logical device zero after admission'


def load(path):
    spec = importlib.util.spec_from_file_location('smoke_answer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


def synchronize():
    if args.device != 'cpu':
        torch.npu.synchronize()


def check(run, x):
    actual = run(x)
    synchronize()
    expected = torch.tensor([i + 1 for i in range(x.numel())], dtype=torch.float32).reshape(x.shape)
    assert actual.device == x.device and actual.dtype == x.dtype
    assert torch.equal(actual.cpu(), expected), 'Independent arithmetic oracle rejected candidate'


def sample(run, x):
    synchronize()
    start = time.perf_counter()
    run(x)
    synchronize()
    elapsed = time.perf_counter() - start
    assert math.isfinite(elapsed) and elapsed > 0
    return elapsed


baseline = load(Path(__file__).parents[1] / 'reference' / 'answer.py')
candidate = baseline if args.mode == 'baseline' else load(Path.cwd() / 'answer.py')
shapes = [(1, 16)] if args.single else [(1, 17), (2, 33), (3, 65)]
timings = []
negative_rejected = 0
for shape in shapes:
    x = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape).to(args.device)
    check(candidate, x)
    if args.mode in ('baseline', 'performance'):
        check(baseline, x)
        if args.mode == 'baseline':
            try:
                check(load(Path(__file__).parents[1] / 'wrong' / 'answer.py'), x)
            except AssertionError:
                negative_rejected += 1
            else:
                raise AssertionError('Wrong control unexpectedly passed')
        sample(baseline, x)
        if args.mode == 'performance':
            sample(candidate, x)
        for repeat in range(3):
            pair = (baseline, candidate) if repeat % 2 == 0 else (candidate, baseline)
            timings.extend(sample(run, x) for run in pair)
print(json.dumps({'passed': True, 'mode': args.mode, 'device': args.device,
                  'hello': args.mode == 'correctness', 'cases': len(shapes),
                  'seconds': timings, 'negativeRejected': negative_rejected}), flush=True)
