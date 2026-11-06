# Numerical contract adapted from Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0; see packs/upstream/LICENSE.
"""Finite-floating CANNBench comparison, including its native CPU controls.

Contract: cann/cann-bench f0488d95b4d44a6a0da1b8cd99e91fbd06181962,
src/kernel_eval/utils/compare.py:431-717 and thresholds.py:54-131.
Immutable source: https://raw.gitcode.com/cann/cann-bench/raw/<revision>/<path>
SHA256 compare.py: 863f68629192ea4e2ee7a25b5705d3473f380a8ce2e12d4b5ea563ea8ed7101c
SHA256 thresholds.py: 6a26f1709a4dbb72de21da19165bb1655797053939d34ab125f3fd7c935e92dd
The existing outer checks still reject nonfinite or structurally invalid output.
"""
import torch
from .inputs import call, sparse_outputs, tensors


# (small-value boundary, absolute-error cutoff, cancellation boundary/zero).
# These are the pinned framework's dtype rules, not workload tolerance overrides.
REGIONS = {'float16': (2**-11, 2**-16, 2**-5),
           'bfloat16': (2**-8, 2**-16, 2**-3),
           'float32': (2**-14, 2**-30, 2**-8)}


def native_reference(workload, module, args, kwargs, oracle_outputs=None):
    """Plain golden at case precision; computed once before target timing."""
    if workload not in ('lact_ttt_chunk', 'tensor_program_vm',
                        'sparse_lightning_indexer_grad_kl_loss_enhance'):
        return None
    if workload == 'sparse_lightning_indexer_grad_kl_loss_enhance' and oracle_outputs is not None:
        # Sparse's pinned CPU oracle and plain math are literally the same FP32
        # function on the same values. Only the benchmark output casts differ.
        return tensors(sparse_outputs(oracle_outputs, args))
    return tensors(call(workload, module, args, kwargs))


def finite_comparison(actual, oracle, threshold, native=None):
    dtype = str(actual.dtype).split('.')[-1]
    small_boundary, absolute_cutoff, cancel_boundary = REGIONS[dtype]
    golden = oracle.detach().to(dtype=actual.dtype).to(device='cpu', dtype=torch.float64)
    output = actual.detach().to(device='cpu', dtype=torch.float64)
    cpu = golden if native is None else native.detach().to(device='cpu', dtype=torch.float64)
    if any(t.shape != golden.shape or not torch.isfinite(t).all() for t in (output, golden, cpu)):
        raise ValueError('Invalid finite comparison reference or output')
    if not output.numel():
        return {'passed': True, 'mere': 0., 'mare': 0.}
    difference = (output-golden).abs()
    magnitude = golden.abs()
    relative = difference/(magnitude+1e-7)
    mere, mare = relative.mean().item(), relative.max().item()
    if mere < threshold and mare < 10*threshold:
        return {'passed': True, 'mere': mere, 'mare': mare}
    small = magnitude < small_boundary
    cancel = (output.abs() < cancel_boundary) & (magnitude >= small_boundary) & (magnitude < cancel_boundary)
    normal = ~small & ~cancel
    cpu_difference = (cpu-golden).abs()
    cpu_relative = cpu_difference/(magnitude+1e-7)
    mismatch, cpu_mismatch = relative > 10*threshold, cpu_relative > 10*threshold
    count = lambda mask: int(mask.sum().item())
    small_errors = count(small & (difference > absolute_cutoff))
    small_cpu_errors = count(small & (cpu_difference > absolute_cutoff))
    cancel_errors, cancel_cpu_errors = count(cancel & mismatch), count(cancel & cpu_mismatch)
    normal_errors, normal_cpu_errors = count(normal & mismatch), count(normal & cpu_mismatch)
    # Pinned issue #92: normal errors require a genuinely imperfect CPU control;
    # a clean CPU normal region never licenses a candidate normal-region error.
    normal_passed = normal_errors == 0 or (normal_cpu_errors > 0 and normal_errors/normal_cpu_errors <= 2)
    passed = (normal_passed and small_errors/max(small_cpu_errors, 1) <= 2
              and cancel_errors/max(cancel_cpu_errors, 1) <= 2)
    return {'passed': passed, 'mere': mere, 'mare': mare,
            'smallErrors': small_errors, 'smallCpuErrors': small_cpu_errors,
            'cancelErrors': cancel_errors, 'cancelCpuErrors': cancel_cpu_errors,
            'normalErrors': normal_errors, 'normalCpuErrors': normal_cpu_errors}
