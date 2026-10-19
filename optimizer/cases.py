"""Serializable, deterministic input contracts; no submitted source executes here."""
from __future__ import annotations

import copy
import hashlib
import json
import math


DTYPES = {"float32", "float16", "bfloat16", "float64", "int64", "int32", "int16", "int8", "uint8", "bool"}
DISTRIBUTIONS = {"normal", "uniform", "zeros", "ones", "alternating"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _hint(hint):
    if not isinstance(hint, dict):
        raise ValueError("input hint must be an object")
    if "scalar" in hint:
        if set(hint) != {"scalar"} or not isinstance(hint["scalar"], (int, float, bool, str, type(None))):
            raise ValueError("scalar must be a JSON primitive")
        digest(hint)
        return
    shape = hint.get("shape")
    if not isinstance(shape, list) or any(type(d) is not int or d < 1 for d in shape):
        raise ValueError("shape must contain positive integer dimensions")
    if hint.get("dtype") not in DTYPES:
        raise ValueError("unsupported dtype")
    if math.prod(shape) > 16_777_216:
        raise ValueError("case exceeds 16M elements per tensor")
    if hint.get("distribution", "normal") not in DISTRIBUTIONS:
        raise ValueError("unsupported distribution")
    if hint.get("layout", "contiguous") not in {"contiguous", "transposed"}:
        raise ValueError("unsupported layout")
    if hint.get("layout") == "transposed" and len(shape) < 2:
        raise ValueError("transposed layout requires rank >= 2")
    if "strides" in hint:
        strides = hint["strides"]
        if len(strides) != len(shape) or any(type(s) is not int or s < 1 for s in strides):
            raise ValueError("strides must be positive and match rank")
        span = 1
        for stride, length in sorted(zip(strides, shape)):
            if length > 1 and stride < span:
                raise ValueError("overlapping strides are unsupported")
            span += (length - 1) * stride
        if span > 16_777_216:
            raise ValueError("strided storage exceeds 16M elements")
    constraints = hint.get("constraints", {})
    if set(constraints) - {"min_shape", "max_shape", "multiple_of"}:
        raise ValueError("unknown shape constraint")
    for name, bounds in constraints.items():
        if len(bounds) != len(shape) or any(type(n) is not int or n < 1 for n in bounds):
            raise ValueError("constraint bounds must be positive and match rank")
        if any((d < b if name == "min_shape" else d > b if name == "max_shape" else d % b != 0)
               for d, b in zip(shape, bounds)):
            raise ValueError("shape violates " + name)
    low, high = hint.get("low", 0 if hint["dtype"] == "uint8" else -4), hint.get("high", 5)
    if not isinstance(low, (float, int)) or not isinstance(high, (float, int)) or not low < high:
        raise ValueError("low/high must be finite increasing numbers")
    digest(hint)


def validate_submission(spec):
    spec = copy.deepcopy(spec)
    if not isinstance(spec.get("source"), str) or not spec["source"].strip():
        raise ValueError("source is required")
    if not isinstance(spec.get("entrypoint"), str) or not spec["entrypoint"].isidentifier():
        raise ValueError("entrypoint must be a named function or Module class")
    if not isinstance(spec.get("init_args", []), list) or not isinstance(spec.get("init_kwargs", {}), dict):
        raise ValueError("constructor args/kwargs must be JSON list/object")
    signatures = spec.get("signatures")
    if not isinstance(signatures, list) or not signatures:
        raise ValueError("at least one representative signature is required")
    for signature in signatures:
        if not isinstance(signature.get("args", []), list) or not isinstance(signature.get("kwargs", {}), dict):
            raise ValueError("signature args/kwargs must be list/object")
        for hint in [*signature.get("args", []), *signature.get("kwargs", {}).values()]:
            _hint(hint)
    digest(spec)
    return spec


def generate_cases(spec, max_cases=24, seed=0):
    spec = validate_submission(spec)
    if not 1 <= max_cases <= 24 or len(spec["signatures"]) > max_cases:
        raise ValueError("representative signatures must fit the 1..24 case budget")
    cases, rejected, seen = [], [], set()

    def add(signature, reason, representative=False):
        signature = copy.deepcopy(signature)
        signature.setdefault("args", [])
        signature.setdefault("kwargs", {})
        identity = digest(signature)
        if identity in seen and not representative:
            return
        try:
            for hint in [*signature["args"], *signature["kwargs"].values()]:
                _hint(hint)
        except ValueError as exc:
            rejected.append({"signature": signature, "reason": reason, "error": str(exc)})
            return
        seen.add(identity)
        case = {**signature, "seed": seed + len(cases), "reason": reason,
                "representative": representative, "id": f"case-{len(cases):03d}"}
        case["sha256"] = digest(case)
        cases.append(case)

    for signature in spec["signatures"]:
        add(signature, "user representative", True)
    for signature in spec["signatures"]:
        for change in ("small", "odd", "below-alignment", "above-alignment", "zeros", "ones", "alternating"):
            if len(cases) >= max_cases:
                break
            variant = copy.deepcopy(signature)
            for hint in [*variant.get("args", []), *variant.get("kwargs", {}).values()]:
                if "shape" not in hint:
                    continue
                if change in DISTRIBUTIONS:
                    hint["distribution"] = change
                else:
                    transform = {"small": lambda d: 1, "odd": lambda d: d if d % 2 else d + 1,
                                 "below-alignment": lambda d: max(1, ((d + 31) // 32) * 32 - 1),
                                 "above-alignment": lambda d: ((d + 31) // 32) * 32 + 1}[change]
                    hint["shape"] = [transform(d) for d in hint["shape"]]
            add(variant, change)
    return {"cases": cases, "rejected": rejected}


def materialize_case(case, device="cpu"):
    """Generate on CPU with a private RNG, then move outside measurement capture."""
    import torch

    generator = torch.Generator().manual_seed(case["seed"])

    def make(hint):
        if "scalar" in hint:
            return hint["scalar"]
        _hint(hint)
        shape, dtype = hint["shape"], getattr(torch, hint["dtype"])
        distribution = hint.get("distribution", "normal")
        low, high = hint.get("low", 0 if dtype == torch.uint8 else -4), hint.get("high", 5)
        if distribution in {"zeros", "ones"}:
            tensor = getattr(torch, distribution)(shape, dtype=dtype)
        elif distribution == "alternating":
            tensor = (torch.arange(math.prod(shape)).reshape(shape) % 2).to(dtype)
        elif dtype == torch.bool:
            tensor = torch.randint(0, 2, shape, generator=generator).bool()
        elif not dtype.is_floating_point:
            tensor = torch.randint(int(low), int(high), shape, generator=generator, dtype=dtype)
        elif distribution == "uniform":
            tensor = (torch.rand(shape, generator=generator) * (high-low) + low).to(dtype)
        else:
            tensor = torch.randn(shape, generator=generator).to(dtype)
        strides = hint.get("strides")
        if hint.get("layout") == "transposed" and strides is None:
            order = list(range(len(shape)))
            order[-1], order[-2] = order[-2], order[-1]
            tensor = tensor.permute(order).contiguous().permute(order)
        elif strides is not None:
            tensor = torch.empty_strided(shape, strides, dtype=dtype).copy_(tensor)
        return tensor.to(device)

    return [make(hint) for hint in case["args"]], {k: make(hint) for k, hint in case["kwargs"].items()}
