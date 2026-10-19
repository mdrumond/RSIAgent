"""Torch worker-only execution and host-owned baseline/comparison policy.

Never execute submissions inside the dashboard process. The caller owns process
isolation, limits, and device admission. This module is not a Python sandbox.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import ModuleType

from .cases import digest, generate_cases, materialize_case, validate_submission


# Adapted from demo/benchmarks/cannbot/verify.py (Huawei, CANN Open Software
# License Agreement v2), get_limits / _compute_accuracy_state / compare.
POLICY = {"name": "cannbot-three-checks-v1", "source": "CANNBot verify.py",
          "source_sha256": "0d80e9e78f8e0c7bc7861dfb2bd5dc9058103ffe5c187db098a8b36e2252b429",
          "required_matched_ratio": 0.9,
          "float32": [2**-14, 2**-30, 2**-13, 1e-3, 2**-13],
          "float16": [2**-11, 2**-16, 2**-10, 9e-2, 2**-10],
          "bfloat16": [2**-8, 2**-16, 2**-7, 1e-1, 2**-7]}


class ReferenceError(ValueError):
    """A retained reference/correctness failure with optional leaf evidence."""

    def __init__(self, message, metrics=None):
        super().__init__(message)
        self.metrics = metrics or {}


def _leaves(value, path="output"):
    import torch
    if isinstance(value, torch.Tensor) or value is None:
        return [(path, value)]
    if isinstance(value, (tuple, list)):
        return [leaf for i, child in enumerate(value) for leaf in _leaves(child, f"{path}[{i}]")]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return [leaf for k, child in value.items() for leaf in _leaves(child, f"{path}.{k}")]
    raise ReferenceError(f"{path}: only tensors, None, and nested tensor containers are supported")


def _tensor_identity(tensor):
    import torch
    data = tensor.detach().cpu().contiguous()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "strides": list(tensor.stride()),
            "sha256": hashlib.sha256(data.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def compare_outputs(expected, actual, inputs=(), non_compute=False):
    """Check all nested leaves; enforce shape/dtype before pinned CANNBot rules."""
    import torch
    input_tensors = [x for x in inputs if isinstance(x, torch.Tensor)]
    float_input = any(x.dtype.is_floating_point for x in input_tensors)
    evidence = []

    def compare(a, b, path):
        if type(a) is not type(b) and not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            raise ReferenceError(f"{path}: output structure/type differs")
        if isinstance(a, (tuple, list, dict)):
            if len(a) != len(b) or (isinstance(a, dict) and set(a) != set(b)):
                raise ReferenceError(f"{path}: output structure differs")
            for key in (a if isinstance(a, dict) else range(len(a))):
                compare(a[key], b[key], f"{path}.{key}")
            return
        if a is None:
            return
        if not isinstance(a, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
            raise ReferenceError(f"{path}: tensor shape/dtype differs or unsupported output")
        if a.is_complex():
            raise ReferenceError(f"{path}: complex outputs are unsupported")
        a, b = a.detach().cpu().contiguous().flatten(), b.detach().cpu().contiguous().flatten()
        metrics = {"path": path, "dtype": str(a.dtype), "elements": a.numel()}
        if non_compute:
            passed = torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            metrics["category"] = "non_compute"
        else:
            if (not torch.equal(torch.isnan(a), torch.isnan(b)) or
                    not torch.equal(torch.isinf(a), torch.isinf(b)) or
                    not torch.equal(a[torch.isinf(a)], b[torch.isinf(b)])):
                raise ReferenceError(f"{path}: NaN/Inf positions or signs differ", metrics)
            finite = torch.isfinite(a) & torch.isfinite(b)
            a, b = a[finite], b[finite]
            if a.dtype == torch.bool or (not a.dtype.is_floating_point and not float_input):
                passed = torch.equal(a, b)
                metrics["category"] = "exact"
            elif not a.dtype.is_floating_point:
                passed = bool(((a.to(torch.int64) - b.to(torch.int64)).abs() <= 1).all())
                metrics["category"] = "quant_fp_to_int"
            elif a.numel() == 0:
                passed = True
                metrics["category"] = "no_finite_values"
            else:
                sv, small_error, relative, atol, rtol = POLICY.get(str(a.dtype).removeprefix("torch."), POLICY["float32"])
                golden, actual_f = a.float(), b.float()
                diff, magnitude = (actual_f - golden).abs(), golden.abs()
                rel_error = diff / (magnitude + 1e-7)
                matched = torch.where(magnitude < sv, diff <= small_error, rel_error <= relative)
                matched_ratio = int(matched.sum().item()) / matched.numel()
                checks = {"max_error_cap": bool((diff <= atol + rtol*magnitude).all()),
                          "matched_ratio": matched_ratio >= POLICY["required_matched_ratio"],
                          "MERE": rel_error.mean().item() < relative}
                metrics.update(category="float_compute", checks=checks, matched_ratio=matched_ratio,
                               MERE=rel_error.mean().item(), max_abs_diff=diff.max().item())
                passed = all(checks.values())
        evidence.append(metrics)
        if not passed:
            raise ReferenceError(f"{path}: pinned CANNBot comparison failed", metrics)

    compare(expected, actual, "output")
    return {"passed": True, "policy": POLICY["name"], "leaves": evidence}


def _clone(value):
    return copy.deepcopy(value)


def _value_state(value):
    import torch
    if isinstance(value, torch.Tensor):
        return _tensor_identity(value)
    if isinstance(value, (list, tuple)):
        return [_value_state(v) for v in value]
    if isinstance(value, dict):
        return {k: _value_state(v) for k, v in value.items()}
    if isinstance(value, (str, bool, int, float, type(None))):
        return value
    return repr(value)


def _input_state(args, kwargs):
    return _value_state({"args": args, "kwargs": kwargs})


def _model_state(model):
    import torch
    if not isinstance(model, torch.nn.Module):
        return {"globals": {k: _value_state(v) for k, v in getattr(model, "__globals__", {}).items()
                            if not k.startswith("__") and not callable(v) and not isinstance(v, ModuleType)},
                "closure": [_value_state(cell.cell_contents) for cell in (getattr(model, "__closure__", None) or [])]}
    attributes = {}
    for name, module in model.named_modules():
        attributes[name] = {k: _value_state(v) for k, v in vars(module).items() if not k.startswith("_")}
    return {"tensors": {k: _tensor_identity(v) for k, v in model.state_dict().items()}, "attributes": attributes}


@dataclass
class PreparedReference:
    spec: dict
    model: object
    cases: list
    manifest: dict

    def invoke(self, *args, **kwargs):
        import torch
        with torch.inference_mode():
            return self.model(*args, **kwargs)


def _run_checked(prepared, args, kwargs):
    import torch
    before, state_before = _input_state(args, kwargs), _model_state(prepared.model)
    rng_before = torch.get_rng_state()
    try:
        output = prepared.invoke(*args, **kwargs)
    finally:
        if before != _input_state(args, kwargs) or state_before != _model_state(prepared.model):
            raise ReferenceError("reference mutates inputs or module state")
        if not torch.equal(rng_before, torch.get_rng_state()):
            raise ReferenceError("random/nondeterministic forward is unsupported")
    return output


def prepare_reference(spec, device="cpu", max_cases=24, seed=0):
    """Execute in an isolated worker; reject reference side effects and freeze cases."""
    import torch
    spec = validate_submission(spec)
    tree = ast.parse(spec["source"])
    if any(isinstance(node, ast.Attribute) and node.attr in {"backward", "requires_grad_", "grad"}
           for node in ast.walk(tree)):
        raise ReferenceError("training/backward is unsupported")
    namespace = {"__name__": "optimizer_torch_reference", "torch": torch}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        exec(compile(tree, "<optimizer-reference>", "exec"), namespace)
        model = namespace.get(spec["entrypoint"])
        if isinstance(model, type):
            if not issubclass(model, torch.nn.Module):
                raise ReferenceError("class entrypoint must inherit torch.nn.Module")
            model = model(*spec.get("init_args", []), **spec.get("init_kwargs", {})).to(device).eval()
            model.requires_grad_(False)
        elif spec.get("init_args") or spec.get("init_kwargs"):
            raise ReferenceError("constructor args require a Module class")
        if not callable(model):
            raise ReferenceError("entrypoint is not callable")
        suite = generate_cases(spec, max_cases=max_cases, seed=seed)
        prepared = PreparedReference(spec, model, [], {"policy": copy.deepcopy(POLICY), "seed": seed,
                    "source_sha256": hashlib.sha256(spec["source"].encode()).hexdigest(),
                    "rejected": suite["rejected"], "cases": []})
        for case in suite["cases"]:
            try:
                args, kwargs = materialize_case(case, device)
                before = _input_state(args, kwargs)
                output = _clone(_run_checked(prepared, args, kwargs))
                again_args, again_kwargs = materialize_case(case, device)
                again = _run_checked(prepared, again_args, again_kwargs)
                try:
                    compare_outputs(output, again, non_compute=True)
                except ReferenceError as exc:
                    raise ReferenceError("reference is nondeterministic: " + str(exc)) from exc
                _leaves(output)
                prepared.cases.append({"case": case, "args": args, "kwargs": kwargs, "output": output})
                prepared.manifest["cases"].append({**case, "inputs": before,
                    "outputs": {path: _tensor_identity(x) if x is not None else None for path, x in _leaves(output)}})
            except ReferenceError:
                raise
            except Exception as exc:
                if case["representative"]:
                    raise ReferenceError(f"representative {case['id']} failed: {type(exc).__name__}: {exc}") from exc
                prepared.manifest["rejected"].append({"case": case, "error": f"{type(exc).__name__}: {exc}"})
        prepared.manifest["suite_sha256"] = digest(prepared.manifest["cases"])
        prepared.manifest["state"] = _model_state(model)
        return prepared


def freeze_reference(prepared, directory):
    """Write a new immutable baseline bundle; refuse overwriting any existing files."""
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = copy.deepcopy(prepared.manifest)
    with (directory / "reference.py").open("x") as stream:
        stream.write(prepared.spec["source"])
    with (directory / "baseline.pt").open("xb") as stream:
        torch.save({"cases": prepared.cases, "state_dict": prepared.model.state_dict()
                    if isinstance(prepared.model, torch.nn.Module) else {}}, stream)
    manifest["baseline_sha256"] = hashlib.sha256((directory / "baseline.pt").read_bytes()).hexdigest()
    manifest["submission"] = prepared.spec
    with (directory / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
    return manifest


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Freeze a deterministic Torch reference in a worker")
    parser.add_argument("submission", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-cases", type=int, default=24)
    args = parser.parse_args(argv)
    prepared = prepare_reference(json.loads(args.submission.read_text()), device=args.device, max_cases=args.max_cases)
    manifest = freeze_reference(prepared, args.output)
    print(json.dumps({"cases": len(manifest["cases"]), "rejected": len(manifest["rejected"]),
                      "suite_sha256": manifest["suite_sha256"], "baseline_sha256": manifest["baseline_sha256"]}))


if __name__ == "__main__":
    main()
