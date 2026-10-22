"""Host-observed reproducers, separate from admitted reusable knowledge."""
from pathlib import Path
import json

from optimizer.jobs import atomic, canonical


def phase1_policy(dtype):
    tolerance = {'float16': 1e-2, 'bfloat16': 2e-2, 'float32': 1e-5,
                 'float8_e4m3fn': 0.0, 'float8_e5m2': 0.0}[dtype]
    return dict(comparison='math.isclose; complete physical output including padding',
                atol=tolerance, rtol=tolerance)


def retain_phase1_failure(artifacts, spec, source, classification, description, payloads, dtype):
    """Retain the first observed failure for each source; later repairs cannot replace it."""
    index = artifacts.root / 'failure-cases.json'
    rows = json.loads(index.read_bytes()) if index.is_file() else []
    if any(row['candidate_sha256'] == source for row in rows):
        return
    refs = {'source': {'sha256': source}}
    refs.update({role: {'sha256': artifacts.put_json(value)} for role, value in payloads.items()})
    descriptor = dict(schema='failure-case-v1', classification=classification,
        description=description, candidate_sha256=source, project_spec=spec.as_dict(),
        policy=phase1_policy(dtype), artifacts=refs)
    if 'inputs' not in refs or 'expected' not in refs:
        from benchmarks.a5kernels.phase1_project import materialize_inputs, evaluate_reference
        from benchmarks.a5kernels.phase1_project_executor import _tensor_payload, _expected_payload
        inputs = materialize_inputs(spec)
        for role, value in dict(inputs={'tensors': [_tensor_payload(x) for x in inputs]},
                                expected=_expected_payload(evaluate_reference(spec, inputs))).items():
            refs[role] = {'sha256': artifacts.put_json(value)}
    if 'inputs' in refs and 'expected' in refs:
        descriptor['project_contract'] = dict(graph=spec.graph.as_dict(), dtype=spec.dtype.value,
            inputs_sha256=refs['inputs']['sha256'], expected_sha256=refs['expected']['sha256'],
            policy='phase1-classify-execution-v1')
    rows.append(dict(candidate_sha256=source, sha256=artifacts.put_json(descriptor)))
    atomic(index, canonical(rows))


def tensor_payload(value):
    """Lossless logical values plus original layout; previews are explanatory only."""
    import torch
    if isinstance(value, torch.Tensor):
        host = value.detach().cpu().contiguous()
        raw = host.reshape(-1).view(torch.uint8).numpy().tobytes()
        preview = host.reshape(-1)[:8].tolist()
        return dict(encoding='logical-contiguous-bytes-hex', dtype=str(value.dtype),
            shape=list(value.shape), strides=list(value.stride()), storage_offset=value.storage_offset(),
            bytes_hex=raw.hex(), preview=[repr(x) if isinstance(x, (float, complex)) else x for x in preview])
    if isinstance(value, dict):
        return {key: tensor_payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return dict(container=type(value).__name__, items=[tensor_payload(item) for item in value])
    return value


def optimizer_failure(prepared, row, actual, candidate_sha256, error):
    from optimizer.reference import POLICY
    import torch
    state = prepared.model.state_dict() if isinstance(prepared.model, torch.nn.Module) else {}
    return dict(schema='failure-case-v1', classification='host-mismatch', description=str(error),
        candidate_sha256=candidate_sha256, suite_sha256=prepared.manifest['suite_sha256'],
        case=row['case'], policy=POLICY, inputs=tensor_payload(dict(args=row['args'], kwargs=row['kwargs'])),
        state=tensor_payload(state), expected=tensor_payload(row['output']), observed=tensor_payload(actual))
