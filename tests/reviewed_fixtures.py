import base64
import json
from types import SimpleNamespace
from benchmarks.a5kernels.evidence import canonical_bytes
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
from test_a5_phase1_adaptive import _project, _memory as project_memory
from optimizer.jobs import canonical, sha

def snapshot(identity="reviewed-seed"):
    seed = _project("phase1-copy", "gm-ub-movement")
    return SimpleNamespace(snapshot_id=identity, verify=lambda: None,
                           memory_entries=lambda: [project_memory(ProjectSpec.from_mapping(seed)).as_dict()])

def project(ordinal):
    value = _project(f"new-project-{ordinal}", "simd-vector")
    value["shape"] = [256 + ordinal]
    for tensor in value["graph"]["inputs"]:
        tensor["shape"] = value["shape"]
    return value

def seed(entries=()):
    artifact = b'original kernel code'
    body = dict(schema='optimizer-memory-snapshot-v1', memory_entries=list(entries),
                files={'source.c': dict(sha256=sha(artifact), base64=base64.b64encode(artifact).decode())})
    return {**body, 'snapshot_id': sha(canonical(body))}

def proposal(source='import torch\ndef forward(a,b,r): return torch.relu(a @ b + r)'):
    return dict(task_id='matmul-residual-relu', rationale='new composed kernel', citations=[],
                submission=dict(source=source, entrypoint='forward', signatures=[dict(args=[
                    dict(shape=[4, 8], dtype='float32'), dict(shape=[8, 16], dtype='float32'),
                    dict(shape=[4, 16], dtype='float32')])]))
