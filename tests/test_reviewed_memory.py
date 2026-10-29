import hashlib

import json

from types import SimpleNamespace

import pytest

from benchmarks.ascend.attention import ATTENTION_CASES, attention_reference, build_execution

from benchmarks.a5kernels.evidence import canonical_bytes

from benchmarks.a5kernels.phase1_provider import DirectCompletion

from benchmarks.ascend.reviewed_memory import admit_native_snapshot, source_cell_id, REVIEW_SCHEMA
from tests.test_reviewed_catlass_memory import _source, GUIDE, ENVIRONMENT

SOURCE = 'extern "C" __global__ __aicore__ void dense_attention(GM_ADDR gm_q, GM_ADDR gm_k, GM_ADDR gm_v, GM_ADDR gm_output) {}'

def native_source(tmp_path, *, profiling=False, partial=False):
    root, source, review_path, memory = _source(tmp_path, partial=partial)
    identity = dict(schema="ascend-native-adaptive-identity-v1", language="ascend-c",
                    target="Ascend910_9382", arch="dav-c220", execution_profile="bz-a3-1",
                    guide_sha256=hashlib.sha256(GUIDE.encode()).hexdigest(), cpl_skills_revision="1" * 40,
                    model=source.model.as_dict(), dtypes=["float16", "bfloat16"],
                    environment_identity=dict(cann_version="test-cann", profiler_version="test-profiler"),
                    profiling_guidance="Use the native profiler" if profiling else None,
                    profiling_mode="new" if profiling else "without-profiling-guidance",
                    measurement_actions={"profiling": profiling, "timing": False},
                    kdb_collection=None)
    (root / "campaign-identity.json").write_bytes(canonical_bytes(identity))
    review = json.loads(review_path.read_bytes())
    review.update(schema=REVIEW_SCHEMA, cell_id=source_cell_id(identity))
    review_path.write_bytes(canonical_bytes(review))
    return root, review_path, memory

def test_native_handoff_is_immutable_and_rejects_a5(tmp_path):
    root, review, memory = native_source(tmp_path)
    snapshot = admit_native_snapshot(root, b"", review, tmp_path / "reviewed-memory")
    assert snapshot.memory_entries() == (memory,)
    assert snapshot.destination.as_dict()["language"] == "ascend-c"
    assert snapshot.destination.as_dict()["target"] == "a3"
    (root / "state.json").unlink()
    snapshot.verify()
    snapshot.memory_path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        snapshot.verify()

def test_a5_identity_is_not_admitted_as_native(tmp_path):
    root, source, review, _ = _source(tmp_path)
    with pytest.raises(ValueError):
        admit_native_snapshot(root, b"", review, tmp_path / "reviewed-memory")

pytestmark = pytest.mark.usefixtures("scripted_pi")
