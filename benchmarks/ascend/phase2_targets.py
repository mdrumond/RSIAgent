"""Fixed Phase 2 goals qualified by executing a reference with the host oracle."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping
from types import MappingProxyType

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import canonical_digest
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
from benchmarks.a5kernels.phase1_project import (
    classify_execution, evaluate_reference, materialize_inputs,
)
from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
from benchmarks.ascend.targets import CurriculumPolicy


def _runtime(value: Mapping[str, str]) -> dict[str, str]:
    value = dict(value)
    if not {"target", "runtime", "device", "toolchain", "product", "compiler"} <= value.keys() or any(
        not isinstance(key, str) or not isinstance(item, str) or not item.strip()
        for key, item in value.items()
    ):
        raise ValueError("runtime identity requires target, runtime, device, toolchain, product and compiler")
    return value


def _timings(rows):
    rows = [dict(row) for row in rows]
    for row in rows:
        if set(row) != {"value", "unit", "conditions", "evidence_sha256"} or (
            isinstance(row["value"], bool) or not isinstance(row["value"], (int, float))
            or not math.isfinite(row["value"]) or row["value"] <= 0
            or row["unit"] not in {"ns", "us", "ms"}
            or not isinstance(row["conditions"], str) or not row["conditions"].strip()
        ):
            raise ValueError("reference timing requires positive device duration, unit and conditions")
    return rows


@dataclass(frozen=True)
class TargetContract:
    """One immutable ProjectSpec and its retained working reference.

    Numerical tolerances and reproducible inputs belong to the existing host
    oracle. A working reference establishes feasibility, not an achievable speedup.
    The artifact store is host-owned; curriculum proposals cannot create targets.
    """

    spec: ProjectSpec
    runtime: Mapping[str, str]
    reference_sha256: str

    def __post_init__(self):
        if not isinstance(self.spec, ProjectSpec):
            raise TypeError("target spec must be ProjectSpec")
        object.__setattr__(self, "runtime", MappingProxyType(_runtime(self.runtime)))
        if (not isinstance(self.reference_sha256, str) or len(self.reference_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.reference_sha256)):
            raise ValueError("reference identity must be SHA-256")

    @property
    def target_id(self):
        return canonical_digest(self.as_dict())

    def as_dict(self):
        return {"spec": self.spec.as_dict(), "runtime": dict(self.runtime),
                "reference_sha256": self.reference_sha256}

    @classmethod
    def from_mapping(cls, value):
        if set(value) != {"spec", "runtime", "reference_sha256"}:
            raise ValueError("target contract fields changed")
        return cls(ProjectSpec.from_mapping(value["spec"]), value["runtime"], value["reference_sha256"])

    def prompt_dict(self):
        return {"target_id": self.target_id, "spec": self.spec.as_dict(),
                "runtime": dict(self.runtime),
                "correctness": {"input_generator": "materialize_inputs(ProjectSpec); exact spec including ID pins deterministic values",
                    "output_shape": list(self.spec.graph.output_shape),
                    "atol_and_rtol": {"float16": 1e-2, "bfloat16": 2e-2, "float8_e4m3fn": 0.0, "float8_e5m2": 0.0}.get(
                        evaluate_reference(self.spec, materialize_inputs(self.spec)).dtype, 1e-5),
                    "oracle": "evaluate_reference; full physical output including padding"},
                "optimization": "Reduce device kernel runtime while preserving correctness; no required speedup."}

    def verify(self, artifacts: ContentAddressedArtifacts, runtime: Mapping[str, str],
               policy: CurriculumPolicy | None = None):
        if _runtime(runtime) != self.runtime:
            raise ValueError("target runtime changed; qualify the reference again")
        if policy is not None:
            if policy.target != self.runtime["target"]:
                raise ValueError("target policy and runtime differ")
            policy.admit(self.spec)
        reference = artifacts.read_json(self.reference_sha256)
        if (reference["schema"] != "ascend-phase2-target-reference-v1"
                or reference["spec"] != self.spec.as_dict()
                or reference["runtime"] != dict(self.runtime)):
            raise ValueError("working reference does not match target contract")
        if not artifacts.read_bytes(reference["source_sha256"]).strip():
            raise ValueError("working reference source is empty")
        payloads = {name: artifacts.read_json(digest)
                    for name, digest in reference["artifacts"].items()}
        expected = evaluate_reference(self.spec, materialize_inputs(self.spec))
        # Recompute the same host contract, rather than trusting a saved verified flag.
        from benchmarks.a5kernels.phase1_project_executor import _expected_payload, _tensor_payload
        if (payloads["inputs"] != {"tensors": [_tensor_payload(t) for t in materialize_inputs(self.spec)]}
                or payloads["expected"] != _expected_payload(expected)):
            raise ValueError("reference inputs or oracle changed")
        observed = payloads["observed"]
        verdict = classify_execution(expected, compile_succeeded=observed["compile_passed"],
            runtime_error=None if observed["runtime_passed"] is True else "reference did not run",
            actual=observed["output"])
        if not verdict.verified:
            raise ValueError("target reference is not host-verified")
        for timing in _timings(reference["timings"]):
            artifacts.read_bytes(timing["evidence_sha256"])


def qualify_target(spec: ProjectSpec, reference_source: str, runtime: Mapping[str, str],
                   executor: ProjectExecutor, artifacts: ContentAddressedArtifacts, *,
                   timings=(), policy: CurriculumPolicy | None = None) -> TargetContract:
    """Execute a reference before enabling a target; no actor-supplied pass flags."""
    runtime = _runtime(runtime)
    if policy is not None:
        if policy.target != runtime["target"]:
            raise ValueError("target policy and runtime differ")
        policy.admit(spec)
    result = executor.execute(spec, reference_source)
    if result.prepared.spec != spec or not result.evidence.verified:
        raise ValueError("target reference must execute and pass the host oracle")
    reference = {"schema": "ascend-phase2-target-reference-v1", "spec": spec.as_dict(),
        "runtime": runtime, "source_sha256": artifacts.put_bytes(reference_source.encode()),
        "artifacts": {name: artifacts.put_json(payload) for name, payload in result.artifacts.items()},
        "timings": _timings(timings)}
    contract = TargetContract(spec, runtime, artifacts.put_json(reference))
    contract.verify(artifacts, runtime, policy)
    return contract
