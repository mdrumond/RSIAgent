"""Curriculum admission policy selected once by target composition.

An admitted dtype is a configured experiment scope, not a hardware support
claim. The caller must validate its selected compiler/device independently.
"""

from copy import deepcopy
from dataclasses import dataclass

from benchmarks.a5kernels.phase1_curriculum import ProjectSpec


METRIC_SWEEP_GUIDANCE = """
Curriculum task selection is not an autotuning queue. Do not schedule separate
projects or tasks that merely vary one parameter (a=1, a=2, a=4, ...) to collect
the same metric. Request one bounded experiment covering the values through the
route's existing supported sweep or batch fields. For ProjectSpec 1-D timing,
use timing-sweep with timing_lengths listing the values and request-timing;
profile_metric selects an available measurement, not an arbitrary parameter sweep.
For executable optimizer tasks, group representative values in one submission's
signatures within existing correctness-case bounds. Do not invent sweep fields.
If the required sweep or metric is unavailable, explain the capability limit and
defer that measurement or choose another meaningful question; do not simulate a
sweep by scheduling a sequence of nearly identical tasks. Distinct correctness
cases and genuine failure isolation remain valid controlled experiments, including
shape or alignment changes. This rule governs curriculum selection, not legitimate
internal optimizer candidate or tuning attempts.
"""


_A3_CAPABILITIES = (
    "gm-ub-movement", "simd-vector", "padded-tail", "vector-reduction",
    "vector-mask", "cube-tiling", "failure-repair", "timing-sweep", "profiling",
)
_A3_FOCI = (
    "pipe-scalar", "pipe-vector", "pipe-cube", "pipe-mte1", "pipe-mte2",
    "pipe-mte3", "pipe-fix", "inter-pipe-sync", "cross-core-sync",
    "sync-correctness", "sync-performance", "tool-msprof",
)


@dataclass(frozen=True)
class CurriculumPolicy:
    target: str
    dtypes: tuple[str, ...]
    capabilities: tuple[str, ...]
    learning_foci: tuple[str, ...]
    system: str
    guidance: str

    def admit(self, spec: ProjectSpec) -> None:
        if self.target == "a3" and spec.profile_metric is not None and spec.profile_metric.value != "pipe-utilization":
            raise ValueError("native profiling metric unavailable; only pipe-utilization is supported")
        if spec.dtype.value not in self.dtypes:
            raise ValueError(f"dtype unavailable for configured {self.target} curriculum")
        if any(item.value not in self.capabilities for item in spec.capabilities):
            raise ValueError(f"capability unavailable for {self.target} curriculum")
        if any(item not in self.learning_foci for item in spec.learning_focus):
            raise ValueError(f"learning focus unavailable for {self.target} curriculum")

    def operation_capabilities(self) -> dict[str, tuple[str, ...]]:
        """Map target operation metadata to the shared semantic graph."""
        native = self.target == "a3"
        return {
            "simd-vector": ("add", "multiply", "relu"),
            "vector-reduction" if native else "simt-reduction": ("reduce-sum",),
            "vector-mask" if native else "simt-mask": ("mask-positive",),
            "cube-tiling": ("matmul",),
        }

    def grammar(self, legacy: dict[str, object]) -> dict[str, object]:
        grammar = deepcopy(legacy)
        grammar["capabilities"] = list(self.capabilities)
        grammar["dtypes"] = list(self.dtypes)
        grammar["optional_project_fields"] = {"learning_focus": list(self.learning_foci)}
        grammar.pop("low_precision_rules", None)
        grammar["output_dtype"] = "input dtype; host reference owns rounding and accumulation"
        if self.target == "a3":
            grammar["profile_metrics"] = ["pipe-utilization"]
        rules = grammar["capability_operation_rules"]
        for label in ("simd-vector", "simt-reduction", "simt-mask", "cube-tiling"):
            rules.pop(label, None)
        rules.update({label: list(ops) for label, ops in self.operation_capabilities().items()})
        return grammar


def a3_curriculum_policy(*, dtypes: tuple[str, ...] = ("float32",)) -> CurriculumPolicy:
    """Conservative native A3 scope; expand only after independent validation."""
    if (not dtypes or not isinstance(dtypes, tuple)
            or len(set(dtypes)) != len(dtypes)
            or any(item not in {"float16", "bfloat16", "float32"} for item in dtypes)):
        raise ValueError("A3 dtype scope must contain unique float16/bfloat16/float32 names")
    return CurriculumPolicy(
        "a3", dtypes, _A3_CAPABILITIES, _A3_FOCI,
        "You are the provider-neutral A3 Ascend C Phase-1 Curriculum Agent. "
        "Select generic host-verifiable projects, never held-out benchmarks. "
        "Wave siblings share pre-wave memory and cannot depend on each other. "
        "Return exactly one allowed decision and JSON without Markdown.",
        "Build an adaptive A3 Ascend C curriculum using only the supplied grammar "
        "and configured dtypes. Write concrete experiments with Question; Controlled "
        "change; Tools; Measurements; Expected lesson. Study tensor movement, vector "
        "arithmetic, native vector masking/reduction, padded shapes and Cube matrix "
        "work. Test legal producer/consumer ordering with controlled correct variants. "
        "Consult the native programming guide for exact APIs. Use host output to "
        "establish correctness, never source or compiler success. Use request-profile "
        "for supported msprof measurements and request-timing for verified sources. "
        "Keep launch/synchronize latency distinct from instrumented device duration; "
        "missing metrics are unavailable, not zero. Profile only available tools; "
        "never infer product support from another target. Measure final correct "
        "sources without debug instrumentation. Select diverse reusable lessons, "
        "not repeated IDs or renamed hypotheses." + METRIC_SWEEP_GUIDANCE,
    )
