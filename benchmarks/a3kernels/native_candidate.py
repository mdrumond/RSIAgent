"""Host-owned, operation-neutral native ABI for A3 candidates.

Only device source is agent-authored. Shapes, input data, launch configuration,
and compiler architecture are supplied by the host. Numerical grading belongs
to the shared project oracle, not this transport contract.
"""

from dataclasses import asdict, dataclass
from functools import cached_property
from importlib.resources import files
import json
import math
import re

from .phase1_evidence import A3_EXECUTION_PROFILES, canonical_digest
from .phase1_protocol import SourceFile


@dataclass(frozen=True)
class NativeTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    values: tuple[float, ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        if not re.fullmatch(r"[A-Za-z_]\w*", self.name):
            raise ValueError("tensor name must be an identifier")
        if not self.shape or any(type(n) is not int or n < 1 for n in self.shape):
            raise ValueError("tensor shape must contain positive dimensions")
        if self.dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError("native tensor dtype must be float16, bfloat16 or float32")
        if self.values is not None:
            object.__setattr__(self, "values", tuple(self.values))
            if len(self.values) != self.numel or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in self.values
            ):
                raise ValueError("tensor values must match shape and be finite")

    @property
    def numel(self):
        return math.prod(self.shape)


@dataclass(frozen=True)
class NativeCandidatePlan:
    source: str
    kernel_name: str
    inputs: tuple[NativeTensor, ...]
    output: NativeTensor
    request_id: str
    attempt_id: str
    project_id: str
    execution_profile: str
    target: str
    arch: str
    block_count: int = 1
    logical_device: int = 0

    def __post_init__(self):
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if not self.inputs or any(t.values is None for t in self.inputs):
            raise ValueError("native inputs require host-owned values")
        if self.output.values is not None:
            raise ValueError("output must not supply an expected answer")
        names = [t.name for t in (*self.inputs, self.output)]
        if len(set(names)) != len(names):
            raise ValueError("tensor names must be unique")
        if not re.fullmatch(r"[A-Za-z_]\w*", self.kernel_name):
            raise ValueError("kernel name must be an identifier")
        if self.execution_profile not in A3_EXECUTION_PROFILES:
            raise ValueError("native plan requires an A3 execution profile")
        if not self.target or not self.arch:
            raise ValueError("native plan requires discovered target and compiler architecture")
        if not all((self.request_id, self.attempt_id, self.project_id)):
            raise ValueError("native plan requires request, attempt and project identifiers")
        if self.logical_device != 0 or type(self.block_count) is not int or not 1 <= self.block_count <= 32:
            raise ValueError("native launch requires logical device 0 and 1..32 blocks")
        pattern = r"\s*".join(re.escape(t) for t in re.findall(r'"[^"]*"|\w+|[^\w\s]', self.signature))
        if len(re.findall(pattern, self.source)) != 1:
            raise ValueError("candidate must define the declared native signature exactly once")
        # Freeze staged host bytes when the plan is created, not when a later
        # replay happens to read installed resources after an update.
        self.files

    @property
    def signature(self):
        args = ", ".join(f"GM_ADDR {t.name}" for t in (*self.inputs, self.output))
        return f'extern "C" __global__ __aicore__ void {self.kernel_name}({args})'

    @property
    def input_payload(self):
        return {"inputs": [asdict(t) for t in self.inputs], "output": asdict(self.output)}

    @property
    def padded_length(self):
        return self.output.numel

    @cached_property
    def files(self):
        root = files("benchmarks.a3kernels.candidate_runtime")
        build = dict(arch=self.arch, compiler="bisheng", output="a3_candidate.so", source="kernel.cpp", target=self.target, kernel_name=self.kernel_name)
        return tuple(sorted((
            SourceFile("candidate.cpp", self.source),
            SourceFile("build.json", json.dumps(build, sort_keys=True)),
            SourceFile("host_wrapper.inc", self._wrapper()),
            SourceFile("host_driver.py", root.joinpath("native_driver.py").read_text()),
            SourceFile("legacy_driver.py", root.joinpath("host_driver.py").read_text()),
            SourceFile("a3_profile_driver.py", root.joinpath("a3_profile_driver.py").read_text()),
            SourceFile("native_measurement_driver.py", root.joinpath("native_measurement_driver.py").read_text()),
        )))

    @property
    def source_fingerprint(self):
        return canonical_digest({f.relative_path: f.sha256 for f in self.files})

    @property
    def execution_id(self):
        return canonical_digest(dict(plan=asdict(self), source_fingerprint=self.source_fingerprint))

    def _wrapper(self):
        types = {"float16": "torch::kFloat16", "bfloat16": "torch::kBFloat16", "float32": "torch::kFloat32"}
        allocations = []
        for i, tensor in enumerate(self.inputs):
            allocations.append(f'auto input_{i} = inputs.at({i});')
        shape = ", ".join(map(str, self.output.shape))
        allocations.append(f'auto native_output = torch::empty({{{shape}}}, inputs.at(0).options().dtype({types[self.output.dtype]}));')
        pointers = ", ".join([*(f"(GM_ADDR)input_{i}.data_ptr()" for i in range(len(self.inputs))), "(GM_ADDR)native_output.data_ptr()"])
        return '''#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
// RSI_A3_CANDIDATE_SOURCE_SLOT
torch::Tensor NativeLaunch(const std::vector<torch::Tensor>& inputs) {
''' + "\n".join(allocations) + f'''
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  auto launch = [=]() -> int {{
    ::{self.kernel_name}<<<{self.block_count}, nullptr, stream>>>({pointers});
    return 0;
  }};
  at_npu::native::OpCommand::RunOpApi("RSIA3NativeCandidate", launch);
  return native_output;
}}
TORCH_LIBRARY(rsi_a3native, registry) {{
  registry.def("launch(Tensor[] inputs) -> Tensor");
}}
TORCH_LIBRARY_IMPL(rsi_a3native, PrivateUse1, registry) {{
  registry.impl("launch", NativeLaunch);
}}
'''
