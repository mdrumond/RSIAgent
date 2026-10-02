#include <ATen/Operators.h>
#include <cstdint>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace rsi_a3candidate_host {
torch::Tensor VectorAdd(const torch::Tensor &a, const torch::Tensor &b,
                        int64_t padded_length, int64_t block_count) {
  TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1,
              "a must be an NPU tensor");
  TORCH_CHECK(b.device() == a.device() && a.sizes() == b.sizes(),
              "inputs must have the same NPU shape");
  TORCH_CHECK(a.scalar_type() == torch::kFloat32 &&
                  b.scalar_type() == torch::kFloat32,
              "inputs must use float32");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
              "inputs must be contiguous");
  TORCH_CHECK(a.numel() > 0 && a.numel() <= 4096,
              "vector length must be in [1, 4096]");
  TORCH_CHECK(padded_length == a.numel(),
              "padded length must equal the allocated tensor length");
  TORCH_CHECK(block_count >= 1 && block_count <= 32,
              "block count must be in [1, 32]");
  auto output = torch::empty_like(a);
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  const auto count = static_cast<uint32_t>(a.numel());
  const auto launch_blocks = static_cast<uint32_t>(block_count);
  const uint32_t buffer_bytes = (count * sizeof(float) + 31U) & ~31U;
  auto launch = [=]() -> int {
    ::vector_add<<<launch_blocks, nullptr, stream>>>(
        (GM_ADDR)a.data_ptr(), (GM_ADDR)b.data_ptr(),
        (GM_ADDR)output.data_ptr(), count, buffer_bytes);
    return 0;
  };
  at_npu::native::OpCommand::RunOpApi("RSIA3CandidateVectorAdd", launch);
  return output;
}
TORCH_LIBRARY(rsi_a3candidates, registry) {
  registry.def("vector_add(Tensor a, Tensor b, int padded_length, int block_count) -> Tensor");
}
TORCH_LIBRARY_IMPL(rsi_a3candidates, PrivateUse1, registry) {
  registry.impl("vector_add", VectorAdd);
}
}  // namespace rsi_a3candidate_host
