// Trusted binding: candidate tools cannot modify this file.
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <cmath>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "kernel.cpp"

torch::Tensor MatmulAdd(const torch::Tensor &a, const torch::Tensor &b,
                        const torch::Tensor &c) {
  TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1,
              "A must be on NPU");
  TORCH_CHECK(b.device() == a.device() && c.device() == a.device(),
              "inputs must share an NPU device");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2,
              "inputs must be matrices");
  TORCH_CHECK(a.scalar_type() == torch::kFloat16 &&
                  b.scalar_type() == torch::kFloat16 &&
                  c.scalar_type() == torch::kFloat32,
              "A/B must be FP16 and C must be FP32");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
              "inputs must be contiguous");
  const uint32_t m = a.size(0), k = a.size(1), n = b.size(1);
  TORCH_CHECK(m > 0 && n > 0 && k > 0 && m <= 512 && n <= 512 && k <= 512,
              "dimensions must be in [1,512]");
  TORCH_CHECK(b.size(0) == k && c.size(0) == m && c.size(1) == n,
              "matrix shapes are incompatible");
  auto output = torch::full_like(c, NAN);
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  const uint32_t blocks = ((m + 63U) / 64U) * ((n + 63U) / 64U);
  auto launch = [=]() -> int {
    MatmulAddTile<<<blocks, nullptr, stream>>>(
        (GM_ADDR)a.data_ptr(), (GM_ADDR)b.data_ptr(), (GM_ADDR)c.data_ptr(),
        (GM_ADDR)output.data_ptr(), m, n, k);
    return 0;
  };
  at_npu::native::OpCommand::RunOpApi("RSIPiMatmulAdd", launch);
  return output;
}

TORCH_LIBRARY(rsi_pi_matmul, registry) {
  registry.def("matmul_add(Tensor a, Tensor b, Tensor c) -> Tensor");
}
TORCH_LIBRARY_IMPL(rsi_pi_matmul, PrivateUse1, registry) {
  registry.impl("matmul_add", MatmulAdd);
}
