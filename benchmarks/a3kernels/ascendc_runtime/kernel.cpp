#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "kernel_operator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace rsi_a3kernels {

__global__ __aicore__ void VectorAddKernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR output, uint32_t count,
    uint32_t buffer_bytes) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::QuePosition::VECIN> a_buffer;
  AscendC::TBuf<AscendC::QuePosition::VECIN> b_buffer;
  AscendC::TBuf<AscendC::QuePosition::VECOUT> output_buffer;
  pipe.InitBuffer(a_buffer, buffer_bytes);
  pipe.InitBuffer(b_buffer, buffer_bytes);
  pipe.InitBuffer(output_buffer, buffer_bytes);

  AscendC::GlobalTensor<float> a_global;
  AscendC::GlobalTensor<float> b_global;
  AscendC::GlobalTensor<float> output_global;
  a_global.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a), count);
  b_global.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b), count);
  output_global.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output), count);
  auto a_local = a_buffer.Get<float>();
  auto b_local = b_buffer.Get<float>();
  auto output_local = output_buffer.Get<float>();

  AscendC::DataCopyExtParams copy{};
  copy.blockCount = 1;
  copy.blockLen = count * sizeof(float);
  copy.srcStride = 0;
  copy.dstStride = 0;
  AscendC::DataCopyPadExtParams<float> padding{false, 0, 0, 0};
  AscendC::DataCopyPad(a_local, a_global, copy, padding);
  AscendC::DataCopyPad(b_local, b_global, copy, padding);
  auto loaded = pipe.FetchEventID<AscendC::HardEvent::MTE2_V>();
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(loaded);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(loaded);
  AscendC::Add(output_local, a_local, b_local, count);
  auto computed = pipe.FetchEventID<AscendC::HardEvent::V_MTE3>();
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(computed);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(computed);
  AscendC::DataCopyPad(output_global, output_local, copy);
}

torch::Tensor VectorAdd(const torch::Tensor &a, const torch::Tensor &b) {
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

  auto output = torch::empty_like(a);
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  const auto count = static_cast<uint32_t>(a.numel());
  const uint32_t buffer_bytes = (count * sizeof(float) + 31U) & ~31U;
  auto launch = [=]() -> int {
    VectorAddKernel<<<1, nullptr, stream>>>(
        (GM_ADDR)a.data_ptr(), (GM_ADDR)b.data_ptr(),
        (GM_ADDR)output.data_ptr(), count, buffer_bytes);
    return 0;
  };
  at_npu::native::OpCommand::RunOpApi("RSIA3VectorAdd", launch);
  return output;
}

TORCH_LIBRARY(rsi_a3kernels, registry) {
  registry.def("vector_add(Tensor a, Tensor b) -> Tensor");
}
TORCH_LIBRARY_IMPL(rsi_a3kernels, PrivateUse1, registry) {
  registry.impl("vector_add", VectorAdd);
}

}  // namespace rsi_a3kernels
