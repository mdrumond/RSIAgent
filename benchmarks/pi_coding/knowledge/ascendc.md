# Ascend C starting knowledge

This is portable guidance distilled from inspected CANN examples, not a solution kernel. The task contract and installed compiler remain authoritative. The historical comparison used a different prompt containing source excerpts; this file reproduces the methodology, not those exact prompt bytes.

- Ascend C device kernels may need an explicit kernel task type when compiler inference fails. Native CANN examples place `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)` inside cube-only kernels and `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` inside vector-only kernels. Select the type appropriate to the implementation; these declarations are not interchangeable.
- CATLASS matmul-plus-add examples separate tiled matrix multiplication from an elementwise-add epilogue. Their host launcher, template libraries, tile sizes and output types are example-specific. Do not copy them into a device-only benchmark without checking its build interface and dtype contract.
- Preserve the benchmark entrypoint and trusted host launcher. Work within the installed Ascend C toolchain; do not assume CATLASS or Blaze headers are available to candidate builds.
- Treat aligned dimensions, partial M/N tiles, reduction tails, local buffer capacities, transfer alignment and synchronization as separate correctness concerns. Compiler feedback and every numerical case must pass before claiming completion.
- A compile pass is not a numerical pass. A numerical pass is not evidence of high throughput. Keep changes focused on the most recent authoritative diagnostic.

Provenance: CANN `ops-tensor`, `examples/mat_mul/mat_mul_basic/mat_mul_basic.cpp`; CANN `ops-transformer`, `mc2/moe_distribute_combine_setup/op_kernel/moe_distribute_combine_setup.cpp` and the corresponding teardown example; CATLASS `examples/03_matmul_add/matmul_add.cpp`. These examples were inspected on 2026-10-16 and identify themselves with the CANN Open Software License Agreement Version 2. These notes are an independently written summary; no third-party implementation is bundled.
