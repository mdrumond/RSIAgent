# FP8 and MXFP8 Phase-1 projects

The host supports `float8_e4m3fn`, `float8_e5m2`, `mxfp8_e4m3fn`, and
`mxfp8_e5m2`, in addition to existing FP16/BF16/FP32 projects. INT8 is deferred.
`DTYPE` is the underlying FP8 type. `OUTPUT_DTYPE` preserves FP8 for copy-only graphs. For all other plain FP8
graphs, quantized FP8 inputs are widened to FP32 before computation, all
intermediates and outputs are FP32, including arithmetic, masking, reductions,
and Cube. MXFP8 outputs are also FP32. This avoids overflowing FP8
intermediates in ordinary arithmetic graphs. Existing FP16/BF16/FP32
semantics remain unchanged. Plain FP8 registers can convert
to FP32/FP16/BF16 and back with explicit width/packing selectors; no E8M0 vector conversion is assumed. Dense width-changing casts must account
for `CastParams(reg_slot=...)` selecting packed lanes; a round trip alone
cannot validate the packing. See the pinned `vector_ops/fp8_full_lanes.py`
for separate full-lane widening and strided-mask checks.

`fp8_copy.py` is a one-block 256-element copy graph using GM-to-UB-to-GM movement. `mxfp8_matmul.py` is a one-block `(16,64,16)` matrix project with
two graph inputs `(16,64)` / `(64,16)` and a single `matmul` node. Host-owned
scale arguments precede the output: `gm_input0, gm_input1, gm_scale0,
gm_scale1, gm_output`. These are validation fixtures, not general tuned kernels.
Compilation and host comparison on the selected target determine correctness.

MXFP8 uses one E8M0 scale for each 32 elements along K. The group count G is
padded even, padded scales have byte127 (1.0), and the reference uses quantized
FP8 values multiplied by these exact scales. A is row-major; B is physically
transposed and exported column-major with logical shape `(K,N)`. Scale A has
physical/logical `(M,G)` and RowMajorMxScaleA. Scale B has physical `(N,G)`,
logical `(G,N)`, and ColMajorMxScaleB. This layout pair requires no pair
interleave. E8M0 bytes are reinterpreted with `element_type=tla.Float8E8M0`;
the two scale-bearing L1-to-L0 copies feed `tla.mmad_mx`, not `tla.mmad`.

Both low-precision Cube forms currently require exactly one ordered matmul node
and no physical-padding capability. Logical K tails are supported by the host
scale packing; a candidate must still implement its own legal device tail path.
The fixed fixture above covers aligned K=64 only.

Source: [pinned Catlass MX example](https://raw.gitcode.com/cann/catlass/raw/698379e0d4b1b94d12728e7181df2cf83a958e8f/python/tla_dsl/examples/end_to_end/mx_mmad/mx_matmul.py).
