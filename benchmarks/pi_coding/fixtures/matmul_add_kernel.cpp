#include "kernel_operator.h"

// One 64x64 output tile per block. Reduce K in chunks of 32.
// A[M,K] and B[K,N]: half. C[M,N] and output[M,N]: float.
extern "C" __global__ __aicore__ void MatmulAddTile(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR output,
    uint32_t m, uint32_t n, uint32_t k) {
  // TODO: implement tiled D = A @ B + C, including edge tiles.
}
