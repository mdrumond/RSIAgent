#include "kernel_operator.h"

// One 64x64 output tile per block. Reduce K in chunks of 32.
// A[M,K] and B[K,N]: half. C[M,N] and output[M,N]: float.
extern "C" __global__ __aicore__ void MatmulAddTile(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR output,
    uint32_t m, uint32_t n, uint32_t k) {

  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace AscendC;
  InitSocState();

  const uint32_t TILE = 64;
  const uint32_t KC = 32;

  uint32_t nTiles = (n + TILE - 1) / TILE;
  uint32_t blockIdx = GetBlockIdx();
  uint32_t tileRow = blockIdx / nTiles;
  uint32_t tileCol = blockIdx % nTiles;

  uint32_t rowStart = tileRow * TILE;
  uint32_t colStart = tileCol * TILE;
  if (rowStart >= m || colStart >= n) {
    return;
  }
  uint32_t rowCnt = (rowStart + TILE <= m) ? TILE : (m - rowStart);
  uint32_t colCnt = (colStart + TILE <= n) ? TILE : (n - colStart);

  GlobalTensor<half> aGm;
  GlobalTensor<half> bGm;
  GlobalTensor<float> cGm;
  GlobalTensor<float> dGm;
  aGm.SetGlobalBuffer((__gm__ half *)a);
  bGm.SetGlobalBuffer((__gm__ half *)b);
  cGm.SetGlobalBuffer((__gm__ float *)c);
  dGm.SetGlobalBuffer((__gm__ float *)output);

  TPipe pipe;
  TBuf<TPosition::VECCALC> accBuf, aBuf, bBuf, oBuf;
  pipe.InitBuffer(accBuf, TILE * TILE * sizeof(float));
  pipe.InitBuffer(aBuf, TILE * KC * sizeof(half));
  pipe.InitBuffer(bBuf, KC * TILE * sizeof(half));
  pipe.InitBuffer(oBuf, TILE * sizeof(float));
  LocalTensor<float> acc = accBuf.Get<float>();
  LocalTensor<half> aL = aBuf.Get<half>();
  LocalTensor<half> bL = bBuf.Get<half>();
  LocalTensor<float> oL = oBuf.Get<float>();

  Duplicate(acc, 0.0f, TILE * TILE);
  PipeBarrier<PIPE_ALL>();

  DataCopyPadExtParams<half> padH{false, 0, 0, (half)0};
  DataCopyPadExtParams<float> padF{false, 0, 0, 0.0f};

  for (uint32_t kk = 0; kk < k; kk += KC) {
    uint32_t kc = (kk + KC <= k) ? KC : (k - kk);
    for (uint32_t r = 0; r < rowCnt; ++r) {
      DataCopyExtParams ep;
      ep.blockCount = 1;
      ep.blockLen = kc * (uint32_t)sizeof(half);
      ep.srcStride = 0;
      ep.dstStride = 0;
      ep.rsv = 0;
      DataCopyPad(aL[r * KC], aGm[(rowStart + r) * k + kk], ep, padH);
    }
    for (uint32_t kx = 0; kx < kc; ++kx) {
      DataCopyExtParams ep;
      ep.blockCount = 1;
      ep.blockLen = colCnt * (uint32_t)sizeof(half);
      ep.srcStride = 0;
      ep.dstStride = 0;
      ep.rsv = 0;
      DataCopyPad(bL[kx * TILE], bGm[(kk + kx) * n + colStart], ep, padH);
    }
    PipeBarrier<PIPE_ALL>();

    for (uint32_t r = 0; r < rowCnt; ++r) {
      for (uint32_t j = 0; j < colCnt; ++j) {
        float s = 0.0f;
        for (uint32_t kx = 0; kx < kc; ++kx) {
          s += (float)aL.GetValue(r * KC + kx) *
               (float)bL.GetValue(kx * TILE + j);
        }
        acc.SetValue(r * TILE + j, acc.GetValue(r * TILE + j) + s);
      }
    }
    PipeBarrier<PIPE_ALL>();
  }

  for (uint32_t r = 0; r < rowCnt; ++r) {
    DataCopyExtParams ep;
    ep.blockCount = 1;
    ep.blockLen = colCnt * (uint32_t)sizeof(float);
    ep.srcStride = 0;
    ep.dstStride = 0;
    ep.rsv = 0;
    DataCopyPad(oL, cGm[(rowStart + r) * n + colStart], ep, padF);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t j = 0; j < colCnt; ++j) {
      oL.SetValue(j, oL.GetValue(j) + acc.GetValue(r * TILE + j));
    }
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(dGm[(rowStart + r) * n + colStart], oL, ep);
    PipeBarrier<PIPE_ALL>();
  }
}
