// **The AMX matrix product, behind a single door.**
//
// `BNNSMatMul` has been deprecated since macOS 15 in favor of `BNNSGraph`. The migration was measured:
// equal throughput on the engine's shapes, but `BNNSGraph` requires a compiled CoreML model per
// GEMM shape. So we keep `BNNSMatMul` (2.26 TFLOP/s against 1.99 for `cblas_sgemm`, measured),
// and the deprecation is acknowledged here, once, instead of a warning at every call.
// The two matrix functions pass their arguments through unchanged: same calls, same bits. The third, the softmax's
// NEON pass, lives here because Swift's SIMD types ran it five times slower.
#pragma once
#include <Accelerate/Accelerate.h>
#include <stdbool.h>

#pragma clang assume_nonnull begin

ssize_t amx_matmul_workspace(bool transA, bool transB, float alpha,
                             const BNNSNDArrayDescriptor *A, const BNNSNDArrayDescriptor *B,
                             const BNNSNDArrayDescriptor *C,
                             const BNNSFilterParameters *_Nullable params);

int amx_matmul(bool transA, bool transB, float alpha,
               const BNNSNDArrayDescriptor *A, const BNNSNDArrayDescriptor *B,
               const BNNSNDArrayDescriptor *C, void *_Nullable workspace,
               const BNNSFilterParameters *_Nullable params);

// **The softmax's numerator and denominator in one pass** (NEON; `HeadSplitAttention.exponentiate`):
// row[j] = exp(scale * row[j] + offset) in place, the sum of the row returned. `offset` is -scale * max, so
// every argument is <= 0. Cephes' expf (relative error < 2e-7), fused multiply-adds, 8 fp32 partial sums.
float amx_softmax_exponentiate(float *row, long count, float scale, float offset);

#pragma clang assume_nonnull end
