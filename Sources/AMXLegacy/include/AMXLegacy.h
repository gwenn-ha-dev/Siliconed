// **The AMX matrix product, behind a single door.**
//
// `BNNSMatMul` has been deprecated since macOS 15 in favor of `BNNSGraph`. M39 measured the migration:
// equal throughput on the engine's shapes, but `BNNSGraph` requires a compiled CoreML model per
// GEMM shape. So we keep `BNNSMatMul` (2.26 TFLOP/s against 1.99 for `cblas_sgemm`, M3/M5),
// and the deprecation is acknowledged here, once, instead of a warning at every call.
// Both functions pass their arguments through unchanged: same calls, same bits.
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

#pragma clang assume_nonnull end
