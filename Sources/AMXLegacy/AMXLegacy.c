#include "AMXLegacy.h"

// See the header: the migration to `BNNSGraph` was measured (M39) and rejected.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

ssize_t amx_matmul_workspace(bool transA, bool transB, float alpha,
                             const BNNSNDArrayDescriptor *A, const BNNSNDArrayDescriptor *B,
                             const BNNSNDArrayDescriptor *C, const BNNSFilterParameters *params) {
    return BNNSMatMulWorkspaceSize(transA, transB, alpha, A, B, C, params);
}

int amx_matmul(bool transA, bool transB, float alpha,
               const BNNSNDArrayDescriptor *A, const BNNSNDArrayDescriptor *B,
               const BNNSNDArrayDescriptor *C, void *workspace, const BNNSFilterParameters *params) {
    return BNNSMatMul(transA, transB, alpha, A, B, C, workspace, params);
}

#pragma clang diagnostic pop
