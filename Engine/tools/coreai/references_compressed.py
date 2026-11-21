"""Compressed constants: a weight's int8 codes and scales as constants of the asset, decoded by
`coreai.blockwise_shift_scale` (coreai-torch's `WeightDequantizeModule`), as Apple's recipes write their weights.
Python only to drive Core AI.

    python tools/coreai/references_compressed.py OUT_DIR
"""
import sys
from pathlib import Path

import torch
from coreai_torch._compression.custom_layers import WeightDequantizeModule
from reference import write

out = Path(sys.argv[1])
i8 = torch.int8


def codes(*shape):
    """(i mod 255) − 127, row-major: what the byte-equality test rebuilds."""
    n = 1
    for d in shape: n *= d
    return ((torch.arange(n) % 255) - 127).to(i8).reshape(shape)


def scales(n, dtype):
    """(i + 1) / 64: exact in every float element."""
    return ((torch.arange(n) + 1).float() / 64).to(dtype)


class Linear(torch.nn.Module):
    """y = x · wᵀ, w [out, in] compressed, a scale per output (per row of w)."""
    def __init__(self, k, n, dtype, per_tensor=False):
        super().__init__()
        q = codes(n, k)
        s = scales(1, dtype).reshape(()) if per_tensor else scales(n, dtype).reshape(n, 1)
        self.w = WeightDequantizeModule(q, s, input_dtype=i8, output_dtype=dtype)
    def forward(self, x):
        return x @ self.w().T


class Stored(torch.nn.Module):
    """y = x · w, w [in, out] compressed (a weight its container keeps transposed), a scale per output (per column)."""
    def __init__(self, k, n, dtype):
        super().__init__()
        q = codes(k, n)
        s = scales(n, dtype).reshape(1, n)
        self.w = WeightDequantizeModule(q, s, input_dtype=i8, output_dtype=dtype)
    def forward(self, x):
        return x @ self.w()


for dtype, tag in ((torch.float16, "f16"), (torch.bfloat16, "bf16"), (torch.float32, "f32")):
    write(out, f"compressed_linear_{tag}_3_32_16", Linear(32, 16, dtype), (torch.ones(3, 32, dtype=dtype),))
    write(out, f"compressed_stored_{tag}_3_32_16", Stored(32, 16, dtype), (torch.ones(3, 32, dtype=dtype),))
write(out, "compressed_tensor_f16_3_32_16", Linear(32, 16, torch.float16, per_tensor=True), (torch.ones(3, 32, dtype=torch.float16),))


def codes4(*shape):
    """(i mod 15) − 7, row-major: int4 in Apple's symmetric range [−7, 7]."""
    n = 1
    for d in shape: n *= d
    return ((torch.arange(n) % 15) - 7).to(i8).reshape(shape)


class Int4(torch.nn.Module):
    """y = x · wᵀ, w [out, in] in int4 codes, an f16 scale per 32 inputs (coreai-models' "4bit")."""
    def __init__(self, k, n):
        super().__init__()
        self.w = WeightDequantizeModule(codes4(n, k), scales(n * k // 32, torch.float16).reshape(n, k // 32),
                                        input_dtype=torch.int4, output_dtype=torch.float16)
    def forward(self, x):
        return x @ self.w().T


write(out, "compressed_int4_f16_3_64_16", Int4(64, 16), (torch.ones(3, 64, dtype=torch.float16),))
