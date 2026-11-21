"""Quantized-family references: int8 codes with their scales dequantized (and quantized) by `coreai.dequantize` /
`coreai.quantize`, as coreai-torch writes them. Python only to drive Core AI.

    python tools/coreai/references_quantized.py OUT_DIR
"""
import sys
from pathlib import Path

import torch
import coreai_torch._compression.custom_layers  # registers torch.ops.coreai.quantize / dequantize
from reference import write

out = Path(sys.argv[1])
h, b, i8 = torch.float16, torch.bfloat16, torch.int8


class Dequantize(torch.nn.Module):
    def __init__(self, axis, dtype):
        super().__init__()
        self.axis, self.dtype = axis, dtype
    def forward(self, w, w_scale):
        return torch.ops.coreai.dequantize(w, w_scale, axis=self.axis, input_dtype=i8, output_dtype=self.dtype)


class DequantizedProduct(torch.nn.Module):
    """y = x @ dequant(w): codes [k, n], a scale per column (axis 1), per row (axis 0), or one."""
    def __init__(self, axis, dtype):
        super().__init__()
        self.axis, self.dtype = axis, dtype
    def forward(self, x, w, w_scale):
        return x @ torch.ops.coreai.dequantize(w, w_scale, axis=self.axis, input_dtype=i8, output_dtype=self.dtype)


class Quantize(torch.nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis
    def forward(self, x, x_scale):
        return torch.ops.coreai.quantize(x, x_scale, i8, axis=self.axis)


class RoundTrip(torch.nn.Module):
    """dequantize(quantize(x, s), s): the cast into a scaled encoding and the cast back out of it."""
    def __init__(self, axis, dtype):
        super().__init__()
        self.axis, self.dtype = axis, dtype
    def forward(self, x, x_scale):
        q = torch.ops.coreai.quantize(x, x_scale, i8, axis=self.axis)
        return torch.ops.coreai.dequantize(q, x_scale, axis=self.axis, input_dtype=i8, output_dtype=self.dtype)


def codes(*shape):
    return torch.randint(-127, 128, shape, dtype=i8)


for dtype, tag in ((h, "f16"), (b, "bf16")):
    write(out, f"dequantize_col_{tag}_5_7", Dequantize(1, dtype), (codes(5, 7), torch.rand(7, dtype=dtype)))
    write(out, f"dequantize_row_{tag}_5_7", Dequantize(0, dtype), (codes(5, 7), torch.rand(5, dtype=dtype)))
    write(out, f"dequantize_tensor_{tag}_5_7", Dequantize(0, dtype), (codes(5, 7), torch.rand(1, dtype=dtype)))
    write(out, f"dequantize_tensor0d_{tag}_5_7", Dequantize(0, dtype), (codes(5, 7), torch.rand((), dtype=dtype)))
    write(out, f"product_col_{tag}_3_5_7", DequantizedProduct(1, dtype), (torch.randn(3, 5, dtype=dtype), codes(5, 7), torch.rand(7, dtype=dtype)))
    write(out, f"product_row_{tag}_3_5_7", DequantizedProduct(0, dtype), (torch.randn(3, 5, dtype=dtype), codes(5, 7), torch.rand(5, dtype=dtype)))
    write(out, f"quantize_col_{tag}_5_7", Quantize(1), (torch.randn(5, 7, dtype=dtype), torch.rand(7, dtype=dtype) + 0.5))
    write(out, f"quantize_row_{tag}_5_7", Quantize(0), (torch.randn(5, 7, dtype=dtype), torch.rand(5, dtype=dtype) + 0.5))
    write(out, f"quantize_tensor_{tag}_5_7", Quantize(0), (torch.randn(5, 7, dtype=dtype), torch.rand(1, dtype=dtype) + 0.5))
# A linear reads the weight [out, in] with a scale per row (axis 0): the codes dequantized, then transposed.
class DequantizedLinear(torch.nn.Module):
    def forward(self, x, w, w_scale):
        return torch.nn.functional.linear(x, torch.ops.coreai.dequantize(w, w_scale, axis=0, input_dtype=i8, output_dtype=h))
write(out, "linear_row_f16_3_5_7", DequantizedLinear(), (torch.randn(3, 5, dtype=h), codes(7, 5), torch.rand(7, dtype=h)))

# What the decomposition could depend on besides the dtype and the axis: the scale's float (f32), the rank, equal sides.
write(out, "dequantize_col_f32_5_7", Dequantize(1, torch.float32), (codes(5, 7), torch.rand(7)))
write(out, "dequantize_col_f16_4_4", Dequantize(1, h), (codes(4, 4), torch.rand(4, dtype=h)))
write(out, "dequantize_axis2_f16_2_3_4", Dequantize(2, h), (codes(2, 3, 4), torch.rand(4, dtype=h)))
write(out, "dequantize_axis1_f16_2_3_4", Dequantize(1, h), (codes(2, 3, 4), torch.rand(3, dtype=h)))
write(out, "quantize_col_f32_5_7", Quantize(1), (torch.randn(5, 7), torch.rand(7) + 0.5))
write(out, "quantize_axis2_f16_2_3_4", Quantize(2), (torch.randn(2, 3, 4, dtype=h), torch.rand(4, dtype=h) + 0.5))
write(out, "roundtrip_col_f16_5_7", RoundTrip(1, h), (torch.randn(5, 7, dtype=h), torch.rand(7, dtype=h) + 0.5))
write(out, "roundtrip_tensor0d_f16_5_7", RoundTrip(0, h), (torch.randn(5, 7, dtype=h), torch.rand((), dtype=h) + 0.5))
write(out, "roundtrip_row_bf16_5_7", RoundTrip(0, b), (torch.randn(5, 7, dtype=b), torch.rand(5, dtype=b) + 0.5))
write(out, "dequantize_tensor0d_f32_5_7", Dequantize(0, torch.float32), (codes(5, 7), torch.rand(())))
write(out, "product_col_f16_16_64_64", DequantizedProduct(1, h), (torch.randn(16, 64, dtype=h), codes(64, 64), torch.rand(64, dtype=h)))

# Two weights: whether coreai-torch shares equal constants between the two dequantizes, and where each lies.
# Equal constants are shared between the two dequantizes (coreai-torch's own order of constants for weights of other
# shapes is not derivable: [3-wide pair, 7-wide pair, axis] for (7, 3) weights, the second weight's first).
class TwoSame(torch.nn.Module):
    def forward(self, x, w1, w1_scale, w2, w2_scale):
        dq = lambda w, s: torch.ops.coreai.dequantize(w, s, axis=1, input_dtype=i8, output_dtype=h)
        return (x @ dq(w1, w1_scale)) * (x @ dq(w2, w2_scale))
write(out, "pair_f16_3_5_7", TwoSame(), (torch.randn(3, 5, dtype=h), codes(5, 7), torch.rand(7, dtype=h), codes(5, 7), torch.rand(7, dtype=h)))
