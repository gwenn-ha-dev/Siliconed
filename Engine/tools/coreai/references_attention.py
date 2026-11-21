"""References of the Attention family: `Op.attention` as coreai-torch writes torch's `scaled_dot_product_attention`
(the engine's [B, H, L, D] operands are torch's own layout, so no reshape surrounds the composite) and `Op.hadamard`
as a reshape and one product per factor. Python only to drive Core AI.

    python tools/coreai/references_attention.py OUT_DIR
    python tools/coreai/swiftrefs.py OUT_DIR coreAIAttentionReferences > Tests/SiliCoreTests/CoreAIAttentionReferences.swift
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from reference import write

out = Path(sys.argv[1])
F = torch.nn.functional
DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}


class Attention(torch.nn.Module):
    def __init__(self, scale=8 ** -0.5, causal=False):
        super().__init__()
        self.scale, self.causal = scale, causal

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, scale=self.scale)


class Masked(torch.nn.Module):
    def __init__(self, scale=8 ** -0.5):
        super().__init__()
        self.scale = scale

    def forward(self, q, k, v, m):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=self.scale)


def qkv(dt, b, h, hk, lq, lk, d):
    return (torch.randn(b, h, lq, d, dtype=dt), torch.randn(b, hk, lk, d, dtype=dt), torch.randn(b, hk, lk, d, dtype=dt))


for name, dt in DTYPES.items():
    write(out, f"sdpa_{name}_1_2_5_8", Attention(), qkv(dt, 1, 2, 2, 5, 5, 8))
    write(out, f"sdpa_scale_{name}_1_2_5_8", Attention(scale=0.25), qkv(dt, 1, 2, 2, 5, 5, 8))
    write(out, f"sdpa_causal_{name}_1_2_5_8", Attention(causal=True), qkv(dt, 1, 2, 2, 5, 5, 8))
write(out, "sdpa_f16_2_2_3_5_8", Attention(), qkv(torch.float16, 2, 2, 2, 3, 5, 8))
write(out, "sdpa_mask_f16_1_2_5_8", Masked(), qkv(torch.float16, 1, 2, 2, 5, 5, 8) + (torch.randn(1, 1, 5, 5, dtype=torch.float16),))
write(out, "sdpa_mask_full_f16_1_2_5_8", Masked(), qkv(torch.float16, 1, 2, 2, 5, 5, 8) + (torch.randn(1, 2, 5, 5, dtype=torch.float16),))


# The Hadamard transform: one product by the group's matrix K = (F1 x ... x Fk) (the last factor on the fastest digit),
# written transposed (y = x . K^T) so that the product has the axis last: groups of G made a digit of their own (a
# reshape, when the axis holds more than one), the axis moved last and back by a permutation (when it is not last).
# Then 1/sqrt(G) inside K when G is a power of 4 (1/sqrt(G) = 2^-k exactly: no second rounding), else as a scalar product.
SYLVESTER = [[1, 1], [1, -1]]
REGULAR = [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]


def kron(factors):
    k = torch.ones(1, 1)
    for f in factors:
        k = torch.kron(k, torch.tensor(f, dtype=torch.float32))
    return k


class Hadamard(torch.nn.Module):
    def __init__(self, shape, axis, factors, dtype):
        super().__init__()
        self.shape, self.axis = shape, axis
        k = kron(factors)
        self.g = k.shape[0]
        self.fold = round(math.log(self.g, 4)) >= 0 and 4 ** round(math.log(self.g, 4)) == self.g
        self.scale = 1 / self.g ** 0.5
        self.register_buffer("kt", (k.T * (self.scale if self.fold else 1)).to(dtype))

    def forward(self, x):
        shape, a, g = self.shape, self.axis, self.g
        groups = shape[a] // g
        v = x
        if groups > 1:
            v = v.reshape(shape[:a] + [groups, g] + shape[a + 1:])
            a += 1
        last = v.dim() - 1
        if a != last:
            perm = list(range(a)) + list(range(a + 1, last + 1)) + [a]
            v = v.permute(perm)
        flat = v.shape
        v = (v.reshape(-1, g) @ self.kt).reshape(flat)
        if a != last:
            v = v.permute([perm.index(i) for i in range(last + 1)])
        v = v.reshape(shape)
        return v if self.fold else v * self.scale


def had(name, shape, axis, factors, dtype=torch.float16):
    write(out, name, Hadamard(shape, axis, factors, dtype), (torch.randn(*shape, dtype=dtype),))


had("hadamard_f16_3_8_s3", [3, 8], 1, [SYLVESTER] * 3)
had("hadamard_bf16_3_8_s3", [3, 8], 1, [SYLVESTER] * 3, dtype=torch.bfloat16)
had("hadamard_group_f16_3_8_s2", [3, 8], 1, [SYLVESTER] * 2)
had("hadamard_regular_f16_3_8", [3, 8], 1, [REGULAR])
had("hadamard_axis0_f16_8_3_s3", [8, 3], 0, [SYLVESTER] * 3)
had("hadamard_axis1_f16_2_8_3_s3", [2, 8, 3], 1, [SYLVESTER] * 3)
had("hadamard_axis1_group_f16_2_8_3_s2", [2, 8, 3], 1, [SYLVESTER] * 2)
had("hadamard_last_f16_2_3_8_s3", [2, 3, 8], 2, [SYLVESTER] * 3)
had("hadamard_fold_axis0_f16_4_3_s2", [4, 3], 0, [SYLVESTER] * 2)
had("hadamard_axis0_group_f16_8_3_s2", [8, 3], 0, [SYLVESTER] * 2)
had("hadamard_axis2_f16_2_3_4_5_s2", [2, 3, 4, 5], 2, [SYLVESTER] * 2)
had("hadamard_fold_group_axis1_bf16_2_8_3_s2", [2, 8, 3], 1, [SYLVESTER] * 2, dtype=torch.bfloat16)
had("hadamard_f32_3_8_s3", [3, 8], 1, [SYLVESTER] * 3, dtype=torch.float32)
had("hadamard_axis0_f16_4_3_5_s2", [4, 3, 5], 0, [SYLVESTER] * 2)
had("hadamard_fold_axis0_group_f16_8_3_5_s2", [8, 3, 5], 0, [SYLVESTER] * 2)
