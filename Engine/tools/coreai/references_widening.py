"""References for an operand that widens exactly into its result (f16 or bf16 read by an f32 op), as coreai-torch writes
it: a conversion, then the op. The engine's graph says nothing of it (an f32 op reads a narrower float as its value);
the lowering writes the conversion. Python only to drive Core AI.

    python tools/coreai/references_widening.py OUT_DIR
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from reference import write

out = Path(sys.argv[1])
h, b, f = torch.float16, torch.bfloat16, torch.float32
F = torch.nn.functional


class Add(torch.nn.Module):
    def forward(self, x, y): return x + y.float()


class Mul(torch.nn.Module):
    def forward(self, x, y): return x * y.float()


class Linear(torch.nn.Module):
    def forward(self, x, w): return F.linear(x, w.float())


class MatMul(torch.nn.Module):
    def forward(self, x, w): return x @ w.float()


write(out, "widen_add_bf16_3_5_5", Add(), (torch.rand(3, 5, dtype=f), torch.rand(5, dtype=b)))
write(out, "widen_mul_f16_3_5", Mul(), (torch.rand(3, 5, dtype=f), torch.rand(3, 5, dtype=h)))
write(out, "widen_linear_bf16_3_5_7", Linear(), (torch.rand(3, 5, dtype=f), torch.rand(7, 5, dtype=b)))
write(out, "widen_matmul_f16_3_5_7", MatMul(), (torch.rand(3, 5, dtype=f), torch.rand(5, 7, dtype=h)))
