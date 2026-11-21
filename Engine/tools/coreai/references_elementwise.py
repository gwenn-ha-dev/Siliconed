"""The Elementwise family's references: the unaries, binaries, comparisons, select, constants and iota beyond the core,
as coreai-torch writes them. Python only to drive Core AI.

    python tools/coreai/references_elementwise.py OUT_DIR
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from reference import write

out = Path(sys.argv[1])
h, b, f = torch.float16, torch.bfloat16, torch.float32
F = torch.nn.functional


def tag(d):
    return str(d).split(".")[1]


def unary(name, fn, dtypes=(h,)):
    class M(torch.nn.Module):
        def forward(self, x): return fn(x)
    for d in dtypes:
        write(out, f"{name}_{tag(d)}_3_5", M(), (torch.rand(3, 5, dtype=d) + 0.5,))


unary("neg", torch.neg, (h, b, f))
unary("abs", torch.abs)
unary("square", torch.square)
unary("sqrt", torch.sqrt, (h, f))
unary("rsqrt", torch.rsqrt)
unary("reciprocal", torch.reciprocal)
unary("log", torch.log)
unary("sin", torch.sin)
unary("cos", torch.cos)
unary("tanh", torch.tanh)
unary("sigmoid", torch.sigmoid)
unary("gelu", lambda x: F.gelu(x, approximate="tanh"), (h, b))
unary("geluerf", lambda x: F.gelu(x), (h, b))


def operand(shape, d):
    return torch.randint(0, 4, shape, dtype=d) if not d.is_floating_point else torch.rand(*shape, dtype=d) + 0.5


def binary(name, fn, dtypes=(h,), shapes=((3, 5), (3, 5))):
    class M(torch.nn.Module):
        def forward(self, x, y): return fn(x, y)
    for d in dtypes:
        write(out, f"{name}_{tag(d)}_" + "_".join("x".join(map(str, s)) for s in shapes), M(),
              tuple(operand(s, d) for s in shapes))


binary("maximum", torch.maximum, (h, b))
binary("minimum", torch.minimum)
binary("pow", torch.pow, (h, f))
binary("maximum", torch.maximum, (h,), ((3, 1), (1, 5)))
binary("pow", torch.pow, (h,), ((3, 5), (5,)))
for n in ("lt", "le", "eq", "ne", "ge", "gt"):
    binary(n, getattr(torch, n), (h, b) if n == "lt" else (h,))
binary("lt", torch.lt, (h,), ((3, 1), (1, 5)))


class Select(torch.nn.Module):
    def forward(self, c, x, y): return torch.where(c, x, y)


for d in (h, b):
    write(out, f"select_{tag(d)}_3_5", Select(), (torch.rand(3, 5) > 0.5, torch.rand(3, 5, dtype=d), torch.rand(3, 5, dtype=d)))
write(out, "select_bcast_3_5", Select(), (torch.rand(3, 1) > 0.5, torch.rand(3, 5, dtype=h), torch.rand(1, 5, dtype=h)))


class CmpSel(torch.nn.Module):
    def forward(self, x, y): return torch.where(x < y, x, y)


write(out, "cmpsel_3_5", CmpSel(), (torch.rand(3, 5, dtype=h), torch.rand(3, 5, dtype=h)))


class Full(torch.nn.Module):
    def forward(self, x): return x + torch.full((3, 5), 0.75, dtype=x.dtype)


write(out, "full_h_3_5", Full(), (torch.rand(3, 5, dtype=h),))


class Zeros(torch.nn.Module):
    def forward(self, x): return x * torch.zeros(3, 5, dtype=x.dtype)


write(out, "zeros_h_3_5", Zeros(), (torch.rand(3, 5, dtype=h),))


class Iota1(torch.nn.Module):
    def forward(self, x): return x + torch.arange(5, dtype=x.dtype).reshape(1, 5).expand(3, 5)


write(out, "iota1_h_3_5", Iota1(), (torch.rand(3, 5, dtype=h),))


class Iota0(torch.nn.Module):
    def forward(self, x): return x + torch.arange(3, dtype=x.dtype).reshape(3, 1).expand(3, 5)


write(out, "iota0_h_3_5", Iota0(), (torch.rand(3, 5, dtype=h),))


class IotaI(torch.nn.Module):
    def forward(self, x): return x + torch.arange(5, dtype=torch.int32).reshape(1, 5).expand(3, 5)


write(out, "iota1_i32_3_5", IotaI(), (torch.ones(3, 5, dtype=torch.int32),))


# Integers: the masks and positions an index builds (comparison, selection, maximum, a constant).
i32 = torch.int32
binary("lt", torch.lt, (i32,))
binary("ge", torch.ge, (i32,))
binary("eq", torch.eq, (i32,))
binary("maximum", torch.maximum, (i32,))
write(out, "select_int32_3_5", Select(), (torch.rand(3, 5) > 0.5, torch.ones(3, 5, dtype=i32), torch.zeros(3, 5, dtype=i32)))


class FullI(torch.nn.Module):
    def forward(self, x): return x + torch.full((3, 5), 7, dtype=x.dtype)


write(out, "full_i32_3_5", FullI(), (torch.ones(3, 5, dtype=i32),))

# Where the shapes differ: which operands `where` broadcasts and what it computes first.
write(out, "select_cond_row_3_5", Select(), (torch.rand(3, 1) > 0.5, torch.rand(3, 5, dtype=h), torch.rand(3, 5, dtype=h)))
write(out, "select_xy_3_5", Select(), (torch.rand(3, 5) > 0.5, torch.rand(3, 1, dtype=h), torch.rand(1, 5, dtype=h)))
write(out, "select_y_rank1_3_5", Select(), (torch.rand(3, 5) > 0.5, torch.rand(3, 5, dtype=h), torch.rand(5, dtype=h)))

# A one-element tensor: no broadcast inside the composites, so their own dates show.
class Sigmoid(torch.nn.Module):
    def forward(self, x): return torch.sigmoid(x)


write(out, "sigmoid_float16_1", Sigmoid(), (torch.rand(1, dtype=h),))

# A scalar operand of `where` (a rank-0 constant: a comparison with zero, a fill): no shape of its own to compute.
class Relu(torch.nn.Module):
    def forward(self, x): return torch.where(x > 0, x, torch.zeros((), dtype=x.dtype))


class ReluX(torch.nn.Module):
    def forward(self, x): return torch.where(x > 0, torch.zeros((), dtype=x.dtype), x)


write(out, "relu_3_5", Relu(), (torch.rand(3, 5, dtype=h),))
write(out, "relux_3_5", ReluX(), (torch.rand(3, 5, dtype=h),))

# Which shape constants `where` writes, and in what order, when the condition or x is a scalar.
write(out, "select_x0_yrow_3_5", Select(), (torch.rand(3, 5) > 0.5, torch.rand((), dtype=h), torch.rand(5, dtype=h)))
write(out, "select_x0_ycol_3_5", Select(), (torch.rand(3, 5) > 0.5, torch.rand((), dtype=h), torch.rand(3, 1, dtype=h)))
write(out, "select_c0_3_5", Select(), (torch.rand(()) > 0.5, torch.rand(3, 5, dtype=h), torch.rand(3, 5, dtype=h)))
write(out, "select_c0x0_3_5", Select(), (torch.rand(()) > 0.5, torch.rand((), dtype=h), torch.rand(3, 5, dtype=h)))
