"""Small Core AI modules, stripped of debug locations, written as assets: the references the engine's bytecode writer is
judged against. Python only to drive Core AI.

    python tools/coreai/references.py OUT_DIR
"""
import sys, torch, coreai_torch
from pathlib import Path
from coreai._compiler._mlir_libs._mlir import passmanager

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
h = torch.float16

def write(name, module, args):
    ep = torch.export.export(module.eval(), args).run_decompositions(coreai_torch.get_decomp_table())
    prog = coreai_torch.TorchConverter().add_exported_program(ep).to_coreai()
    mm = prog._module._mlir_module
    with mm.context:
        passmanager.PassManager.parse("builtin.module(strip-debuginfo)").run(mm.operation)
    (out / (name + ".mlir")).write_text(mm.operation.get_asm(print_generic_op_form=True))
    prog.save_asset(out / (name + ".aimodel"))

class Product(torch.nn.Module):
    def forward(self, x, w): return x @ w
class Affine(torch.nn.Module):
    def forward(self, x, w, b): return x @ w + b
class Silu(torch.nn.Module):
    def forward(self, x): return torch.nn.functional.silu(x)
class Mul(torch.nn.Module):
    def forward(self, a, b): return a * b
class Add(torch.nn.Module):
    def forward(self, a, b): return a + b

write("product_256_512_512", Product(), (torch.randn(256, 512, dtype=h), torch.randn(512, 512, dtype=h)))
write("product_3_5_7", Product(), (torch.randn(3, 5, dtype=h), torch.randn(5, 7, dtype=h)))
write("product_f32_3_5_7", Product(), (torch.randn(3, 5), torch.randn(5, 7)))
write("affine_3_5_7", Affine(), (torch.randn(3, 5, dtype=h), torch.randn(5, 7, dtype=h), torch.randn(7, dtype=h)))
write("silu_3_5", Silu(), (torch.randn(3, 5, dtype=h),))
write("mul_3_5", Mul(), (torch.randn(3, 5, dtype=h), torch.randn(3, 5, dtype=h)))
write("add_3_5", Add(), (torch.randn(3, 5, dtype=h), torch.randn(3, 5, dtype=h)))

# The element types' and the other ops' codes, one thing varied at a time.
class Sub(torch.nn.Module):
    def forward(self, a, b): return a - b
class Div(torch.nn.Module):
    def forward(self, a, b): return a / b
class Exp(torch.nn.Module):
    def forward(self, x): return torch.exp(x)
class Block(torch.nn.Module):
    def forward(self, x, w1, w3, w2): return (torch.nn.functional.silu(x @ w1) * (x @ w3)) @ w2

write("add_bf16_3_5", Add(), (torch.randn(3, 5, dtype=torch.bfloat16), torch.randn(3, 5, dtype=torch.bfloat16)))
write("add_i32_3_5", Add(), (torch.ones(3, 5, dtype=torch.int32), torch.ones(3, 5, dtype=torch.int32)))
write("add_i64_3_5", Add(), (torch.ones(3, 5, dtype=torch.int64), torch.ones(3, 5, dtype=torch.int64)))
write("add_both_3_1_1_5", Add(), (torch.randn(3, 1, dtype=h), torch.randn(1, 5, dtype=h)))
write("sub_3_5", Sub(), (torch.randn(3, 5, dtype=h), torch.randn(3, 5, dtype=h)))
write("div_3_5", Div(), (torch.randn(3, 5, dtype=h), torch.randn(3, 5, dtype=h)))
write("exp_3_5", Exp(), (torch.randn(3, 5, dtype=h),))
write("block_4_8_16", Block(), (torch.randn(4, 8, dtype=h), torch.randn(8, 16, dtype=h), torch.randn(8, 16, dtype=h), torch.randn(16, 8, dtype=h)))

# A weight as a constant (a parameter of the module): where its bytes go.
class ProductConstant(torch.nn.Module):
    def __init__(self, k, n, dtype=h):
        super().__init__()
        self.w = torch.nn.Parameter(torch.arange(k * n, dtype=dtype).reshape(k, n) / 64, requires_grad=False)
    def forward(self, x): return x @ self.w
class TwoConstants(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.arange(35, dtype=h).reshape(5, 7) / 64, requires_grad=False)
        self.w2 = torch.nn.Parameter(torch.arange(21, dtype=h).reshape(7, 3) / 32, requires_grad=False)
    def forward(self, x): return (x @ self.w1) @ self.w2

write("constant_3_5_7", ProductConstant(5, 7), (torch.randn(3, 5, dtype=h),))
write("constant_3_64_64", ProductConstant(64, 64), (torch.randn(3, 64, dtype=h),))
write("constants_3_5_7_3", TwoConstants(), (torch.randn(3, 5, dtype=h),))
# 256 elements or more: a blob of the resource section (its key a hash seeded per process: the test gives it).
write("constant_f32_3_1_256", ProductConstant(1, 256, torch.float32), (torch.randn(3, 1),))

# What Z's products need: a linear (x · wᵀ, the weight stored [out, in]) and a conversion.
class Linear(torch.nn.Module):
    def forward(self, x, w): return torch.nn.functional.linear(x, w)
class Cast(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype
    def forward(self, x): return x.to(self.dtype)

write("linear_3_5_7", Linear(), (torch.randn(3, 5, dtype=h), torch.randn(7, 5, dtype=h)))
write("cast_bf16_f16_3_5", Cast(h), (torch.randn(3, 5, dtype=torch.bfloat16),))
write("cast_f16_f32_3_5", Cast(torch.float32), (torch.randn(3, 5, dtype=h),))
class Transpose(torch.nn.Module):
    def forward(self, w): return w.T
write("transpose_7_5", Transpose(), (torch.randn(7, 5, dtype=h),))
