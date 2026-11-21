"""The Reductions family's references: coreai-torch's own assets for sum, mean, max, min, argmax, softmax and the two
normalizations, one thing varied at a time (the element type, the axes, keep or drop). Python only to drive Core AI.

    python tools/coreai/references_reductions.py OUT_DIR
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import coreai_torch
from coreai_torch.composite_ops import RMSNorm, RMSNormImpl

sys.path.insert(0, str(Path(__file__).parent))
from reference import write

out = Path(sys.argv[1])
# Two graphs in one asset (`main` invokes a composite of its own, named with a per-process hash): read, not rebuilt.
composites = out / "composites"
h, b, f = torch.float16, torch.bfloat16, torch.float32
EPS = 1e-5


class Reduce(torch.nn.Module):
    def __init__(self, kind, dim, keep):
        super().__init__()
        self.kind, self.dim, self.keep = kind, dim, keep

    def forward(self, x):
        match self.kind:
            case "sum": return x.sum(self.dim, keepdim=self.keep)
            case "mean": return x.mean(self.dim, keepdim=self.keep)
            case "max": return x.amax(self.dim, keepdim=self.keep)
            case "min": return x.amin(self.dim, keepdim=self.keep)
            case "argmax": return x.argmax(self.dim, keepdim=self.keep)


class Softmax(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x): return F.softmax(x, dim=self.dim)


class RMSDecomposed(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x): return F.rms_norm(x, self.dims, eps=EPS)


class LayerNorm(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x): return F.layer_norm(x, self.dims, eps=EPS)


tag = {h: "f16", b: "bf16", f: "f32"}
for kind in ["sum", "mean", "max", "min", "argmax"]:
    write(out, f"{kind}_1_keep_3_5", Reduce(kind, 1, True), (torch.randn(3, 5, dtype=h),))
    write(out, f"{kind}_1_drop_3_5", Reduce(kind, 1, False), (torch.randn(3, 5, dtype=h),))
    write(out, f"{kind}_0_drop_3_5", Reduce(kind, 0, False), (torch.randn(3, 5, dtype=h),))
    write(out, f"{kind}_2_drop_2_3_5", Reduce(kind, 2, False), (torch.randn(2, 3, 5, dtype=h),))
for kind in ["sum", "mean", "max", "min"]:
    write(out, f"{kind}_0_2_keep_2_3_5", Reduce(kind, [0, 2], True), (torch.randn(2, 3, 5, dtype=h),))
    write(out, f"{kind}_0_2_drop_2_3_5", Reduce(kind, [0, 2], False), (torch.randn(2, 3, 5, dtype=h),))
    write(out, f"{kind}_bf16_1_drop_3_5", Reduce(kind, 1, False), (torch.randn(3, 5, dtype=b),))
    write(out, f"{kind}_f32_1_drop_3_5", Reduce(kind, 1, False), (torch.randn(3, 5, dtype=f),))
write(out, "sum_0_1_drop_3_5", Reduce("sum", [0, 1], False), (torch.randn(3, 5, dtype=h),))
write(out, "mean_0_1_drop_3_5", Reduce("mean", [0, 1], False), (torch.randn(3, 5, dtype=h),))
write(out, "sum_1_2_drop_2_3_5", Reduce("sum", [1, 2], False), (torch.randn(2, 3, 5, dtype=h),))
write(out, "mean_1_drop_2_3_5", Reduce("mean", 1, False), (torch.randn(2, 3, 5, dtype=h),))
write(out, "argmax_bf16_1_drop_3_5", Reduce("argmax", 1, False), (torch.randn(3, 5, dtype=b),))
write(out, "argmax_f32_1_drop_3_5", Reduce("argmax", 1, False), (torch.randn(3, 5, dtype=f),))

for t in (h, b, f):
    write(out, f"softmax_{tag[t]}_3_5", Softmax(-1), (torch.randn(3, 5, dtype=t),))
write(out, "softmax_0_f16_3_5", Softmax(0), (torch.randn(3, 5, dtype=h),))
write(out, "softmax_1_f16_2_3_5", Softmax(1), (torch.randn(2, 3, 5, dtype=h),))

for t in (h, b, f):
    write(out, f"rms_{tag[t]}_3_5", RMSDecomposed((5,)), (torch.randn(3, 5, dtype=t),))
    write(composites, f"layernorm_{tag[t]}_3_5", LayerNorm((5,)), (torch.randn(3, 5, dtype=t),))
write(out, "rms_f16_2_3_5_last2", RMSDecomposed((3, 5)), (torch.randn(2, 3, 5, dtype=h),))
write(composites, "layernorm_f16_2_3_5_last2", LayerNorm((3, 5)), (torch.randn(2, 3, 5, dtype=h),))


class LayerNormInline(torch.nn.Module):
    """Layer norm spelled out in the composite's own arithmetic (f32 statistics, one rounding), without gamma and beta:
    what coreai-torch's `layer_norm` composite holds, minus the composite and its affine terms."""
    def __init__(self, dims):
        super().__init__()
        self.axes = tuple(range(-len(dims), 0))

    def forward(self, x):
        x32 = x.to(torch.float32)
        centered = x32 - x32.mean(self.axes, keepdim=True)
        variance = (centered * centered).mean(self.axes, keepdim=True)
        return (centered * torch.rsqrt(variance + EPS)).to(x.dtype)


for t in (h, b, f):
    write(out, f"layernorminline_{tag[t]}_3_5", LayerNormInline((5,)), (torch.randn(3, 5, dtype=t),))
write(out, "layernorminline_f16_2_3_5_last2", LayerNormInline((3, 5)), (torch.randn(2, 3, 5, dtype=h),))

# The RMSNorm composite (externalized, as a model would be exported): a graph of its own, invoked by `main`.
def write_composite(name, module, args, specs):
    composites.mkdir(parents=True, exist_ok=True)
    from coreai._compiler._mlir_libs._mlir import passmanager
    ep = lambda m: torch.export.export(m, args).run_decompositions(coreai_torch.get_decomp_table())
    prog = coreai_torch.TorchConverter().add_pytorch_module(module.eval(), export_fn=ep, externalize_modules=specs).to_coreai()
    mm = prog._module._mlir_module
    with mm.context:
        passmanager.PassManager.parse("builtin.module(strip-debuginfo)").run(mm.operation)
    (composites / (name + ".mlir")).write_text(mm.operation.get_asm(print_generic_op_form=True))
    prog.save_asset(composites / (name + ".aimodel"))


from coreai_torch import ExternalizeSpec
for t in (h, f):
    write_composite(f"rmscomposite_{tag[t]}_3_5", RMSNorm(5, eps=EPS).to(t), (torch.randn(3, 5, dtype=t),),
                    [ExternalizeSpec(RMSNormImpl, composite_op_name="rms_norm", composite_attrs=["eps"])])
