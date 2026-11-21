"""Kernel-family references: a Metal kernel of ours inside a Core AI graph (`coreai.metal4_kernel`), as coreai-torch
writes it — each operand through `copy_with_constraints`, the result through `copy_discarding_constraints`. The one
kernel the engine hosts is the int8 x int8 product; the source here is its shape, not its tuning. Python only to
drive Core AI.

    python tools/coreai/references_kernel.py OUT_DIR

coreai-torch names a kernel `<name>_<8 random letters>`: the name each reference got is printed, and written beside
the asset (`NAME.kernel`), for the byte-equality test to give the engine's writer the same one.
"""
import re
import sys
from pathlib import Path

import torch
import coreai_torch
from coreai.authoring import MetalParameter
from coreai._compiler._mlir_libs._mlir import passmanager

out = Path(sys.argv[1])
i8, f32, b16 = torch.int8, torch.float32, torch.bfloat16


def product_source(rows, cols, sg, result):
    """int8 codes A [M, K] and B [K, N] summed in int32 by matmul2d, each sum scaled by sa[row] * sb[column] in f32 and
    stored in the result's element."""
    return f"""
        const int M = a.get_extent(1), N = c.get_extent(0);
        constexpr auto d = matmul2d_descriptor({rows}, {cols}, static_cast<int>(dynamic_extent), false, false, false);
        matmul2d<d, execution_simdgroups<{sg}>> op;
        const int x = int(code.x) * {cols}, y = int(code.y) * {rows};
        auto ta = a.slice(0, y);
        auto tb = b.slice(x, 0);
        auto t = op.template get_destination_cooperative_tensor<decltype(ta), decltype(tb), int32_t>();
        for (uint16_t i = 0; i < t.get_capacity(); ++i) {{ if (t.is_valid_element(i)) t[i] = 0; }}
        op.run(ta, tb, t);
        for (uint16_t i = 0; i < t.get_capacity(); ++i) {{
            if (!t.is_valid_element(i)) continue;
            const auto at = t.get_multidimensional_index(i);
            const int column = x + at[0], row = y + at[1];
            if (column < N && row < M) c[column, row] = {result}(float(t[i]) * (sa[row] * sb[column]));
        }}
    """


def product(m, n, dtype, rows=64, cols=64, sg=4):
    def defn(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
        return ((a.float() @ b.float()) * sa[:, None] * sb[None, :]).to(dtype)
    tiles = ((n + cols - 1) // cols, (m + rows - 1) // rows)
    kernel = coreai_torch.TorchMetalKernel(
        name="int8_product", input_names=["a", "b", "sa", "sb"], result_names=["c"],
        src=product_source(rows, cols, sg, "bfloat" if dtype == b16 else "float"), torch_defn=defn, helper_src=None,
        metal_params=[MetalParameter(name="code", dtype="uint3", attr="threadgroup_position_in_grid")])
    grid, group = (tiles[0] * 32 * sg, tiles[1], 1), (32 * sg, 1, 1)
    return kernel, lambda a, b, sa, sb: kernel(a, b, sa, sb, threads_per_grid=grid, threads_per_thread_group=group, result_shapes=[[m, n]])


class Product(torch.nn.Module):
    def __init__(self, m, n, dtype):
        super().__init__()
        kernel, self.call = product(m, n, dtype)
        self.kernels = [kernel]
    def forward(self, a, b, sa, sb):
        return self.call(a, b, sa, sb)


class Affine(torch.nn.Module):
    """The product among Core AI's own ops: its result added to a bias."""
    def __init__(self, m, n, dtype):
        super().__init__()
        kernel, self.call = product(m, n, dtype)
        self.kernels = [kernel]
    def forward(self, a, b, sa, sb, bias):
        return self.call(a, b, sa, sb) + bias


def write(name, module, args):
    out.mkdir(parents=True, exist_ok=True)
    ep = torch.export.export(module.eval(), args).run_decompositions(coreai_torch.get_decomp_table())
    prog = coreai_torch.TorchConverter().register_custom_kernels(module.kernels).add_exported_program(ep).to_coreai()
    mm = prog._module._mlir_module
    with mm.context:
        passmanager.PassManager.parse("builtin.module(strip-debuginfo)").run(mm.operation)
    text = mm.operation.get_asm(print_generic_op_form=True)
    (out / (name + ".mlir")).write_text(text)
    (out / (name + ".kernel")).write_text(re.search(r'kernel_name = "(\w+)"', text).group(1))
    prog.save_asset(out / (name + ".aimodel"))
    print(name, (out / (name + ".kernel")).read_text())


def codes(*shape):
    return torch.randint(-127, 128, shape, dtype=i8)


write("kernel_product_f32_3_5_7", Product(3, 7, f32), (codes(3, 5), codes(5, 7), torch.rand(3), torch.rand(7)))
write("kernel_product_bf16_3_5_7", Product(3, 7, b16), (codes(3, 5), codes(5, 7), torch.rand(3), torch.rand(7)))
write("kernel_affine_bf16_3_5_7", Affine(3, 7, b16), (codes(3, 5), codes(5, 7), torch.rand(3), torch.rand(7), torch.rand(7, dtype=b16)))
# Two rows of tiles: the grid is no longer the group, three constants instead of two.
write("kernel_product_f32_70_5_7", Product(70, 7, f32), (codes(70, 5), codes(5, 7), torch.rand(70), torch.rand(7)))
