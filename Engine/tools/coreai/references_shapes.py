"""The Shapes family's references (reshape, slice, concat, broadcast, pad, gather, scatter-add, slice assignment),
written by coreai-torch with the shared writer: what `CoreAIShapes.swift` must equal, byte for byte. Python only to
drive Core AI.

    python tools/coreai/references_shapes.py OUT_DIR [NAME...]
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from reference import write

out = Path(sys.argv[1])
only = sys.argv[2:]
h = torch.float16


def case(name, module, args):
    if only and name not in only:
        return
    try:
        write(out, name, module, args)
    except Exception as e:
        print("FAILED", name, type(e).__name__, str(e)[:300])


def fn(f):
    class M(torch.nn.Module):
        def forward(self, *a): return f(*a)
    return M()


x = torch.randn(3, 5, dtype=h)
y = torch.randn(2, 3, 5, dtype=h)
bf = torch.bfloat16
i32 = lambda *v: torch.tensor(v, dtype=torch.int32)
i64 = lambda *v: torch.tensor(v, dtype=torch.int64)

# reshape
case("reshape_3_5_to_15", fn(lambda t: t.reshape(15)), (x,))
case("reshape_2_3_5_to_6_5", fn(lambda t: t.reshape(6, 5)), (y,))
case("reshape_2_3_5_to_3_10", fn(lambda t: t.reshape(3, 10)), (y,))
case("reshape_f32_3_5_to_5_3", fn(lambda t: t.reshape(5, 3)), (torch.randn(3, 5),))
case("reshape_bf16_3_5_to_5_3", fn(lambda t: t.reshape(5, 3)), (torch.randn(3, 5, dtype=bf),))
case("reshape_i32_3_5_to_5_3", fn(lambda t: t.reshape(5, 3)), (torch.ones(3, 5, dtype=torch.int32),))

# slice
case("slice_3_5_axis1", fn(lambda t: t[:, 1:4]), (x,))
case("slice_3_5_axis0", fn(lambda t: t[1:3]), (x,))
case("slice_3_5_both", fn(lambda t: t[1:3, 2:5]), (x,))
case("slice_2_3_5_mid", fn(lambda t: t[:, 1:3, :]), (y,))
case("slice_2_3_5_all", fn(lambda t: t[1:2, 0:2, 1:4]), (y,))
case("slice_3_5_whole", fn(lambda t: t[:, 0:5]), (x,))
case("slice_f32_3_5", fn(lambda t: t[:, 1:4]), (torch.randn(3, 5),))
case("slice_bf16_3_5", fn(lambda t: t[:, 1:4]), (torch.randn(3, 5, dtype=bf),))
case("slice_i32_3_5", fn(lambda t: t[:, 1:4]), (torch.ones(3, 5, dtype=torch.int32),))
case("narrow_3_5", fn(lambda t: t.narrow(1, 2, 2)), (x,))

# concat
case("concat_3_5_axis0", fn(lambda a, b: torch.cat([a, b], 0)), (x, torch.randn(2, 5, dtype=h)))
case("concat_3_5_axis1", fn(lambda a, b: torch.cat([a, b], 1)), (x, torch.randn(3, 2, dtype=h)))
case("concat3_3_5_axis1", fn(lambda a, b, c: torch.cat([a, b, c], 1)), (x, torch.randn(3, 2, dtype=h), torch.randn(3, 1, dtype=h)))
case("concat_2_3_5_axis2", fn(lambda a, b: torch.cat([a, b], 2)), (y, torch.randn(2, 3, 2, dtype=h)))
case("concat_f32_3_5", fn(lambda a, b: torch.cat([a, b], 1)), (torch.randn(3, 5), torch.randn(3, 2)))
case("concat_bf16_3_5", fn(lambda a, b: torch.cat([a, b], 1)), (torch.randn(3, 5, dtype=bf), torch.randn(3, 2, dtype=bf)))
case("concat_i32_3_5", fn(lambda a, b: torch.cat([a, b], 1)), (torch.ones(3, 5, dtype=torch.int32), torch.ones(3, 2, dtype=torch.int32)))

# broadcast
case("expand_1_5_to_3_5", fn(lambda t: t.expand(3, 5)), (torch.randn(1, 5, dtype=h),))
case("expand_3_1_to_3_5", fn(lambda t: t.expand(3, 5)), (torch.randn(3, 1, dtype=h),))
case("expand_5_to_3_5", fn(lambda t: t.expand(3, 5)), (torch.randn(5, dtype=h),))
case("expand_5_to_2_3_5", fn(lambda t: t.expand(2, 3, 5)), (torch.randn(5, dtype=h),))
case("expand_1_1_to_3_5", fn(lambda t: t.expand(3, 5)), (torch.randn(1, 1, dtype=h),))
case("expand_f32_1_5", fn(lambda t: t.expand(3, 5)), (torch.randn(1, 5),))
case("expand_3_5_same", fn(lambda t: t.expand(3, 5)), (x,))

# pad
case("pad_3_5_last", fn(lambda t: F.pad(t, (1, 2))), (x,))
case("pad_3_5_both", fn(lambda t: F.pad(t, (1, 2, 2, 1))), (x,))
case("pad_3_5_value", fn(lambda t: F.pad(t, (1, 2), value=-2.5)), (x,))
case("pad_3_5_value1", fn(lambda t: F.pad(t, (1, 2, 0, 1), value=1.0)), (x,))
case("pad_3_5_before", fn(lambda t: F.pad(t, (2, 0))), (x,))
case("pad_2_3_5", fn(lambda t: F.pad(t, (1, 1, 0, 2, 1, 0))), (y,))
case("pad_f32_3_5", fn(lambda t: F.pad(t, (1, 2), value=-2.5)), (torch.randn(3, 5),))
case("pad_bf16_3_5", fn(lambda t: F.pad(t, (1, 2), value=-2.5)), (torch.randn(3, 5, dtype=bf),))
case("pad_i32_3_5", fn(lambda t: F.pad(t, (1, 2), value=7)), (torch.ones(3, 5, dtype=torch.int32),))

# gather
table = torch.randn(7, 5, dtype=h)
case("index_select_7_5_i32", fn(lambda t, i: torch.index_select(t, 0, i)), (table, i32(1, 3, 0, 6)))
case("index_select_7_5_i64", fn(lambda t, i: torch.index_select(t, 0, i)), (table, i64(1, 3, 0, 6)))
case("index_7_5_i32", fn(lambda t, i: t[i]), (table, i32(1, 3, 0, 6)))
case("index_7_5_i64", fn(lambda t, i: t[i]), (table, i64(1, 3, 0, 6)))
case("index_7_5_2d_i32", fn(lambda t, i: t[i]), (table, torch.tensor([[1, 3], [0, 6]], dtype=torch.int32)))
case("embedding_7_5_i32", fn(lambda t, i: F.embedding(i, t)), (table, i32(1, 3, 0, 6)))
case("embedding_7_5_2d_i32", fn(lambda t, i: F.embedding(i, t)), (table, torch.tensor([[1, 3], [0, 6]], dtype=torch.int32)))
case("index_select_axis1_3_7", fn(lambda t, i: torch.index_select(t, 1, i)), (torch.randn(3, 7, dtype=h), i32(1, 3, 0, 6)))
case("index_select_axis2_2_3_7", fn(lambda t, i: torch.index_select(t, 2, i)), (torch.randn(2, 3, 7, dtype=h), i32(1, 3, 0)))
case("index_select_f32_7_5", fn(lambda t, i: torch.index_select(t, 0, i)), (torch.randn(7, 5), i32(1, 3, 0, 6)))
case("index_select_bf16_7_5", fn(lambda t, i: torch.index_select(t, 0, i)), (torch.randn(7, 5, dtype=bf), i32(1, 3, 0, 6)))
case("gather_dim1_3_7", fn(lambda t, i: torch.gather(t, 1, i)),
     (torch.randn(3, 7, dtype=h), torch.tensor([[1, 3], [0, 6], [2, 2]], dtype=torch.int32)))

# scatter-add
base = torch.randn(7, 5, dtype=h)
case("index_add_7_5_i32", fn(lambda b, i, u: torch.index_add(b, 0, i, u)), (base, i32(1, 3, 1, 6), torch.randn(4, 5, dtype=h)))
case("index_add_7_5_i64", fn(lambda b, i, u: torch.index_add(b, 0, i, u)), (base, i64(1, 3, 1, 6), torch.randn(4, 5, dtype=h)))
case("index_add_axis1_3_7", fn(lambda b, i, u: torch.index_add(b, 1, i, u)), (torch.randn(3, 7, dtype=h), i32(1, 3, 1, 6), torch.randn(3, 4, dtype=h)))
case("index_add_f32_7_5", fn(lambda b, i, u: torch.index_add(b, 0, i, u)), (torch.randn(7, 5), i32(1, 3, 1, 6), torch.randn(4, 5)))
case("scatter_add_7_5_i32", fn(lambda b, i, u: torch.scatter_add(b, 0, i, u)),
     (base, torch.tensor([[1, 3, 1, 6, 2], [0, 0, 4, 4, 5]], dtype=torch.int32), torch.randn(2, 5, dtype=h)))
case("index_put_acc_7_5_i32", fn(lambda b, i, u: b.index_put((i,), u, accumulate=True)), (base, i32(1, 3, 1, 6), torch.randn(4, 5, dtype=h)))

# slice assignment (cache fill)
cache = torch.randn(3, 8, dtype=h)
case("slice_scatter_3_8_axis1", fn(lambda c, u: torch.slice_scatter(c, u, 1, 2, 4)), (cache, torch.randn(3, 2, dtype=h)))
case("slice_scatter_3_8_axis0", fn(lambda c, u: torch.slice_scatter(c, u, 0, 1, 3)), (cache, torch.randn(2, 8, dtype=h)))


def assign(c, u):
    c = c.clone()
    c[:, 2:4] = u
    return c


case("assign_3_8_axis1", fn(assign), (cache, torch.randn(3, 2, dtype=h)))


class DynAssign(torch.nn.Module):
    def forward(self, c, u, pos):
        n = pos.item()
        torch._check_is_size(n)
        torch._check(n + 2 <= 8)
        return torch.slice_scatter(c, u, 1, n, n + 2)


case("slice_scatter_dynamic_3_8", DynAssign(), (cache, torch.randn(3, 2, dtype=h), torch.tensor(2, dtype=torch.int32)))
case("index_copy_3_8_axis1", fn(lambda c, i, u: torch.index_copy(c, 1, i, u)), (cache, i32(2, 3), torch.randn(3, 2, dtype=h)))
case("slice_scatter_f32_3_8", fn(lambda c, u: torch.slice_scatter(c, u, 1, 2, 4)), (torch.randn(3, 8), torch.randn(3, 2)))

# scatter-add: coreai-torch has no index_add / scatter_add, only scatter with reduce="add" (indices as wide as the updates)
case("scatter_reduce_axis0_7_5", fn(lambda b, i, u: torch.scatter(b, 0, i.reshape(4, 1).expand(4, 5), u, reduce="add")),
     (base, i32(1, 3, 1, 6), torch.randn(4, 5, dtype=h)))
case("scatter_reduce_axis1_3_7", fn(lambda b, i, u: torch.scatter(b, 1, i.reshape(1, 4).expand(3, 4), u, reduce="add")),
     (torch.randn(3, 7, dtype=h), i32(1, 3, 1, 6), torch.randn(3, 4, dtype=h)))
case("scatter_reduce_axis0_f32_7_5", fn(lambda b, i, u: torch.scatter(b, 0, i.reshape(4, 1).expand(4, 5), u, reduce="add")),
     (torch.randn(7, 5), i32(1, 3, 1, 6), torch.randn(4, 5)))
case("scatter_reduce_direct_7_5", fn(lambda b, i, u: torch.scatter(b, 0, i, u, reduce="add")),
     (base, torch.tensor([[1, 3, 1, 6, 2], [0, 0, 4, 4, 5]], dtype=torch.int32), torch.randn(2, 5, dtype=h)))
case("index_select_1d_7", fn(lambda t, i: torch.index_select(t, 0, i)), (torch.randn(7, dtype=h), i32(1, 3, 0, 6)))


class DynAssignAt(torch.nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, c, u, pos):
        n = pos.item()
        torch._check_is_size(n)
        torch._check(n + 2 <= c.shape[self.dim])
        return torch.slice_scatter(c, u, self.dim, n, n + 2)


case("slice_scatter_dynamic_8_3_axis0", DynAssignAt(0), (torch.randn(8, 3, dtype=h), torch.randn(2, 3, dtype=h), torch.tensor(2, dtype=torch.int32)))
case("slice_scatter_dynamic_2_8_3_axis1", DynAssignAt(1), (torch.randn(2, 8, 3, dtype=h), torch.randn(2, 2, 3, dtype=h), torch.tensor(2, dtype=torch.int32)))

# which constant of a broadcast comes first (sizes or dims) depends on something not yet understood: more samples
case("expand_1_1_5_to_2_3_5", fn(lambda t: t.expand(2, 3, 5)), (torch.randn(1, 1, 5, dtype=h),))
case("expand_1_3_1_to_2_3_4", fn(lambda t: t.expand(2, 3, 4)), (torch.randn(1, 3, 1, dtype=h),))
case("expand_1_1_to_1_5", fn(lambda t: t.expand(1, 5)), (torch.randn(1, 1, dtype=h),))
case("expand_4_1_1_to_4_3_5", fn(lambda t: t.expand(4, 3, 5)), (torch.randn(4, 1, 1, dtype=h),))
case("expand_1_to_3_5", fn(lambda t: t.expand(3, 5)), (torch.randn(1, dtype=h),))
case("expand_1_5_to_2_3_5", fn(lambda t: t.expand(2, 3, 5)), (torch.randn(1, 5, dtype=h),))
case("expand_2_1_to_2_7", fn(lambda t: t.expand(2, 7)), (torch.randn(2, 1, dtype=h),))
case("expand_1_5_to_9_5", fn(lambda t: t.expand(9, 5)), (torch.randn(1, 5, dtype=h),))
