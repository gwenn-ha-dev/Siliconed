"""The Convolution family's references: small torch convolutions exported by coreai-torch, written as assets
(`reference.write`) for the byte-equality test. Python only to drive Core AI.

    python tools/coreai/references_convolution.py OUT_DIR
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from reference import write

out = Path(sys.argv[1])
h = torch.float16


class Conv(torch.nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.kw = kw

    def forward(self, x, w):
        d = x.dim() - 2
        f = (F.conv1d, F.conv2d, F.conv3d)[d - 1]
        return f(x, w, **self.kw)


def case(name, x, w, dtype=h, **kw):
    write(out, name, Conv(**kw), (torch.randn(*x, dtype=dtype), torch.randn(*w, dtype=dtype)))


# 2-D: input [1, 4, 5, 6], kernel 3x3.
case("conv2d_f16", (1, 4, 5, 6), (8, 4, 3, 3))
case("conv2d_pad1", (1, 4, 5, 6), (8, 4, 3, 3), padding=1)
case("conv2d_pad_hw", (1, 4, 5, 6), (8, 4, 3, 3), padding=(1, 2))
case("conv2d_stride2_pad1", (1, 4, 5, 6), (8, 4, 3, 3), stride=2, padding=1)
case("conv2d_stride_hw", (1, 4, 5, 6), (8, 4, 3, 3), stride=(1, 2))
case("conv2d_dilation2", (1, 4, 7, 8), (8, 4, 3, 3), dilation=2)
case("conv2d_dilation_pad", (1, 4, 7, 8), (8, 4, 3, 3), dilation=2, padding=2)
case("conv2d_groups2", (1, 4, 5, 6), (8, 2, 3, 3), groups=2, padding=1)
case("conv2d_depthwise", (1, 4, 5, 6), (4, 1, 3, 3), groups=4, padding=1)
case("conv2d_1x1", (1, 4, 5, 6), (8, 4, 1, 1))
case("conv2d_batch2", (2, 4, 5, 6), (8, 4, 3, 3), padding=1)
case("conv2d_f32", (1, 4, 5, 6), (8, 4, 3, 3), dtype=torch.float32, padding=1)
case("conv2d_bf16", (1, 4, 5, 6), (8, 4, 3, 3), dtype=torch.bfloat16, padding=1)
case("conv2d_pad1_f16_big", (1, 16, 32, 32), (16, 16, 3, 3), padding=1)
# 1-D and 3-D.
case("conv1d_f16", (1, 4, 9), (8, 4, 3))
case("conv1d_pad1", (1, 4, 9), (8, 4, 3), padding=1)
case("conv1d_stride2_pad1", (1, 4, 9), (8, 4, 3), stride=2, padding=1)
case("conv1d_groups2_dil2", (1, 4, 9), (8, 2, 3), groups=2, dilation=2, padding=2)
case("conv3d_f16", (1, 4, 4, 5, 6), (8, 4, 3, 3, 3))
case("conv3d_pad1", (1, 4, 4, 5, 6), (8, 4, 3, 3, 3), padding=1)
case("conv3d_stride2_pad1", (1, 4, 4, 5, 6), (8, 4, 3, 3, 3), stride=2, padding=1)
case("conv3d_groups2_dil2", (1, 4, 6, 7, 8), (8, 2, 3, 3, 3), groups=2, dilation=2, padding=2)
# Without padding: the order of the constants differs (no pad op between them).
case("conv2d_stride2", (1, 4, 7, 8), (8, 4, 3, 3), stride=2)
case("conv2d_dilation2_nopad", (1, 4, 7, 8), (8, 4, 3, 3), dilation=2)
case("conv2d_groups2_nopad", (1, 4, 5, 6), (8, 2, 3, 3), groups=2)
case("conv2d_all_nopad", (1, 4, 9, 10), (8, 2, 3, 3), groups=2, stride=(2, 1), dilation=(1, 2))
case("conv1d_stride2", (1, 4, 9), (8, 4, 3), stride=2)
case("conv1d_dil2_groups2", (1, 4, 9), (8, 2, 3), dilation=2, groups=2)
case("conv3d_stride2", (1, 4, 6, 7, 8), (8, 4, 3, 3, 3), stride=2)
case("conv3d_all_nopad", (1, 4, 6, 7, 8), (8, 2, 3, 3, 3), groups=2, stride=(2, 1, 1), dilation=(1, 2, 1))
case("conv2d_pad_asym_stride", (1, 4, 9, 10), (8, 2, 3, 3), groups=2, stride=(2, 1), dilation=(1, 2), padding=(1, 0))
