"""The state references: a torch buffer mutated in place becomes a Core AI state (`!coreai.handle`), read with
`read_handle` and written with `write_handle`, the order kept by tokens from one `create_token`; the graph returns
each state's last token, named by the state, before its own results. Only graphs with no constant are here: coreai-torch
hoists a module's constants above the reads in an order the engine does not derive, so a state beside
an `updateSlice` is judged by its run, not its bytes.

    python tools/coreai/references_state.py .build/coreai/state
    python tools/coreai/swiftrefs.py .build/coreai/state coreAIStateReferences > Tests/SiliCoreTests/CoreAIStateReferences.swift
"""
import sys
from pathlib import Path

import torch
from reference import write

h = torch.float16


class Add(torch.nn.Module):
    """Written, never read again: the token of the write is the result."""
    def __init__(self): super().__init__(); self.register_buffer("c", torch.zeros(4, dtype=h))
    def forward(self, x):
        self.c.add_(x)
        return x * x


class AddRead(torch.nn.Module):
    """Written, then read: a second `read_handle` after the write, its token the result."""
    def __init__(self): super().__init__(); self.register_buffer("c", torch.zeros(4, dtype=h))
    def forward(self, x):
        self.c.add_(x)
        return self.c * x


class Two(torch.nn.Module):
    """Two states, each its own reads and writes from one `create_token`."""
    def __init__(self):
        super().__init__()
        self.register_buffer("a", torch.zeros(4, dtype=h))
        self.register_buffer("b", torch.zeros(4, dtype=h))
    def forward(self, x):
        self.a.add_(x)
        self.b.add_(x * x)
        return x - x


out = Path(sys.argv[1])
x = torch.randn(4, dtype=h)
write(out, "state_add_4", Add(), (x,))
write(out, "state_add_read_4", AddRead(), (x,))
write(out, "state_two_4", Two(), (x,))
