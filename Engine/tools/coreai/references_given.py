"""The references of a graph whose weights are arguments, in the shapes the engine gives Core AI's GPU a weight in
(`Aicode.given`): its own when its rows are a multiple of 64 bytes (`block_4_32_64`, the block of `block_4_8_16`
at such shapes), flat and reshaped before anything reads it otherwise (`product_given_flat_3_5_7`).

    python tools/coreai/references_given.py DIR
    python tools/coreai/swiftrefs.py DIR coreAIGivenReferences > Tests/SiliCoreTests/CoreAIGivenReferences.swift
"""
import sys
from pathlib import Path

import torch

from reference import write

out = Path(sys.argv[1])
h = torch.float16


class Block(torch.nn.Module):
    def forward(self, x, w1, w3, w2): return (torch.nn.functional.silu(x @ w1) * (x @ w3)) @ w2


class FlatProduct(torch.nn.Module):
    def forward(self, x, w): return x @ w.reshape(5, 7)


write(out, "block_4_32_64", Block(), (torch.randn(4, 32, dtype=h), torch.randn(32, 64, dtype=h), torch.randn(32, 64, dtype=h), torch.randn(64, 32, dtype=h)))
write(out, "product_given_flat_3_5_7", FlatProduct(), (torch.randn(3, 5, dtype=h), torch.randn(35, dtype=h)))
