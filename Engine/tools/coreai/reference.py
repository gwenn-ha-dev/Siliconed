"""The shared writer of reference modules (as `references.py` writes its own): a torch module exported by coreai-torch,
debug locations stripped, saved as `NAME.aimodel` with its generic text `NAME.mlir` beside it. A family's references
live in `tools/coreai/references_<family>.py`, each calling `write`; `swiftrefs.py` turns a directory of them into
the Swift dictionary its byte-equality test reads. Python only to drive Core AI.
"""
from pathlib import Path

import torch
import coreai_torch
from coreai._compiler._mlir_libs._mlir import passmanager


def write(out: Path, name: str, module: torch.nn.Module, args: tuple) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ep = torch.export.export(module.eval(), args).run_decompositions(coreai_torch.get_decomp_table())
    prog = coreai_torch.TorchConverter().add_exported_program(ep).to_coreai()
    mm = prog._module._mlir_module
    with mm.context:
        passmanager.PassManager.parse("builtin.module(strip-debuginfo)").run(mm.operation)
    (out / (name + ".mlir")).write_text(mm.operation.get_asm(print_generic_op_form=True))
    prog.save_asset(out / (name + ".aimodel"))
