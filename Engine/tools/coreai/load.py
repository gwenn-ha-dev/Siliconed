"""Core AI's own loader re-reading an asset's bytecode (main.mlirb), printed as generic coreai text, or its error.
The judge of the engine's bytecode writer: the same text as the reference's. Python only to drive Core AI.

    python tools/coreai/load.py FILE.mlirb [...]
"""
import sys
from pathlib import Path
from coreai._compiler.context import Context
from coreai._compiler._mlir_libs._coreaiIR._bindings.compiler import load_module_from_bytecode

for name in sys.argv[1:]:
    context = Context()._mlir_context
    try:
        module = load_module_from_bytecode(context=context, path=Path(name))
        print(module.operation.get_asm(print_generic_op_form=True))
    except Exception as error:
        print("refused:", error)
