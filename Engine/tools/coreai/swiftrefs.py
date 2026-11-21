"""A directory of reference assets (`reference.write`) as the Swift dictionary a byte-equality test reads.

    python tools/coreai/swiftrefs.py DIR VARIABLE > Tests/SiliCoreTests/CoreAI<Family>References.swift
"""
import base64
import sys
from pathlib import Path

root, variable = Path(sys.argv[1]), sys.argv[2]
print("// coreai-torch's own assets (main.mlirb, debug locations stripped), written by `tools/coreai/references_*.py`:")
print("// what the engine's bytecode writer must equal, byte for byte. Base64; regenerate with the scripts, never by hand.")
print()
print(f"let {variable}: [String: String] = [")
for asset in sorted(root.glob("*.aimodel")):
    data = base64.b64encode((asset / "main.mlirb").read_bytes()).decode()
    print(f'    "{asset.stem}": "{data}",')
print("]")
