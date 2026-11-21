"""Fingerprints of the goldens: how a reader knows their oracle run is ours without our files.

The goldens are not published (gigabytes, and weights under licences that do not allow it): the
scripts beside this one rebuild them. What is published is their fingerprint, `fingerprints.json`:
for every tensor of every golden, its dtype, its shape and a few statistics. A reader who rebuilt the
goldens compares them with it:

    python tools/oracle/fingerprints.py compare [--goldens DIR]     what differs from ours, and by how much
    python tools/oracle/fingerprints.py write   [--goldens DIR]     the maintainer, after regenerating
    python tools/oracle/fingerprints.py add --only TEXT [TEXT …]    the maintainer: only the goldens not yet
                                                                    published whose path holds one of the TEXTs

Integers (token identifiers, masks, positions) must match exactly: they are compared by a hash of
their bytes. Floats are compared by their statistics, with a tolerance, never bit for bit: PyTorch
on the CPU does not promise the same float32 bits from one machine to another (a different BLAS, a
different vector width, even a different memory layout changes the order of a sum), so two honest
oracle runs differ by rounding. That rounding is far below what `siliconed check` judges — its
thresholds are relative errors of 5e-4 per channel — and the tolerance here is set between the two.
A golden that is missing is said, not failed: one may rebuild only the stages one checks.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.dont_write_bytecode = True
from common import goldens_directory  # noqa: E402

HERE = Path(__file__).resolve().parent
PUBLISHED = HERE / "fingerprints.json"

# The relative tolerance on a float tensor's norm and mean absolute value. Rounding moves them by
# ~1e-6; the free-running trajectories of the oracle, which chain eight evaluations, by more. 1e-3 was too
# loose for one evaluation: another file of published weights moved a model output's norm by 5.6e-4 and
# passed unseen.
TOLERANCE = 1e-4
TRAJECTORY_TOLERANCE = 1e-3


def tolerance_for(relative: str, tolerance: float | None) -> float:
    """The tolerance asked, else the one for this golden: looser for a chain of evaluations."""
    if tolerance is not None:
        return tolerance
    return TRAJECTORY_TOLERANCE if "trajectory" in relative else TOLERANCE


def fingerprint(tensor: torch.Tensor) -> dict:
    entry = {"dtype": str(tensor.dtype).removeprefix("torch."), "shape": list(tensor.shape)}
    if tensor.dtype.is_floating_point:
        values = tensor.double()
        entry["norm"] = values.norm().item()
        entry["mean_abs"] = values.abs().mean().item() if values.numel() else 0.0
        entry["max_abs"] = values.abs().max().item() if values.numel() else 0.0
    else:
        entry["sha256"] = hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
    return entry


def goldens(directory: Path):
    """Every golden under `directory`, by its path relative to it; the test images are inputs, not goldens."""
    for path in sorted(directory.rglob("*.safetensors")):
        relative = path.relative_to(directory).as_posix()
        if not relative.startswith("inputs/"):
            yield relative, path


def read(path: Path) -> dict:
    with safe_open(str(path), "pt") as f:
        return {name: fingerprint(f.get_tensor(name)) for name in f.keys()}


def save(table: dict) -> None:
    """One golden per line, in key order: a regenerated golden changes its own line and no other, so a
    diff of this file reads. Any JSON layout is read back, this one is what is written."""
    lines = [f" {json.dumps(relative)}: {json.dumps(table[relative], sort_keys=True)}" for relative in sorted(table)]
    PUBLISHED.write_text("{\n" + ",\n".join(lines) + "\n}\n")


def write(directory: Path) -> int:
    table = {relative: read(path) for relative, path in goldens(directory)}
    save(table)
    print(f"{len(table)} goldens → {PUBLISHED.name}")
    return 0


def add(directory: Path, only: list[str]) -> int:
    """New goldens (a path with one of `only`) join the published table; the others are left as they are."""
    table = json.loads(PUBLISHED.read_text())
    added = [relative for relative, path in goldens(directory)
             if relative not in table and any(text in relative for text in only)]
    for relative in added:
        table[relative] = read(dict(goldens(directory))[relative])
        print(f"  + {relative}")
    save(table)
    print(f"{len(added)} goldens added → {PUBLISHED.name} ({len(table)} in all)")
    return 0


def relative_gap(ours: float, theirs: float) -> float:
    scale = max(abs(ours), abs(theirs))
    return 0.0 if scale == 0 else abs(ours - theirs) / scale


def compare(directory: Path, asked: float | None) -> int:
    published = json.loads(PUBLISHED.read_text())
    present = dict(goldens(directory))
    failures, compared, worst = [], 0, (0.0, "")
    for relative, expected in sorted(published.items()):
        if relative not in present:
            continue
        compared += 1
        found = read(present[relative])
        tolerance = tolerance_for(relative, asked)
        for name, ours in expected.items():
            theirs = found.get(name)
            where = f"{relative} · {name}"
            if theirs is None:
                failures.append(f"{where}: missing")
            elif (theirs["dtype"], theirs["shape"]) != (ours["dtype"], ours["shape"]):
                failures.append(f"{where}: {theirs['dtype']} {theirs['shape']}, ours {ours['dtype']} {ours['shape']}")
            elif "sha256" in ours:
                if theirs["sha256"] != ours["sha256"]:
                    failures.append(f"{where}: integers differ")
            else:
                gap = max(relative_gap(ours[k], theirs[k]) for k in ("norm", "mean_abs"))
                worst = max(worst, (gap, where))
                if not gap <= tolerance:
                    failures.append(f"{where}: statistics {gap:.1e} apart (tolerance {tolerance:.0e})")
        for name in sorted(set(found) - set(expected)):
            failures.append(f"{relative} · {name}: not in ours")
    unknown = sorted(set(present) - set(published))
    for relative in unknown:
        print(f"  not fingerprinted: {relative}")
    print(f"{compared} of {len(published)} goldens present and compared"
          + (f"; the floats' largest gap {worst[0]:.1e} ({worst[1]})" if worst[1] else ""))
    for failure in failures:
        print(f"  ✗ {failure}")
    if compared == 0:
        print("— nothing to compare: no golden of ours in this folder")
        return 2
    print("✓ the goldens are ours, up to rounding" if not failures else f"✗ {len(failures)} difference(s)")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=["compare", "write", "add"])
    parser.add_argument("--only", nargs="+", action="extend", default=[], help="add: the texts a new golden's path must hold one of")
    parser.add_argument("--goldens", help="the goldens folder (default: $SILICONED_GOLDENS, else ./goldens)")
    parser.add_argument("--tolerance", type=float, default=None,
                        help=f"compare: one tolerance for every golden (default {TOLERANCE:.0e}, {TRAJECTORY_TOLERANCE:.0e} for a trajectory)")
    arguments = parser.parse_args()
    directory = goldens_directory(arguments.goldens)
    if arguments.action == "add":
        return add(directory, arguments.only) if arguments.only else parser.error("add needs --only")
    return write(directory) if arguments.action == "write" else compare(directory, arguments.tolerance)


if __name__ == "__main__":
    sys.exit(main())
