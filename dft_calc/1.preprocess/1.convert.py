#!/usr/bin/env python3
"""Convert SIESTA grid outputs into per-channel HDF5 files."""

import argparse
import sys
from pathlib import Path

import h5py


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import siestaio  # noqa: E402


GRID_FILES = {
    "target": "Molecular.RHO",
    "feature1": "Molecular.IOCH",
    "feature2": "rhoatom",
    "feature3": "rhoover",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert completed SIESTA calculations to raw HDF5 channels."
    )
    parser.add_argument(
        "calculations",
        type=Path,
        help="Directory whose immediate children are calculation case directories.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output directory for target/ and feature1/ through feature3/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace HDF5 files that already exist.",
    )
    return parser.parse_args()


def case_directories(path):
    cases = sorted(item for item in path.iterdir() if item.is_dir())
    if not cases:
        raise ValueError("No calculation case directories found in %s" % path)
    return cases


def validate_case(case):
    missing = [name for name in GRID_FILES.values() if not (case / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "%s is missing required grid file(s): %s"
            % (case, ", ".join(missing))
        )


def write_grid(source, destination, overwrite):
    if destination.exists() and not overwrite:
        raise FileExistsError(
            "%s already exists; pass --overwrite to replace it" % destination
        )
    grid = siestaio.readGrid(str(source))[2][0]
    with h5py.File(str(destination), "w") as handle:
        handle.create_dataset("value", data=grid)


def main():
    args = parse_args()
    calculations = args.calculations.expanduser().resolve()
    output = args.output.expanduser().resolve()
    cases = case_directories(calculations)

    for channel in GRID_FILES:
        (output / channel).mkdir(parents=True, exist_ok=True)

    for index, case in enumerate(cases, start=1):
        validate_case(case)
        for channel, filename in GRID_FILES.items():
            write_grid(
                case / filename,
                output / channel / (case.name + ".h5"),
                args.overwrite,
            )
        print("[%d/%d] %s" % (index, len(cases), case.name), flush=True)


if __name__ == "__main__":
    main()
