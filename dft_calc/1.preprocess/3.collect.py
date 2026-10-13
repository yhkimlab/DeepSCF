#!/usr/bin/env python3
"""Apply a model's fixed feature scaling and assemble DeepSCF HDF5 files."""

import argparse
from pathlib import Path

import h5py
import numpy as np


CHANNELS = ("target", "feature1", "feature2", "feature3")
SCALE_KEYS = tuple("%s%d" % (prefix, index) for prefix in ("mean", "std") for index in range(1, 5))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assemble model-ready HDF5 files using one fixed scale.txt."
    )
    parser.add_argument(
        "data",
        type=Path,
        help="Directory containing raw target/ and feature1/ through feature3/.",
    )
    parser.add_argument("scale", type=Path, help="scale.txt for the target model.")
    parser.add_argument("output", type=Path, help="Output dataset directory.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace assembled HDF5 files that already exist.",
    )
    return parser.parse_args()


def read_scale(path):
    values = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            words = line.split()
            if not words:
                continue
            if len(words) != 3 or words[1] != "=" or words[0] not in SCALE_KEYS:
                raise ValueError("Invalid scale entry at %s:%d" % (path, line_number))
            values[words[0]] = float(words[2])
    missing = sorted(set(SCALE_KEYS) - set(values))
    if missing:
        raise ValueError("%s is missing: %s" % (path, ", ".join(missing)))
    return values


def channel_files(data, channel):
    return {path.stem: path for path in sorted((data / channel).glob("*.h5"))}


def validate_channels(files):
    reference = set(files["target"])
    if not reference:
        raise ValueError("No target HDF5 files were found")
    for channel in CHANNELS[1:]:
        current = set(files[channel])
        if current != reference:
            raise ValueError(
                "%s case IDs do not match target (missing=%s, extra=%s)"
                % (channel, sorted(reference - current), sorted(current - reference))
            )
    return sorted(reference)


def read_value(path):
    with h5py.File(str(path), "r") as handle:
        return handle["value"][:]


def main():
    args = parse_args()
    data = args.data.expanduser().resolve()
    scale = read_scale(args.scale.expanduser().resolve())
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    files = {channel: channel_files(data, channel) for channel in CHANNELS}
    case_ids = validate_channels(files)

    for index, case_id in enumerate(case_ids, start=1):
        destination = output / (case_id + ".h5")
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                "%s already exists; pass --overwrite to replace it" % destination
            )

        target = read_value(files["target"][case_id])
        features = []
        for channel_index, channel in enumerate(CHANNELS[1:], start=2):
            values = read_value(files[channel][case_id])
            features.append(
                (values - scale["mean%d" % channel_index])
                / scale["std%d" % channel_index]
            )

        with h5py.File(str(destination), "w") as handle:
            handle.create_dataset("target", data=target)
            handle.create_dataset("feature", data=np.stack(features))
        print("[%d/%d] %s" % (index, len(case_ids), case_id), flush=True)


if __name__ == "__main__":
    main()
