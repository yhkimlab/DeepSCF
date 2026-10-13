#!/usr/bin/env python3
"""Calculate the sample-averaged normalization used by DeepSCF."""

import argparse
from pathlib import Path

import h5py
import numpy as np


CHANNELS = ("target", "feature1", "feature2", "feature3")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create scale.txt from raw per-channel HDF5 files."
    )
    parser.add_argument(
        "data",
        type=Path,
        help="Directory containing target/ and feature1/ through feature3/.",
    )
    parser.add_argument("output", type=Path, help="Output scale.txt path.")
    return parser.parse_args()


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
                % (
                    channel,
                    sorted(reference - current),
                    sorted(current - reference),
                )
            )
    return sorted(reference)


def average_sample_statistics(paths):
    means = []
    standard_deviations = []
    for path in paths:
        with h5py.File(str(path), "r") as handle:
            values = handle["value"][:]
        means.append(float(np.mean(values)))
        standard_deviations.append(float(np.std(values)))
    # Keep the original pipeline's sequential sample-wise accumulation.
    return sum(means) / len(means), sum(standard_deviations) / len(standard_deviations)


def main():
    args = parse_args()
    data = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    files = {channel: channel_files(data, channel) for channel in CHANNELS}
    case_ids = validate_channels(files)

    statistics = []
    for channel in CHANNELS:
        paths = [files[channel][case_id] for case_id in case_ids]
        statistics.append(average_sample_statistics(paths))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for index, (mean, _) in enumerate(statistics, start=1):
            handle.write("mean%d = %.15f\n" % (index, mean))
        for index, (_, std) in enumerate(statistics, start=1):
            handle.write("std%d = %.15f\n" % (index, std))

    print("Wrote %s from %d cases" % (output, len(case_ids)))


if __name__ == "__main__":
    main()
