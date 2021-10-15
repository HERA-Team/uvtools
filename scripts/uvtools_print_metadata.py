#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2021 HERA collaboration
# Licensed under the MIT License

"""Print all the antennas with data in a UVData readable file (e.g. uhv5)."""

import argparse

import numpy as np
from pyuvdata import UVData

# setup argparse
a = argparse.ArgumentParser(
    description="A command-line script for printing all the antennas with visibilities "
    "associated with them in a UVData readable file."
)
a.add_argument(
    "file",
    type=str,
    help="pyuvdata-compatible file to print antennas from.",
)

# get args
args = a.parse_args()

uvd = UVData()

# read in the metadata only
uvd.read(args.file, read_data=False)

print(np.union1d(uvd.ant_1_array, uvd.ant_2_array))