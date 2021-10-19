#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2021 HERA collaboration
# Licensed under the MIT License

"""
Print visibilities from a UVData readable file (e.g. uhv5).

Prints visibilities as a function of time for a given baseline and channel number.
"""

import argparse

import numpy as np
from pyuvdata import UVData
import pyuvdata.utils as uvutils

# setup argparse
a = argparse.ArgumentParser(
    description="A command-line script for printing visibilities from a UVData "
    "readable file given a baseline (specified as antenna numbers and polarization) "
    "and channel number."
)
a.add_argument(
    "file",
    type=str,
    help="pyuvdata-compatible file to print visibilities from.",
)
a.add_argument(
    "ant_1",
    type=int,
    help="First antenna in baseline.",
)
a.add_argument(
    "ant_2",
    type=int,
    help="Second antenna in baseline.",
)
a.add_argument(
    "pol",
    type=str,
    help="Polarization.",
)
a.add_argument(
    "freq_chan",
    type=int,
    help="Frequency channel.",
)

# get args
args = a.parse_args()

uvd = UVData()

# read in just the desired data
uvd.read(
    args.file,
    bls=(args.ant_1, args.ant_2, args.pol),
    freq_chans=[args.freq_chan],
)

data = np.squeeze(uvd.data_array)
print(' '.join([str(vis) for vis in data]))