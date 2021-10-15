#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2021 HERA collaboration
# Licensed under the MIT License

"""Print the correlation matrix for a list of antennas from a sum and diff file."""

import argparse

import numpy as np
from pyuvdata import UVData
import pyuvdata.utils as uvutils

# setup argparse
a = argparse.ArgumentParser(
    description="A command-line script for the correlation metric from a sum and diff "
    "file given a list of antennas."
)
a.add_argument(
    "sum_file",
    type=str,
    help="The sum file to use.",
)
a.add_argument(
    "diff_file",
    type=str,
    help="The diff file to use.",
)
a.add_argument(
    "pol",
    type=str,
    help="Polarization.",
)
a.add_argument(
    "--ant_list",
    nargs="*",
    type=int,
    default=None,
    help="Antennas to include in the matrix",
)
a.add_argument(
    "--threshold",
    type=float,
    default=None,
    help="Threshold to use for annotations.",
)

# get args
args = a.parse_args()

uvd_sum = UVData()
uvd_diff = UVData()

if args.ant_list is not None:
    select_ants = np.asarray(args.ant_list)
else:
    select_ants = None

# read in just the desired data from the sum file
uvd_sum.read(
    args.sum_file,
    antenna_nums=select_ants,
    polarizations=[args.pol],
)
uvd_sum.reorder_blts(conj_convention="ant1<ant2")

# read in just the desired data from the sum file
uvd_diff.read(
    args.diff_file,
    antenna_nums=select_ants,
    polarizations=[args.pol],
)
uvd_diff.reorder_blts(conj_convention="ant1<ant2")

evens = uvd_sum.data_array + uvd_diff.data_array
odds = uvd_sum.data_array - uvd_diff.data_array

evens = evens / np.abs(evens)
odds = odds / np.abs(odds)

# reshape to be Nbls, Ntimes, Nfreqs
# this is raw correlator data, so the baseline number changes faster than the time
evens = np.reshape(evens, (uvd_sum.Ntimes, uvd_sum.Nbls, uvd_sum.Nfreqs))
odds = np.reshape(odds, (uvd_sum.Ntimes, uvd_sum.Nbls, uvd_sum.Nfreqs))

corr_metric = evens * np.conj(odds)

corr_metric = np.abs(np.nanmean(np.nanmean(corr_metric, axis=2), axis=0))

ant1_vals = np.reshape(uvd_sum.ant_1_array, (uvd_sum.Ntimes, uvd_sum.Nbls))[0, :]

ant1_unique = np.unique(ant1_vals)
print("    " + " ".join([f"{an:4n}" for an in ant1_unique]))
if args.threshold is None:
    for ant_ind, ant in enumerate(ant1_unique):
        corrs = corr_metric[np.nonzero(ant1_vals == ant)[0]]
        
        print(f"{ant:4n}" + "     "*ant_ind, " ".join([f"{corr:.2f}" for corr in corrs]))
else:
    for ant_ind, ant in enumerate(ant1_unique):
        corrs = corr_metric[np.nonzero(ant1_vals == ant)[0]]
        corrs = ["  x  " if corr > args.threshold else "  .  " for corr in corrs ]
        
        print(f"{ant:4n}" + "     "*ant_ind, "".join(corrs))
