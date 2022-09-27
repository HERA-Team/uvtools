#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2021 HERA collaboration
# Licensed under the MIT License

"""Print the correlation matrix from a sum and diff file."""

import argparse

import numpy as np
from pyuvdata import UVData
import pyuvdata.utils as uvutils

try:
    from hera_mc import cm_hookup, mc
    node_info = True
except ImportError:
    node_info = False

# setup argparse
a = argparse.ArgumentParser(
    description="A command-line script for printing the correlation metric from a "
    "sum and diff file."
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
    help="Threshold to use for annotations. If None (default) the correlation metric "
    "is printed, otherwise an 'x' indicates a value over threshold and a '.' indicates "
    "a value at or below the threshold.",
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

# read in just the desired data from the sum file
uvd_diff.read(
    args.diff_file,
    antenna_nums=select_ants,
    polarizations=[args.pol],
)

ants_data_unique = np.union1d(uvd_sum.ant_1_array, uvd_sum.ant_2_array)

if node_info:
    # get mapping of antenna numbers to node/snap numbers
    ant_node_strs = []
    ant_1_node_strs = np.zeros(uvd_sum.ant_1_array.shape, dtype=int)
    ant_2_node_strs = np.zeros(uvd_sum.ant_2_array.shape, dtype=int)
    with mc.MCSessionWrapper(session=None) as session: 
        for ant in ants_data_unique:
            hookup = cm_hookup.Hookup(session)
            ant_dict = hookup.get_hookup('HH')
            key = 'HH%i:A' % (ant)
            pol = 'E'
            # node is a string
            node = ant_dict[key].get_part_from_type('node')['E<ground'][1:]
            # snapLoc is a 2-tuple, the first element is a string for snap location (0-3),
            # the second is the node as an integer
            snapLoc = (ant_dict[key].hookup[f'{pol}<ground'][-1].downstream_input_port[-1], ant)
            # snapInput is a 2-tuple, the first element is a string for snap input (0-10+),
            # the second is the node as an integer
            snapInput = (ant_dict[key].hookup[f'{pol}<ground'][-2].downstream_input_port[1:], ant)

            ant_node_str = node + snapLoc[0] + snapInput[0].zfill(2)
            ant_node_strs.append(ant_node_str)

            ant_1_node_strs[np.nonzero(uvd_sum.ant_1_array == ant)] = int(ant_node_str)
            ant_2_node_strs[np.nonzero(uvd_sum.ant_2_array == ant)] = int(ant_node_str)

    # first conjugate to ensure we have the baselines we want for the upper triangle
    conj_index_array = np.where(ant_1_node_strs > ant_2_node_strs)[0]
    if conj_index_array.size > 0:
        uvd_sum.conjugate_bls(convention=conj_index_array)
        uvd_diff.conjugate_bls(convention=conj_index_array)

        # redo the ant string mapping after the conjugation
        for ant_ind, ant in enumerate(ants_data_unique):
            ant_node_str = ant_node_strs[ant_ind]
            ant_1_node_strs[np.nonzero(uvd_sum.ant_1_array == ant)] = int(ant_node_str)
            ant_2_node_strs[np.nonzero(uvd_sum.ant_2_array == ant)] = int(ant_node_str)

    ant_argsort = np.argsort(ant_node_strs)

    ants_data_sorted = ants_data_unique[ant_argsort]
    ant_node_strs_sorted = np.asarray(ant_node_strs)[ant_argsort]

    # lexsort uses the listed arrays from last to first
    # (so the primary sort is on the last one)
    bl_index_array = np.lexsort((ant_2_node_strs, ant_1_node_strs, uvd_sum.time_array))
        
    uvd_sum.reorder_blts(bl_index_array)
    uvd_diff.reorder_blts(bl_index_array)
else:
    uvd_sum.reorder_blts(conj_convention="ant1<ant2")
    uvd_diff.reorder_blts(conj_convention="ant1<ant2")
    ants_data_sorted = np.sort(ants_data_unique)

evens = uvd_sum.data_array + uvd_diff.data_array
odds = uvd_sum.data_array - uvd_diff.data_array

evens = evens / np.abs(evens)
odds = odds / np.abs(odds)

# reshape to be Nbls, Ntimes, Nfreqs
# We sorted this so that time changes slowest, then baseline
evens = np.reshape(evens, (uvd_sum.Ntimes, uvd_sum.Nbls, uvd_sum.Nfreqs))
odds = np.reshape(odds, (uvd_sum.Ntimes, uvd_sum.Nbls, uvd_sum.Nfreqs))

corr_metric = evens * np.conj(odds)
corr_metric = np.abs(np.nanmean(np.nanmean(corr_metric, axis=2), axis=0))

ant1_vals = np.reshape(uvd_sum.ant_1_array, (uvd_sum.Ntimes, uvd_sum.Nbls))[0, :]
ant2_vals = np.reshape(uvd_sum.ant_2_array, (uvd_sum.Ntimes, uvd_sum.Nbls))[0, :]

if node_info:
    print("node       " + "   ".join([a_str[:2] for a_str in ant_node_strs_sorted]))
    print("     ant " + " ".join([f"{an:4n}" for an in ants_data_sorted]))
else:
    print("ant  " + " ".join([f"{an:4n}" for an in ants_data_sorted]))
if args.threshold is None:
    for ant_ind, ant in enumerate(ants_data_sorted):
        this_row = np.nonzero(ant1_vals == ant)[0]
        assert np.allclose(ant2_vals[this_row], ants_data_sorted[ant_ind:]), "something went wrong with sorting!"

        corrs = corr_metric[this_row]
        if node_info:
            node = ant_node_strs_sorted[ant_ind][:2]
            print(f"{node}  " + f"{ant:4n}" + "     "*ant_ind, " ".join([f"{corr:.2f}" for corr in corrs]))
        else:
            print(f"{ant:4n}" + "     "*ant_ind, " ".join([f"{corr:.2f}" for corr in corrs]))
else:
    for ant_ind, ant in enumerate(ants_data_sorted):
        this_row = np.nonzero(ant1_vals == ant)[0]
        assert np.allclose(ant2_vals[this_row], ants_data_sorted[ant_ind:]), "something went wrong with sorting!"

        corrs = corr_metric[this_row]
        corrs = ["  x  " if corr > args.threshold else "  .  " for corr in corrs ]
        
        if node_info:
            node = ant_node_strs_sorted[ant_ind][:2]
            print(f"{node}  " + f"{ant:4n} " + "     "*ant_ind, "".join(corrs))
        else:
            print(f"{ant:4n} " + "     "*ant_ind, "".join(corrs))
