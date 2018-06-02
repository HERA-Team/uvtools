import numpy as np
import os
import glob


def search_data(templates, pols, matched_pols=False, reverse_nesting=False):
    """
    Glob-parse data templates to search for data files.

    Parameters
    ---------- 
    templates : str or list
        A glob-parsable search string, or list of such strings, with a {pol}
        spot for string formatting. Ex. ["zen.even.{pol}.LST.*.HH.uv"]

    pols : str or list
        A polarization string, or list of polarization strings, to search for.
        Ex. ["xx", "yy"]

    matched_pols : boolean
        If True, only use datafiles that are present for all polarizations.

    reverse_nesting : boolean
        If True, flip the nesting of datafiles to be polarization-datafile.
        By default, the output is polarization-datafile.

    Returns
    -------
    datafiles : list
        A nested list of paths to datafiles. By default, the structure is
        polarization-datafile nesting. If reverse_nesting, then the structure
        is flipped to datafile-polarization structure.
    """
    # type check
    if isinstance(templates, (str, np.str)):
        templates = [templates]
    if isinstance(pols, (str, np.str)):
        pols = [pols]
    # search for datafiles
    datafiles = []
    for p in pols:
        dfs = []
        for t in templates:
            df = glob.glob(t.format(pol=p))
            if len(df) > 0:
                dfs.extend(df)
        if len(dfs) > 0:
            datafiles.append(sorted(dfs))
    # get unique files
    allfiles = [item for sublist in datafiles for item in sublist]
    unique_files = set()
    for f in allfiles:
        for p in pols:
            if ".{pol}.".format(pol=p) in f:
                unique_files.update(set([f.replace(".{pol}.".format(pol=p), ".{pol}.")]))
                continue
    unique_files = sorted(unique_files)
    # check for unique files with all pols
    if matched_pols:
        Npols = len(pols)
        _templates = []
        for f in unique_files:
            goodfile = True
            for p in pols:
                if f.format(pol=p) not in allfiles:
                    goodfile = False
            if goodfile:
                _templates.append(f)

        datafiles = search_data(_templates, pols, matched_pols=False, reverse_nesting=False)
    # reverse nesting if desired
    if reverse_nesting:
        datafiles = []
        for f in unique_files:
            dfs = []
            for p in pols:
                df = f.format(pol=p)
                if df in allfiles:
                    dfs.append(df)
            datafiles.append(dfs)

    return datafiles
