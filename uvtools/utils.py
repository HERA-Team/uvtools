import numpy as np
import glob


def search_data(templates, pols, matched_pols=False, reverse_nesting=False, flatten=False):
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
        If True, flip the nesting of datafiles to be datafile-polarization.
        By default, the output is polarization-datafile.

    flatten : boolean
        If True, flatten the nested output datafiles to a single hierarchy.

    Returns
    -------
    datafiles : list
        A nested list of paths to datafiles. By default, the structure is
        polarization-datafile nesting. If reverse_nesting, then the structure
        is flipped to datafile-polarization structure.

    datapols : list
        List of polarizations for each file in datafile
    """
    # type check
    if isinstance(templates, (str, np.str)):
        templates = [templates]
    if isinstance(pols, (str, np.str, np.integer, int)):
        pols = [pols]
    # search for datafiles
    datafiles = []
    datapols = []
    for pol in pols:
        dps = []
        dfs = []
        for template in templates:
            _dfs = glob.glob(template.format(pol=pol))
            if len(_dfs) > 0:
                dfs.extend(_dfs)
                dps.extend([pol for df in _dfs])
        if len(dfs) > 0:
            datafiles.append(sorted(dfs))
            datapols.append(dps)
    # get unique files
    allfiles = [item for sublist in datafiles for item in sublist]
    allpols = [item for sublist in datapols for item in sublist]
    unique_files = set()
    for _file in allfiles:
        for pol in pols:
            if ".{pol}.".format(pol=pol) in _file:
                unique_files.update(set([_file.replace(".{pol}.".format(pol=pol), ".{pol}.")]))
                break
    unique_files = sorted(unique_files)
    # check for unique files with all pols
    if matched_pols:
        Npols = len(pols)
        _templates = []
        for _file in unique_files:
            goodfile = True
            for pol in pols:
                if _file.format(pol=pol) not in allfiles:
                    goodfile = False
            if goodfile:
                _templates.append(_file)

        # achieve goal by calling search_data with new _templates that are polarization matched
        datafiles, datapols = search_data(_templates, pols, matched_pols=False, reverse_nesting=False)
    # reverse nesting if desired
    if reverse_nesting:
        datafiles = []
        datapols = []
        for _file in unique_files:
            dfs = []
            dps = []
            for pol in pols:
                df = _file.format(pol=pol)
                if df in allfiles:
                    dfs.append(df)
                    dps.append(pol)
            datafiles.append(dfs)
            datapols.append(dps)
    # flatten
    if flatten:
        datafiles = [item for sublist in datafiles for item in sublist]
        datapols = [item for sublist in datapols for item in sublist]

    return datafiles, datapols
