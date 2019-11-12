from numpy.fft import fft, fftshift
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

def FFT(data, axis):
    """Convenient function for performing a FFT along an axis.

    Parameters
    ----------
    data : np.ndarray
       An array of data, assumed to not be ordered in the numpy FFT convention.
       Typically the data_array of a UVData object.

    axis : int
        The axis to perform the FFT over.

    Returns
    -------
    data_fft : np.ndarray
        The Fourier transform of the data along the specified axis. The array
        has the same shape as the original data, with the same ordering.
    """

    return fftshift(fft(fftshift(data, axis), axis=axis), axis)

def get_fourier_freqs(times):
    """A function for generating Fourier frequencies given 'times'.

    Parameters
    ----------
    times : np.ndarray, shape=(Ntimes,)
        An array of parameter values. These are nominally referred to as times,
        but may be frequencies or other parameters for which a Fourier dual can
        be defined. This function assumes a uniform sampling rate.

    Returns
    -------
    freqs : np.ndarray, shape=(Ntimes,)
        An array of coordinates dual to the input coordinates. Similar to the
        output of np.fft.fftfreq, but ordered so that the zero frequency is in
        the center of the array.
    """
    # get the number of samples and the sample rate
    N = len(times)
    dt = np.mean(np.diff(times))

    # get the Nyquist frequency
    f_nyq = 1.0 / (2 * dt)

    # return the frequency array
    return np.linspace(-f_nyq, f_nyq, N, endpoint=False)
