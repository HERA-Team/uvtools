from numpy.fft import fft, fftshift, ifftshift
import numpy as np
import glob

from . import dspec

# for checking UVData pair metadata in plot.plot_diff_x functions
class MetadataError(ValueError):
    pass

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

def FFT(data, axis=-1, taper=None, **kwargs):
    """Convenient function for performing a FFT along an axis.

    Parameters
    ----------
    data : np.ndarray
       An array of data, assumed to not be ordered in the numpy FFT convention.
       Typically the data_array of a UVData object.

    axis : int, optional
        The axis to perform the FFT over. Default is the last axis of the array.

    taper : str, optional
        Choice of taper (windowing function) to apply to the data. Default is 
        to use no taper.

    **kwargs
        Keyword arguments for generating the taper. Passed directly to 
        ``uvtools.dspec.gen_window``.

    Returns
    -------
    data_fft : np.ndarray
        The Fourier transform of the data along the specified axis. The array
        has the same shape as the original data, with the same ordering.
    """
    # Ensure that the trick for reshaping the window works for 1-D arrays.
    axis = np.arange(data.ndim)[axis]
    window = dspec.gen_window(taper, data.shape[axis], **kwargs)
    new_shape = tuple([1 if ax != axis else -1 for ax in range(data.ndim)])
    window.shape = new_shape

    # It's not obvious what the correct way to shift the data is; numpy docs
    # state that fft(A) gives the Fourier transform, with correct phases and
    # amplitudes, with the convention that positive frequencies are first,
    # followed by negative frequencies (i.e. fftshift(fft(A)) gives the right
    # answer ordered to be monotonically increasing in Fourier frequency).
    # Contrary to that, there are discussions on stackoverflow that suggest
    # the phases will be incorrect *unless* the data is subjected to an
    # ifftshift before being transformed; however, some simple tests suggest
    # that shifting the data prior to transforming actually creates phase
    # instabilities and conjugates the transformed data relative to what is
    # expected. So we perform no shifting prior to transforming.
    return fftshift(fft(window * data, axis=axis), axis)

def fourier_freqs(times):
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

def check_uvd_pair_metadata(uvd1, uvd2):
    """Check that the relevant metadata agrees for `uvd1` and `uvd2`.

    This check ensures that both ``UVData`` objects have the same number 
    of blts and frequencies. It also checks to make sure that the time 
    and frequency arrays agree to within the mean integration time and 
    channel width, respectively. Finally, the check ensures that both 
    ``UVData`` objects have the same baseline vectors, currently set 
    to the default tolerance for ``np.isclose``. Note that this check 
    does not ensure that both ``UVData`` objects have the same 
    polarization arrays.

    Parameters
    ----------
    uvd1, uvd2 : pyuvdata.UVData
        UVData objects containing the visibilities that are being compared
        have sufficiently similar metadata.
    """
    # make sure that both UVData objects have the same number of blts/freqs
    if not uvd1.Nblts == uvd2.Nblts:
        raise MetadataError("The number of baseline-times disagree.")

    if not uvd1.Nfreqs == uvd2.Nfreqs:
        raise MetadataError("The number of frequencies disagree.")

    # helper function; mean separation in array values for two arrays x1, x2
    dx = lambda x1, x2 : 0.5 * (np.mean(np.diff(x1)) + np.mean(np.diff(x2)))

    t1vals = np.unique(uvd1.time_array)
    t2vals = np.unique(uvd2.time_array)
    if uvd1.Ntimes > 1 and uvd2.Ntimes > 1:
        if not np.all(
            np.isclose(t1vals, t2vals, rtol=0, atol=dx(t1vals, t2vals))
        ):
            raise MetadataError(
                "Time values disagree more than the mean integration time."
            )

    f1vals = uvd1.freq_array[0]
    f2vals = uvd2.freq_array[0]
    if uvd1.Nfreqs > 1 and uvd2.Nfreqs > 1:
        if not np.all(np.isclose(f1vals, f2vals, atol=dx(f1vals, f2vals))):
            raise MetadataError(
                "Frequency values disagree more than the mean channel width."
            )

    bls1 = uvd1.uvw_array
    bls2 = uvd2.uvw_array
    if not np.all(np.isclose(bls1, bls2)):
        raise MetadataError(
            "Baseline arrays do not agree or are not correctly phased."
        )

    if not uvd1.vis_units == uvd2.vis_units:
        raise MetadataError("The visibility units do not agree.")

def diff(vis1, vis2, mode):
    """Calculate a specified type of difference between two visibilities.

    Parameters
    ----------
    vis1, vis2 : ndarray of complex
        Arrays of visibility data to be differenced.

    mode : str
        Type of difference to take. Possible options are as follows:
            'abs' : Return the difference of the visibility amplitudes
            'phs' : Return the difference of the visibility phases
            'complex' : Return the amplitude of the complex difference

    Returns
    -------
    visdiff : ndarray of float
        Difference of `vis1` and `vis2` calculated according to `mode`.

    """
    if mode == "abs":
        return np.abs(vis1) - np.abs(vis2)
    elif mode == "phs":
        return np.angle(vis1) - np.angle(vis2)
    elif mode == "complex":
        return np.abs(vis1 - vis2)
    else:
        raise ValueError("You have not specified an accepted differencing " \
                         "mode. The accepted modes are 'abs', 'phs', and " \
                         "'complex'. See the ``utils.diff`` docstring for " \
                         "details on what each mode does.")
