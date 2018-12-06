import aipy
import numpy as np
from six.moves import range

def wedge_width(bl_len, sdf, nchan, standoff=0., horizon=1.):
    '''Return the (upper,lower) delay bins that geometrically correspond to the sky.
    Variable names preserved for backward compatability with capo/PAPER analysis.
    
    Arguments:
        bl_len: length of baseline (in 1/[sdf], typically ns)
        sdf: frequency channel width (typically in GHz)
        nchan: number of frequency channels
        standoff: fixed additional delay beyond the horizon (same units as bl_len)
        horizon: proportionality constant for bl_len where 1 is the horizon (full light travel time)

    Returns:
        uthresh, lthresh: bin indices for filtered bins started at uthresh (which is filtered)
            and ending at lthresh (which is a negative integer and also not filtered)
            Designed for area = np.ones(nchan, dtype=np.int); area[uthresh:lthresh] = 0
    '''
    bl_dly = horizon * bl_len + standoff
    return calc_width(bl_dly, sdf, nchan)    


def calc_width(filter_size, real_delta, nsamples):
    '''Calculate the upper and lower bin indices of a fourier filter,
    assuming mode ordering convention of np.fft.ifft

    Arguments:
        filter_size: the half-width (i.e. the width of the positive part) of the region in fourier 
            space, symmetric about 0, that is filtered out. In units of 1/[real_delta].
            Alternatively, can be fed as len-2 tuple specifying the negative and positive bound
            of the filter in fourier space respectively.
        real_delta: the bin width in real space
        nsamples: the number of samples in the array to be filtered

    Returns:
        uthresh, lthresh: bin indices for filtered bins started at uthresh (which is filtered)
            and ending at lthresh (which is a negative integer and also not filtered).
            Designed for area = np.ones(nsamples, dtype=np.int); area[uthresh:lthresh] = 0
    '''
    if isinstance(filter_size, (list, tuple)):
        _, l = calc_width(np.abs(filter_size[1]), real_delta, nsamples)
        u, _ = calc_width(np.abs(filter_size[0]), real_delta, nsamples)
        return (u, l)        
    bin_width = 1.0 / (real_delta * nsamples)
    w = int(np.around(filter_size / bin_width))
    uthresh, lthresh = w + 1, -w
    if lthresh == 0: 
        lthresh = nsamples
    return (uthresh, lthresh)


def high_pass_fourier_filter(data, wgts, filter_size, real_delta, clean2d=False, tol=1e-9, window='none', 
                             skip_wgt=0.1, maxiter=100, gain=0.1, alpha=0.2):
    '''Apply a highpass fourier filter to data. Uses aipy.deconv.clean. Default is a 1D clean
    on the last axis of data.

    Arguments:
        data: 1D or 2D (real or complex) numpy array to be filtered.
            (Unlike previous versions, it is NOT assumed that weights have already been multiplied
            into the data.)
        wgts: real numpy array of linear multiplicative weights with the same shape as the data. 
        filter_size: the half-width (i.e. the width of the positive part) of the region in fourier 
            space, symmetric about 0, that is filtered out. In units of 1/[real_delta].
            Alternatively, can be fed as len-2 tuple specifying the negative and positive bound
            of the filter in fourier space respectively (for asymmetric CLEANing).
            For 2D cleaning, a len-2 list must be fed with each element following the rules
            above for each CLEAN dimension.
        real_delta: the bin width in real space of the dimension to be filtered.
            If 2D cleaning, then real_delta must also be a len-2 list.
        clean2d : bool, if True perform 2D clean, else perform a 1D clean on last axis.
        tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
        window: window function for filtering applied to the filtered axis. 
            See aipy.dsp.gen_window for options. If clean2D, can be fed as a list
            specifying the window for each axis in data.
        skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that 
            time. Only works properly when all weights are all between 0 and 1.
        maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
        gain: The fraction of a residual used in each iteration. If this is too low, clean takes
            unnecessarily long. If it is too high, clean does a poor job of deconvolving.
        alpha : float, if window is 'tukey', this is its alpha parameter.

    Returns:
        d_mdl: best fit low-pass filter components (CLEAN model) in real space
        d_res: best fit high-pass filter components (CLEAN residual) in real space
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    '''
    # type checks
    dndim = data.ndim
    assert dndim == 1 or dndim == 2, "data must be a 1D or 2D ndarray"
    if clean2d:
        assert dndim == 2, "data must be 2D for 2D clean"
        assert isinstance(filter_size, (tuple, list)), "filter_size must be list or tuple for 2D clean"
        assert len(filter_size) == 2, "len(filter_size) must equal 2 for 2D clean"
        assert isinstance(filter_size[0], (int, np.integer, float, np.float, list, tuple)) \
            and isinstance(filter_size[1], (int, np.integer, float, np.float, list, tuple)), "elements of filter_size must be floats or lists"
        assert isinstance(real_delta, (tuple, list)), "real_delta must be list or tuple for 2D clean"
        assert len(real_delta) == 2, "len(real_delta) must equal 2 for 2D clean"
    else:
        assert isinstance(real_delta, (int, np.integer, float, np.float)), "if not clean2d, real_delta must be a float"
        assert isinstance(window, (str, np.str)), "If not clean2d, window must be a string"

    # 1D clean
    if not clean2d:
        # setup _d and _w arrays
        if window == 'tukey':
            win_kwargs = {'alpha': alpha}
        else:
            win_kwargs = {}
        # this backwards way of constructing a window is done
        # b/c aipy returns just 1 when window is none
        nchan = data.shape[-1]
        _window = np.empty(nchan)
        _window[:] = aipy.dsp.gen_window(nchan, window=window, **win_kwargs)
        if dndim == 2:
            _window = _window[None, :]
        _d = np.fft.ifft(data * wgts * _window, axis=-1)
        _w = np.fft.ifft(wgts * _window, axis=-1)

        # calculate area array
        area = np.ones(nchan, dtype=np.int)
        uthresh, lthresh = calc_width(filter_size, real_delta, nchan)
        area[uthresh:lthresh] = 0

        # run clean
        if dndim == 1:
            # For 1D data array run once
            _d_cl, info = aipy.deconv.clean(_d, _w, area=area, tol=tol, stop_if_div=False, maxiter=maxiter, gain=gain)
            d_mdl = np.fft.fft(_d_cl)
            del info['res']
        elif data.ndim == 2:
            # For 2D data array, iterate
            info = []
            d_mdl = np.empty_like(data)
            for i in range(data.shape[0]):
                if _w[i, 0] < skip_wgt:
                    d_mdl[i] = 0 # skip highly flagged (slow) integrations
                    info.append({'skipped': True})
                else:
                    _d_cl, info_here = aipy.deconv.clean(_d[i], _w[i], area=area, tol=tol, stop_if_div=False, maxiter=maxiter, gain=gain)
                    d_mdl[i] = np.fft.fft(_d_cl)
                    del info_here['res']
                    info.append(info_here)

    # 2D clean on 2D data
    else:
        # setup _d and _w arrays
        if isinstance(window, (list, tuple)):
            w1, w2 = window
        else:
            w1, w2 = window, window
        if w1 == 'tukey':
            w1_kwargs = {'alpha': alpha}
        else:
            w1_kwargs = {}
        if w2 == 'tukey':
            w2_kwargs = {'alpha': alpha}
        else:
            w2_kwargs = {}

        # this backwards way of constructing a window is done
        # b/c aipy returns just 1 when window is none
        _w1, _w2 = np.empty(data.shape[0], dtype=np.float), np.empty(data.shape[1], dtype=np.float)
        _w1[:] = aipy.dsp.gen_window(data.shape[0], window=w1, **w1_kwargs)
        _w2[:] = aipy.dsp.gen_window(data.shape[1], window=w2, **w2_kwargs)
        _window = _w1[:, None] * _w2[None, :]
        _d = np.fft.ifft2(data * wgts * _window, axes=(0, 1))
        _w = np.fft.ifft2(wgts * _window, axes=(0, 1))

        # calculate area array
        a1 = np.ones(data.shape[0], dtype=np.int)
        uthresh, lthresh = calc_width(filter_size[0], real_delta[0], data.shape[0])
        a1[uthresh:lthresh] = 0
        a2 = np.ones(data.shape[1], dtype=np.int)
        uthresh, lthresh = calc_width(filter_size[1], real_delta[1], data.shape[1])
        a2[uthresh:lthresh] = 0
        area = np.outer(a1, a2)

        # run clean
        _d_cl, info = aipy.deconv.clean(_d, _w, area=area, tol=tol, stop_if_div=False, maxiter=maxiter, gain=gain)
        d_mdl = np.fft.fft2(_d_cl, axes=(0, 1))
        del info['res']

    d_res = data - d_mdl

    return d_mdl, d_res, info


def delay_filter(data, wgts, bl_len, sdf, standoff=0., horizon=1., min_dly=0.0, tol=1e-4,
                 window='none', skip_wgt=0.5, maxiter=100, gain=0.1, **win_kwargs):
    '''Apply a wideband delay filter to data. Variable names preserved for 
        backward compatability with capo/PAPER analysis.

    Arguments:
        data: 1D or 2D (real or complex) numpy array where last dimension is frequency.
            (Unlike previous versions, it is NOT assumed that weights have already been multiplied
            into the data.)
        wgts: real numpy array of linear multiplicative weights with the same shape as the data. 
        bl_len: length of baseline (in 1/[sdf], typically ns)
        sdf: frequency channel width (typically in GHz)
        standoff: fixed additional delay beyond the horizon (same units as bl_len)
        horizon: proportionality constant for bl_len where 1 is the horizon (full light travel time)
        min_dly: a minimum delay used for cleaning: if bl_dly < min_dly, use min_dly. same units as bl_len
        tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
        window: window function for filtering applied to the filtered axis. 
            See aipy.dsp.gen_window for options.        
        skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that 
            time. Only works properly when all weights are all between 0 and 1.
        maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
        gain: The fraction of a residual used in each iteration. If this is too low, clean takes
            unnecessarily long. If it is too high, clean does a poor job of deconvolving.
        win_kwargs : any keyword arguments for the window function selection in aipy.dsp.gen_window.
            Currently, the only window that takes a kwarg is the tukey window with a alpha=0.5 default.

    Returns:
        d_mdl: best fit low-pass filter components (CLEAN model) in the frequency domain
        d_res: best fit high-pass filter components (CLEAN residual) in the frequency domain
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    '''
    # get bl delay
    bl_dly = _get_bl_dly(bl_len, horizon=horizon, standoff=standoff, min_dly=min_dly)

    # run fourier filter
    return high_pass_fourier_filter(data, wgts, bl_dly, sdf, tol=tol, window=window,
                                    skip_wgt=skip_wgt, maxiter=maxiter, gain=gain, **win_kwargs)


def fringe_filter(data, wgts, max_frate, dt, tol=1e-4, skip_wgt=0.5, maxiter=100,
                  gain=0.1, window='none', **win_kwargs):
    """
    Run a CLEAN deconvolution along the time axis.

    Args:
        data : 1D or 2D data array. If 2D, shape=(Ntimes, Nfreqs)
        wgts : 1D or 2D weight array.
        max_frate : float, maximum fringe-rate (i.e. frequency) to CLEAN, units of 1/[dt]
        dt : float, time-bin width of data
        tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
        window: window function for filtering applied to the filtered axis. 
            See aipy.dsp.gen_window for options.        
        skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that 
            time. Only works properly when all weights are all between 0 and 1.
        maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
        gain: The fraction of a residual used in each iteration. If this is too low, clean takes
            unnecessarily long. If it is too high, clean does a poor job of deconvolving.
        win_kwargs : any keyword arguments for the window function selection in aipy.dsp.gen_window.
            Currently, the only window that takes a kwarg is the tukey window with a alpha=0.5 default.

    Returns:
        mdl : CLEAN model, in the time domain
        res : residual, in the time domain
        info : CLEAN info
    """
    # run fourier filter
    mdl, res, info = high_pass_fourier_filter(data.T, wgts.T, max_frate, dt, tol=tol, window=window,
                                              skip_wgt=skip_wgt, maxiter=maxiter, gain=gain, **win_kwargs)
    return mdl.T, res.T, info


def vis_filter(data, wgts, max_frate=None, dt=None, bl_len=None, sdf=None, standoff=0.0, horizon=1., min_dly=0., 
               tol=1e-4, window='none', maxiter=100, gain=1e-1, skip_wgt=0.5, **win_kwargs):
    """
    A generalized interface to delay and/or fringe-rate 1D CLEAN functions, or a full 2D clean
    if both bl_len & sdf and max_frate & dt variables are specified.

    Args:
        data : 1D or 2D data array. If 2D has shape=(Ntimes, Nfreqs)
        wgts : float weight array, matching shape of data
        max_frate : float, maximum fringe-rate (i.e. frequency) to CLEAN, units of 1/[dt]
        dt : float, time-bin width [sec]
        bl_len: length of baseline (in 1/[sdf], typically ns)
        sdf: frequency channel width (typically in GHz)
        standoff: fixed additional delay beyond the horizon (same units as bl_len)
        horizon: proportionality constant for bl_len where 1 is the horizon (full light travel time)
        min_dly: a minimum delay used for cleaning: if bl_dly < min_dly, use min_dly. same units as bl_len
        tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
        window: window function for filtering applied to the filtered axis. 
            See aipy.dsp.gen_window for options.        
        skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that 
            time. Only works properly when all weights are all between 0 and 1.
        maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
        gain: The fraction of a residual used in each iteration. If this is too low, clean takes
            unnecessarily long. If it is too high, clean does a poor job of deconvolving.
        win_kwargs : any keyword arguments for the window function selection in aipy.dsp.gen_window.
            Currently, the only window that takes a kwarg is the tukey window with a alpha=0.5 default.

    Returns:
        mdl : CLEAN model in data space
        res : CLEAN residual in data space
        info : CLEAN info
    """
    # type checks
    timeclean = False
    if dt is not None or max_frate is not None:
        timeclean = True
        assert max_frate is not None and dt is not None, "Must specify both max_frate and dt"

    freqclean = False
    if sdf is not None or bl_len is not None:
        freqclean = True
        assert sdf is not None and bl_len is not None, "Must specify both bl_len and sdf"

    clean2d = timeclean and freqclean

    # 1D clean
    if not clean2d:
        # time clean
        if timeclean:
            mdl, res, info = fringe_filter(data, wgts, max_frate, dt, tol=tol, skip_wgt=skip_wgt, maxiter=maxiter,
                                           gain=gain, window=window, **win_kwargs)

        # freq clean
        elif freqclean:
            mdl, res, info = delay_filter(data, wgts, bl_len, sdf, standoff=standoff, horizon=horizon, tol=tol,
                                          min_dly=min_dly, gain=gain, maxiter=maxiter, skip_wgt=skip_wgt, window=window, **win_kwargs)

    # 2D clean
    else:
        # get bl delay
        bl_dly = _get_bl_dly(bl_len, horizon=horizon, standoff=standoff, min_dly=min_dly)

        # 2D clean
        mdl, res, info = high_pass_fourier_filter(data, wgts, (max_frate, bl_dly), (dt, sdf), tol=tol, window=window,
                                                  maxiter=maxiter, gain=gain, clean2d=True, **win_kwargs)

    return mdl, res, info


def _get_bl_dly(bl_len, horizon=1., standoff=0., min_dly=0.):
    # construct baseline delay
    bl_dly = horizon * bl_len + standoff

    # check minimum delay
    bl_dly = np.max([bl_dly, min_dly])

    return bl_dly



#import binning

# def delay_filter_aa(aa, data, wgts, i, j, sdf, phs2lst=False, jds=None, 
#         skip_wgt=0.5, lst_res=binning.DEFAULT_LST_RES, standoff=0., horizon=1., 
#         tol=1e-4, window='none', maxiter=100):
#     '''Use information from AntennaArray object to delay filter data, with the
#     option to phase data to an lst bin first.  Arguments are the same as for
#     delay_filter and binning.phs2lstbin.  Returns mdl, residual, and info
#     in the frequency domain.'''
#     if phs2lst:
#         data = binning.phs2lstbin(data, aa, i, j, jds=jds, lst_res=lst_res)
#     bl = aa.get_baseline(i,j)
#     return delay_filter(data, wgts, np.linalg.norm(bl), sdf, 
#             standoff=standoff, horizon=horizon, tol=tol, window=window, 
#             skip_wgt=skip_wgt, maxiter=maxiter)


# XXX is this a used function?
#def delayfiltercov(C,horizon_bins=5,eig_cut_dnr=2):
    #delay filter a spectral covariance matrix
    #horizon_bins = distance delay=0 to be retained, ie the size of the wedge in bins
    # eig_cut_dnr = retain eigenvalues with a dynamic range of  median(dnr)*eig_cut_dnr 
    # where dnr is max(dspec eigenvector)/mean(abs(dpsec eigenvector outside horizon))    
    #
    # returns filtered_covariance,matching_projection matrix
    #S,V = np.linalg.eig(C)
    #dV = np.fft.ifft(V,axis=0)
    #calculate eigenvalue cut, selecting only those eigenvectors with strong delay spectrum signals
    #dnr = np.max(np.abs(dV),axis=0)/np.mean(np.abs(dV)[horizon_bins:-horizon_bins,:],axis=0)
    #median_dnr = np.median(dnr)
    #eig_cut_dnr *= median_dnr
    #S[dnr<eig_cut_dnr] = 0 #apply eigenvalue cut
    #mask outside wedge
    #dV[horizon_bins:-horizon_bins,:] = 0 # mask out stuff outside the horizon
    #V_filtered = np.fft.fft(dV,axis=0)
    #return filtered covariance and its matching projection matrix
    #return np.einsum('ij,j,jk',V_filtered,S,V_filtered.T),np.einsum('ij,j,jk',V_filtered,S!=0,V_filtered.T)

