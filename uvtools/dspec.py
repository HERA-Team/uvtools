# -*- coding: utf-8 -*-
# Copyright (c) 2018 The HERA Collaboration
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import

import aipy
import numpy as np
from six.moves import range
from scipy.signal import windows
from warnings import warn
from scipy.optimize import leastsq, lsq_linear
import copy

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
    '''Calculate the upper and lower bin indices of a fourier filter

    Arguments:
        filter_size: the half-width (i.e. the width of the positive part) of the region in fourier
            space, symmetric about 0, that is filtered out. In units of 1/[real_delta].
            Alternatively, can be fed as len-2 tuple specifying the absolute value of the negative
            and positive bound of the filter in fourier space respectively.
            Example: (20, 40) --> (-20 < tau < 40)
        real_delta: the bin width in real space
        nsamples: the number of samples in the array to be filtered

    Returns:
        uthresh, lthresh: bin indices for filtered bins started at uthresh (which is filtered)
            and ending at lthresh (which is a negative integer and also not filtered).
            Designed for area = np.ones(nsamples, dtype=np.int); area[uthresh:lthresh] = 0
    '''
    if isinstance(filter_size, (list, tuple, np.ndarray)):
        _, l = calc_width(np.abs(filter_size[0]), real_delta, nsamples)
        u, _ = calc_width(np.abs(filter_size[1]), real_delta, nsamples)
        return (u, l)
    bin_width = 1.0 / (real_delta * nsamples)
    w = int(np.around(filter_size / bin_width))
    uthresh, lthresh = w + 1, -w
    if lthresh == 0:
        lthresh = nsamples
    return (uthresh, lthresh)


def high_pass_fourier_filter(data, wgts, filter_size, real_delta, clean2d=False, tol=1e-9, window='none',
                             skip_wgt=0.1, maxiter=100, gain=0.1, filt2d_mode='rect', alpha=0.5,
                             edgecut_low=0, edgecut_hi=0, add_clean_residual=False, mode='clean', cache={},
                             fg_deconv_method='clean', deconv_dayenu_foregrounds=False, fg_restore_size=None,
                             fg_deconv_fundamental_period=None):
    '''Apply a highpass fourier filter to data. Uses aipy.deconv.clean. Default is a 1D clean
    on the last axis of data.

    Arguments:
        data: 1D or 2D (real or complex) numpy array to be filtered.
            (Unlike previous versions, it is NOT assumed that weights have already been multiplied
            into the data.)
        wgts: real numpy array of linear multiplicative weights with the same shape as the data.
        filter_size: the half-width (i.e. the width of the positive part) of the region in fourier
            space, symmetric about 0, that is filtered out. In units of 1/[real_delta].
            Alternatively, can be fed as len-2 tuple specifying the absolute value of the negative
            and positive bound of the filter in fourier space respectively.
            Example: (20, 40) --> (-20 < tau < 40)
         real_delta: the bin width in real space of the dimension to be filtered.
            If 2D cleaning, then real_delta must also be a len-2 list.
        clean2d : bool, if True perform 2D clean, else perform a 1D clean on last axis.
        tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
        window: window function for filtering applied to the filtered axis.
            See dspec.gen_window for options. If clean2D, can be fed as a list
            specifying the window for each axis in data.
        skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
            time. Only works properly when all weights are all between 0 and 1.
        maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
        gain: The fraction of a residual used in each iteration. If this is too low, clean takes
            unnecessarily long. If it is too high, clean does a poor job of deconvolving.
        alpha : float, if window is 'tukey', this is its alpha parameter.
        filt2d_mode : str, only applies if clean2d == True. options = ['rect', 'plus']
            If 'rect', a 2D rectangular filter is constructed in fourier space (default).
            If 'plus', the 'rect' filter is first constructed, but only the plus-shaped
            slice along 0 delay and fringe-rate is kept.
        edgecut_low : int, number of bins to consider zero-padded at low-side of the FFT axis,
            such that the windowing function smoothly approaches zero. For 2D cleaning, can
            be fed as a tuple specifying edgecut_low for first and second FFT axis.
        edgecut_hi : int, number of bins to consider zero-padded at high-side of the FFT axis,
            such that the windowing function smoothly approaches zero. For 2D cleaning, can
            be fed as a tuple specifying edgecut_hi for first and second FFT axis.
        add_clean_residual : bool, if True, adds the CLEAN residual within the CLEAN bounds
            in fourier space to the CLEAN model. Note that the residual actually returned is
            not the CLEAN residual, but the residual in input data space.
        mode : string,
             choose from ['clean','dayenu','dft_interp']
             use aipy.deconv.clean if 'clean'
             use 'dayenu' if 'dayenu'
             if 'dft_interp', then interpolates flagged channels with DFT modes.
        cache : dict, optional dictionary for storing pre-computed filtering matrices in linear
            cleaning.
        deconv_dayenu_foregrounds : bool, if True, then apply clean to data - residual where
            res is the data-vector after applying a linear clean filter.
            This allows for in-painting flagged foregrounds without introducing
            clean artifacts into EoR window. If False, mdl will still just be the
            difference between the original data vector and the residuals after
            applying the linear filter.
        fg_deconv_method : string, can be 'leastsq' or 'clean'. If 'leastsq', deconvolve difference between data and linear residual
            by performing linear least squares fitting of data - linear resid to dft modes in filter window.
            If 'clean', obtain deconv fg model using perform a hogboem clean of difference between data and linear residual.
        fg_restore_size: float, optional, allow user to only restore foregrounds subtracted by linear filter
             within a region of this size. If None, set to filter_size.
             This allows us to avoid the problem that if we have RFI flagging and apply a linear filter
             that is larger then the horizon then the foregrounds that we fit might actually include super
             -horizon flagging side-lobes and restoring them will introduce spurious structure.
        fg_deconv_fundamental_period: int, optional
            fundamental period of Fourier modes to fit too.
            if none, default to length of data vector.

    Returns:
        d_mdl: CLEAN model -- best fit low-pass filter components (CLEAN model) in real space
        d_res: CLEAN residual -- difference of data and d_mdl, nulled at flagged channels
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    '''
    # type checks
    dndim = data.ndim
    if fg_restore_size is None:
        fg_restore_size = filter_size
    if fg_deconv_fundamental_period is None:
        fg_deconv_fundamental_period = data.shape[-1]
    if isinstance(fg_deconv_fundamental_period, (float, int, np.float, np.int)):
        fg_deconv_fundamental_period = [fg_deconv_fundamental_period]

    assert dndim == 1 or dndim == 2, "data must be a 1D or 2D ndarray"

    if not mode in ['clean', 'dayenu', 'dft_interp']:
        raise ValueError("mode must be in ['clean', 'dayenu', 'dft_interp']")

    if clean2d:
        assert dndim == 2, "data must be 2D for 2D clean"
        assert isinstance(filter_size, (tuple, list)), "filter_size must be list or tuple for 2D clean"
        assert len(filter_size) == 2, "len(filter_size) must equal 2 for 2D clean"
        assert isinstance(filter_size[0], (int, np.integer, float, np.float, list, tuple)) \
            and isinstance(filter_size[1], (int, np.integer, float, np.float, list, tuple)), "elements of filter_size must be floats or lists"
        assert isinstance(real_delta, (tuple, list)), "real_delta must be list or tuple for 2D clean"
        assert len(real_delta) == 2, "len(real_delta) must equal 2 for 2D clean"
        if isinstance(edgecut_low, (int, np.integer)):
            edgecut_low = (edgecut_low, edgecut_low)
        if isinstance(edgecut_hi, (int, np.integer)):
            edgecut_hi = (edgecut_hi, edgecut_hi)
        if isinstance(window, (str, np.str)):
            window = (window, window)
        if isinstance(alpha, (float, np.float, int, np.integer)):
            alpha = (alpha, alpha)
    else:
        assert isinstance(real_delta, (int, np.integer, float, np.float)), "if not clean2d, real_delta must be a float"
        assert isinstance(window, (str, np.str)), "If not clean2d, window must be a string"

    # 1D clean
    if not clean2d:
        # setup _d and _w arrays
        win = gen_window(window, data.shape[-1], alpha=alpha, edgecut_low=edgecut_low, edgecut_hi=edgecut_hi)
        if dndim == 2:
            win = win[None, :]
        _d = np.fft.ifft(data * wgts * win, axis=-1)
        _w = np.fft.ifft(wgts * win, axis=-1)

        # calculate area array
        area = np.ones(data.shape[-1], dtype=np.int)
        uthresh, lthresh = calc_width(filter_size, real_delta, data.shape[-1])
        area[uthresh:lthresh] = 0

        ff = [ tol ]
        if isinstance(filter_size, np.float):
            fc = [ 0. ]
            fw = [ filter_size ]
        else:
            half_width =  (filter_size[1]-filter_size[0]) / 2.
            center = (filter_size[0] + filter_size[1]) / 2.
            fc = [ center ]
            fw = [ half_width ]
        if isinstance(fg_restore_size, float):
            fcfg = [ 0. ]
            fwfg = [ fg_restore_size ]
        else:
            fcfg = [ (fg_restore_size[0] + fg_restore_size[1]) / 2.  ]
            fwfg = [ (fg_restore_size[1] - fg_restore_size[0]) / 2. ]
        uthresh_fg, lthresh_fg = calc_width(fg_restore_size, real_delta, data.shape[-1])
        area_fg = np.ones(data.shape[-1], dtype=np.int)
        area_fg[uthresh_fg:lthresh_fg] = 0
        # run clean
        if dndim == 1:
            # For 1D data array run once
            if mode=='clean':
                _d_cl, info = aipy.deconv.clean(_d, _w, area=area, tol=tol, stop_if_div=False, maxiter=maxiter, gain=gain)
                _d_res = info['res']
                del info['res']
            elif mode=='dayenu':
                d_r, info = dayenu_filter(data * wgts * win, wgts * win, delta_data=real_delta,
                                                filter_dimensions = [1], filter_centers=fc, filter_half_widths=fw, filter_factors=ff, cache=cache)
                _d_res = np.fft.ifft(d_r)
                if deconv_dayenu_foregrounds:
                    if fg_deconv_method == 'clean':
                        _d_cl, info_fg = aipy.deconv.clean(_d - _d_res, _w, area=area_fg, tol=tol, stop_if_div=False, maxiter=maxiter, gain=gain)
                        del info_fg['res']
                        info_fg['maxiter'] = maxiter
                        info_fg['gain'] = gain
                        info['fg_deconv'] = info_fg
                    elif fg_deconv_method == 'leastsq':
                        nmin = int((fcfg[0] - fwfg[0]) * real_delta * fg_deconv_fundamental_period[-1])
                        nmax = int((fcfg[0] + fwfg[0]) * real_delta * fg_deconv_fundamental_period[-1])
                        info['fg_deconv'] = {'method':'leastsq','nmin':nmin, 'nmax':nmax}
                        d_cl, _, _ = delay_filter_leastsq_1d( (data * wgts * win - d_r).squeeze(), flags=(wgts==0.).squeeze(), sigma=1.,
                                                            nmax=(nmin, nmax), freq_units=True, even_modes=True, fundamental_period=fg_deconv_fundamental_period[-1])
                        _d_cl = np.fft.ifft(d_cl)
                else:
                    _d_cl = _d - _d_res
            elif mode=='dft_interp':
                nmin = int((fcfg[0] - fwfg[0]) * real_delta * fg_deconv_fundamental_period[-1])
                nmax = int((fcfg[0] + fwfg[0]) * real_delta * fg_deconv_fundamental_period[-1])
                info['fg_deconv'] = {'method':'dft_interp','nmin':nmin, 'nmax':nmax}
                d_cl, _, _ = delay_filter_leastsq_1d( (data * wgts * win ).squeeze(), flags=(wgts==0.).squeeze(), sigma=1.,
                                                    nmax=(nmin, nmax), freq_units=True, even_modes=True, fundamental_period=fg_deconv_fundamental_period[-1])
                _d_cl = np.fft.ifft(d_cl)
                _d_res = _d  - _d_cl

        elif data.ndim == 2:
            # For 2D data array, iterate
            info = []
            _d_cl = np.empty_like(data)
            _d_res = np.empty_like(data)
            for i in range(data.shape[0]):
                if _w[i, 0] < skip_wgt:
                    _d_cl[i] = 0  # skip highly flagged (slow) integrations
                    _d_res[i] = _d[i]
                    info.append({'skipped': True})
                else:
                    if mode=='clean':
                        _cl, info_here = aipy.deconv.clean(_d[i], _w[i], area=area, tol=tol, stop_if_div=False, maxiter=maxiter, gain=gain)
                        _d_cl[i] = _cl
                        _d_res[i] = info_here['res']
                        del info_here['res']
                        info.append(info_here)
                    elif mode=='dayenu':
                        d_r, info_here = dayenu_filter(data[i] * wgts[i] * win, wgts[i] * win, delta_data=real_delta,
                                                            filter_dimensions=[1], filter_centers=fc, filter_half_widths=fw, filter_factors=ff, cache=cache)
                        _d_res[i] = np.fft.ifft(d_r)
                        if deconv_dayenu_foregrounds:
                            if fg_deconv_method == 'clean':
                                _d_cl[i], info_fg = aipy.deconv.clean(_d[i] - _d_res[i], _w[i], area=area_fg, tol=tol, stop_if_div=False, maxiter=maxiter, gain=gain)
                                del info_fg['res']
                                info_fg['maxiter'] = maxiter
                                info_fg['gain'] = gain
                                info_here['fg_deconv'] = info_fg
                            elif fg_deconv_method == 'leastsq':
                                nmin = int((fcfg[0] - fwfg[0]) * real_delta * fg_deconv_fundamental_period[-1])
                                nmax = int((fcfg[0] + fwfg[0]) * real_delta * fg_deconv_fundamental_period[-1])
                                info_here['fg_deconv'] = {'method':'leastsq','nmin':nmin, 'nmax':nmax}
                                d_cl, _, _ = delay_filter_leastsq_1d( (data[i] * wgts[i] * win - d_r).squeeze(), flags=(wgts[i]==0.).squeeze(), sigma=1.,
                                                                    nmax=(nmin, nmax), freq_units=True, even_modes=True, fundamental_period=fg_deconv_fundamental_period[-1])
                                _d_cl[i] = np.fft.ifft(d_cl)
                        else:
                            _d_cl[i] = _d[i] - _d_res[i]
                        info.append(info_here)

                    elif mode=='dft_interp':
                        info_here = {}
                        nmin = int((fcfg[0] - fwfg[0]) * real_delta * fg_deconv_fundamental_period[-1])
                        nmax = int((fcfg[0] + fwfg[0]) * real_delta * fg_deconv_fundamental_period[-1])
                        info_here['fg_deconv'] = {'method':'dft_interp','nmin':nmin, 'nmax':nmax}
                        d_cl, _, _ = delay_filter_leastsq_1d( (data[i] * wgts[i] * win ).squeeze(), flags=(wgts[i]==0.).squeeze(), sigma=1.,
                                                            nmax=(nmin, nmax), freq_units=True, even_modes=True, fundamental_period=fg_deconv_fundamental_period[-1])
                        _d_cl[i] = np.fft.ifft(d_cl)
                        _d_res[i] = _d[i] - _d_cl[i]
                        info.append(info_here)

    # 2D clean on 2D data
    else:
        # setup _d and _w arrays
        win1 = gen_window(window[0], data.shape[0], alpha=alpha[0], edgecut_low=edgecut_low[0], edgecut_hi=edgecut_hi[0])
        win2 = gen_window(window[1], data.shape[1], alpha=alpha[1], edgecut_low=edgecut_low[1], edgecut_hi=edgecut_hi[1])
        win = win1[:, None] * win2[None, :]
        _d = np.fft.ifft2(data * wgts * win, axes=(0, 1))
        _w = np.fft.ifft2(wgts * win, axes=(0, 1))

        # calculate area array
        a1 = np.ones(data.shape[0], dtype=np.int)
        uthresh, lthresh = calc_width(filter_size[0], real_delta[0], data.shape[0])
        a1[uthresh:lthresh] = 0
        a2 = np.ones(data.shape[1], dtype=np.int)
        uthresh, lthresh = calc_width(filter_size[1], real_delta[1], data.shape[1])
        a2[uthresh:lthresh] = 0
        area = np.outer(a1, a2)
        # the following lines are only necessary for linear filtering.
        uthresh_fg, lthresh_fg = calc_width(fg_restore_size[0], real_delta[0], data.shape[0])
        a1fg = np.ones(data.shape[0], dtype=np.int)
        a1fg[uthresh_fg:lthresh_fg] = 0
        uthresh_fg, lthresh_fg = calc_width(fg_restore_size[1], real_delta[1], data.shape[1])
        a2fg = np.ones(data.shape[-1], dtype=np.int)
        a2fg[uthresh_fg:lthresh_fg] = 0
        area_fg = np.outer(a1fg, a2fg)

        fc = []
        fw = []
        fcfg = []
        fwfg = []
        for fs in range(2):
            if isinstance(filter_size[fs], tuple) or isinstance(filter_size[fs], list):
                fct = [ (filter_size[fs][1] + filter_size[fs][0]) / 2.  ]
                fwt = [ (filter_size[fs][1] - filter_size[fs][0]) / 2.  ]
            else:
                fct = [ 0. ]
                fwt = [ filter_size[fs]  ]
            if isinstance(fg_restore_size[fs], float):
                fcfgt = [ 0. ]
                fwfgt = [ fg_restore_size[fs] ]
            else:
                fcfgt = [ (fg_restore_size[fs][0] + fg_restore_size[fs][1]) / 2.  ]
                fwfgt = [ (fg_restore_size[fs][1] - fg_restore_size[fs][0]) / 2. ]
            fc.append(fct)
            fw.append(fwt)
            fcfg.append(fcfgt)
            fwfg.append(fwfgt)
        ff = [ [tol],[tol] ]

        # check for filt2d_mode
        if filt2d_mode == 'plus':
            _area = np.zeros(data.shape, dtype=np.int)
            _area_fg = np.zeros_like(_area)
            if not mode=='dayenu':
                _area[:, 0] = area[:, 0]
                _area[0, :] = area[0, :]
                _area_fg[:, 0] = area_fg[:, 0]
                _area_fg[0, :] = area_fg[0, :]
            else:
                _area_fg[a1 == 1.,:] = 1.
                _area_fg[:, a2 == 1.] = 1.
            area = _area
            area_fg = _area_fg
        elif filt2d_mode == 'rect':
            pass
        else:
            raise ValueError("Didn't recognize filt2d_mode {}".format(filt2d_mode))

        # run clean
        if mode == 'clean':
            _d_cl, info = aipy.deconv.clean(_d, _w, area=area, tol=tol, stop_if_div=False, maxiter=maxiter, gain=gain)
            _d_res = info['res']
            del info['res']
        elif mode == 'dayenu':
            assert filt2d_mode == "plus", "2d linear deconvolution only supports filt2d_mode == 'plus'."

            d_r, info = dayenu_filter(data * wgts * win, wgts * win, delta_data=[real_delta[0],real_delta[1]], filter_centers=fc, filter_half_widths=fw,
                                         filter_factors=ff, cache=cache, filter_dimensions=[0, 1])
            _d_res = np.fft.ifft2(d_r)
            if deconv_dayenu_foregrounds:
                if fg_deconv_method == 'clean':
                    _d_cl, info_fg = aipy.deconv.clean(_d - _d_res, _w, area=area_fg, tol=tol, stop_if_div=False, maxiter=maxiter, gain=gain)
                    del info_fg['res']
                    info_fg['maxiter'] = maxiter
                    info_fg['gain'] = gain
                    info['fg_deconv'] = info_fg
                elif fg_deconv_method == 'leastsq':
                    nmin = int((fcfg[1][0] - fwfg[1][0]) * real_delta[1] * data.shape[-1])
                    nmax = int((fcfg[1][0] + fwfg[1][0]) * real_delta[1] * data.shape[-1])
                    info['fg_deconv'] = {'method':'leastsq', 'nmin':nmin, 'nmax':nmax}
                    d_cl, _, _ = delay_filter_leastsq(data * wgts * win - d_r, flags=wgts==0., sigma=1.,
                                                      nmax=(nmin, nmax), freq_units=True, even_modes=True)
                    _d_cl = np.fft.ifft(d_cl)

            else:
                _d_cl = _d - _d_res
        elif mode == 'dft_interp':
            raise ValueError("2d clean not yet supported for dft interpolation.")


    # add resid to model in CLEAN bounds
    if add_clean_residual:
        _d_cl += _d_res * area

    # fft back to input space
    if clean2d:
        d_mdl = np.fft.fft2(_d_cl, axes=(0, 1))
        d_res = np.fft.fft2(_d_res, axes=(0, 1))
    else:
        d_mdl = np.fft.fft(_d_cl)
        d_res = np.fft.fft(_d_res)
    # get residual in data space
    if mode =='clean' or mode == 'dft_interp':
        d_res = (data - d_mdl) * ~np.isclose(wgts * win, 0.0)

    return d_mdl, d_res, info

def dayenu_filter(data, wgts, filter_dimensions, filter_centers, filter_half_widths, filter_factors, delta_data=None, cache = {}, user_frequencies=None):
    '''Apply a linear delay filter to waterfall data.
        Due to performance reasons, linear filtering only supports separable delay/fringe-rate filters.
    Arguments:
        data: 1D or 2D (real or complex) numpy array where last dimension is frequency.
        Does not assume that weights have already been multiplied!
        wgts: real numpy array of linear multiplicative weights with the same shape as the data.
        delta_data: float, list
            the width of data bins. Typically Hz: float. if 2d clean, should be 2-tuple or 2-list
        filter_dimensions: list
            list of integers indicating data dimensions to filter. Must be 0, 1, or -1
        filter_centers: float, list, or 1d numpy array of delays at which to center filter windows
            Typically in units of (seconds)
        filter_half_widths: float, list, or 1d numpy array of half-widths of each delay filtere window
            with centers specified by filter_centers.
            Typically in units of (seconds)
        filter_factors: float, list, or 1d numpy array of factors by which filtering should be
            applied within each filter window specified in filter_centers and
            filter_half_widths. If a float or length-1 list/ndarray is provided,
            the same filter factor will be used in every filter window.
        cache: optional dictionary for storing pre-computed delay filter matrices.
        user_frequencies: optional
            array-like list of arbitrary frequencies. If this is supplied, evaluate sinc_downweight_mat at these frequencies
            instead of linear array of nchan. For 2d clean, should be supplied as a 2-tuple or 2-list.
    Returns:
        data: array, 2d clean residual with data filtered along the frequency direction.
        info: dictionary with filtering parameters and a list of skipped_times and skipped_channels

    '''
    # check that data and weight shapes are consistent.
    d_shape = data.shape
    w_shape = wgts.shape
    d_dim = data.ndim
    w_dim = wgts.ndim
    assert not (delta_data is None and user_frequencies is None), "Error: no delta_data or user_frequencies provided. delta_data must be specified if no user_frequencies are specified. user_frequencies must be specified if delta_data is not specified."
    if not (d_dim == 1 or d_dim == 2):
        raise ValueError("number of dimensions in data array does not "
                         "equal 1 or 2! data dim = %d"%(d_dim))
    if not (w_dim == 1 or w_dim == 2):
        raise ValueError("number of dimensions in wgts array does not "
                         "equal 1 or 2! wght dim = %d"%(w_dim))
    if not w_dim == d_dim:
        raise ValueError("number of dimensions in data array does not equal "
                         "number of dimensions in weights array."
                         "data.dim == %d, wgts.dim == %d"%(d_dim, w_dim))
    for dim in range(d_dim):
        if not d_shape[dim] == w_shape[dim]:
            raise ValueError("number of elements along data dimension %d, nel=%d"
                             "does not equal the number of elements along weight"
                             "dimension %d, nel = %d"%(dim, d_shape[dim], dim, w_shape[dim]))
    #convert 1d data to 2d data to save lines of code.
    if d_dim == 1:
        data = np.asarray([data])
        wgts = np.asarray([wgts])
        data_1d = True
        # 1d data will result in nonsensical filtering along zeroth axis.
        filter_dimensions=[1]

    else:
        data_1d = False
    nchan = data.shape[1]
    ntimes = data.shape[0]
    # Check that inputs are tiples or lists
    assert isinstance(filter_dimensions, (list,tuple,int)), "filter_dimensions must be a list or tuple"
    # if filter_dimensions are supplied as a single integer, convert to list (core code assumes lists).
    if isinstance(filter_dimensions, int):
        filter_dimensions = [filter_dimensions]
    # check that filter_dimensions is no longer then 2 elements
    assert len(filter_dimensions) in [1, 2], "length of filter_dimensions cannot exceed 2"
    # make sure filter_dimensions are 0 or 1.
    for dim in filter_dimensions:
        assert isinstance(dim,int), "only integers are valid filter dimensions"
    # make sure that all filter dimensions are valid for the supplied data.
    assert np.all(np.abs(np.asarray(filter_dimensions)) < data.ndim), "invalid filter dimensions provided, must be 0 or 1/-1"
    # convert filter dimensions to a list of integers (incase the dimensions were supplied as floats)
    filter_dimensions=list(np.unique(np.asarray(filter_dimensions)).astype(int))
    # will only filter each dim a single time.
    # now check validity of other inputs. We perform the same check over multiple
    # inputs by iterating over a list with their names.
    check_vars = [filter_centers, filter_half_widths, filter_factors]
    check_names = ['filter_centers', 'filter_half_widths', 'filter_factors']
    for anum, aname, avar in zip(range(len(check_vars)),check_names,check_vars):
        # If any of these inputs is a float or numpy array, convert to a list.
        if isinstance(avar, np.ndarray):
            check_vars[anum] = list(avar)
        elif isinstance(avar, np.float):
            check_vars[anum] = [avar]

    filter_centers,filter_half_widths,filter_factors = check_vars
    # Next, perform some checks that depend on the filtering dimensions provided.
    if 0 in filter_dimensions and 1 in filter_dimensions:
        for avar,aname in zip(check_vars,check_names):
            err_msg = "2d clean specified! %s must be a length-2 list of lists for 2d clean"%aname
            # if we are going to filter in dimension 1 and 0, make sure that each input
            # listed in check_vars is a length-2 list of lists.
            if len(avar) == 2:
                if not (isinstance(avar[0], list) and isinstance(avar[1], list)):
                    raise ValueError(err_msg)
            else:
                raise ValueError(err_msg)
            assert (isinstance(delta_data,(tuple,list,np.ndarray)) and len(delta_data) == 2) or delta_data is None, "For 2d filtering, delta_data must be a 2d long list or tuple or ndarray"
            assert (isinstance(user_frequencies,(tuple,list,np.ndarray)) and len(user_frequencies) == 2) or user_frequencies is None, "For 2d filtering, user_frequencies must be 2d long list or tuple or ndarray"
            if user_frequencies is None:
                user_frequencies = [None, None]
            if delta_data is None:
                delta_data = [None, None]
        for ff_num,ff_list in zip([0,1],filter_factors):
            # we allow the user to provide a single filter factor for multiple
            # filtering windows on a single dimension. This code
            # iterates through each dimension and if a single filter_factor is provided
            # it converts the filter_factor list to a list of filter_factors with the same
            # length as filter_centers.
            if len(ff_list) == 1:
                ff_list = [ff_list[0] for m in range(len(filter_centers[ff_num]))]
    else:
        assert isinstance(user_frequencies,(list,np.ndarray)) or user_frequencies is None, "for 1d clean, provide a list or numpy array for user_frequencies"
        if not delta_data is None:
            assert isinstance(delta_data, (float,np.float, int, np.int)), "for 1d clean, provide a float or integer for delta_data"
        # If we are going to filter along a single dimensions.
        if len(filter_factors) == 1:
            # extend filter factor list of user supplied a float or len-1 list.
            filter_factors = [filter_factors[0] for m in range(len(filter_centers))]
        if 0 in filter_dimensions:
            # convert 1d input-lists to
            # a list of lists for core-code to operate on.
            filter_factors = [filter_factors,[]]
            filter_centers = [filter_centers,[]]
            filter_half_widths = [filter_half_widths,[]]
            user_frequencies = [user_frequencies,None]
            delta_data = [delta_data,0.]
        elif 1 in filter_dimensions:
            # convert 1d input-lists to
            # a list of lists for core-code to operate on.
            filter_factors = [[],filter_factors]
            filter_centers = [[],filter_centers]
            filter_half_widths = [[],filter_half_widths]
            delta_data = [0., delta_data]
            user_frequencies = [None, user_frequencies]
    check_vars = [filter_centers, filter_half_widths, filter_factors]
    # Now check that the number of filter factors = number of filter widths
    # = number of filter centers for each dimension.
    for fs in filter_dimensions:
        for aname1,avar1 in zip(check_names,check_vars):
            for aname2,avar2 in zip(check_names,check_vars):
                if not len(avar1[fs]) == len(avar2[fs]):
                    raise ValueError("Number of elements in %s-%d must equal the"
                                     " number of elements %s-%d!"%(aname1, fs, aname2, fs))

    info = {'filter_centers':filter_centers, 'filter_half_widths':filter_half_widths, 'filter_factors': filter_factors,
            'delta_data':delta_data, 'data_shape':data.shape, 'filter_dimensions': filter_dimensions, 'user_frequencies':user_frequencies}
    skipped = [[],[]]
    # in the lines below, we iterate over the time dimension. For each time, we
    # compute a lazy covariance matrix (filter_mat) from the weights (wght) and
    # a sinc downweight matrix. (dayenu_mat_inv). We then attempt to
    # take the psuedo inverse to get a filtering matrix that removes foregrounds.
    # we do this for the zeroth and first filter dimension.
    output = copy.deepcopy(data)
    #this loop iterates through dimensions to iterate over (fs is the non-filter
    #axis).
    for fs in filter_dimensions:
        if fs == 0:
            _d, _w = output.T, wgts.T
        else:
            _d, _w = output, wgts
        #if the axis orthogonal to the iteration axis is to be filtered, then
        #filter it!.
        for sample_num, sample, wght in zip(range(data.shape[fs-1]), _d, _w):
            if user_frequencies[fs] is None:
                filter_key = (data.shape[fs], delta_data[fs], ) + tuple(filter_centers[fs]) + \
                tuple(filter_half_widths[fs]) + tuple(filter_factors[fs]) + tuple(wght.tolist()) + ('inverse',)
            else:
                filter_key = (data.shape[fs], ) + tuple(user_frequencies[fs]) + tuple(filter_centers[fs]) + \
                tuple(filter_half_widths[fs]) + tuple(filter_factors[fs]) + tuple(wght.tolist()) + ('inverse',)
            if not filter_key in cache:
                #only calculate filter matrix and psuedo-inverse explicitly if they are not already cached
                #(saves calculation time).
                wght_mat = np.outer(wght.T, wght)
                if not user_frequencies[fs] is None:
                    df=1.
                else:
                    df=delta_data[fs]
                filter_mat = dayenu_mat_inv(nchan=data.shape[fs], df=df, filter_centers=filter_centers[fs],
                                                     filter_half_widths=filter_half_widths[fs],
                                                     filter_factors=filter_factors[fs], cache=cache,
                                                     user_frequencies=user_frequencies[fs]) * wght_mat
                try:
                    #Try taking psuedo-inverse. Occasionally I've encountered SVD errors
                    #when a lot of channels are flagged. Interestingly enough, I haven't
                    #I'm not sure what the precise conditions for the error are but
                    #I'm catching it here.
                    cache[filter_key] = np.linalg.pinv(filter_mat)
                except np.linalg.LinAlgError:
                    #if SVD fails to converge, set filter matrix to to lots of nans and skip it
                    #during multiplication.
                    cache[filter_key] = np.ones((data.shape[fs], data.shape[fs]), dtype=complex) * np.nan
            #if matrix is already cached,
            filter_mat = cache[filter_key]
            if not np.all(filter_mat == 9e99):
                if fs == 0:
                    output[:, sample_num] = np.dot(filter_mat, sample)
                elif fs == 1:
                    output[sample_num] = np.dot(filter_mat, sample)
            else:
                skipped[fs-1].append(sample_num)

    #1d data will only be filtered across "channels".
    if data_1d and ntimes == 1:
        output = output[0]
    info['skipped_time_steps'] = skipped[0]
    info['skipped_channels'] = skipped[1]
    return output, info


def delay_filter(data, wgts, bl_len, sdf, standoff=0., horizon=1., min_dly=0.0, tol=1e-4,
                 window='none', skip_wgt=0.5, maxiter=100, gain=0.1, edgecut_low=0, edgecut_hi=0,
                 alpha=0.5, add_clean_residual=False, mode='clean', cache={},
                 deconv_dayenu_foregrounds=False, fg_deconv_method='clean',
                 fg_restore_size=None, fg_deconv_fundamental_period=None):
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
            See dspec.gen_window for options.
        skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
            time. Only works properly when all weights are all between 0 and 1.
        maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
        gain: The fraction of a residual used in each iteration. If this is too low, clean takes
            unnecessarily long. If it is too high, clean does a poor job of deconvolving.
        edgecut_low : int, number of bins to consider zero-padded at low-side of the FFT axis,
            such that the windowing function smoothly approaches zero. For 2D cleaning, can
            be fed as a tuple specifying edgecut_low for first and second FFT axis.
        edgecut_hi : int, number of bins to consider zero-padded at high-side of the FFT axis,
            such that the windowing function smoothly approaches zero. For 2D cleaning, can
            be fed as a tuple specifying edgecut_hi for first and second FFT axis.
        alpha : float, if window is tukey this is its alpha parameter.
        add_clean_residual : bool, if True, adds the residual within the CLEAN bounds
            in fourier space to the CLEAN model (and sets residual within CLEAN bounds to zero).
            This is more in-line with a standard filtering operation, rather than a CLEAN operation.
            If False, residual is not added to the CLEAN model.
        mode : string,
             choose from ['clean','dayenu','dft_interp']
             use aipy.deconv.clean if 'clean'
             use 'dayenu' if 'dayenu'
             if 'dft_interp', then interpolates flagged channels with DFT modes.
        cache : dict, optional dictionary for storing pre-computed filtering matrices in linear
            cleaning.
        deconv_dayenu_foregrounds : bool, if True, then apply clean to data - residual where
            res is the data-vector after applying a linear clean filter.
            This allows for in-painting flagged foregrounds without introducing
            clean artifacts into EoR window. If False, mdl will still just be the
            difference between the original data vector and the residuals after
            applying the linear filter.
        fg_deconv_method : string, can be 'leastsq' or 'clean'. If 'leastsq', deconvolve difference between data and linear residual
            by performing linear least squares fitting of data - linear resid to dft modes in filter window.
            If 'clean', obtain deconv fg model using perform a hogboem clean of difference between data and linear residual.
        fg_restore_size: float, optional, allow user to only restore foregrounds subtracted by linear filter
            within a region of this size. If None, set to filter_size.
            This allows us to avoid the problem that if we have RFI flagging and apply a linear filter
            that is larger then the horizon then the foregrounds that we fit might actually include super
            -horizon flagging side-lobes and restoring them will introduce spurious structure.
        fg_deconv_fundamental_period: int, optional
            fundamental period of Fourier modes to fit too.
            if none, default to length of data vector.

    Returns:
        d_mdl: CLEAN model -- best fit low-pass filter components (CLEAN model) in real space
        d_res: CLEAN residual -- difference of data and d_mdl, nulled at flagged channels
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    '''
    # print deprecation warning
    warn("Warning: dspec.delay_filter will soon be deprecated in favor of filtering.vis_filter",
         DeprecationWarning)

    # get bl delay
    bl_dly = _get_bl_dly(bl_len, horizon=horizon, standoff=standoff, min_dly=min_dly)

    return high_pass_fourier_filter(data, wgts, bl_dly, sdf, tol=tol, window=window, edgecut_low=edgecut_low,
                                    edgecut_hi=edgecut_hi, skip_wgt=skip_wgt, maxiter=maxiter, gain=gain,
                                    fg_deconv_method=fg_deconv_method, alpha=alpha,
                                    add_clean_residual=add_clean_residual, mode=mode,
                                    cache=cache, deconv_dayenu_foregrounds=deconv_dayenu_foregrounds,
                                    fg_restore_size=fg_restore_size,
                                    fg_deconv_fundamental_period=fg_deconv_fundamental_period)


def fringe_filter(data, wgts, max_frate, dt, tol=1e-4, skip_wgt=0.5, maxiter=100, gain=0.1,
                  window='none', edgecut_low=0, edgecut_hi=0, alpha=0.5, add_clean_residual=False,
                  mode='clean', cache = {}, deconv_dayenu_foregrounds=False,
                  fg_deconv_method='clean', fg_restore_size=None,fg_deconv_fundamental_period=None):
    """
    Run a CLEAN deconvolution along the time axis.

    Args:
        data : 1D or 2D data array. If 2D, shape=(Ntimes, Nfreqs)
        wgts : 1D or 2D weight array.
        max_frate : float, maximum fringe-rate (i.e. frequency) to CLEAN, units of 1/[dt]
        dt : float, time-bin width of data
        tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
        window: window function for filtering applied to the filtered axis.
            See gen_window for options.
        skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
            time. Only works properly when all weights are all between 0 and 1.
        maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
        gain: The fraction of a residual used in each iteration. If this is too low, clean takes
            unnecessarily long. If it is too high, clean does a poor job of deconvolving.
        edgecut_low : int, number of bins to consider zero-padded at low-side of the FFT axis,
            such that the windowing function smoothly approaches zero. For 2D cleaning, can
            be fed as a tuple specifying edgecut_low for first and second FFT axis.
        edgecut_hi : int, number of bins to consider zero-padded at high-side of the FFT axis,
            such that the windowing function smoothly approaches zero. For 2D cleaning, can
            be fed as a tuple specifying edgecut_hi for first and second FFT axis.
        alpha : float, if window is tukey this is its alpha parameter.
        add_clean_residual : bool, if True, adds the residual within the CLEAN bounds
            in fourier space to the CLEAN model (and sets residual within CLEAN bounds to zero).
            This is more in-line with a standard filtering operation, rather than a CLEAN operation.
            If False, residual is not added to the CLEAN model.
        mode : string,
             choose from ['clean','dayenu','dft_interp']
             use aipy.deconv.clean if 'clean'
             use 'dayenu' if 'dayenu'
             if 'dft_interp', then interpolates flagged channels with DFT modes.
        cache : dict, optional dictionary for storing pre-computed filtering matrices in linear
            cleaning.
        deconv_dayenu_foregrounds : bool, if True, then apply clean to data - residual where
            res is the data-vector after applying a linear clean filter.
            This allows for in-painting flagged foregrounds without introducing
            clean artifacts into EoR window. If False, mdl will still just be the
            difference between the original data vector and the residuals after
            applying the linear filter.
        cache : dict, optional dictionary for storing pre-computed filtering matrices in linear
            cleaning.
        deconv_dayenu_foregrounds : bool, if True, then apply clean to data - residual where
            res is the data-vector after applying a linear clean filter.
            This allows for in-painting flagged foregrounds without introducing
            clean artifacts into EoR window. If False, mdl will still just be the
            difference between the original data vector and the residuals after
            applying the linear filter.
        fg_deconv_method : string, can be 'leastsq' or 'clean'. If 'leastsq', deconvolve difference between data and linear residual
            by performing linear least squares fitting of data - linear resid to dft modes in filter window.
            If 'clean', obtain deconv fg model using perform a hogboem clean of difference between data and linear residual.
        fg_restore_size: float, optional, allow user to only restore foregrounds subtracted by linear filter
            within a region of this size. If None, set to filter_size.
            This allows us to avoid the problem that if we have RFI flagging and apply a linear filter
            that is larger then the horizon then the foregrounds that we fit might actually include super
            -horizon flagging side-lobes and restoring them will introduce spurious structure.
        fg_deconv_fundamental_period: int, optional
            fundamental period of Fourier modes to fit too.
            if none, default to length of data vector.

    Returns:
        d_mdl: CLEAN model -- best fit low-pass filter components (CLEAN model) in real space
        d_res: CLEAN residual -- difference of data and d_mdl, nulled at flagged channels
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    """
    # print deprecation warning
    warn("Warning: dspec.fringe_filter will soon be deprecated in favor of filtering.fringe_filter",
         DeprecationWarning)

    # run fourier filter
    mdl, res, info = high_pass_fourier_filter(data.T, wgts.T, max_frate, dt, tol=tol, window=window, edgecut_low=edgecut_low, fg_deconv_method=fg_deconv_method,
                                              edgecut_hi=edgecut_hi, skip_wgt=skip_wgt, maxiter=maxiter, gain=gain, deconv_dayenu_foregrounds=deconv_dayenu_foregrounds,
                                              alpha=alpha, add_clean_residual=add_clean_residual, mode=mode, cache=cache,
                                              fg_restore_size=fg_restore_size, fg_deconv_fundamental_period=fg_deconv_fundamental_period)
    return mdl.T, res.T, info


def vis_filter(data, wgts, max_frate=None, dt=None, bl_len=None, sdf=None, standoff=0.0, horizon=1., min_dly=0.,
               tol=1e-4, window='none', maxiter=100, gain=1e-1, skip_wgt=0.5, filt2d_mode='rect',
               edgecut_low=0, edgecut_hi=0, alpha=0.5, add_clean_residual=False, mode='clean', cache={},
               deconv_dayenu_foregrounds=False, fg_deconv_method='clean', fg_restore_size=None,
               fg_deconv_fundamental_period=None):
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
            See gen_window for options.
        skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
            time. Only works properly when all weights are all between 0 and 1.
        maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
        gain: The fraction of a residual used in each iteration. If this is too low, clean takes
            unnecessarily long. If it is too high, clean does a poor job of deconvolving.
        filt2d_mode : str, only applies if clean2d == True. options = ['rect', 'plus']
            If 'rect', a 2D rectangular filter is constructed in fourier space (default).
            If 'plus', the 'rect' filter is first constructed, but only the plus-shaped
            slice along 0 delay and fringe-rate is kept.
        edgecut_low : int, number of bins to consider zero-padded at low-side of the FFT axis,
            such that the windowing function smoothly approaches zero. For 2D cleaning, can
            be fed as a tuple specifying edgecut_low for first and second FFT axis.
        edgecut_hi : int, number of bins to consider zero-padded at high-side of the FFT axis,
            such that the windowing function smoothly approaches zero. For 2D cleaning, can
            be fed as a tuple specifying edgecut_hi for first and second FFT axis.
        alpha : float, if window is tukey, this is its alpha parameter.
        add_clean_residual : bool, if True, adds the residual within the CLEAN bounds
            in fourier space to the CLEAN model (and sets residual within CLEAN bounds to zero).
            This is more in-line with a standard filtering operation, rather than a CLEAN operation.
            If False, residual is not added to the CLEAN model.
        mode : string,
             choose from ['clean','dayenu','dft_interp']
             use aipy.deconv.clean if 'clean'
             use 'dayenu' if 'dayenu'
             if 'dft_interp', then interpolates flagged channels with DFT modes.
        cache : dict, optional dictionary for storing pre-computed filtering matrices in linear
            cleaning.
        deconv_dayenu_foregrounds : bool, if True, then apply clean to data - residual where
            res is the data-vector after applying a linear clean filter.
            This allows for in-painting flagged foregrounds without introducing
            clean artifacts into EoR window. If False, mdl will still just be the
            difference between the original data vector and the residuals after
            applying the linear filter.
        fg_deconv_method : string, can be 'leastsq' or 'clean'. If 'leastsq', deconvolve difference between data and linear residual
            by performing linear least squares fitting of data - linear resid to dft modes in filter window.
            If 'clean', obtain deconv fg model using perform a hogboem clean of difference between data and linear residual.
        fg_restore_size: float, optional, allow user to only restore foregrounds subtracted by linear filter
            within a region of this size. If None, set to filter_size.
            This allows us to avoid the problem that if we have RFI flagging and apply a linear filter
            that is larger then the horizon then the foregrounds that we fit might actually include super
            -horizon flagging side-lobes and restoring them will introduce spurious structure.
        fg_deconv_fundamental_period: int, optional
            fundamental period of Fourier modes to fit too.
            if none, default to length of data vector.
    Returns:
        d_mdl: CLEAN model -- best fit low-pass filter components (CLEAN model) in real space
        d_res: CLEAN residual -- difference of data and d_mdl, nulled at flagged channels
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    """
    # print deprecation warning
    warn("Warning: dspec.vis_filter will soon be deprecated in favor of filtering.vis_filter",
         DeprecationWarning)

    # type checks
    timeclean = False
    if dt is not None or max_frate is not None:
        timeclean = True
        assert max_frate is not None and dt is not None, "Must specify both max_frate and dt for time cleaning"

    freqclean = False
    if sdf is not None or bl_len is not None:
        freqclean = True
        assert sdf is not None and bl_len is not None, "Must specify both bl_len and sdf for frequency cleaning"

    clean2d = timeclean and freqclean

    # 1D clean
    if not clean2d:
        # time clean
        if timeclean:
            mdl, res, info = high_pass_fourier_filter(data.T, wgts.T, max_frate, dt, tol=tol, window=window, edgecut_low=edgecut_low,
                                                      edgecut_hi=edgecut_hi, skip_wgt=skip_wgt, maxiter=maxiter,
                                                      gain=gain, mode=mode, fg_deconv_method=fg_deconv_method,
                                                      alpha=alpha, add_clean_residual=add_clean_residual, cache=cache,
                                                      deconv_dayenu_foregrounds=deconv_dayenu_foregrounds, fg_restore_size=fg_restore_size)
            mdl, res = mdl.T, res.T

        # freq clean
        elif freqclean:
            bl_dly = _get_bl_dly(bl_len, horizon=horizon, standoff=standoff, min_dly=min_dly)
            mdl, res, info = high_pass_fourier_filter(data, wgts, bl_dly, sdf, tol=tol, window=window, edgecut_low=edgecut_low,
                                                      edgecut_hi=edgecut_hi, skip_wgt=skip_wgt, maxiter=maxiter, gain=gain,
                                                      mode=mode, fg_deconv_method=fg_deconv_method,
                                                      alpha=alpha, add_clean_residual=add_clean_residual,
                                                      cache=cache, deconv_dayenu_foregrounds=deconv_dayenu_foregrounds, fg_restore_size=fg_restore_size)

    # 2D clean
    else:
        # get bl delay
        bl_dly = _get_bl_dly(bl_len, horizon=horizon, standoff=standoff, min_dly=min_dly)

        # 2D clean
        mdl, res, info = high_pass_fourier_filter(data, wgts, (max_frate, bl_dly), (dt, sdf), tol=tol, window=window, edgecut_low=edgecut_low,
                                                  mode = mode, edgecut_hi=edgecut_hi, maxiter=maxiter,
                                                  gain=gain, clean2d=True, filt2d_mode=filt2d_mode,
                                                  fg_deconv_method=fg_deconv_method, fg_restore_size=fg_restore_size,
                                                  alpha=alpha, add_clean_residual=add_clean_residual, cache=cache,
                                                  deconv_dayenu_foregrounds=deconv_dayenu_foregrounds)

    return mdl, res, info


def _get_bl_dly(bl_len, horizon=1., standoff=0., min_dly=0.):
    # construct baseline delay
    bl_dly = horizon * bl_len + standoff

    # check minimum delay
    bl_dly = np.max([bl_dly, min_dly])

    return bl_dly


def gen_window(window, N, alpha=0.5, edgecut_low=0, edgecut_hi=0, normalization=None, **kwargs):
    """
    Generate a 1D window function of length N.

    Args:
        window : str, window function
        N : int, number of channels for windowing function.
        edgecut_low : int, number of bins to consider as zero-padded at the low-side
            of the array, such that the window smoothly connects to zero.
        edgecut_hi : int, number of bins to consider as zero-padded at the high-side
            of the array, such that the window smoothly connects to zero.
        alpha : if window is 'tukey', this is its alpha parameter.
        normalization : str, optional
            set to 'rms' to divide by rms and 'mean' to divide by mean.
    """
    if normalization is not None:
        if normalization not in ["mean", "rms"]:
            raise ValueError("normalization must be one of ['rms', 'mean']")
    # parse multiple input window or special windows
    w = np.zeros(N, dtype=np.float)
    Ncut = edgecut_low + edgecut_hi
    if Ncut >= N:
        raise ValueError("Ncut >= N for edgecut_low {} and edgecut_hi {}".format(edgecut_low, edgecut_hi))
    if edgecut_hi > 0:
        edgecut_hi = -edgecut_hi
    else:
        edgecut_hi = None
    if window in ['none', None, 'None', 'boxcar', 'tophat']:
        w[edgecut_low:edgecut_hi] = windows.boxcar(N - Ncut)
    elif window in ['blackmanharris', 'blackman-harris', 'bh', 'bh4']:
        w[edgecut_low:edgecut_hi] =  windows.blackmanharris(N - Ncut)
    elif window in ['hanning', 'hann']:
        w[edgecut_low:edgecut_hi] =  windows.hann(N - Ncut)
    elif window == 'tukey':
        w[edgecut_low:edgecut_hi] =  windows.tukey(N - Ncut, alpha)
    elif window in ['blackmanharris-7term', 'blackman-harris-7term', 'bh7']:
        # https://ieeexplore.ieee.org/document/293419
        a_k = [0.27105140069342, 0.43329793923448, 0.21812299954311, 0.06592544638803, 0.01081174209837,
              0.00077658482522, 0.00001388721735]
        w[edgecut_low:edgecut_hi] = windows.general_cosine(N - Ncut, a_k, True)
    elif window in ['cosinesum-9term', 'cosinesum9term', 'cs9']:
        # https://ieeexplore.ieee.org/document/940309
        a_k = [2.384331152777942e-1, 4.00554534864382e-1, 2.358242530472107e-1, 9.527918858383112e-2,
               2.537395516617152e-2, 4.152432907505835e-3, 3.68560416329818e-4, 1.38435559391703e-5,
               1.161808358932861e-7]
        w[edgecut_low:edgecut_hi] = windows.general_cosine(N - Ncut, a_k, True)
    elif window in ['cosinesum-11term', 'cosinesum11term', 'cs11']:
        # https://ieeexplore.ieee.org/document/940309
        a_k = [2.151527506679809e-1, 3.731348357785249e-1, 2.424243358446660e-1, 1.166907592689211e-1,
               4.077422105878731e-2, 1.000904500852923e-2, 1.639806917362033e-3, 1.651660820997142e-4,
               8.884663168541479e-6, 1.938617116029048e-7, 8.482485599330470e-10]
        w[edgecut_low:edgecut_hi] = windows.general_cosine(N - Ncut, a_k, True)
    else:
        try:
            # return any single-arg window from windows
            w[edgecut_low:edgecut_hi] = getattr(windows, window)(N - Ncut)
        except AttributeError:
            raise ValueError("Didn't recognize window {}".format(window))
    if normalization == 'rms':
        w /= np.sqrt(np.mean(np.abs(w)**2.))
    if normalization == 'mean':
        w /= w.mean()
    return w


def fourier_operator(dsize, nmax, nmin=None, freq_units=False, even_modes=False, L=None):
    """
    Return a complex Fourier analysis operator for a given data dimension and number of Fourier modes.

    Parameters
    ----------
    dsize : int
        Size of data array.

    nmax : int
        Maximum Fourier mode number. Modes will be constructed between
        [nmin, nmax], for a total of (nmax - min) + 1 modes.
    nmin : int, optional, default nmin = nmax
        minimum integer of fourier mode numbers. Modes will be constructed between
        [nmin, nmax] for total of (nmax - nmin) + 1 modes.
    freq_units : bool,
        if False, then fourier modes are given by e^(-m * n * j / N)
        if True, then fourier modes are given by e^(-m * n * j / N * 2 * pi)
        where N is fundamental period.
    even_modes : bool, optional, default = False
        instead of 2n + 1 modes, use 2n modes from -n, n-1 as per usual.
    L : int, optional, default = None
        fundamental period of Fourier modes to fit too.
        if none, default to ndata.
    Returns
    -------
    F : array_like
        Fourier matrix operator, of shape (Nmodes, Ndata)
    """
    nu = np.arange(dsize)
    if L is None:
        if not even_modes:
            L = nu[-1] - nu[0]
        else:
            L = dsize
    if nmin is None:
        nmin = -nmax
    # Construct frequency array (*not* in physical frequency units)

    if freq_units:
        L  = L / (2. * np.pi)
    # Build matrix operator for complex Fourier basis
    if even_modes:
        n = np.arange(nmin, nmax)
    else:
        n = np.arange(nmin, nmax + 1)
    F = np.array([np.exp(-1.j * _n * nu / L) for _n in n])
    return F


def fourier_model(cn, Nfreqs):
    """
    Calculate a 1D (complex) Fourier series model from a set of complex coefficients.

    Parameters
    ----------
    coeffs : array_like
        Array of complex Fourier coefficients, ordered from (-n, n), where n is
        the highest harmonic mode in the model.

    Nfreqs : int
        Number of frequency channels to model. The Fourier modes are integer
        harmonics within this frequency window.

    Returns
    -------
    model : array_like
        Fourier model constructed from the input harmonic coefficients.
        Shape: (Nfreqs,).
    """
    try:
        cn_shape = cn.shape
    except AttributeError:
        raise ValueError("cn must be a 1D array")
    if len(cn.shape) != 1:
        raise ValueError("cn must be a 1D array")
    nmax = (cn.size - 1) // 2  # Max. harmonic

    # Build matrix operator for complex Fourier basis
    F = fourier_operator(dsize=Nfreqs, nmax=nmax)

    # Return model
    return np.dot(cn, F)


def delay_filter_leastsq_1d(data, flags, sigma, nmax, add_noise=False, freq_units = False,
                            cn_guess=None, use_linear=True, operator=None, even_modes=False, fundamental_period=None):
    """
    Fit a smooth model to 1D complex-valued data with flags, using a linear
    least-squares solver. The model is a Fourier series up to a specified
    order. As well as calculating a best-fit model, this will also return a
    copy of the data with flagged regions filled in ('in-painted') with the
    smooth solution.

    Optionally, you can also add an uncorrelated noise realization on top of
    the smooth model in the flagged region.

    Parameters
    ----------
    data : array_like, complex
        Complex visibility array as a function of frequency, with shape
        (Nfreqs,).

    flags : array_like, bool
        Boolean flags with the same shape as data.

    sigma : float or array_like
        Noise standard deviation, in the same units as the data. If float,
        assumed to be homogeneous in frequency. If array_like, must have
        the same shape as the data.

        Note that the choice of sigma will have some bearing on how sensitive
        the fits are to small-scale variations.

    nmax: int or 2-tuple of ints
        Max. order of Fourier modes to fit. A model with complex Fourier modes
        between [-n, n] will be fitted to the data, where the Fourier basis
        functions are ~ exp(-i 2 pi n nu / (Delta nu). If 2-tuple fit [-n0, n1].

    add_noise : bool, optional
        Whether to add an unconstrained noise realization to the in-painted areas.
        This uses sigma to set the noise standard deviation. Default: False.

    cn_guess : array_like, optional
        Initial guess for the series coefficients. If None, zeros will be used.
        A sensible choice of cn_guess can speed up the solver significantly.
        Default: None.

    use_linear : bool, optional
        Whether to use a fast linear least-squares solver to fit the Fourier
        coefficients, or a slower generalized least-squares solver.
        Default: True.

    operator : array_like, optional
        Fourier basis operator matrix. This is used to pass in a pre-computed
        matrix operator when calling from other functions, e.g. from
        delay_filter_leastsq. Operator must have shape (Nmodes, Nfreq), where
        Nmodes = 2*nmax + 1. A complex Fourier basis will be automatically
        calculated if no operator is specified.
    freq_units : bool, optional, default = False
        if False, then fourier modes are given by e^(-m * n * j / N)
        if True, then fourier modes are given by e^(-m * n * j / N * 2 * pi)
    even_modes : bool, optional, default = False
        instead of 2n + 1 modes, use 2n modes from -n, n-1 as per usual.

    fundamental_period : int, optional, default = None
        fundamental period of Fourier modes to fit too.
        if none, default to ndata.

    Returns
    -------
    model : array_like
        Best-fit model, composed of a sum of Fourier modes.

    model_coeffs : array_like
        Coefficients of Fourier modes, ordered from modes [-nmax, +nmax].

    data_out : array_like
        In-painted data.
    """
    # Construct Fourier basis operator if not specified
    if isinstance(nmax, tuple) or isinstance(nmax, list):
        nmin = nmax[0]
        nmax = nmax[1]
        assert isinstance(nmin, int) and isinstance(nmax, int), "Provide integers for nmax and nmin"
    elif isinstance(nmax, int):
        nmin = -nmax
    if operator is None:
        F = fourier_operator(dsize=data.size, nmin = nmin, nmax=nmax, freq_units=freq_units, even_modes=even_modes, L=fundamental_period)
    else:
        F = operator
        if even_modes:
            cshape = nmax - nmin
        else:
            cshape = nmax - nmin + 1
        if F.shape[0] != cshape:
            raise ValueError("Fourier basis operator has the wrong shape. "
                             "Must have shape (Nmodes, Nfreq).")
    # Turn flags into a mask
    w = np.logical_not(flags)

    # Define model and likelihood function
    def model(cn, F):
        return np.dot(cn, F)
    if even_modes:
        nmodes = nmax - nmin
    else:
        nmodes = nmax - nmin + 1

    # Initial guess for Fourier coefficients (real + imaginary blocks)
    cn_in = np.zeros(2 * nmodes)
    if cn_guess is not None:
        if cn_in.size != 2 * cn_guess.size:
            raise ValueError("cn_guess must be of size %s" % (cn_in.size / 2))
        cn_in[:cn_guess.shape[0]] = cn_guess.real
        cn_in[cn_guess.shape[0]:] = cn_guess.imag

    # Make sure sigma is the right size for matrix broadcasting
    if isinstance(sigma, np.ndarray):
        mat_sigma = np.tile(sigma, (nmodes, 1)).T
    else:
        mat_sigma = sigma

    # Run least-squares fit
    if use_linear:
        # Solve as linear system
        A = np.atleast_2d(w).T * F.T
        res = lsq_linear(A / mat_sigma ** 2., w * data / sigma ** 2.)
        cn_out = res.x
    else:
        # Use full non-linear leastsq fit
        def loglike(cn):
            """
            Simple log-likelihood, assuming Gaussian data. Calculates:
                logL = -0.5 [w*(data - model)]^2 / sigma^2.
            """
            # Need to do real and imaginary parts separately, otherwise
            # leastsq() fails
            _delta = w * (data - model(cn[:nmodes] + 1.j * cn[nmodes:], F))
            delta = np.concatenate((_delta.real / sigma, _delta.imag / sigma))
            return -0.5 * delta**2.

        # Do non-linear least-squares calculation
        cn, stat = leastsq(loglike, cn_in)
        cn_out = cn[:nmodes] + 1.j * cn[nmodes:]

    # Inject smooth best-fit model into masked areas
    bf_model = model(cn_out, F)
    data_out = data.copy()
    data_out[flags] = bf_model[flags]

    # Add noise to in-painted regions if requested
    if add_noise:
        noise = np.random.randn(np.sum(flags)) \
            + 1.j * np.random.randn(np.sum(flags))
        if isinstance(sigma, np.ndarray):
            data_out[flags] += sigma[flags] * noise
        else:
            data_out[flags] += sigma * noise

    # Return coefficients and best-fit model
    return bf_model, cn_out, data_out


def delay_filter_leastsq(data, flags, sigma, nmax, add_noise=False, freq_units = False,
                         cn_guess=None, use_linear=True, operator=None, even_modes=False, fundamental_period=None):
    """
    Fit a smooth model to each 1D slice of 2D complex-valued data with flags,
    using a linear least-squares solver. The model is a Fourier series up to a
    specified order. As well as calculating a best-fit model, this will also
    return a copy of the data with flagged regions filled in ('in-painted')
    with the smooth solution.

    Optionally, you can also add an uncorrelated noise realization on top of
    the smooth model in the flagged region.

    N.B. This is just a wrapper around delay_filter_leastsq_1d() but with some
    time-saving precomputations. It fits to each 1D slice of the data
    individually, and does not perform a global fit to the 2D data.

    Parameters
    ----------
    data : array_like, complex
        Complex visibility array as a function of frequency, with shape
        (Ntimes, Nfreqs).

    flags : array_like, bool
        Boolean flags with the same shape as data.

    sigma : float or array_like
        Noise standard deviation, in the same units as the data. If float,
        assumed to be homogeneous in frequency. If array_like, must have
        the same shape as the data.

        Note that the choice of sigma will have some bearing on how sensitive
        the fits are to small-scale variations.

    nmax: int
        Max. order of Fourier modes to fit. A model with complex Fourier modes
        between [-n, n] will be fitted to the data, where the Fourier basis
        functions are ~ exp(-i 2 pi n nu / (Delta nu).

    add_noise : bool, optional
        Whether to add an unconstrained noise realization to the in-painted areas.
        This uses sigma to set the noise standard deviation. Default: False.

    cn_guess : array_like, optional
        Initial guess for the series coefficients of the first row of the
        input data. If None, zeros will be used. Default: None.

    use_linear : bool, optional
        Whether to use a fast linear least-squares solver to fit the Fourier
        coefficients, or a slower generalized least-squares solver.
        Default: True.

    operator : array_like, optional
        Fourier basis operator matrix. Must have shape (Nmodes, Nfreq), where
        Nmodes = 2*nmax + 1. A complex Fourier basis will be used by default.

    freq_units : bool, optional, default = False
        if False, then fourier modes are given by e^(-m * n * j / N)
        if True, then fourier modes are given by e^(-m * n * j / N * 2 * pi)
    even_modes : bool, optional, default False
        instead of 2n + 1 modes, use 2n modes from -n, n-1 as per usual.
    fundamental_period : int, optional, default = None
        fundamental period of Fourier modes to fit too.
        if none, default to ndata.

    Returns
    -------
    model : array_like
        Best-fit model, composed of a sum of Fourier modes. Same shape as the
        data.

    model_coeffs : array_like
        Coefficients of Fourier modes, ordered from modes [-n, +n].

    data_out : array_like
        In-painted data.
    """
    if isinstance(nmax, tuple) or isinstance(nmax, list):
        nmin = nmax[0]
        nmax = nmax[1]
        assert isinstance(nmin, int) and isinstance(nmax, int), "Provide integers for nmax and nmin"
    elif isinstance(nmax, int):
        nmin = -nmax
    # Construct and cache Fourier basis operator (for speed)
    if operator is None:
        F = fourier_operator(dsize=data.shape[1], nmax=nmax, nmin=nmin, freq_units=freq_units, even_modes=even_modes, L=fundamental_period)
    else:
        # delay_filter_leastsq_1d will check for correct dimensions
        F = operator
    if even_modes:
        nmodes = nmax - nmin
    else:
        nmodes = nmax - nmin + 1
    # Array to store in-painted data
    inp_data = np.zeros(data.shape, dtype=np.complex)
    cn_array = np.zeros((data.shape[0], nmodes), dtype=np.complex)
    mdl_array = np.zeros(data.shape, dtype=np.complex)

    # Loop over array
    cn_out = None
    for i in range(data.shape[0]):
        bf_model, cn_out, data_out = delay_filter_leastsq_1d(
            data[i], flags[i], sigma=sigma, nmax=(nmin, nmax), add_noise=add_noise, even_modes=even_modes,
            use_linear=use_linear, cn_guess=cn_out, operator=F, freq_units=freq_units, fundamental_period=fundamental_period)
        inp_data[i, :] = data_out
        cn_array[i, :] = cn_out
        mdl_array[i, :] = bf_model

    return mdl_array, cn_array, inp_data


def fourier_interpolation_operator(x, filter_centers, filter_half_widths,
                                 filter_factors, cache={}, fundamental_period=None, xc=None):
    """
    Calculates Fourier operator with multiple flexible delay windows to fit data, potentially with arbitrary
    user provided frequencies.

    A_{nu tau} = e^{- 2 * pi * i * nu * tau / B}

    Parameters
    ----------
    x:
        x values to evaluate operator at
    filter_centers: float or list
        float or list of floats of centers of delay filter windows in nanosec
    filter_half_widths: float or list
        float or list of floats of half-widths of delay filter windows in nanosec
    filter_factors: float or list
        float or list of floats of filtering factors.
    cache: dictionary, optional dictionary storing filter matrices with keys
    (nchan, df, ) + (filter_centers) + (filter_half_widths) + \
    (filter_factors)
    B: fundamental period of fourier modes to use for fitting. units of 1/x. For standard DFT, this is bandwidth.
    """
    #if no fundamental fourier period is provided, set fundamental period equal to measurement
    #bandwidth.
    if fundamental_period is None:
        fundamental_period = np.median(np.diff(x)) * len(x)
    if xc is None:
        xc = x[int(np.round(len(x)/2))]
    filter_centers, filter_half_widths, _ = parse_check_fourier_operator_inputs(filter_centers,
                                                                                filter_half_widths,
                                                                                user_frequencies)
    #each column is a fixed delay
    opkey = ('fourier_interpolation_operator',) + tuple(x) + tuple(filter_centers) + tuple(filter_half_widths)
    if not opkey in cache:
        amat = []
        for fc, fw in zip(filter_centers,filter_half_widths):
            bs = np.ceil(fw * fundamental_period)
            dlys = fc + np.arange(-bs, bs) / fundamental_period
            xg, dg = np.meshgrid(x-xc, dlys, indexing='ij')
            fblock = np.exp(2j * np.pi * dg * xg)
            amat.append(fblock)
        cache[opkey] = np.hstack(amat)
    return cache[opkey]



def delay_interpolation_matrix(nchan, ndelay, wgts, fundamental_period=None, cache={}, taper='none', return_diagnostics=False):
    """
    Compute a foreground interpolation matrix.

    Computes a foreground interpolation matrix that, when applied to data,
    interpolates over flagged channels with delays between
    -ndelay / fundamental_period, ndelay / fundamental_period

    The computed Matrix is equal to F = A @ [ A^T @ W @ A]^{-1} @ A^T W
    where A is an nchan \times 2ndelay  design matrix
    y = A \tilde{y}
    y is the frequency representation of data and \tilde{y} is
    a 2xndelay vector holding the data's fourier coefficients. W is a diagonal
    matrix of frequency-data weights. The net effect of F, when applied to flagged
    data, is to solve for the fourier coefficients fitting unflagged channels
    ([ A^T @ W @ A]^{-1} @ A^T W solves the linear least squares problem) and then return
    the unflagged Fourier transform by apply A @ to the fitted coefficients, resulting
    in data that is linearly interpolated.

    !!! THIS FUNCTION WILL BE DEPRECATED BY INTERPOLATION_MATRIX !!!

    Parameters
    ----------
    nchan: int
        Number of frequency channels to interpolate over.
    ndelay: int
        number of delays to use in interpolation.
    wgts: float array
        wgts to be applied to each frequency channel.
        must have length equal to nchan.
        in addition, wgts should have more nonezero values then there are
        degrees of freedom (delay modes) to solve for.
    fundamental_period: float, optional
        the fundamental period of reconstructed delays. Default: nchan,
        tends to give well conditioned matrices.
        I find that 2 x nchan gives the best results (AEW).
    cache: dict, optional
        optional cache holding pre-computed matrices
    taper: string, optional
        use a taper to fit.
    Returns
    ----------
    (nchan, nchan) numpy array
        that can be used to interpolate over channel gaps.
    """
    if not len(wgts) == nchan:
        raise ValueError("nchan must equal length of wgts")
    if fundamental_period is None: #recommend 2 x nchan or nchan.
        fundamental_period = 2*nchan #this tends to give well conditioned matrices.
    if not np.sum((np.abs(wgts) > 0.).astype(float)) >= 2*ndelay:
        raise ValueError("number of unflagged channels must be greater then or equal to number of delays")
    matkey = (nchan, ndelay, fundamental_period) + tuple(wgts)
    if matkey not in cache or return_diagnostics:
        frequencies, delays = np.meshgrid(np.arange(nchan)-nchan/2, np.arange(-ndelay,ndelay), indexing='ij')
        delays = delays / fundamental_period
        a_mat = np.exp(2j * delays * frequencies * np.pi) / fundamental_period
        wmat = np.diag(wgts * gen_window(taper, nchan)).astype(complex)
        cmat = np.dot(a_mat.T, (a_mat.T * wgts).T)
        if np.linalg.cond(cmat)>=1e9:
            warn('Warning!!!!: Poorly conditioned matrix! Your linear inpainting IS WRONG!'
                  'Fix this by adjusting fundamental tones!')
        cmati = np.linalg.inv(cmat)
        tmat = np.dot(cmati,(a_mat.T * wgts))
        a_mat = np.dot(a_mat, tmat)
        cache[matkey] = a_mat
    a_mat = cache[matkey]
    if not return_diagnostics:
        return a_mat
    else:
        return a_mat, cmat, cmati, tmat

def parse_check_fourier_operator_inputs(filter_centers, filter_half_widths,user_frequencies):
    """
    Parse and check floats or lists of filter window parameters.

    Parameters
    ----------------------------------------------------
    filter_centers: float or list
        float or list of floats of centers of delay filter windows in nanosec
    filter_half_widths: float or list
        float or list of floats of half-widths of delay filter windows in nanosec
    filter_factors: float or list
        float or list of floats of filtering factors.
    Returns
    ----------------------------------------------------
    Filter parameters with types checked and converted to lists.
    """
    if isinstance(filter_centers, float) or isinstance(filter_factors, int):
        filter_centers = [filter_centers]
    if isinstance(filter_half_widths, float) or isinstance(filter_factors, int):
        filter_half_widths = [filter_half_widths]
    assert user_frequencies is None or isinstance(user_frequencies,(list, np.ndarray)),"user provided frequencies must be ndarray or list"
    return filter_centers, filter_half_widths

def dayenu_mat_inv(x, filter_centers, filter_half_widths,
                            filter_factors, cache={}, wrap=False, wrap_interval=1,
                            nwraps=1000, no_regularization=False):
    """
    Computes the inverse of sinc weights for a baseline.
    This form of weighting is diagonal in delay-space and down-weights tophat regions.

    Parameters
    ----------
    x: array like
        array-like list of arbitrary frequencies. If this is supplied, evaluate sinc_downweight_mat at these frequencies
        instead of linear array of nchan.
    filter_centers: float or list
        float or list of floats of centers of delay filter windows in nanosec
    filter_half_widths: float or list
        float or list of floats of half-widths of delay filter windows in nanosec
    filter_factors: float or list
        float or list of floats of filtering factors.
    cache: dictionary, optional dictionary storing filter matrices with keys
    tuple(x) + (filter_centers) + (filter_half_widths) + \
    (filter_factors)


    !!!-------------
    WARNING: The following parameters are intended for theoretical
    studies of how inverse sinc-weighting functions
    but should not be changed from defaults in practical data analysis!
    !!!------------
        wrap: bool, If true, add a wrap around, equivalent to situation
              where we want sinc weights to be the IDFT of a diagonal matrix
        wrap_interval: integer, interval of wrap around in units of nf * df (bandwidth)
        nwraps: number of wraps to include.
        no_regularization: bool,  if True, do not include diagonal regularization.

    Returns
    ----------
     (nchan, nchan) complex inverse of the tophat filtering matrix assuming that the delay-space covariance is diagonal and zero outside
         of the horizon
    """
    if isinstance(filter_factors,float) or isinstance(filter_factors, int):
        filter_factors = [filter_factors]
    filter_centers, filter_half_widths = parse_check_fourier_operator_inputs(filter_centers,
                                                                            filter_half_widths, x)
    nchan = len(user_frequencies)
    filter_key = tuple(x) + tuple(filter_centers) + \
    tuple(filter_half_widths) + tuple(filter_factors) + (wrap, wrap_interval, nwraps, no_regularization)

    if not filter_key in cache:
        fx, fy = np.meshgrid(x,x)
        sdwi_mat = np.identity(fx.shape[0]).astype(np.complex128)
        if no_regularization:
            sdwi_mat *= 0.
        for fc, fw, ff in zip(filter_centers, filter_half_widths, filter_factors):
            if not ff == 0:
                if not wrap:
                    sdwi_mat = sdwi_mat + np.sinc( 2. * (fx-fy) * fw ).astype(np.complex128)\
                            * np.exp(-2j * np.pi * (fx-fy) * df * fc) / ff
                else:
                    for wnum in np.arange(-nwraps//2, nwraps//2):
                        offset = nchan * wnum * wrap_interval
                        sdwi_mat = sdwi_mat + \
                        np.sinc( 2. *  (fx-fy - offset) * fw  ).astype(np.complex128)\
                        * np.exp(-2j * np.pi * (fx-fy - offset) * df * fc) / ff
    else:
        sdwi_mat = cache[filter_key]
    return sdwi_mat
