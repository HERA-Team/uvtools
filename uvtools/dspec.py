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

#DEFAULT PARAMETERS FOR CLEANs
CLEAN_DEFAULTS={'tol':1e-9, 'window':{False:'none',True:['none', 'none']},
 'alpha':.5, 'maxiter':100, 'gain':0.1,
 'edgecut_low':{True:[0, 0],False:0}, 'edgecut_hi':{True:[0, 0],False:0},
 'add_clean_residual':False, 'filt2d_mode':'rect'}
 #In the above dictionary, some fields are different depending on whether
 #filter2d is True or False (whether we do 2d filtering or not). These parameters,
 #are listed below in DEFAULT_FILT2D. In CLEAN_DEFAULTS they are set to a dictionary
 #where True references default param values for 2d filtering and False references
 #default param values for 1d filtering.
DEFAULT_FILT2D = ['edgecut_hi', 'edgecut_low', 'window']
CLEAN_KEYS = list(CLEAN_DEFAULTS.keys())
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

def _fourier_filter_hash(filter_centers, filter_half_widths,
                         filter_factors, x, w=None, hash_decimal=10, **kwargs):
    '''
    Generate a hash key for a fourier filter

    Parameters
    ----------
        filter_centers: list,
                        list of floats for filter centers
        filter_half_widths: list
                        list of float filter half widths (in fourier space)

        filter_factors: list
                        list of float filter factors
        x: the x-axis of the data to be subjected to the hashed filter.
        w: optional vector of float weights to hash to. default, none
        hash_decimal: number of decimals to use for floats in key.
        kwargs: additional hashable elements the user would like to
                include in their filter key.
    '''
    filter_key = tuple(np.round(x,hash_decimal))\
    + tuple(np.round(np.asarray(filter_centers) * np.mean(np.diff(x)) * len(x), hash_decimal))\
    + tuple(np.round(np.asarray(filter_half_widths) * np.mean(np.diff(x)) * len(x), hash_decimal))\
    + tuple(np.round(np.asarray(filter_factors) * 1e9, hash_decimal))
    if w is not None:
        filter_key = filter_key + tuple(np.round(w.tolist(), hash_decimal))
    filter_key = filter_key + tuple([kwargs[k] for k in kwargs])
    return filter_key





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

def fourier_filter(x, data, wgts, filter_centers, filter_half_widths, suppression_factors,
                   mode, filter2d, fitting_options, cache=None, filter_dim=1, skip_wgt=0.1,
                   max_contiguous_edge_flags=10):
                   '''
                   A filtering function that tries to wrap up all functionality of high_pass_fourier_filter
                   and add support for additional linear fitting options.
                   It can filter 1d or 2d data with x-axis(es) x and wgts in fourier domain
                   rectangular windows centered at filter_centers or filter_half_widths
                   perform filtering along any of 2 dimensions in 2d or 1d!
                   the 'dft' and 'dayenu' modes support irregularly sampled data.
                   Parameters
                   -----------
                   x: array-like
                      Array of floats giving x-values of data. Depending on the chosen method, this data may need to be equally spaced.
                      If performing a 2d clean, a 2-list or 2-tuple of np.ndarrays with x-values should be provided.
                    data: array-like
                        1d or 2d numpy.ndarray of complex data to filter.
                    wgts: array-like
                        1d or 2d numpy.ndarray of real weights. Must be the same shape as data.
                    filter_centers: array-like
                        if not 2dfilter: 1d np.ndarray or list or tuple of floats
                        specifying centers of rectangular fourier regions to filter.
                        If 2dfilter: should be a 2-list or 2-tuple. Each element
                        should be a list or tuple or np.ndarray of floats that include
                        centers of rectangular regions to filter.
                    filter_half_widths: array-like
                        if not 2dfilter: 1d np.ndarray or list of tuples of floats
                        specifying the half-widths of rectangular fourier regions to filter.
                        if 2dfilter: should be a 2-list or 2-tuple. Each element should
                        be a list or tuple or np.ndarray of floats that include centers
                        of rectangular bins.
                    suppression_factors: array-like
                        if not 2dfilter: 1d np.ndarray or list of tuples of floats
                        specifying the fractional residuals of model to leave in the data.
                        For example, 1e-6 means that the filter will leave in 1e-6 of data fitted
                        by the model.
                        if 2dfilter: should be a 2-list or 2-tuple. Each element should
                        be a list or tuple or np.ndarray of floats that include centers
                        of rectangular bins.
                    mode: string
                        specify filtering mode. Currently supported are
                        'clean', iterative clean
                        'dpss_lsq', dpss fitting using scipy.optimize.lsq_linear
                        'dft_lsq', dft fitting using scipy.optimize.lsq_linear
                        'dpss_matrix', dpss fitting using direct lin-lsq matrix
                                       computation. Slower then lsq but provides linear
                                       operator that can be used to propagate
                                       statistics and the matrix is cached so
                                       on average, can be faster for data with
                                       many similar flagging patterns.
                        'dft_matrix', dft fitting using direct lin-lsq matrix
                                      computation. Slower then lsq but provides
                                      linear operator that can be used to propagate
                                      statistics and the matrix is cached so
                                      on average, can be faster for data with
                                      many similar flagging patterns.
                                      !!!WARNING: In my experience,
                                      'dft_matrix' option is numerical unstable.!!!
                                      'dpss_matrix' works much better.
                        'dayenu', apply dayenu filter to data. Does not
                                 deconvolve subtracted foregrounds.
                        'dayenu_dft_leastsq', apply dayenu filter to data
                                 and deconvolve subtracted foregrounds using
                                'dft_leastsq' method (see above).
                        'dayenu_dpss_leastsq', apply dayenu filter to data
                                 and deconvolve subtracted foregrounds using
                                 'dpss_leastsq' method (see above)
                        'dayenu_dft_matrix', apply dayenu filter to data
                                 and deconvolve subtracted foregrounds using
                                'dft_matrix' mode (see above).
                                !!!WARNING: dft_matrix mode is often numerically
                                unstable. I don't recommend it!
                        'dayenu_dpss_matrix', apply dayenu filter to data
                                 and deconvolve subtracted foregrounds using
                                 'dpss_matrix' method (see above)
                        'dayenu_clean', apply dayenu filter to data. Deconvolve
                                 subtracted foregrounds with 'clean'.
                    filter2d: bool
                        specify whether filtering will be performed in 2d or 1d.
                        If filter is 1d, it will be applied across the -1 axis.
                    fitting_options: dict
                        dictionary with options for fitting techniques.
                        if filter2d is true, this should be a 2-tuple or 2-list
                        of dictionaries. The dictionary for each dimension must
                        specify the following for each fitting method.
                        * 'dft':
                            'fundamental_period': float or 2-tuple
                                The fundamental_period of dft modes to fit. This is the
                                Fourier resolution of fitted fourier modes equal to
                                1/FP where FP is the fundamental period. For a standard
                                delay DFT FP = B where B is the visibility bandwidth
                                FP also sets the number of
                                modes fit within each window in 'filter_half_widths' will
                                equal fw / fundamental_period where fw is the filter width.
                                if filter2d, must provide a 2-tuple with fundamental_period
                                of each dimension.
                        * 'dayenu':
                            No parameters necessary if you are only doing 'dayenu'.
                            For 'dayenu_dpss', 'dayenu_dft', 'dayenu_clean' see below
                            and use the appropriate fitting options for each method.
                        * 'dpss':
                            'eigenval_cutoff': array-like
                                list of sinc_matrix eigenvalue cutoffs to use for included dpss modes.
                            'nterms': array-like
                                list of integers specifying the order of the dpss sequence to use in each
                                filter window.
                            'edge_supression': array-like
                                specifies the degree of supression that must occur to tones at the filter edges
                                to calculate the number of DPSS terms to fit in each sub-window.
                            'avg_suppression': list of floats, optional
                                specifies the average degree of suppression of tones inside of the filter edges
                                to calculate the number of DPSS terms. Similar to edge_supression but instead checks
                                the suppression of a since vector with equal contributions from all tones inside of the
                                filter width instead of a single tone.
                        *'clean':
                             defaults can be accessed in dspec.CLEAN_DEFAULTS
                             'tol': float,
                                clean tolerance. 1e-9 is standard.
                             'maxiter' : int
                                maximum number of clean iterations. 100 is standard.
                             'filt2d_mode' : string
                                if 'rect', clean withing a rectangular region of Fourier space given
                                by the intersection of each set of windows.
                                if 'plus' only clean the plus-shaped shape along
                                zero-delay and fringe rate.
                            'edgecut_low' : int, number of bins to consider zero-padded at low-side of the FFT axis,
                                such that the windowing function smoothly approaches zero. For 2D cleaning, can
                                be fed as a tuple specifying edgecut_low for first and second FFT axis.
                            'edgecut_hi' : int, number of bins to consider zero-padded at high-side of the FFT axis,
                                such that the windowing function smoothly approaches zero. For 2D cleaning, can
                                be fed as a tuple specifying edgecut_hi for first and second FFT axis.
                            'add_clean_residual' : bool, if True, adds the CLEAN residual within the CLEAN bounds
                                in fourier space to the CLEAN model. Note that the residual actually returned is
                                not the CLEAN residual, but the residual in input data space.
                            'window' : window function for filtering applied to the filtered axis.
                                See dspec.gen_window for options. If clean2D, can be fed as a list
                                specifying the window for each axis in data.
                            'gain': The fraction of a residual used in each iteration. If this is too low, clean takes
                                unnecessarily long. If it is too high, clean does a poor job of deconvolving.
                            'alpha': float, if window is 'tukey', this is its alpha parameter.

                    cache: dict, optional
                        dictionary for caching fitting matrices.

                    filter_dim, int optional
                        specify dimension to filter. default 1,
                        and if 2d filter, will use both dimensions.

                    max_contiguous_edge_flags : int, optional
                        if the number of contiguous samples at the edge is greater then this
                        at either side, skip.
                    skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
                        Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
                        time. Only works properly when all weights are all between 0 and 1.
                    Returns
                    ---------
                        d_mdl: array-like
                            model -- best fit real-space model of data.
                        d_res: array-like
                            residual -- difference of data and model, nulled at flagged channels
                        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
                              if the mode is 'dayenu', 'dpss', or 'dft', then info has the following sub-dicts.
                              clean uses a different info dict structure because of downstream code assumptions that are not
                              sufficiently general to describe the other methods. We should eventually migrate clean assumptions
                              to this format.
                              * 'status': dict holding two sub-dicts status of filtering on each time/frequency step.
                                        - 'axis_0'/'axis_1': dict holding the status of time filtering for each time/freq step. Keys are integer index
                                                    of each step and values are a string that is either 'success' or 'skipped'.
                              * 'filter_params': dict holding the filtering parameters for each axis with the following sub-dicts.
                                        - 'axis_0'/'axis_1': dict holding filtering parameters for filtering over each respective axis.
                                                    - 'mode': the filtering mode used to filter the time axis ('dayenu', 'dpss_leastsq' 'dpss_method')
                                                    - 'basis': (if using dpss/dft) gives the filtering basis.
                                                    - 'filter_centers': centers of filtering windows.
                                                    - 'filter_half_widths': half-widths of filtering regions for each axis.
                                                    - 'suppression_factors': amount of suppression for each filtering region.
                                                    - 'basis_options': the basis options used for dpss/dft mode. See dft_operator and dpss_operator for
                                                                       more details.
                                                    - 'x': vector of x-values used to generate the filter.
                   '''
                   if cache is None:
                       cache = {}
                   supported_modes=['clean', 'dft_leastsq', 'dpss_leastsq', 'dft_matrix', 'dpss_matrix', 'dayenu',
                                    'dayenu_dft_leastsq', 'dayenu_dpss_leastsq', 'dayenu_dpss_matrix',
                                    'dayenu_dft_matrix', 'dayenu_clean']
                   if not mode in supported_modes:
                       raise ValueError("Need to supply a mode in supported modes:%s"%(str(supported_modes)))
                   mode = mode.split('_')
                   ndim_data = len(data.shape)
                   ndim_wgts = len(wgts.shape)
                   if not ndim_wgts == ndim_data:
                       raise ValueError("Number of dimensions in weights, %d does not equal number of dimensions in data, %d!"%(ndim_wgts, ndim_data))
                   #The core code of this method will always assume 2d data
                   if ndim_data == 1:
                       data = np.asarray([data])
                       wgts = np.asarray([wgts])
                   if not filter2d and filter_dim == 0:
                       data = data.T
                       wgts = wgts.T
                   if mode[0] == 'dayenu':
                       if filter2d:
                           filter_dim_d = [0, 1]
                       else:
                           filter_dim_d = [1]
                       residual, info = dayenu_filter(x=x, data=data, wgts=wgts, filter_dimensions=filter_dim_d,
                                                     filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                                                     filter_factors=suppression_factors, cache=cache, skip_wgt=skip_wgt,
                                                     max_contiguous_edge_flags=max_contiguous_edge_flags)
                       model = data - residual
                       if len(mode) > 1:
                           model, _, info_deconv = fit_basis_2d(x=x, data=model, filter_centers=filter_centers, filter_dims=filter_dim_d,
                                                                 skip_wgt=skip_wgt, basis=mode[1], method=mode[2], wgts=wgts, basis_options=fitting_options,
                                                                 filter_half_widths=filter_half_widths, suppression_factors=suppression_factors,
                                                                 cache=cache, max_contiguous_edge_flags=max_contiguous_edge_flags)
                           info['info_deconv']=info_deconv

                   elif mode[0] == 'dft' or mode[0] == 'dpss':
                        if filter2d:
                            filter_dim_d = [0, 1]
                        else:
                            filter_dim_d = [1]
                        model, residual, info = fit_basis_2d(x=x, data=data, filter_centers=filter_centers, filter_dims=filter_dim_d,
                                                            skip_wgt=skip_wgt, basis=mode[0], method=mode[1], wgts=wgts, basis_options=fitting_options,
                                                            filter_half_widths=filter_half_widths, suppression_factors=suppression_factors,
                                                            cache=cache, max_contiguous_edge_flags=max_contiguous_edge_flags)
                   elif mode[0] == 'clean':
                        #Unpack all of the clean parameters from
                        #fitting_options. This is to preserve default behavior
                        #in high_pass_fourier_filter
                        #check that fittig options are in allowed keys
                        for param in fitting_options:
                            if not param in CLEAN_KEYS:
                                raise ValueError("Invalid CLEAN param provided: %s\n"
                                                 "Valid params are %s"%(param, str(CLEAN_KEYS)))
                        for param in CLEAN_DEFAULTS:
                            if not param in fitting_options:
                                if not param in DEFAULT_FILT2D:
                                    fitting_options[param] = CLEAN_DEFAULTS[param]
                                else:
                                    fitting_options[param] = CLEAN_DEFAULTS[param][filter2d]

                        #arguments for 2d clean should be supplied as
                        #2-list. For code economy, we expand 1d arguments to 2d
                        #including the data and weights to 1 x N arrays.

                        if not filter2d:
                            #pad = [0, pad]
                            _x = [np.zeros(data.shape[0]), np.fft.fftfreq(len(x), x[1]-x[0])]
                            x = [np.zeros(data.shape[0]), x]
                            edgecut_hi = [ 0, fitting_options['edgecut_hi'] ]
                            edgecut_low = [ 0, fitting_options['edgecut_low']]
                            filter_centers = [[0.], copy.deepcopy(filter_centers)]
                            filter_half_widths = [[9e99], copy.deepcopy(filter_half_widths)]
                            window_opt = ['none', fitting_options['window']]
                        else:
                            if not np.all(np.isclose(np.diff(x[1]), np.mean(np.diff(x[1])))):
                                raise ValueError("Data must be equally spaced for CLEAN mode!")
                            _x = [np.fft.fftfreq(len(x[m]), x[m][1]-x[m][0]) for m in range(2)]
                            edgecut_hi = fitting_options['edgecut_hi']
                            edgecut_low = fitting_options['edgecut_low']
                            window_opt = fitting_options['window']
                        for m in range(2):
                            if not np.all(np.isclose(np.diff(x[m]), np.mean(np.diff(x[m])))):
                                raise ValueError("Data must be equally spaced for CLEAN mode!")
                        window = [gen_window(window_opt[m], data.shape[m], alpha=fitting_options['alpha'], normalization='mean',
                                           edgecut_low=edgecut_low[m], edgecut_hi=edgecut_hi[m]) for m in range(2)]
                        window[0] = np.atleast_2d(window[0]).T
                        area_vecs = [ np.zeros(len(_x[m])) for m in range(2) ]
                        #set area equal to one inside of filtering regions
                        tol = fitting_options['tol']
                        maxiter = fitting_options['maxiter']
                        gain = fitting_options['gain']
                        add_clean_residual = fitting_options['add_clean_residual']
                        filt2d_mode = fitting_options['filt2d_mode']
                        if filt2d_mode == 'rect' or not filter2d:
                            for m in range(2):
                                for fc, fw in zip(filter_centers[m], filter_half_widths[m]):
                                    area_vecs[m][np.abs(_x[m] - fc)<=fw] = 1.
                            #if filtering windows are rectangular,
                            #we can just take outer products
                            area = np.outer(area_vecs[0], area_vecs[1])
                        elif filt2d_mode == 'plus' and filter2d:
                            area = np.zeros(data.shape)
                            #construct and add a 'plus' for each filtering window pair in each dimension.
                            for fc0, fw0 in zip(filter_centers[0], filter_half_widths[0]):
                                for fc1, fw1 in zip(filter_centers[1], filter_half_widths[1]):
                                    area_temp = np.zeros(area.shape)
                                    if fc0 >= _x[0].min() and fc0 <= _x[0].max():
                                        area_temp[np.argmin(np.abs(_x[0]-fc0)),(_x[1] - fc1) <= fw1]=1.
                                    if fc1 >= _x[1].min() and fc1 <= _x[1].max():
                                        area_temp[(_x[0] - fc0) <= fw0, np.argmin(np.abs(_x[1]-fc1))]=1.
                                    area += area_temp
                            area = (area>0.).astype(int)
                        else:
                            raise ValueError("%s is not a valid filt2d_mode! choose from ['rect', 'plus']"%(filt2d_mode))
                        if filter2d:
                            _wgts = np.fft.ifft2(window[0] * wgts * window[1])
                            _data = np.fft.ifft2(window[0] * data * wgts * window[1])
                        else:
                            _wgts = np.fft.ifft(window[0] * wgts * window[1], axis=1)
                            _data = np.fft.ifft(window[0] * wgts * data * window[1], axis=1)
                        _d_cl = np.zeros_like(_data)
                        _d_res = np.zeros_like(_data)
                        if not filter2d:
                            info = []
                            for i, _d, _w, _a in zip(np.arange(_data.shape[0]).astype(int), _data, _wgts, area):
                                # we skip steps that might trigger infinite CLEAN loops or divergent behavior.
                                # if the weights sum up to a value close to zero (most of the data is flagged)
                                # or if the data itself is close to zero.
                                if _w[0] < skip_wgt or np.all(np.isclose(_d, 0.)):
                                    _d_cl[i] = 0.
                                    _d_res[i] = _d
                                    info.append({'skipped':True})
                                else:
                                    _d_cl[i], _info = aipy.deconv.clean(_d, _w, area=_a, tol=tol, stop_if_div=False,
                                                                    maxiter=maxiter, gain=gain)
                                    _d_res[i] = _info['res']
                                    _info['skipped'] = False
                                    del(_info['res'])
                                    info.append(_info)
                        elif filter2d:
                                # we skip 2d cleans if all the data is close to zero (which can cause an infinite clean loop)
                                # or the weights are all equal to zero which can also lead to a clean loop.
                                # the maximum of _wgts should be the average value of all cells in 2d wgts.
                                # since it is the 2d fft of wgts.
                                if not np.all(np.isclose(_data, 0.)) and np.abs(_wgts).max() > skip_wgt:
                                    _d_cl, info = aipy.deconv.clean(_data, _wgts, area=area, tol=tol, stop_if_div=False,
                                                                    maxiter=maxiter, gain=gain)
                                    _d_res = info['res']
                                    del(info['res'])
                                else:
                                    info = {'skipped':True}
                                    _d_cl = np.zeros_like(_data)
                                    _d_res = np.zeros_like(_d_cl)
                        if add_clean_residual:
                            _d_cl = _d_cl + _d_res * area
                        if filter2d:
                            model = np.fft.fft2(_d_cl)
                            residual = np.fft.fft2(_d_res)
                        else:
                            model = np.fft.fft(_d_cl, axis=1)
                            residual = np.fft.fft(_d_res, axis=1)
                        #transpose back if filtering the 0th dimension.
                   if not filter2d and filter_dim == 0:
                        model = model.T
                        residual = residual.T
                        data = data.T
                        wgts = wgts.T
                        if not mode[0] == 'clean':
                            # downstream code assumes a certain format for clean info dictionaries
                            # so right now, I only perform this switching for non clean mode.
                            # eventually, it would be nice to standardize clean too.
                            # but that needs to happen after we verify that this does not
                            # break said downstream code.
                            for k in info:
                                if not k == 'info_deconv':
                                    info[k]['axis_0'] = copy.deepcopy(info[k]['axis_1'])
                                    info[k]['axis_1'] = {}
                        # if we deconvolve the subtracted foregrounds in dayenu
                        # then provide fitting options for the deconvolution.
                        if 'info_deconv' in info:
                            for k in info['info_deconv']:
                                info['info_deconv'][k]['axis_0'] = copy.deepcopy(info['info_deconv'][k]['axis_1'])
                                info['info_deconv'][k]['axis_1'] = {}
                   return model, residual, info


#TODO: Add DPSS interpolation function to this.
def high_pass_fourier_filter(data, wgts, filter_size, real_delta, clean2d=False, tol=1e-9, window='none',
                             skip_wgt=0.1, maxiter=100, gain=0.1, filt2d_mode='rect', alpha=0.5,
                             edgecut_low=0, edgecut_hi=0, add_clean_residual=False, mode='clean', cache=None,
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
            this sets the resolution in Fourier space. A standard DFT has a resolution
            of 1/N_{FP} = 1/N between fourier modes so that the DFT operator is
            D_{mn} = e^{-2 \pi i m n / N_{FP}}. fg_deconv_fundamental_period
            is N_{FP}.

    Returns:
        d_mdl: CLEAN model -- best fit low-pass filter components (CLEAN model) in real space
        d_res: CLEAN residual -- difference of data and d_mdl, nulled at flagged channels
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    '''
    if cache is None:
        cache = {}
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
            if mode == 'clean':
                _d_cl, info = aipy.deconv.clean(_d, _w, area=area, tol=tol, stop_if_div=False, maxiter=maxiter, gain=gain)
                _d_res = info['res']
                del info['res']
            elif mode == 'dayenu':
                d_r, info = dayenu_filter(np.arange(len(data))-len(data)/2*real_delta,
                                         data * wgts * win, wgts * win, max_contiguous_edge_flags=len(data)-1,
                                         filter_dimensions = [1], filter_centers=fc, filter_half_widths=fw, filter_factors=ff, cache=cache, skip_wgt=skip_wgt)
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
                info ={'fg_deconv': {'method':'dft_interp','nmin':nmin, 'nmax':nmax}}
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
                    if mode == 'clean':
                        _cl, info_here = aipy.deconv.clean(_d[i], _w[i], area=area, tol=tol, stop_if_div=False, maxiter=maxiter, gain=gain)
                        _d_cl[i] = _cl
                        _d_res[i] = info_here['res']
                        del info_here['res']
                        info.append(info_here)
                    elif mode == 'dayenu':
                        d_r, info_here = dayenu_filter(np.arange(len(data[i]))*real_delta,
                                                       data[i] * wgts[i] * win, wgts[i] * win, skip_wgt=skip_wgt,
                                                       filter_dimensions=[1], filter_centers=fc, max_contiguous_edge_flags=len(data[i])-1,
                                                       filter_half_widths=fw, filter_factors=ff, cache=cache)
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
                                d_cl, _, _ = delay_filter_leastsq_1d( (data[i] * wgts[i] * win - d_r).squeeze(), flags=np.isclose(wgts[i], 0).squeeze(), sigma=1.,
                                                                    nmax=(nmin, nmax), freq_units=True, even_modes=True, fundamental_period=fg_deconv_fundamental_period[-1])
                                _d_cl[i] = np.fft.ifft(d_cl)
                        else:
                            _d_cl[i] = _d[i] - _d_res[i]
                        info.append(info_here)

                    elif mode == 'dft_interp':
                        info_here = {}
                        nmin = int((fcfg[0] - fwfg[0]) * real_delta * fg_deconv_fundamental_period[-1])
                        nmax = int((fcfg[0] + fwfg[0]) * real_delta * fg_deconv_fundamental_period[-1])
                        info_here['fg_deconv'] = {'method':'dft_interp','nmin':nmin, 'nmax':nmax}
                        d_cl, _, _ = delay_filter_leastsq_1d( (data[i] * wgts[i] * win ).squeeze(), flags=np.isclose(wgts[i], 0).squeeze(), sigma=1.,
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
            if not mode == 'dayenu':
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

            d_r, info = dayenu_filter([(np.arange(data.shape[0])-data.shape[0]/2)*real_delta[0],
                                        (np.arange(data.shape[1])-data.shape[1]/2)*real_delta[1]],
                                        data * wgts * win, wgts * win, filter_centers=fc, filter_half_widths=fw,
                                        filter_factors=ff, cache=cache, filter_dimensions=[0, 1], skip_wgt=skip_wgt,
                                        max_contiguous_edge_flags=np.min([data.shape[0], data.shape[1]])-1)
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
    if mode == 'clean' or mode == 'dft_interp':
        d_res = (data - d_mdl) * ~np.isclose(wgts * win, 0.0)

    return d_mdl, d_res, info

def dayenu_filter(x, data, wgts, filter_dimensions, filter_centers, filter_half_widths, filter_factors,
                  cache = {}, return_matrices=True, hash_decimal=10, skip_wgt=0.1, max_contiguous_edge_flags=10):
    '''
    Apply a linear delay filter to waterfall data.
    Due to performance reasons, linear filtering only supports separable delay/fringe-rate filters.

    Arguments
    ---------
    x: array-like or length-2 list/tuples that are array-like
        x-values for each data point in dimension to be filtered.
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
    return_matrices: bool,
        if True, return a dict referencing every every filtering matrix used.
    hash_decimal: number of decimals to hash x to
    skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
        Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
        time. Only works properly when all weights are all between 0 and 1.
    Returns
    -------
    data: array, 2d clean residual with data filtered along the frequency direction.
    info: dictionary with filtering parameters and a list of skipped_times and skipped_channels
          has the following fields
         * 'status': dict holding two sub-dicts status of filtering on each time/frequency step.
                   - 'axis_0'/'axis_1': dict holding the status of time filtering for each time/freq step. Keys are integer index
                               of each step and values are a string that is either 'success' or 'skipped'.
         * 'filter_params': dict holding the filtering parameters for each axis with the following sub-dicts.
                   - 'axis_0'/'axis_1': dict holding filtering parameters for filtering over each respective axis.
                               - 'filter_centers': centers of filtering windows.
                               - 'filter_half_widths': half-widths of filtering regions for each axis.
                               - 'suppression_factors': amount of suppression for each filtering region.
                               - 'x': vector of x-values used to generate the filter.
    '''
    # check that data and weight shapes are consistent.
    d_shape = data.shape
    w_shape = wgts.shape
    d_dim = data.ndim
    w_dim = wgts.ndim
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
    if not isinstance(x, (np.ndarray,list, tuple)):
        raise ValueError("x must be a numpy array, list, or tuple")
    # Check that inputs are tiples or lists
    if not isinstance(filter_dimensions, (list,tuple,int)):
        raise ValueError("filter_dimensions must be a list or tuple")
    # if filter_dimensions are supplied as a single integer, convert to list (core code assumes lists).
    if isinstance(filter_dimensions, int):
        filter_dimensions = [filter_dimensions]
    # check that filter_dimensions is no longer then 2 elements
    if not len(filter_dimensions) in [1, 2]:
        raise ValueError("length of filter_dimensions cannot exceed 2")
    # make sure filter_dimensions are 0 or 1.
    for dim in filter_dimensions:
        if not isinstance(dim,int):
            raise ValueError("only integers are valid filter dimensions")
    # make sure that all filter dimensions are valid for the supplied data.
    if not np.all(np.abs(np.asarray(filter_dimensions)) < data.ndim):
        raise ValueError("invalid filter dimensions provided, must be 0 or 1/-1")
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
            if not len(x) == 2:
                raise ValueError("For 2d filtering, x must be 2d long list or tuple or ndarray")
            for j in range(2):
                if not isinstance(x[j], (tuple, list, np.ndarray)):
                    raise ValueError("x[%d] must be a tuple, list or numpy array."%(j))
                x[j]=np.asarray(x[j])
        for ff_num,ff_list in zip([0,1],filter_factors):
            # we allow the user to provide a single filter factor for multiple
            # filtering windows on a single dimension. This code
            # iterates through each dimension and if a single filter_factor is provided
            # it converts the filter_factor list to a list of filter_factors with the same
            # length as filter_centers.
            if len(ff_list) == 1:
                ff_list = [ff_list[0] for m in range(len(filter_centers[ff_num]))]


    else:
        if len(filter_factors) == 1:
            # extend filter factor list of user supplied a float or len-1 list.
            filter_factors = [filter_factors[0] for m in range(len(filter_centers))]
        if 0 in filter_dimensions:
            # convert 1d input-lists to
            # a list of lists for core-code to operate on.
            filter_factors = [filter_factors,[]]
            filter_centers = [filter_centers,[]]
            filter_half_widths = [filter_half_widths,[]]
            x = [x,None]
        elif 1 in filter_dimensions:
            # convert 1d input-lists to
            # a list of lists for core-code to operate on.
            filter_factors = [[],filter_factors]
            filter_centers = [[],filter_centers]
            filter_half_widths = [[],filter_half_widths]
            x = [None, x]
    check_vars = [filter_centers, filter_half_widths, filter_factors]
    # Now check that the number of filter factors = number of filter widths
    # = number of filter centers for each dimension.
    for fs in filter_dimensions:
        for aname1,avar1 in zip(check_names,check_vars):
            for aname2,avar2 in zip(check_names,check_vars):
                if not len(avar1[fs]) == len(avar2[fs]):
                    raise ValueError("Number of elements in %s-%d must equal the"
                                     " number of elements %s-%d!"%(aname1, fs, aname2, fs))

    info = {'status':{'axis_0':{}, 'axis_1':{}}, 'filter_params':{'axis_0':{}, 'axis_1':{}}}
    for fs in range(2):
        info['filter_params']['axis_%d'%fs]['filter_centers'] = filter_centers[fs]
        info['filter_params']['axis_%d'%fs]['filter_half_widths'] = filter_half_widths[fs]
        info['filter_params']['axis_%d'%fs]['x'] = x[fs]
        info['filter_params']['axis_%d'%fs]['mode'] = 'dayenu'
    skipped = [[],[]]
    # in the lines below, we iterate over the time dimension. For each time, we
    # compute a lazy covariance matrix (filter_mat) from the weights (wght) and
    # a sinc downweight matrix. (dayenu_mat_inv). We then attempt to
    # take the psuedo inverse to get a filtering matrix that removes foregrounds.
    # we do this for the zeroth and first filter dimension.
    output = copy.deepcopy(data)
    #this loop iterates through dimensions to iterate over (fs is the non-filter
    #axis).
    filter_matrices=[{},{}]
    #check filter factors for zeros and negative numbers
    for ff in filter_factors:
        for fv in ff:
            if fv <= 0.:
                raise ValueError("All filter factors must be greater than zero! You provided %.2e :(!"%(fv))
    for fs in filter_dimensions:
        if fs == 0:
            _d, _w = output.T, wgts.T
        else:
            _d, _w = output, wgts
        #if the axis orthogonal to the iteration axis is to be filtered, then
        #filter it!.
        for sample_num, sample, wght in zip(range(data.shape[fs-1]), _d, _w):
            filter_key = _fourier_filter_hash(filter_centers=filter_centers[fs], filter_half_widths=filter_half_widths[fs],
                                              filter_factors=filter_factors[fs], x=x[fs], w=wght,
                                              label='dayenu_filter_matrix')
            if not filter_key in cache:
                #only calculate filter matrix and psuedo-inverse explicitly if they are not already cached
                #(saves calculation time).
                if np.count_nonzero(wght) / len(wght) >= skip_wgt and np.count_nonzero(wght[:max_contiguous_edge_flags]) > 0 \
                   and np.count_nonzero(wght[-max_contiguous_edge_flags:]) >0:
                    wght_mat = np.outer(wght.T, wght)
                    filter_mat = dayenu_mat_inv(x=x[fs], filter_centers=filter_centers[fs],
                                                         filter_half_widths=filter_half_widths[fs],
                                                         filter_factors=filter_factors[fs], cache=cache) * wght_mat
                    try:
                        #Try taking psuedo-inverse. Occasionally I've encountered SVD errors
                        #when a lot of channels are flagged. Interestingly enough, I haven't
                        #I'm not sure what the precise conditions for the error are but
                        #I'm catching it here.
                        cache[filter_key] = np.linalg.pinv(filter_mat)
                    except np.linalg.LinAlgError:
                        # skip if we can't invert or psuedo-invert the matrix.
                        cache[filter_key] = None
                else:
                    # skip if we don't meet skip_wegith criterion or continuous edge flags
                    # are to many. This last item isn't really a problem for dayenu
                    # but it's here for consistancy.
                    cache[filter_key] = None

            filter_mat = cache[filter_key]
            if filter_mat is not None:
                if fs == 0:
                    output[:, sample_num] = np.dot(filter_mat, sample)
                elif fs == 1:
                    output[sample_num] = np.dot(filter_mat, sample)
                info['status']['axis_%d'%fs][sample_num] = 'success'
            else:
                skipped[fs-1].append(sample_num)
                info['status']['axis_%d'%fs][sample_num] = 'skipped'
            if return_matrices:
                filter_matrices[fs][sample_num]=filter_mat

    #1d data will only be filtered across "channels".
    if data_1d and ntimes == 1:
        output = output[0]
    return output, info


def delay_filter(data, wgts, bl_len, sdf, standoff=0., horizon=1., min_dly=0.0, tol=1e-4,
                 window='none', skip_wgt=0.5, maxiter=100, gain=0.1, edgecut_low=0, edgecut_hi=0,
                 alpha=0.5, add_clean_residual=False, mode='clean', cache=None,
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
            this sets the resolution in Fourier space. A standard DFT has a resolution
            of 1/N_{FP} = 1/N between fourier modes so that the DFT operator is
            D_{mn} = e^{-2 \pi i m n / N_{FP}}. fg_deconv_fundamental_period
            is N_{FP}.

    Returns:
        d_mdl: CLEAN model -- best fit low-pass filter components (CLEAN model) in real space
        d_res: CLEAN residual -- difference of data and d_mdl, nulled at flagged channels
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    '''
    # print deprecation warning
    warn("Warning: dspec.delay_filter will soon be deprecated in favor of filtering.vis_filter",
         DeprecationWarning)
    if cache is None:
        cache = {}
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
            this sets the resolution in Fourier space. A standard DFT has a resolution
            of 1/N_{FP} = 1/N between fourier modes so that the DFT operator is
            D_{mn} = e^{-2 \pi i m n / N_{FP}}. fg_deconv_fundamental_period
            is N_{FP}.

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
               edgecut_low=0, edgecut_hi=0, alpha=0.5, add_clean_residual=False, mode='clean', cache=None,
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
            this sets the resolution in Fourier space. A standard DFT has a resolution
            of 1/N_{FP} = 1/N between fourier modes so that the DFT operator is
            D_{mn} = e^{-2 \pi i m n / N_{FP}}. fg_deconv_fundamental_period
            is N_{FP}.
    Returns:
        d_mdl: CLEAN model -- best fit low-pass filter components (CLEAN model) in real space
        d_res: CLEAN residual -- difference of data and d_mdl, nulled at flagged channels
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    """
    # print deprecation warning
    warn("Warning: dspec.vis_filter will soon be deprecated in favor of filtering.vis_filter",
         DeprecationWarning)

    # type checks
    if cache is None:
        cache = {}
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


def fit_basis_1d(x, y, w, filter_centers, filter_half_widths,
                basis_options, suppression_factors=None, hash_decimal=10,
                method='leastsq', basis='dft', cache=None):
    """
    A 1d linear-least-squares fitting function for computing models and residuals for fitting of the form
    y_model = A @ c
    where A is a design matrix encoding our choice for a basis functions
    and y_model is a fitted version of the data and c is a set of fitting coefficients determined by
    c = [A^T w A]^{-1} A^T w y
    where y is the original data and w is a diagonal matrix of weights for each channel in y.
    Currently supports fitting of dpss and dft modes.
    Parameters
    ----------
    x: array-like
        x-axis of data to fit.
    y: array-like
        y-axis of data to fit.
    w: array-like
        data weights.
    filter_centers': array-like
        list of floats specifying the centers of fourier windows with which to fit signals
    filter_half_widths': array-like
        list of floats specifying the half-widths of fourier windows to model.
    suprression_factors: array-like, optional
        list of floats for each basis function denoting the fraction of
        of each basis element that should be present in the fitted model
        If none provided, model will include 100% of each mode.
        It is sometimes useful, for renormalization reversability
        to only include 1-\epsilon where \epsilon is a small number of
        each mode in the model.
    hash_decimal: number of decimals to round to for floating point keys.
    basis_options: dictionary
        basis specific options for fitting. The two bases currently supported are dft and dpss whose options
        are as follows:
            * 'dft':
               *'fundamental_period': float or 2-tuple
                The fundamental_period of dft modes to fit. This is the
                Fourier resoltion of fitted fourier modes equal to
                1/FP where FP is the fundamental period. For a standard
                delay DFT FP = B where B is the visibility bandwidth
                FP also sets the number of
                modes fit within each window in 'filter_half_widths' will
                equal fw / fundamental_period where fw is the filter width.
                if filter2d, must provide a 2-tuple with fundamental_period
                of each dimension.
            * 'dpss':
                The basis_options must include one and only one of the four options
                for specifying how to terminate the dpss series in each filter window.
                *'eigenval_cutoff': array-like
                    list of sinc_matrix eigenvalue cutoffs to use for included dpss modes.
                *'nterms': array-like
                    list of integers specifying the order of the dpss sequence to use in each
                    filter window.
                *'edge_supression': array-like
                    specifies the degree of supression that must occur to tones at the filter edges
                    to calculate the number of DPSS terms to fit in each sub-window.
                *'avg_suppression': list of floats, optional
                    specifies the average degree of suppression of tones inside of the filter edges
                    to calculate the number of DPSS terms. Similar to edge_supression but instead checks
                    the suppression of a since vector with equal contributions from all tones inside of the
                    filter width instead of a single tone.
    method: string
        specifies the fitting method to use. We currently support.
            *'leastsq' to perform iterative leastsquares fit to derive model.
                using scipy.optimize.leastsq
            *'matrix' derive model by directly calculate the fitting matrix
                [A^T W A]^{-1} A^T W and applying it to the y vector.


    Returns:
        model: array-like
            Ndata array of complex floats equal to interpolated model
        resid: array-like
            Ndata array of complex floats equal to y - model
        info:
            dictionary containing fitting arguments for reference.
            if 'matrix' method is used. Fields are
                * 'method' : method used to derive fits.
                * 'basis' : basis that the fits are in
                * 'filter_centers' : filtering centers argument
                * 'filter_half_widths' : filter_half_widths argument
                * 'suppression_factors' : suppression_factors argument
                * 'basis_options' : basis specific options dictionary
                                    see dpss_operator and dft_operator.
                * 'amat' : A matrix used for fitting.
                * 'fitting_matrix' : matrix used for fitting (A [ATA]^-1 AT)
                  if the method == 'matrix'.

    """
    if cache is None:
        cache = {}
    info = copy.deepcopy(basis_options)
    if basis.lower() == 'dft':
        amat = dft_operator(x, filter_centers=filter_centers,
                            filter_half_widths=filter_half_widths,
                            cache=cache, **basis_options)
    elif basis.lower() == 'dpss':
        amat, nterms = dpss_operator(x, filter_centers=filter_centers,
                                     filter_half_widths=filter_half_widths,
                                     cache=cache, **basis_options)
        info['nterms'] = nterms
    else:
        raise ValueError("Specify a fitting basis in supported bases: ['dft', 'dpss']")
    if suppression_factors is None:
        suppression_vector = np.ones(amat.shape[1])
    else:
        if basis.lower() == 'dft':
            suppression_vector =  np.hstack([1-sf * np.ones(2*int(np.ceil(fw * basis_options['fundamental_period'])))\
                                             for sf,fw in zip(suppression_factors, filter_half_widths)])
        elif basis.lower() == 'dpss':
            suppression_vector = np.hstack([1-sf * np.ones(nterm) for sf, nterm in zip(suppression_factors, nterms)])
    info['method'] = method
    info['basis'] = basis
    info['filter_centers'] = filter_centers
    info['filter_half_widths'] = filter_half_widths
    info['suppression_factors'] = suppression_factors
    info['basis_options'] = basis_options
    info['amat'] = amat
    wmat = np.diag(w)
    if method == 'leastsq':
        a = np.atleast_2d(w).T * amat
        res = lsq_linear(a, w * y)
        cn_out = res.x
    elif method == 'matrix':
        fm_key = _fourier_filter_hash(filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                                      filter_factors=suppression_vector, x=x, w=w, hash_decimal=hash_decimal,
                                      label='fitting matrix', basis=basis)
        if basis.lower() == 'dft':
            fm_key = fm_key + (basis_options['fundamental_period'], )
        elif basis.lower() == 'dpss':
            fm_key = fm_key + tuple(nterms)
        fmat = fit_solution_matrix(wmat, amat, cache=cache, fit_mat_key=fm_key)
        info['fitting_matrix'] = fmat
        cn_out = fmat @ y
    else:
        raise ValueError("Provided 'method', '%s', is not in ['leastsq', 'matrix']."%(method))
    model = amat @ (suppression_vector * cn_out)
    resid = (y - model) * (np.abs(w) > 0.) #suppress flagged residuals (such as RFI)
    return model, resid, info

def fit_basis_2d(x, data, wgts, filter_centers, filter_half_widths,
                basis_options, suppression_factors=None,
                method='leastsq', basis='dft', cache=None,
                filter_dims = [1], skip_wgt=0.1, max_contiguous_edge_flags=5):
    """
    A 1d linear-least-squares fitting function for computing models and residuals for fitting of the form
    y_model = A @ c
    where A is a design matrix encoding our choice for a basis functions
    and y_model

    Parameters
    ----------
    x: array-like or 2-tuple/2-list
        x-axis of data to fit.
        if more then one filter_dim, must provide 2-tuple or 2-list with x
    data: array-like
        data to fit, should be an Ntimes x Nfreqs array.
    wgts: array-like
        data weights.
    filter_centers': array-like
        list of floats specifying the centers of fourier windows with which to fit signals
    filter_half_widths': array-like
        list of floats specifying the half-widths of fourier windows to model.
    suppression_factors: array-like, optional
        list of floats for each basis function denoting the fraction of
        of each basis element that should be present in the fitted model
        If none provided, model will include 100% of each mode.
        It is sometimes useful, for renormalization reversability
        to only include 1-\epsilon where \epsilon is a small number of
        each mode in the model.
    basis_options: dictionary or 2-tuple
        basis specific options for fitting. The two bases currently supported are dft and dpss whose options
        are as follows:
            * 'dft':
              *'fundamental_period': float or 2-tuple
                The fundamental_period of dft modes to fit. This is the
                Fourier resoltion of fitted fourier modes equal to
                1/FP where FP is the fundamental period. For a standard
                delay DFT FP = B where B is the visibility bandwidth
                FP also sets the number of
                modes fit within each window in 'filter_half_widths' will
                equal fw / fundamental_period where fw is the filter width.
                if filter2d, must provide a 2-tuple with fundamental_period
                of each dimension.
            * 'dpss':
                The basis_options must include one and only one of the four options
                for specifying how to terminate the dpss series in each filter window.
                *'eigenval_cutoff': array-like
                    list of sinc_matrix eigenvalue cutoffs to use for included dpss modes.
                *'nterms': array-like
                    list of integers specifying the order of the dpss sequence to use in each
                    filter window.
                *'edge_supression': array-like
                    specifies the degree of supression that must occur to tones at the filter edges
                    to calculate the number of DPSS terms to fit in each sub-window.
                *'avg_suppression': list of floats, optional
                    specifies the average degree of suppression of tones inside of the filter edges
                    to calculate the number of DPSS terms. Similar to edge_supression but instead checks
                    the suppression of a since vector with equal contributions from all tones inside of the
                    filter width instead of a single tone.
    method: string
        specifies the fitting method to use. We currently support.
            *'leastsq' to perform iterative leastsquares fit to derive model.
                using scipy.optimize.leastsq
            *'matrix' derive model by directly calculate the fitting matrix
                [A^T W A]^{-1} A^T W and applying it to the y vector.

    filter_dim, int optional
        specify dimension to filter. default 1,
        and if 2d filter, will use both dimensions.

    skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
        Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
        time. Only works properly when all weights are all between 0 and 1.

    max_contiguous_edge_flags : int, optional
        if the number of contiguous samples at the edge is greater then this
        at either side, skip .

    Returns
    -------
        model: array-like
            Ndata array of complex floats equal to interpolated model
        resid: array-like
            Ndata array of complex floats equal to y - model
    info: dictionary with filtering parameters and a list of skipped_times and skipped_channels
          has the following fields
         * 'status': dict holding two sub-dicts status of filtering on each time/frequency step.
                   - 'axis_0'/'axis_1': dict holding the status of time filtering for each time/freq step. Keys are integer index
                               of each step and values are a string that is either 'success' or 'skipped'.
         * 'filter_params': dict holding the filtering parameters for each axis with the following sub-dicts.
                   - 'axis_0'/'axis_1': dict holding filtering parameters for filtering over each respective axis.
                               - 'filter_centers': centers of filtering windows.
                               - 'filter_half_widths': half-widths of filtering regions for each axis.
                               - 'suppression_factors': amount of suppression for each filtering region.
                               - 'x': vector of x-values used to generate the filter.
                               - 'basis': (if using dpss/dft) gives the filtering basis.
                               - 'basis_options': the basis options used for dpss/dft mode. See dft_operator and dpss_operator for
                                                  more details.
    """
    if cache is None:
        cache={}
    info = {'status':{'axis_0':{}, 'axis_1':{}}}
    residual = np.zeros_like(data)
    filter2d = (0 in filter_dims and 1 in filter_dims)
    filter_dims = sorted(filter_dims)[::-1]
    #this will only happen if filter_dims is only zero!
    if filter_dims[0] == 0:
        data = data.T
        wgts = wgts.T
    if not filter2d:
        x = [np.zeros_like(x), x]
        filter_centers = [[], copy.deepcopy(filter_centers)]
        filter_half_widths = [[], copy.deepcopy(filter_half_widths)]
        suppression_factors = [[], copy.deepcopy(suppression_factors)]
        basis_options=[{}, basis_options]
    else:
        if not isinstance(basis_options, (tuple,list)) or not len(basis_options) == 2:
            raise ValueError("basis_options must be 2-tuple or 2-list for 2d filtering.")
    #filter -1 dimension
    model = np.zeros_like(data)
    for i, _y, _w, in zip(range(data.shape[0]), data, wgts):
        if np.count_nonzero(_w)/len(_w) >= skip_wgt and np.count_nonzero(_w[:max_contiguous_edge_flags]) > 0 \
                                                        and np.count_nonzero(_w[-max_contiguous_edge_flags:]) >0:
            model[i], _, info_t = fit_basis_1d(x=x[1], y=_y, w=_w, filter_centers=filter_centers[1],
                                            filter_half_widths=filter_half_widths[1],
                                            suppression_factors=suppression_factors[1],
                                            basis_options=basis_options[1], method=method,
                                            basis=basis, cache=cache)
            info['status']['axis_1'][i] = 'success'
        else:
            info['status']['axis_1'][i] = 'skipped'
    #and if filter2d, filter the 0 dimension. Note that we feed in the 'model'
    #set wgts for time filtering to happen on skipped rows
    info['filter_params'] = {'axis_0':{}, 'axis_1':{}}
    if np.all([info['status']['axis_1'][i] == 'success' for i in info['status']['axis_1']]):
        info['filter_params']['axis_1']['method'] = info_t['method']
        info['filter_params']['axis_1']['basis'] = info_t['basis']
        info['filter_params']['axis_1']['filter_centers'] = info_t['filter_centers']
        info['filter_params']['axis_1']['filter_half_widths'] = info_t['filter_half_widths']
        info['filter_params']['axis_1']['suppression_factors'] = info_t['suppression_factors']
        info['filter_params']['axis_1']['basis_options'] = info_t['basis_options']
        info['filter_params']['axis_1']['mode'] = info_t['basis'] + '_' + method
    if filter2d:
        wgts_time = np.ones_like(wgts)
        for i in range(data.shape[0]):
            if info['status']['axis_1'][i] == 'skipped':
                wgts_time[i] = 0.
        for i, _y, _w, in zip(range(model.shape[1]), model.T, wgts_time.T):
            if np.count_nonzero(_w)/len(_w) >= skip_wgt and np.count_nonzero(_w[:max_contiguous_edge_flags]) > 0 \
               and np.count_nonzero(_w[-max_contiguous_edge_flags:]) >0:
                model.T[i], _, info_t = fit_basis_1d(x=x[0], y=_y, w=_w, filter_centers=filter_centers[0],
                                                                 filter_half_widths=filter_half_widths[0],
                                                                 suppression_factors=suppression_factors[0],
                                                                 basis_options=basis_options[0], method=method,
                                                                 basis=basis, cache=cache)
                info['status']['axis_0'][i] = 'success'
            else:
                info['status']['axis_0'][i] = 'skipped'
        if np.all([info['status']['axis_0'][i] == 'success' for i in info['status']['axis_0']]):
            info['filter_params']['axis_0']['method'] = info_t['method']
            info['filter_params']['axis_0']['basis'] = info_t['basis']
            info['filter_params']['axis_0']['filter_centers'] = info_t['filter_centers']
            info['filter_params']['axis_0']['filter_half_widths'] = info_t['filter_half_widths']
            info['filter_params']['axis_0']['suppression_factors'] = info_t['suppression_factors']
            info['filter_params']['axis_0']['basis_options'] = info_t['basis_options']

    residual = (data - model) * (np.abs(wgts) > 0).astype(float)
    #this will only happen if filter_dims is only zero!
    if filter_dims[0] == 0:
        data = data.T
        wgts = wgts.T
        model = model.T
        residual = residual.T
        for k in info:
            info[k]['axis_0'] = copy.deepcopy(info[k]['axis_1'])
            info[k]['axis_1'] = {}
    return model, residual, info


def fit_solution_matrix(weights, design_matrix, cache=None, hash_decimal=10, fit_mat_key=None):
    """
    Calculate the linear least squares solution matrix
    from a design matrix, A and a weights matrix W
    S = [A^T W A]^{-1} A^T W

    Parameters
    ----------
    weights: array-like
        ndata x ndata matrix of data weights
    design_matrx: array-like
        ndata x n_fit_params matrix transforming fit_parameters to data
    cache: optional dictionary
        optional dictionary storing pre-computed fitting matrix.
    hash_decimal: int optional
        the number of decimals to use in hash for caching. default is 10
    fit_mat_key: optional hashable variable
        optional key. If none is used, hash fit matrix against design and
        weighting matrix.

    Returns
    -----------
        array-like
        n_fit_params x n_fit_params matrix
        S = [A^T W A]^{-1} A ^T W
    """
    if cache is None:
        cache = {}
    ndata = weights.shape[0]
    if not weights.shape[0] == weights.shape[1]:
        raise ValueError("weights must be a square matrix")
    if not design_matrix.shape[0] == ndata:
        raise ValueError("weights matrix incompatible with design_matrix!")
    if fit_mat_key is None:
            opkey = ('fitting_matrix',) + tuple(np.round(weights.flatten(), hash_decimal))\
                    +tuple(np.round(design_matrix.flatten(), hash_decimal))
    else:
        opkey = fit_mat_key

    if not opkey in cache:
        #check condition number
        cmat = np.conj(design_matrix.T) @ weights @ design_matrix
        #should there be a conjugation!?!
        if np.linalg.cond(cmat)>=1e9:
            warn('Warning!!!!: Poorly conditioned matrix! Your linear inpainting IS WRONG!')
            cache[opkey] = np.linalg.pinv(cmat) @ np.conj(design_matrix.T) @ weights
        else:
            try:
                cache[opkey] = np.linalg.inv(cmat) @ np.conj(design_matrix.T) @ weights
            except np.linalg.LinAlgError as error:
                print(error)
                cache[opkey] = None
    return cache[opkey]


def dpss_operator(x, filter_centers, filter_half_widths, cache=None, eigenval_cutoff=None,
        edge_suppression=None, nterms=None, avg_suppression=None, xc=None, hash_decimal=10):
    """
    Calculates DPSS operator with multiple delay windows to fit data. Frequencies
    must be equally spaced (unlike Fourier operator). Users can specify how the
    DPSS series fits are cutoff in each delay-filtering window with one (and only one)
    of three conditions: eigenvalues in sinc matrix fall below a thresshold (eigenval_cutoff),
    user specified number of DPSS terms (nterms), xor the suppression of fourier
    tones at the filter edge by a user specified amount (edge_supression).

    Parameters
    ----------
    x: array-like
        x values to evaluate operator at
    filter_centers: array-like
        list of floats of centers of delay filter windows in nanosec
    filter_half_widths: array-like
        list of floats of half-widths of delay filter windows in nanosec
    cache: dictionary, optional
        dictionary for storing operator matrices with keys
        tuple(x) + tuple(filter_centers) + tuple(filter_half_widths)\
         + (series_cutoff_name,) = tuple(series_cutoff_values)
    eigenval_cutoff: list of floats, optional
        list of sinc matrix eigenvalue cutoffs to use for included dpss modes.
    nterms: list of integers, optional
        integer specifying number of dpss terms to include in each delay fitting block.
    edge_suppression: list of floats, optional
        specifies the degree of supression that must occur to tones at the filter edges to
        calculate the number of DPSS terms to fit in each sub-window.
    avg_suppression: list of floats, optional
        specifies the average degree of suppression of tones inside of the filter edges
        to calculate the number of DPSS terms. Similar to edge_suppression but instead
        checks the suppression of a sinc vector with equal contributions from
        all tones inside of the filter width instead of a single tone.
    xc: float optional
    hash_decimal: number of decimals to round for floating point dict keys.

    Returns
    ----------
    2-tuple
    First element:
        Design matrix for DPSS fitting.   Ndata x (Nfilter_window * nterm)
        transforming from DPSS modes to data.
    Second element:
        list of integers with number of terms for each fourier window specified by filter_centers
        and filter_half_widths
    """
    if cache is None:
        cache = {}
    #conditions for halting.
    crit_labels = ['eigenval_cutoff', 'nterms', 'edge_suppression', 'avg_suppression']
    crit_list = [eigenval_cutoff, nterms, edge_suppression, avg_suppression]
    crit_provided = np.asarray([not crit is None for crit in crit_list]).astype(bool)
    #only allow the user to specify a single condition for cutting off DPSS modes to fit.
    crit_provided_name = [ label for m,label in enumerate(crit_labels) if crit_provided[m] ]
    crit_provided_value = [ crit for m,crit in enumerate(crit_list) if crit_provided[m] ]
    if np.count_nonzero(crit_provided) != 1:
        raise ValueError('Must only provide a single series cutoff condition. %d were provided: %s '%(np.count_nonzero(crit_provided),
                                                                                                 str(crit_provided_name)))
    opkey = _fourier_filter_hash(filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                                 filter_factors=[0.], crit_name=crit_provided_name[0], x=x,
                                 w=None, hash_decimal=hash_decimal,
                                 label='dpss_operator', crit_val=tuple(crit_provided_value[0]))
    if not opkey in cache:
        #check that xs are equally spaced.
        if not np.all(np.isclose(np.diff(x), np.mean(np.diff(x)))):
            #for now, don't support DPSS iterpolation unless x is equally spaced.
            #In principal, I should be able to compute off-grid DPSS points using
            #the fourier integral of the DPSWF
            raise ValueError('x values must be equally spaced for DPSS operator!')
        nf = len(x)
        df = np.abs(x[1]-x[0])
        xg, yg = np.meshgrid(x,x)
        if xc is None:
            xc = x[nf//2]
        #determine cutoffs
        if nterms is None:
            nterms = []
            for fn,fw in enumerate(filter_half_widths):
                dpss_vectors = windows.dpss(nf, nf * df * fw, nf)
                if not eigenval_cutoff is None:
                    smat = np.sinc(2 * fw * (xg-yg)) * 2 * df * fw
                    eigvals = np.sum((smat @ dpss_vectors.T) * dpss_vectors.T, axis=0)
                    nterms.append(np.max(np.where(eigvals>=eigenval_cutoff[fn])))
                if not edge_suppression is None:
                    z0=fw * df
                    edge_tone=np.exp(-2j*np.pi*np.arange(nf)*z0)
                    fit_components = dpss_vectors * (dpss_vectors @ edge_tone)
                    #this is a vector of RMS residuals of a tone at the edge of the delay window being fitted between 0 to nf DPSS components.
                    rms_residuals = np.asarray([ np.sqrt(np.mean(np.abs(edge_tone - np.sum(fit_components[:k],axis=0))**2.)) for k in range(nf)])
                    nterms.append(np.max(np.where(rms_residuals>=edge_suppression[fn])))
                if not avg_suppression is None:
                    sinc_vector=np.sinc(2 * fw * df * (np.arange(nf)-nf/2.))
                    sinc_vector = sinc_vector / np.sqrt(np.mean(sinc_vector**2.))
                    fit_components = dpss_vectors * (dpss_vectors @ sinc_vector)
                    #this is a vector of RMS residuals of vector with equal contributions from all tones within -fw and fw.
                    rms_residuals = np.asarray([ np.sqrt(np.mean(np.abs(sinc_vector - np.sum(fit_components[:k],axis=0))**2.)) for k in range(nf)])
                    nterms.append(np.max(np.where(rms_residuals>=avg_suppression[fn])))
        #next, construct A matrix.
        amat = []
        for fc, fw, nt in zip(filter_centers,filter_half_widths, nterms):
            amat.append(np.exp(2j * np.pi * (yg[:,:nt]-xc) * fc ) * windows.dpss(nf, nf * df * fw, nt).T )
        cache[opkey] = ( np.hstack(amat), nterms )
    return cache[opkey]


def dft_operator(x, filter_centers, filter_half_widths,
                cache=None, fundamental_period=None, xc=None, hash_decimal=10):
    """
    Discrete Fourier operator with multiple flexible delay windows to fit data, potentially with arbitrary
    user provided frequencies.

    A_{nu tau} = e^{- 2 * pi * i * nu * tau / B}

    for a set of taus contained within delay regions centered at filter_centers
    and with half widths of filter_half_widths separated by 1/B where B
    is provided by fundamental_period.

    Parameters
    ----------
    x: array-like floats.
        x values to evaluate operator at
    filter_centers: float or list
        float or list of floats of centers of delay filter windows in nanosec
    filter_half_widths: float or list
        float or list of floats of half-widths of delay filter windows in nanosec
    cache: dictionary, optional dictionary storing operator matrices with keys
    (x) + (filter_centers) + (filter_half_widths) + \
    hash_decimal: int, optional number of decimals to use for floating point keys.

    Returns
    --------
    Ndata x (Nfilter_window * nterm) design matrix transforming DFT coefficients
    to data.

    """
    if cache is None:
        cache = {}
    #if no fundamental fourier period is provided, set fundamental period equal to measurement
    #bandwidth.
    if fundamental_period is None:
        fundamental_period = np.median(np.diff(x)) * len(x)
    if xc is None:
        xc = x[int(np.round(len(x)/2))]
    if isinstance(filter_centers, float):
        filter_centers = [filter_centers]
    if isinstance(filter_half_widths, float):
        filter_half_widths = [filter_half_widths]

    #each column is a fixed delay
    opkey = _fourier_filter_hash(filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                                 filter_factors=[0.], x=x, w=None, hash_decimal=hash_decimal,
                                 label='dft_operator', fperiod=fundamental_period)
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



def delay_interpolation_matrix(nchan, ndelay, wgts, fundamental_period=None, cache=None, window='none'):
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

    !!! THIS FUNCTION WILL BE DEPRECATED BY fit_solution_matrix !!!

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
        fundamental period of Fourier modes to fit too.
        this sets the resolution in Fourier space. A standard DFT has a resolution
        of 1/N_{FP} = 1/N between fourier modes so that the DFT operator is
        D_{mn} = e^{-2 \pi i m n / N_{FP}}. fg_deconv_fundamental_period
        is N_{FP}.
    cache: dict, optional
        optional cache holding pre-computed matrices
    window: string, optional
        use a window to fit.
    Returns
    ----------
    (nchan, nchan) numpy array
        that can be used to interpolate over channel gaps.
    """
    if cache is None:
        cache = {}
    if not len(wgts) == nchan:
        raise ValueError("nchan must equal length of wgts")
    if fundamental_period is None: #recommend 2 x nchan or nchan.
        fundamental_period = 2*nchan #this tends to give well conditioned matrices.
    if not np.sum((np.abs(wgts) > 0.).astype(float)) >= 2*ndelay:
        raise ValueError("number of unflagged channels must be greater then or equal to number of delays")
    matkey = (nchan, ndelay, fundamental_period) + tuple(wgts)
    amat = dft_operator(x=np.arange(nchan)-nchan/2., filter_centers=[0.], filter_half_widths=[ndelay/fundamental_period],
                                          cache=cache, fundamental_period=fundamental_period)
    wmat = np.diag(wgts * gen_window(window, nchan)).astype(complex)
    fs = fit_solution_matrix(wmat, amat)
    if fs is not None:
        return amat @ fs
    else:
        return np.nan * np.ones((nchan, nchan))


def dayenu_mat_inv(x, filter_centers, filter_half_widths,
                            filter_factors, cache=None, wrap=False, wrap_interval=1,
                            nwraps=1000, no_regularization=False, hash_decimal=10):
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
    hash_decimal int, number of decimals to consider when hashing x
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
    if cache is None:
        cache = {}
    if isinstance(filter_factors,(float,int, np.int, np.float)):
        filter_factors = [filter_factors]
    if isinstance(filter_centers, (float, int, np.int, np.float)):
        filter_centers = [filter_centers]
    if isinstance(filter_half_widths, (float, int, np.int, np.float)):
        filter_half_widths = [filter_half_widths]

    nchan = len(x)

    filter_key = _fourier_filter_hash(filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                                         filter_factors=filter_factors, x=x, w=None, hash_decimal=hash_decimal,
                                         label='dayenu_matrix_inverse', wrap=wrap, wrap_interval=wrap_interval,
                                         nwraps=nwraps, no_regularization=no_regularization)
    if not filter_key in cache:
        fx, fy = np.meshgrid(x,x)
        sdwi_mat = np.identity(fx.shape[0]).astype(np.complex128)
        if no_regularization:
            sdwi_mat *= 0.
        for fc, fw, ff in zip(filter_centers, filter_half_widths, filter_factors):
            if not ff == 0:
                if not wrap:
                    sdwi_mat = sdwi_mat + np.sinc( 2. * (fx-fy) * fw ).astype(np.complex128)\
                            * np.exp(-2j * np.pi * (fx-fy) * fc) / ff
                else:
                    bwidth = x[-1] - x[0] + (x[1]-x[0])
                    for wnum in np.arange(-nwraps//2, nwraps//2):
                        offset = bwidth * wnum * wrap_interval
                        sdwi_mat = sdwi_mat + \
                        np.sinc( 2. *  (fx-fy - offset) * fw  ).astype(np.complex128)\
                        * np.exp(-2j * np.pi * (fx-fy - offset) * fc) / ff
        cache[filter_key] = sdwi_mat
    else:
        sdwi_mat = cache[filter_key]
    return sdwi_mat
