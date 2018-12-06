import aipy
import numpy as np
from six.moves import range
from scipy.optimize import leastsq, lsq_linear

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
    '''Calculate the upper and lower bin indices of a fourier filter.

    Arguments:
        filter_size: the half-width (i.e. the width of the positive part) of the region in fourier 
            space, symmetric about 0, that is filtered out. In units of 1/[real_delta].
        real_delta: the bin width in real space
        nsamples: the number of samples in the array to be filtered

    Returns:
        uthresh, lthresh: bin indices for filtered bins started at uthresh (which is filtered)
            and ending at lthresh (which is a negative integer and also not filtered).
            Designed for area = np.ones(nsamples, dtype=np.int); area[uthresh:lthresh] = 0
    '''
    bin_width = 1.0 / (real_delta * nsamples)
    w = int(np.around(filter_size / bin_width))
    uthresh, lthresh = w + 1, -w
    if lthresh == 0: 
        lthresh = nsamples
    return (uthresh, lthresh)


def high_pass_fourier_filter(data, wgts, filter_size, real_delta, tol=1e-9, window='none', 
                             skip_wgt=0.1, maxiter=100, gain=0.1, **win_kwargs):
    '''Apply a highpass fourier filter to data. Uses aipy.deconv.clean. 

    Arguments:
        data: 1D or 2D (real or complex) numpy array to be filtered along the last dimension.
            (Unlike previous versions, it is NOT assumed that weights have already been multiplied
            into the data.)
        wgts: real numpy array of linear multiplicative weights with the same shape as the data. 
        filter_size: the half-width (i.e. the width of the positive part) of the region in fourier 
            space, symmetric about 0, that is filtered out. In units of 1/[real_delta].
        real_delta: the bin width in real space of the dimension to be filtered
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
        d_mdl: best fit low-pass filter components (CLEAN model) in real space
        d_res: best fit high-pass filter components (CLEAN residual) in real space
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    '''
    nchan = data.shape[-1]
    window = aipy.dsp.gen_window(nchan, window=window, **win_kwargs)
    _d = np.fft.ifft(data * wgts * window, axis=-1)
    _w = np.fft.ifft(wgts * window, axis=-1)
    uthresh,lthresh = calc_width(filter_size, real_delta, nchan)
    area = np.ones(nchan, dtype=np.int) 
    area[uthresh:lthresh] = 0
    if data.ndim == 1:
        _d_cl, info = aipy.deconv.clean(_d, _w, area=area, tol=tol, stop_if_div=False, maxiter=maxiter, gain=gain)
        d_mdl = np.fft.fft(_d_cl)
        del info['res']
    elif data.ndim == 2:
        info = []
        d_mdl = np.empty_like(data)
        for i in range(data.shape[0]):
            if _w[i,0] < skip_wgt: 
                d_mdl[i] = 0 # skip highly flagged (slow) integrations
                info.append({'skipped': True})
            else:
                _d_cl, info_here = aipy.deconv.clean(_d[i], _w[i], area=area, tol=tol, stop_if_div=False, maxiter=maxiter, gain=gain)
                d_mdl[i] = np.fft.fft(_d_cl)
                del info_here['res']
                info.append(info_here)
    else: 
        raise ValueError('data must be a 1D or 2D array')
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
    # construct baseline delay
    bl_dly = horizon * bl_len + standoff

    # check minimum delay
    bl_dly = np.max([bl_dly, min_dly])

    # run fourier filter
    return high_pass_fourier_filter(data, wgts, bl_dly, sdf, tol=tol, window=window,
                                    skip_wgt=skip_wgt, maxiter=maxiter, gain=gain, **win_kwargs)


def fourier_operator(dsize, nmax):
    """
    Return a complex Fourier analysis operator for a given data dimension and 
    number of Fourier modes.
    
    Parameters
    ----------
    dsize : int
        Size of data array.
    
    nmax : int
        Maximum Fourier mode number. Modes will be constructed between 
        [-nmax, nmax], for a total of 2*nmax + 1 modes.
    
    Returns
    -------
    F : array_like
        Fourier matrix operator, of shape (Nmodes, Ndata)
    """
    # Construct frequency array (*not* in physical frequency units)
    nu = np.arange(dsize)
    L = nu[-1] - nu[0]
    
    # Build matrix operator for complex Fourier basis
    n = np.arange(-nmax, nmax+1)
    F = np.array([np.exp(-1.j * _n * nu / L) for _n in n])
    return F


def fourier_model(cn, Nfreqs):
    """
    Calculate a 1D (complex) Fourier series model from a set of complex 
    coefficients.
    
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
        if len(cn.shape) != 1: raise ValueError("cn must be a 1D array")
    except:
        raise ValueError("cn must be a 1D array")
    nmax = (cn.size - 1) // 2 # Max. harmonic
    
    # Build matrix operator for complex Fourier basis
    F = fourier_operator(dsize=Nfreqs, nmax=nmax)
    
    # Return model
    return np.dot(cn, F)


def delay_filter_leastsq_1d(data, flags, sigma, nmax, add_noise=False, 
                            cn_guess=None, use_linear=True, operator=None):
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
    
    nmax: int
        Max. order of Fourier modes to fit. A model with complex Fourier modes 
        between [-n, n] will be fitted to the data, where the Fourier basis 
        functions are ~ exp(-i 2 pi n nu / (Delta nu).
    
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
        Fourier basis operator matrix. Must have shape (Nmodes, Nfreq), where 
        Nmodes = 2*nmax + 1.
    
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
    if operator is None:
        F = fourier_operator(dsize=data.size, nmax=nmax)
    else:
        F = operator
        #n = np.arange(-nmax, nmax+1)
        try:
            if F.shape[0] != 2*nmax+1:
                raise ValueError("Fourier basis operator has the wrong shape. "
                                 "Must have shape (Nmodes, Nfreq).")
        except: 
            raise ValueError("Fourier basis operator has the wrong shape. "
                             "Must have shape (Nmodes, Nfreq).")
    
    # Turn flags into a mask
    w = np.logical_not(flags)
    
    # Define model and likelihood function
    model = lambda cn: np.dot(cn, F)
    nmodes = 2*nmax + 1
    
    # Initial guess for Fourier coefficients (real + imaginary blocks)
    cn_in = np.zeros(2*nmodes)
    if cn_guess is not None:
        assert cn_in.size == 2*cn_guess.size, "cn_guess must be of size %s" \
            % (cn_in.size/2)
        cn_in[:cn_guess.shape[0]] = cn_guess.real
        cn_in[cn_guess.shape[0]:] = cn_guess.imag
    
    # Run least-squares fit
    if use_linear:
        # Solve as linear system
        A = np.atleast_2d(w).T * F.T
        res = lsq_linear(A/sigma, w*data/sigma)
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
            _delta = w * (data - model(cn[:nmodes] + 1.j*cn[nmodes:]))
            delta = np.concatenate((_delta.real/sigma, _delta.imag/sigma))
            return -0.5 * delta**2.
        
        # Do non-linear least-squares calculation
        cn, stat = leastsq(loglike, cn_in)
        cn_out = cn[:nmodes] + 1.j*cn[nmodes:]
    
    # Inject smooth best-fit model into masked areas
    bf_model = model(cn_out)
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


def delay_filter_leastsq(data, flags, sigma, nmax, add_noise=False, 
                         cn_guess=None, use_linear=True):
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
    # Construct and cache Fourier basis operator (for speed)
    F = fourier_operator(dsize=data.shape[1], nmax=nmax)
    nmodes = 2*nmax + 1
    
    # Array to store in-painted data
    inp_data = np.zeros(data.shape, dtype=np.complex)
    cn_array = np.zeros((data.shape[0], nmodes), dtype=np.complex)
    mdl_array = np.zeros(data.shape, dtype=np.complex)
    
    # Loop over array
    cn_out = None
    for i in range(data.shape[0]):
        bf_model, cn_out, data_out = delay_filter_leastsq_1d(
                                                data[i], flags[i], sigma=sigma, 
                                                nmax=nmax, add_noise=add_noise, 
                                                use_linear=use_linear,
                                                cn_guess=cn_out, operator=F)
        inp_data[i,:] = data_out
        cn_array[i,:] = cn_out
        mdl_array[i,:] = bf_model
    
    return mdl_array, cn_array, inp_data


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

