import aipy
import numpy as np

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
        uthresh, lthresh: bin indices for upper and lower edges of the fourier filter
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
        uthresh, lthresh: bin indices for upper and lower edges of the fourier filter    
    '''
    bin_width = 1. / (real_delta * nsamples)
    w = int(round(filter_size / bin_width))
    uthresh, lthresh = w + 1, -w
    if lthresh == 0: 
        lthresh = nsamples
    return (uthresh, lthresh)


def high_pass_fourier_filter(data, wgts, filter_size, real_delta, tol=1e-9, window='none', skip_wgt=0.1, maxiter=100):
    '''Apply a highpass fourier filter to data. Uses aipy.deconv.clean. 

    Arguments:
        data: 1D or 2D (real or complex) numpy array to be filtered along the last dimension.
        wgts: real numpy array of multiplicative weights with the same shape as the data. 
        filter_size: the half-width (i.e. the width of the positive part) of the region in fourier 
            space, symmetric about 0, that is filtered out. In units of 1/[real_delta].
        real_delta: the bin width in real space of the dimension to be filtered
        tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
        window: window function for filtering applied to the filtered axis. 
            See aipy.dsp.gen_window for options.
        skip_wgt: skips filtering highly flagged rows (unflagged fraction ~< skip_wgt) 
            Only works properly when all weights are all between 0 and 1.
        maxiter: Maxinum number of iterations for aipy.deconv.clean to converge.

    Returns:
        d_mdl: best fit low-pass filter components (CLEAN model) in real space
        d_res: best fit high-pass filter components (CLEAN residual) in real space
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    '''
    nchan = data.shape[-1]
    window = aipy.dsp.gen_window(nchan, window=window)
    _d = np.fft.ifft(data * wgts * window, axis=-1)
    _w = np.fft.ifft(wgts * wgts * window, axis=-1)
    uthresh,lthresh = calc_width(filter_size, real_delta, nchan)
    area = np.ones(nchan, dtype=np.int); area[uthresh:lthresh] = 0
    if data.ndim == 1:
        _d_cl, info = aipy.deconv.clean(_d, _w, area=area, tol=tol, stop_if_div=False, maxiter=maxiter)
        d_mdl = np.fft.fft(_d_cl)
        del info['res']
    elif data.ndim == 2:
        info = []
        d_mdl = np.empty_like(data)
        for i in xrange(data.shape[0]):
            if _w[i,0] < skip_wgt: 
                d_mdl[i] = 0 # skip highly flagged (slow) integrations
            else:
                _d_cl, info_here = aipy.deconv.clean(_d[i], _w[i], area=area, tol=tol, stop_if_div=False, maxiter=maxiter)
                d_mdl[i] = np.fft.fft(_d_cl)
                del info_here['res']
                info.append(info_here)
    else: 
        raise ValueError('data must be a 1D or 2D array')
    d_res = data - d_mdl * wgts

    return d_mdl, d_res, info


    
def delay_filter(data, wgts, bl_len, sdf, standoff=0., horizon=1., tol=1e-4, 
        window='none', skip_wgt=0.5, maxiter=100):
    '''Apply a wideband delay filter to data. Variable names preserved for 
        backward compatability with capo/PAPER analysis.

    Arguments:
        data: 1D or 2D (real or complex) numpy array to where last dimension is frequency.
        wgts: real numpy array of multiplicative weights with the same shape as the data. 
        bl_len: length of baseline (in 1/[sdf], typically ns)
        sdf: frequency channel width (typically in GHz)
        standoff: fixed additional delay beyond the horizon (same units as bl_len)
        horizon: proportionality constant for bl_len where 1 is the horizon (full light travel time)
        tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
        window: window function for filtering applied to the filtered axis. 
            See aipy.dsp.gen_window for options.        
        skip_wgt: skips filtering highly flagged rows (unflagged fraction ~< skip_wgt) 
            Only works properly when all weights are all between 0 and 1.
        maxiter: Maxinum number of iterations for aipy.deconv.clean to converge.

    Returns:
        d_mdl: best fit low-pass filter components (CLEAN model) in the frequency domain
        d_res: best fit high-pass filter components (CLEAN residual) in the frequency domain
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    '''    
    bl_dly = horizon * bl_len + standoff
    return high_pass_fourier_filter(data, wgts, bl_dly, sdf, tol=tol,window=window,
                                    skip_wgt=skip_wgt, maxiter=maxiter)




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

