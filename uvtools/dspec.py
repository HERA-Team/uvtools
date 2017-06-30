import aipy
import binning
import numpy as np

def wedge_width(bl_len, sdf, nchan, standoff=0., horizon=1.):
    '''Return the (upper,lower) delay bins that geometrically
    correspond to the sky, for bl_len in ns, sdf in GHz, and the number
    of channels nchan.'''
    bl_dly = horizon * bl_len + standoff
    bin_dly = 1. / (sdf * nchan)
    w = int(round(bl_dly/bin_dly))
    uthresh, lthresh = w + 1, -w
    if lthresh == 0: lthresh = nchan
    return (uthresh,lthresh)
    
def delay_filter(data, wgts, bl_len, sdf, standoff=0., horizon=1., tol=1e-4, 
        window='none', skip_wgt=0.5, maxiter=100):
    '''Apply a wideband delay filter to data.  Data are weighted again by wgts,
    windowed, Fourier transformed, and deconvolved allowing clean components
    between lthresh and uthresh.  The mdl, residual, and info are returned in
    frequency domain.'''
    nchan = data.shape[-1]
    window = aipy.dsp.gen_window(nchan, window=window)
    _d = np.fft.ifft(data * wgts * window, axis=-1)
    _w = np.fft.ifft(wgts * wgts * window, axis=-1)
    uthresh,lthresh = wedge_width(bl_len, sdf, nchan, standoff=standoff, horizon=horizon)
    area = np.ones(nchan, dtype=np.int); area[uthresh:lthresh] = 0
    if data.ndim == 1:
        _d_cl, info = aipy.deconv.clean(_d, _w, area=area, tol=tol, stop_if_div=False, maxiter=maxiter)
        d_mdl = np.fft.fft(_d_cl)# + info['res'])
    elif data.ndim == 2:
        d_mdl = np.empty_like(data)
        for i in xrange(data.shape[0]):
            if _w[i,0] < skip_wgt: d_mdl[i] = 0 # skip highly flagged (slow) integrations
            else:
                _d_cl, info = aipy.deconv.clean(_d[i], _w[i], area=area, tol=tol, 
                        stop_if_div=False, maxiter=maxiter)
                d_mdl[i] = np.fft.fft(_d_cl)# + info['res'])
                # XXX info overwritten every i
    else: raise ValueError('data must be a 1D or 2D array')
    d_res = data - d_mdl * wgts
    return d_mdl, d_res, info

def delay_filter_aa(aa, data, wgts, i, j, sdf, phs2lst=False, jds=None, 
        skip_wgt=0.5, lst_res=binning.DEFAULT_LST_RES, standoff=0., horizon=1., 
        tol=1e-4, window='none', maxiter=100):
    '''Use information from AntennaArray object to delay filter data, with the
    option to phase data to an lst bin first.  Arguments are the same as for
    delay_filter and binning.phs2lstbin.  Returns mdl, residual, and info
    in the frequency domain.'''
    if phs2lst:
        data = binning.phs2lstbin(data, aa, i, j, jds=jds, lst_res=lst_res)
    bl = aa.get_baseline(i,j)
    return delay_filter(data, wgts, np.linalg.norm(bl), sdf, 
            standoff=standoff, horizon=horizon, tol=tol, window=window, 
            skip_wgt=skip_wgt, maxiter=maxiter)

# XXX is this a used function?
def delayfiltercov(C,horizon_bins=5,eig_cut_dnr=2):
    #delay filter a spectral covariance matrix
    #horizon_bins = distance delay=0 to be retained, ie the size of the wedge in bins
    # eig_cut_dnr = retain eigenvalues with a dynamic range of  median(dnr)*eig_cut_dnr 
    # where dnr is max(dspec eigenvector)/mean(abs(dpsec eigenvector outside horizon))    
    #
    # returns filtered_covariance,matching_projection matrix
    S,V = np.linalg.eig(C)
    dV = np.fft.ifft(V,axis=0)
    #calculate eigenvalue cut, selecting only those eigenvectors with strong delay spectrum signals
    dnr = np.max(np.abs(dV),axis=0)/np.mean(np.abs(dV)[horizon_bins:-horizon_bins,:],axis=0)
    median_dnr = np.median(dnr)
    eig_cut_dnr *= median_dnr
    S[dnr<eig_cut_dnr] = 0 #apply eigenvalue cut
    #mask outside wedge
    dV[horizon_bins:-horizon_bins,:] = 0 # mask out stuff outside the horizon
    V_filtered = np.fft.fft(dV,axis=0)
    #return filtered covariance and its matching projection matrix
    return np.einsum('ij,j,jk',V_filtered,S,V_filtered.T),np.einsum('ij,j,jk',V_filtered,S!=0,V_filtered.T)

