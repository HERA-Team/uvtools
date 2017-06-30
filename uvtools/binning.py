import numpy as np, aipy

def gen_lst_res(nbins=None, secs=None, dlst=None):
    '''Generate the resolution of an LST bin from (one of) the desired number 
    of bins (nbins), time interval in seconds (secs), or LST interval in
    radians (dlst).  The resulting interval may be slightly different than
    specified by secs or dlst to ensure no partial bins at the 2*pi -> 0 wrap.'''
    if secs is not None:
        assert(dlst is None) # only one of nbins, secs, dlst can be specified
        dlst = 2*np.pi / aipy.const.sidereal_day * secs
    if dlst is not None:
        assert(nbins is None) # only one of nbins, secs, dlst can be specified
        nbins = np.around(2*np.pi / dlst).astype(np.int)
    return 2*np.pi / nbins

DEFAULT_LST_RES = gen_lst_res(secs=30.) # default 30s
DEFAULT_UV_RES = 1.5 # wavelengths
    
def get_lstbins(lst_res=DEFAULT_LST_RES, min_lst=0., max_lst=2*np.pi):
    bins = np.arange(lst_res/2, 2*np.pi, lst_res)
    valid = np.where(np.logical_and(bins >= min_lst, bins < max_lst))
    return bins[valid]

def lstbin(lst, lst_res=DEFAULT_LST_RES):
    '''Chooses an lst bin for a given lst resolution (from gen_lst_res). Takes care of
    tricky wrap cases at 2*pi -> 0.'''
    nbins = np.around(2*np.pi / lst_res).astype(np.int)
    tbin = np.floor((lst % (2*np.pi)) / lst_res).astype(np.int) + .5
    return tbin * lst_res

def uv2bin(u,v,lst,uv_res=DEFAULT_UV_RES, lst_res=DEFAULT_LST_RES):
    '''Round u,v,lst coordinates to nearest (u',v',lst') bin.'''
    ubin, vbin = np.around(u / uv_res).astype(np.int), np.around(v / uv_res).astype(np.int)
    lstb = lstbin(lst, lst_res=lst_res)
    return (ubin*uv_res, vbin*uv_res, lstb)

def rebin_log(x, y, bin=10):
    '''For y=f(x), bin x into log_e bins, and average y over
    these bin sizes.'''
    logx = np.log(np.abs(x))
    hist1,bins = np.histogram(logx, bins=bin, weights=y)
    hist2,bins = np.histogram(logx, bins=bin)
    logx = .5 * (bins[1:] + bins[:-1])
    return np.e**logx, hist1 / np.where(hist2 == 0, 1., hist2)

def jd2lstbin(jds, aa, lst_res=DEFAULT_LST_RES):
    bins = []
    for jd in jds:
        aa.set_jultime(jd)
        bins.append(lstbin(aa.sidereal_time(), lst_res=lst_res))
    return bins

def gen_lstbinphs(aa, i, j, lst_res=DEFAULT_LST_RES):
    '''Return the Delta phase required to shift phase center from lst to lstbin.'''
    lst = aa.sidereal_time()
    lstb = lstbin(lst, lst_res=lst_res)
    zen = aipy.phs.RadioFixedBody(lst, aa.lat)
    zenb = aipy.phs.RadioFixedBody(lstb, aa.lat)
    zen.compute(aa); zenb.compute(aa)
    return aa.gen_phs(zenb, i, j) * aa.gen_phs(zen, i, j).conj()

def phs2lstbin(data, aa, i, j, jds=None, lst_res=DEFAULT_LST_RES):
    '''Phase data to the closest lst bin, which is the point that transits zenith
    at the sidereal time corresponding to the center of an lst bin.'''
    if data.ndim == 1:
        if jds is not None: aa.set_jultime(jds)
        return data * gen_lstbinphs(aa, i, j, lst_res=lst_res)
    elif data.ndim == 2:
        assert(len(jds) == data.shape[0])
        d = np.empty_like(data)
        for i,jd in enumerate(jds):
            aa.set_jultime(jd)
            d[i] = data[i] * gen_lstbinphs(aa, i, j, lst_res=lst_res)
        return d
    else: raise ValueError('data must be a 1D or 2D array')

def gen_phs2lstbin_mfunc(aa, lst_res=DEFAULT_LST_RES):
    def mfunc(uv, p, d, f):
        _, jd, (i,j) = p
        aa.set_jultime(jd)
        if i != j: d = phs2lstbin(d, aa, i, j, lst_res=lst_res)
        return p,d,f
    return mfunc


