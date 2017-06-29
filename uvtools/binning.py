import numpy as np, aipy

LST_RES = 2*np.pi/24
UV_RES = 1.5

def uv2bin(u,v,lst,uv_res=UV_RES, lst_res=LST_RES):
    return (int(np.around(u / uv_res) + 4096) * 8192 + int(np.around(v / uv_res) + 4096)) * 8192 + int(np.around(lst/lst_res))

def bin2uv(bin, uv_res=UV_RES, lst_res=LST_RES):
    bin = int(bin)
    v = ((bin/8192) % 8192 - 4096) * float(uv_res)
    u = (bin / 8192**2 - 4096) * float(uv_res)
    lst = (bin % 8192) * float(lst_res)
    return u,v, lst

def rebin_log(x, y, bin=10):
    '''For y=f(x), bin x into log_e bins, and average y over
    these bin sizes.'''
    logx = np.log(np.abs(x))
    hist1,bins = np.histogram(logx, bins=bin, weights=y)
    hist2,bins = np.histogram(logx, bins=bin)
    logx = .5 * (bins[1:] + bins[:-1])
    return np.e**logx, hist1 / np.where(hist2 == 0, 1., hist2)

def lstbin(lst, lst_res=40.):
    '''Chooses an lst bin for a given lst.  lst_res in seconds'''
    lst_res = lst_res / aipy.const.sidereal_day * (2*np.pi)
    return bin2uv(uv2bin(0,0,lst,lst_res=lst_res),lst_res=lst_res)[-1]

def jd2lstbin(jds, aa, lst_res=40.):
    bins = []
    for jd in jds:
        aa.set_jultime(jd)
        bins.append(lstbin(aa.sidereal_time(), lst_res=lst_res))
    return bins

def gen_lstbinphs(aa, i, j, lst_res=40.):
    '''Return the Delta phase required to shift phase center from lst to lstbin.'''
    lst = aa.sidereal_time()
    lstb = lstbin(lst, lst_res=lst_res)
    zen = aipy.phs.RadioFixedBody(lst, aa.lat)
    zenb = aipy.phs.RadioFixedBody(lstb, aa.lat)
    zen.compute(aa); zenb.compute(aa)
    return aa.gen_phs(zenb, i, j) * aa.gen_phs(zen, i, j).conj()

def phs2lstbin(data, aa, i, j, jds=None, lst_res=40.):
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

def gen_phs2lstbin_mfunc(aa, lst_res=40.):
    def mfunc(uv, p, d, f):
        _, jd, (i,j) = p
        aa.set_jultime(jd)
        if i != j: d = phs2lstbin(d, aa, i, j, lst_res=lst_res)
        return p,d,f
    return mfunc


