import unittest
import aipy as a, numpy as n
import pylab as p
try:
    from mpl_toolkits.basemap import Basemap
except ModuleNotFoundError:
    raise ModuleNotFoundError("Fringe test requires Basemap! Try running 'pip install --user git+https://github.com/matplotlib/basemap.git'")
import uvtools.fringe as fringe

# XXX these aren't actually unittests.  need real ones.
# Currently disabled!

"""
class TestFRFilter(unittest.TestCase):
    def setUp(self):
        self.aa = a.cal.get_aa('psa6240_v003', n.linspace(.1,.2,203))
    def test_get_beam_w_fr(self):
        interps = fringe.get_beam_w_fr(self.aa, (1,4), ref_chan=160)
        t,firs, frbins, frspace = fringe.get_fringe_rate_kernels(interps, 42.9, 401)
        print(n.ones_like(frspace).sum())
        print(n.sum(frspace**2))
        p.subplot(121)
        p.plot(t, firs[160])
        p.plot(t, n.abs(firs[160]))
        p.subplot(122)
        p.plot(frbins, frspace[160])
        p.xlim(-.0005,.0020)
        p.ylim(0.0,1.0)
        p.show()
    def test_get_optimal_kernel_at_ref(self):
        ch = 100
        binwidth=.00005
        freq = self.aa.get_afreqs()[ch]
        bin_edges = n.arange(-.01+binwidth/2,.01,binwidth)
        h = a.healpix.HealpixMap(nside=64)
        xyz = h.px2crd(n.arange(h.map.size), ncrd=3)
        top_x, top_y, top_z = n.dot(self.aa._eq2zen, xyz)
        _bmx = self.aa[0].bm_response((top_x,top_y,top_z), pol='x')[ch]
        _bmy = self.aa[0].bm_response((top_x,top_y,top_z), pol='y')[ch]
        bmxx = n.where(top_z > 0, _bmx**2, 0)
        bmyy = n.where(top_z > 0, _bmy**2, 0)
        bm_I = 0.5 * (bmxx + bmyy)
        lat = self.aa.lat
        DIM = 800
        RES = .5
        SIZE = DIM / RES
        im = a.img.Img(DIM, res=RES)
        def proj_hmap(dat, mode='e'):
            h.map = dat
            #if mode == 'e': x,y,z = im.get_eq(dec=aa.lat, center=(SIZE/2,SIZE/2))
            if mode == 'e': x,y,z = im.get_eq(dec=lat, center=(SIZE/2,SIZE/2))
            elif mode == 't': x,y,z = im.get_top(center=(SIZE/2,SIZE/2))
            else: raise ValueError(mode)
            v = n.logical_not(x.mask)
            x,y,z = x.flatten(), y.flatten(), z.flatten()
            d = h[x,y,z]
            d.shape = (SIZE,SIZE)
            d = n.where(v, d, n.NaN)
            return d

        m = Basemap(projection='ortho', lat_0=lat, lon_0=180, rsphere=1.)

        def plot_hmap(dat, mode='log', mx=0, drng=3):
            d = proj_hmap(dat,'e')
            if mode == 'log': d = n.log10(n.abs(d).clip(10**(mx-drng),n.Inf))
            elif mode == 'real': d = d.real
            m.imshow(d, vmax=mx, vmin=mx-drng, origin='lower', interpolation='bicubic', cmap='jet')
        p.subplot(121)
        plot_hmap(bm_I)
        m.drawmapboundary(linewidth=2)
        m.drawmeridians(n.arange(0, 360, 30), linewidth=2)
        m.drawparallels(n.arange(-90,90,30), linewidth=2)
        print(bm_I.sum()*(4*n.pi/h.npix()),)
        bm_I2 = bm_I**2
        print(bm_I.sum()*(4*n.pi/h.npix()))
        p.subplot(122)
        plot_hmap(bm_I2)
        m.drawmapboundary(linewidth=2)
        m.drawmeridians(n.arange(0, 360, 30), linewidth=2)
        m.drawparallels(n.arange(-90,90,30), linewidth=2)
        p.show()
        xyz = (xyz[1], xyz[0], xyz[2])
        bl = self.aa.get_baseline(1,4,'r') * freq
        print('Baseline:', bl)
        fng = fringe.mk_fng(bl, *xyz)
        h_I, bin_edges = n.histogram(fng, bins=bin_edges, weights=bm_I2)
        h_I = n.sqrt(h_I)
        h_I /= h_I.max()
        p.plot(bin_edges[:-1], h_I, label='rederived')
        h_I, bins, wgt, (cen,wid) = fringe.get_optimal_kernel_at_ref(self.aa, ch, (1,4), binwidth=binwidth)
        p.plot(bins, h_I, label='frf_conv')
        p.plot(bins, fringe.gauss(bins, cen, wid), label='gaussian')
        p.legend(loc='best')
        p.show()
"""

if __name__ == '__main__':
    unittest.main()
