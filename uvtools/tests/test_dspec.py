import unittest
import uvtools.dspec as dspec
import numpy as np, random
import nose.tools as nt
from pyuvdata import UVData
from uvtools.data import DATA_PATH
import os

random.seed(0)

class TestMethods(unittest.TestCase):

    def test_wedge_width(self):
        # Test boundaries of delay bins
        self.assertEqual(dspec.wedge_width(0, .01, 10), (1,10))
        self.assertEqual(dspec.wedge_width(5., .01, 10), (1,10))
        self.assertEqual(dspec.wedge_width( 9., .01, 10), (2,-1))
        self.assertEqual(dspec.wedge_width(10., .01, 10), (2,-1))
        self.assertEqual(dspec.wedge_width(15., .01, 10), (3,-2))
        # test nchan
        self.assertEqual(dspec.wedge_width(10., .01, 20), (3,-2))
        self.assertEqual(dspec.wedge_width(10., .01, 40), (5,-4))
        # test sdf
        self.assertEqual(dspec.wedge_width(10., .02, 10), (3,-2))
        self.assertEqual(dspec.wedge_width(10., .04, 10), (5,-4))
        # test standoff
        self.assertEqual(dspec.wedge_width(100., .001, 100, standoff=4.), (11,-10))
        self.assertEqual(dspec.wedge_width(100., .001, 100, standoff=5.), (11,-10))
        self.assertEqual(dspec.wedge_width(100., .001, 100, standoff=10.), (12,-11))
        self.assertEqual(dspec.wedge_width(100., .001, 100, standoff=15.), (13,-12))
        # test horizon
        self.assertEqual(dspec.wedge_width(100., .001, 100, horizon=.1), (2,-1))
        self.assertEqual(dspec.wedge_width(100., .001, 100, horizon=.5), (6,-5))
        self.assertEqual(dspec.wedge_width(100., .001, 100, horizon=1.5), (16,-15))
        self.assertEqual(dspec.wedge_width(100., .001, 100, horizon=2.), (21,-20))

    def test_delay_filter_dims(self):
        self.assertRaises(AssertionError, dspec.delay_filter, np.zeros((1,2,3)), np.zeros((1,2,3)), 0, .001)

    def test_delay_filter_1D(self):
        NCHAN = 128
        TOL = 1e-6
        data = np.ones(NCHAN, dtype=np.complex)
        wgts = .5*np.ones(NCHAN, dtype=np.complex)
        dmdl, dres, info = dspec.delay_filter(data, wgts, 0., .1/NCHAN, tol=TOL)
        np.testing.assert_allclose(data, dmdl, atol=NCHAN*TOL)
        np.testing.assert_allclose(dres, np.zeros_like(dres), atol=NCHAN*TOL)
        wgts[::16] = 0
        dmdl, dres, info = dspec.delay_filter(data, wgts, 0., .1/NCHAN, tol=TOL)
        np.testing.assert_allclose(data, dmdl, atol=NCHAN*TOL)
        np.testing.assert_allclose(dres, np.zeros_like(dres), atol=NCHAN*TOL)
        data = np.random.normal(size=NCHAN)
        wgts = np.ones_like(data)
        dmdl, dres, info = dspec.delay_filter(data, wgts, 0., .1/NCHAN, tol=1e-9)
        self.assertAlmostEqual(np.average(data), np.average(dmdl), 3)
        self.assertAlmostEqual(np.average(dres), 0, 3)

    def test_delay_filter_2D(self):
        NCHAN = 128
        NTIMES = 10
        TOL = 1e-6
        data = np.ones((NTIMES, NCHAN), dtype=np.complex)
        wgts = np.ones((NTIMES, NCHAN), dtype=np.complex)
        dmdl, dres, info = dspec.delay_filter(data, wgts, 0., .1/NCHAN, tol=TOL)
        np.testing.assert_allclose(data, dmdl, atol=NCHAN*TOL)
        np.testing.assert_allclose(dres, np.zeros_like(dres), atol=NCHAN*TOL)
        wgts[:,::16] = 0;
        wgts*=.9 #tests to make sure wgts**2 normalization works
        dmdl, dres, info = dspec.delay_filter(data, wgts, 0., .1/NCHAN, tol=TOL)
        np.testing.assert_allclose(data, dmdl, atol=NCHAN*TOL)
        np.testing.assert_allclose(dres, np.zeros_like(dres), atol=NCHAN*TOL)
        data = np.array(np.random.normal(size=(NTIMES,NCHAN)),dtype=complex)
        wgts = np.ones_like(data)
        dmdl, dres, info = dspec.delay_filter(data, wgts, 0., .1/NCHAN, tol=1e-9)
        np.testing.assert_allclose(np.average(data,axis=1), np.average(dmdl,axis=1), atol=1e-3)
        np.testing.assert_allclose(np.average(dres,axis=1), 0, atol=1e-3)
    
    def test_skip_wgt(self):
        NCHAN = 128
        NTIMES = 10
        TOL = 1e-6
        data = np.ones((NTIMES, NCHAN), dtype=np.complex)
        wgts = np.ones((NTIMES, NCHAN), dtype=np.complex)
        wgts[0, 0:-4] = 0
        dmdl, dres, info = dspec.delay_filter(data, wgts, 0., .1/NCHAN, tol=TOL, skip_wgt=.1)
        np.testing.assert_allclose(data[1:,:], dmdl[1:,:], atol=NCHAN*TOL)
        np.testing.assert_allclose(dres[1:,:], np.zeros_like(dres)[1:,:], atol=NCHAN*TOL)
        np.testing.assert_allclose(dmdl[0,:], np.zeros_like(dmdl[0,:]), atol=NCHAN*TOL)
        np.testing.assert_allclose(dres[0,:], data[0,:], atol=NCHAN*TOL)
        self.assertEqual(len(info), NTIMES)
        self.assertTrue(info[0]['skipped'])

    def test_calc_width(self):
        # test single filter_size
        nchan = 100
        dt = 10.
        filter_size = 1e-2
        u, l = dspec.calc_width(filter_size, dt, nchan)
        frs = -np.fft.fftfreq(nchan, dt)  # negative b/c of ifft convention
        nt.assert_true(np.all(np.abs(frs[u:l]) > filter_size))

        # test multiple entries in filter_size
        filter_size = (1e-2, 2e-2)
        u, l = dspec.calc_width(filter_size, dt, nchan)
        nt.assert_true(np.all((frs[u:l] < -1e-2) | (frs[u:l] > 2e-2)))


def test_vis_filter():
    # load file
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, "zen.2458042.17772.xx.HH.uvXA"), bls=[(24, 25)])

    # simulate some data in fringe-rate and delay space
    np.random.seed(0)
    dfft = (np.random.normal(0, 1, uvd.Nfreqs * uvd.Ntimes).astype(np.complex) \
         + 1j * np.random.normal(0, 1, uvd.Nfreqs * uvd.Ntimes)).reshape(uvd.Ntimes, uvd.Nfreqs)
    dfft[2, 2] += 200
    dfft[20, 20] += 50
    d = np.fft.fft2(dfft)

    # get snr of modes
    davg = np.mean(np.abs(np.fft.fft(dfft, axis=0)), axis=0)
    std = np.median(davg)
    snr1 = davg[2] / std
    snr2 = davg[20] / std

    # simulate some flags
    f = np.zeros_like(d, dtype=np.bool)
    d[:, 20] += 1e3
    f[:, 20] = True
    d[20, :] += 1e3
    f[20, :] = True

    # get data, wgts
    w = (~f).astype(np.float)
    bl_len = 50 / 2.99e8
    sdf = np.median(np.diff(uvd.freq_array.squeeze()))
    dt = np.median(np.diff(np.unique(uvd.time_array))) * 24 * 3600
    frs = np.fft.fftfreq(uvd.Ntimes, d=dt)
    dlys = np.fft.fftfreq(uvd.Nfreqs, d=sdf) * 1e9

    def get_snr(clean, fftax=1, avgax=0, modes=[2, 20]):
        cfft = np.fft.ifft(clean, axis=fftax)
        cavg = np.mean(np.abs(cfft), axis=avgax)
        std = np.median(cavg)
        return [cavg[m] / std for m in modes]

    # delay filter basic execution
    mdl, res, info = dspec.delay_filter(d, w, bl_len, sdf, standoff=0, horizon=1.0, min_dly=0.0,
                                        tol=1e-4, window='none', skip_wgt=0.1, gain=0.1)
    # assert recovered snr of input modes
    clean = mdl + res * w
    snrs = get_snr(clean)
    nt.assert_true(np.isclose(snrs[0], snr1, atol=2))
    nt.assert_true(np.isclose(snrs[1], snr2, atol=2))

    # test vis filter is the same
    mdl2, res2, info2 = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, standoff=0, horizon=1.0, min_dly=0.0,
                                           tol=1e-4, window='none', skip_wgt=0.1, gain=0.1)
    nt.assert_true(np.isclose(mdl - mdl2, 0.0).all())

    # fringe filter basic execution 
    mdl, res, info = dspec.fringe_filter(d, w, frs[15], dt, tol=1e-4, window='none', skip_wgt=0.1, gain=0.1)

    # assert recovered snr of input modes
    clean = mdl + res * w
    snrs = get_snr(clean)
    nt.assert_true(np.isclose(snrs[0], snr1, atol=2))
    nt.assert_true(np.isclose(snrs[1], snr2, atol=2))

    # test vis filter is the same
    mdl2, res2, info2 = dspec.vis_filter(d, w, max_frate=frs[15], dt=dt, tol=1e-4, window='none', skip_wgt=0.1, gain=0.1)
    nt.assert_true(np.isclose(mdl - mdl2, 0.0).all())

    # try non-symmetric filter
    mdl, res, info = dspec.fringe_filter(d, w, (frs[-20], frs[10]), dt, tol=1e-4, window='none', skip_wgt=0.1, gain=0.1)

    # assert recovered snr of input modes
    clean = mdl + res * w
    snrs = get_snr(clean)
    nt.assert_true(np.isclose(snrs[0], snr1, atol=2))
    nt.assert_true(np.isclose(snrs[1], snr2, atol=2))

    # 2d clean
    mdl, res, info = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, max_frate=frs[15], dt=dt, tol=1e-4, window='none', maxiter=100, gain=1e-1)

    # assert recovered snr of input modes
    clean = mdl + res * w
    snrs = get_snr(clean)
    nt.assert_true(np.isclose(snrs[0], snr1, atol=2))
    nt.assert_true(np.isclose(snrs[1], snr2, atol=2))

    # non-symmetric 2D clean
    mdl, res, info = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, max_frate=(frs[-20], frs[10]), dt=dt, tol=1e-4, window='none', maxiter=100, gain=1e-1)

    # assert recovered snr of input modes
    clean = mdl + res * w
    snrs = get_snr(clean)
    nt.assert_true(np.isclose(snrs[0], snr1, atol=2))
    nt.assert_true(np.isclose(snrs[1], snr2, atol=2))

    # try plus filtmode on 2d clean
    mdl, res, info = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, max_frate=(frs[-20], frs[10]), dt=dt, tol=1e-4, window=('none', 'blackman'), edgecut_low=(0, 5), edgecut_hi=(2, 5), maxiter=100, gain=1e-1, filt2d_mode='plus')

    # assert recovered snr of input modes
    clean = mdl + res * w
    snrs = get_snr(clean)
    nt.assert_true(np.isclose(snrs[0], snr1, atol=2))
    nt.assert_true(np.isclose(snrs[1], snr2, atol=2))

    # exceptions
    nt.assert_raises(ValueError, dspec.vis_filter, d, w, bl_len=bl_len, sdf=sdf, max_frate=(frs[-20], frs[10]), dt=dt, filt2d_mode='foo')


if __name__ == '__main__':
    unittest.main()
