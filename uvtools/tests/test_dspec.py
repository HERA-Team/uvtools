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
        self.assertRaises(ValueError, dspec.delay_filter, np.zeros((1,2,3)), np.zeros((1,2,3)), 0, .001)

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
    
    def test_delay_filter_leastsq(self):
        NCHAN = 128
        NTIMES = 10
        TOL = 1e-7
        data = np.ones((NTIMES, NCHAN), dtype=np.complex)
        flags = np.zeros((NTIMES, NCHAN), dtype=np.bool)
        sigma = 0.1 # Noise level (not important here)
        
        # Fourier coeffs for input data, ordered from (-nmax, nmax)
        cn = np.array([-0.1-0.1j, -0.1+0.1j, -0.3-0.01j, 
                        0.5+0.01j, 
                       -0.3-0.01j, -0.1+0.1j, 0.1-0.1j])
        data *= np.atleast_2d( dspec.fourier_model(cn, NCHAN) )
        
        # Estimate smooth Fourier model on unflagged data
        bf_model, cn_out, data_out = dspec.delay_filter_leastsq(data, flags, 
                                                                sigma, nmax=3, 
                                                                add_noise=False)
        np.testing.assert_allclose(data.real, bf_model.real, atol=NCHAN*TOL)
        
        
        # Estimate smooth Fourier model on data with some flags
        flags[:,10] = True
        flags[:,65:70] = True
        bf_model, cn_out, data_out = dspec.delay_filter_leastsq(data, flags, 
                                                                sigma, nmax=3, 
                                                                add_noise=False)
        np.testing.assert_allclose(data, bf_model, atol=NCHAN*TOL)
        np.testing.assert_allclose(data, data_out, atol=NCHAN*TOL)
        
        # Test 1D code directly
        bf_model, cn_out, data_out = dspec.delay_filter_leastsq_1d(
                                                    data[0], flags[0], sigma, 
                                                    nmax=3, add_noise=False)
        np.testing.assert_allclose(data[0], bf_model, atol=NCHAN*TOL)
        
        # Test 1D code with non-linear leastsq
        bf_model, cn_out, data_out = dspec.delay_filter_leastsq_1d(
                                                    data[0], flags[0], sigma, 
                                                    nmax=3, add_noise=False, 
                                                    use_linear=False)
        np.testing.assert_allclose(data[0], bf_model, atol=NCHAN*TOL)
        
        # Test that noise injection can be switched on
        bf_model, cn_out, data_out = dspec.delay_filter_leastsq_1d(
                                                    data[0], flags[0], sigma, 
                                                    nmax=3, add_noise=True)
        
    
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


def test_vis_filter():
    # load file
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, "zen.all.xx.LST.1.06964.uvA"))
    # get data, wgts
    d = uvd.get_data(24, 25)
    w = (~uvd.get_flags(24, 25)).astype(np.float)
    bl_len = 14.6 / 2.99e8
    sdf = np.median(np.diff(uvd.freq_array.squeeze()))
    # basic execution
    mdl, res, info = dspec.delay_filter(d, w, bl_len, sdf, standoff=50.0, horizon=1.0, min_dly=0.0,
                                        tol=1e-4, window='blackman-harris', skip_wgt=0.1, gain=0.1)
    nt.assert_equal(mdl.shape, (6, 1024))
    nt.assert_equal(res.shape, (6, 1024))


if __name__ == '__main__':
    unittest.main()
