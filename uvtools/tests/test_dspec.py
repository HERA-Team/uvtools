import unittest
import uvtools.dspec as dspec
import numpy as np, random
import nose.tools as nt
from pyuvdata import UVData
from uvtools.data import DATA_PATH
import os
import warnings
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

    def test_fourier_model(self):
        NMAX = 7
        NFREQS = 100
        nmodes = 2*NMAX + 1
        cn = (np.arange(nmodes) + 1.j*np.arange(nmodes)) / float(nmodes)
        model = dspec.fourier_model(cn, NFREQS)

        # Test shape of output model
        self.assertEqual((NFREQS,), model.shape)

        # Test errors
        nt.assert_raises(ValueError, dspec.fourier_model, 3, NFREQS)
        nt.assert_raises(ValueError, dspec.fourier_model, np.empty((3, 3)), NFREQS)

    def test_delay_filter_leastsq(self):
        NCHAN = 128
        NTIMES = 10
        TOL = 1e-7
        data = np.ones((NTIMES, NCHAN), dtype=np.complex)
        flags = np.zeros((NTIMES, NCHAN), dtype=np.bool)
        sigma = 0.1 # Noise level (not important here)

        # Fourier coeffs for input data, ordered from (-nmax, nmax)
        cn = np.array([-0.1-0.1j, -0.1+0.1j, -0.3-0.01j, 0.5+0.01j,
                       -0.3-0.01j, -0.1+0.1j, 0.1-0.1j])
        data *= np.atleast_2d( dspec.fourier_model(cn, NCHAN) )

        # Estimate smooth Fourier model on unflagged data
        bf_model, cn_out, data_out = dspec.delay_filter_leastsq(data, flags,
                                                                sigma, nmax=3,
                                                                add_noise=False)
        np.testing.assert_allclose(data, bf_model, atol=NCHAN*TOL)
        np.testing.assert_allclose(cn, cn_out[0], atol=1e-6)

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
            data[0], flags[0], sigma, nmax=3, add_noise=False)
        np.testing.assert_allclose(data[0], bf_model, atol=NCHAN*TOL)

        # Test 1D code with non-linear leastsq
        bf_model, cn_out, data_out = dspec.delay_filter_leastsq_1d(
            data[0], flags[0], sigma, nmax=3, add_noise=False, use_linear=False)
        np.testing.assert_allclose(data[0], bf_model, atol=NCHAN*TOL)

        # Test that noise injection can be switched on
        bf_model, cn_out, data_out = dspec.delay_filter_leastsq_1d(
            data[0], flags[0], sigma, nmax=3, add_noise=True)
        np.testing.assert_allclose(data[0], bf_model, atol=NCHAN * TOL * sigma)

        # Test with a noise array
        sigma_array = sigma * np.ones_like(data[0])
        bf_model, cn_out, data_out = dspec.delay_filter_leastsq_1d(
            data[0], flags[0], sigma_array, nmax=3, add_noise=True)
        np.testing.assert_allclose(data[0], bf_model, atol=NCHAN * TOL * sigma)

        # Test errors
        nt.assert_raises(ValueError, dspec.delay_filter_leastsq_1d,
                         data[0], flags[0], sigma, nmax=3, operator=np.empty((3, 3)))
        nt.assert_raises(ValueError, dspec.delay_filter_leastsq_1d,
                         data[0], flags[0], sigma, nmax=3, cn_guess=np.array([3]))

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
        np.testing.assert_allclose(dres[0,:], (data * wgts)[0,:], atol=NCHAN*TOL)
        self.assertEqual(len(info), NTIMES)
        self.assertTrue(info[0]['skipped'])

    def test_calc_width(self):
        # test single filter_size
        nchan = 100
        dt = 10.
        filter_size = 1e-2
        u, l = dspec.calc_width(filter_size, dt, nchan)
        frs = np.fft.fftfreq(nchan, dt)  # negative b/c of ifft convention
        nt.assert_true(np.all(np.abs(frs[u:l]) > filter_size))

        # test multiple entries in filter_size
        filter_size = (1e-2, 2e-2)
        u, l = dspec.calc_width(filter_size, dt, nchan)
        nt.assert_true(np.all((frs[u:l] < -1e-2) | (frs[u:l] > 2e-2)))

    def test_gen_window(self):
        for w in ['none', 'blackmanharris', 'hann', 'tukey', 'barthann', 'blackmanharris-7term',
                  'cosinesum-9term', 'cosinesum-11term']:
            win = dspec.gen_window(w, 100)
            nt.assert_true(len(win), 100)
            nt.assert_true(isinstance(win, np.ndarray))
            nt.assert_true(win.min() >= 0.0)
            nt.assert_true(win.max() <= 1.0)
            nt.assert_raises(ValueError, dspec.gen_window, w, 100, normalization='foo')
            win2 = dspec.gen_window(w, 100,normalization='mean')
            nt.assert_true(np.all(np.isclose(win, win2*np.mean(win),atol=1e-6)))
            win3 = dspec.gen_window(w, 100,normalization='rms')
            nt.assert_true(np.all(np.isclose(win, win3*np.sqrt(np.mean(win**2.)),atol=1e-6)))

        nt.assert_raises(ValueError, dspec.gen_window, 'foo', 200)

def test_dayenu_filter():
    nf = 100
    df = 100e3
    freqs = np.arange(-nf//2, nf//2) * df
    #generate some noise
    noise = (np.random.randn(nf) + 1j * np.random.randn(nf))/np.sqrt(2.)
    #a foreground tone and a signal tone
    fg_tone = 1e4 * np.exp(2j * np.pi * freqs * 50e-9)
    sg_tone = 1e2 * np.exp(2j * np.pi * freqs * 1000e-9)
    fg_ns = noise + fg_tone
    fg_sg =  fg_tone + sg_tone
    data_1d = fg_ns
    data_2d = np.array([data_1d, data_1d])
    filter_centers = [0.]
    filter_half_widths = [200e-9]
    filter_factors = [1e-9]
    wghts_1d = np.ones(nf)
    wghts_2d = np.array([wghts_1d, wghts_1d])
    #test functionality for numpy arrays
    dspec.dayenu_filter(data_1d, wghts_1d, [1], np.array(filter_centers), np.array(filter_half_widths),
                        np.array(filter_factors), delta_data=df)
    #provide filter_dimensions as an integer.
    dspec.dayenu_filter(data_1d, wghts_1d, 1, np.array(filter_centers), np.array(filter_half_widths),
                        np.array(filter_factors), delta_data=df)

    #test functionality on floats
    dspec.dayenu_filter(data_1d, wghts_1d,  1, filter_centers[0], filter_half_widths[0],
                        filter_factors[0],delta_data=df)
    filter_half_widths2 = [200e-9, 200e-9]
    filter_centers2 = [0., -1400e-9]
    filter_factors2 = [1e-9, 1e-9]
    #check if throws error when number of filter_half_widths not equal to len filter_centers
    nt.assert_raises(ValueError, dspec.dayenu_filter, data_1d, wghts_1d, [1], filter_centers,
                    filter_half_widths2, filter_factors, delta_data=df)
    #check if throws error when number of filter_half_widths not equal to len filter_factors
    nt.assert_raises(ValueError, dspec.dayenu_filter, data_1d, wghts_1d, 1, filter_centers,
                    filter_half_widths, filter_factors2, delta_data=df)
    #check if error thrown when wghts have different length then data
    nt.assert_raises(ValueError, dspec.dayenu_filter, data_1d, wghts_1d[:-1], 1, filter_centers,
                    filter_half_widths, filter_factors, delta_data=df)
    #check if error thrown when dimension of data does not equal dimension of weights.
    nt.assert_raises(ValueError, dspec.dayenu_filter, data_1d, wghts_2d, 1, filter_centers,
                    filter_half_widths, filter_factors, delta_data=df)
    #check if error thrown if dimension of data does not equal 2 or 1.
    nt.assert_raises(ValueError, dspec.dayenu_filter, np.zeros((10,10,10)), wghts_1d, 1, filter_centers,
                    filter_half_widths, filter_factors, delta_data=df)
    #check if error thrown if dimension of weights does not equal 2 or 1.
    nt.assert_raises(ValueError, dspec.dayenu_filter, wghts_1d, np.zeros((10,10,10)), 1, filter_centers,
                    filter_half_widths, filter_factors, delta_data=df)
    #now filter foregrounds and test that std of residuals are close to std of noise:
    filtered_noise, _ =  dspec.dayenu_filter(data_1d, wghts_1d, [1], filter_centers, filter_half_widths,
                                         filter_factors, delta_data=df)
    #print(np.std((data_1d - fg_tone).real)*np.sqrt(2.))
    #print(np.std((filtered_noise).real)*np.sqrt(2.))
    np.testing.assert_almost_equal( np.std(filtered_noise.real)**2. + np.std(filtered_noise.imag)**2.,
                                  np.std(noise.real)**2. + np.std(noise.imag)**2., decimal = 0)
    #now filter foregrounds and signal and test that std of residuals are close to std of signal.
    filtered_signal, _ =  dspec.dayenu_filter(fg_sg, wghts_1d, [1], filter_centers, filter_half_widths,
                                              filter_factors, delta_data=df)
    np.testing.assert_almost_equal( (np.std(filtered_signal.real)**2. + np.std(filtered_signal.imag)**2.)/1e4,
                                  (np.std(sg_tone.real)**2. + np.std(sg_tone.imag)**2.)/1e4, decimal = 0)
    #Next, we test performing a fringe-rate clean. Generate a 50-meter EW baseline with a single
    #source moving overhead perpindicular to baseline
    TEST_CACHE = {}
    OMEGA_EARTH = 2. * np.pi / 3600. / 24.
    times = np.linspace(-1800, 1800., nf, endpoint = False)
    dt = times[1]-times[0]
    freqs = np.linspace(145e6, 155e6, nf, endpoint=False)
    fg, tg = np.meshgrid(freqs,times)
    signal_2d = 1e6 * np.exp(2j * np.pi * 50. / 3e8 * np.sin(OMEGA_EARTH * tg) * fg)
    noise_2d = np.random.randn(nf,nf)/np.sqrt(2.)\
    + 1j*np.random.randn(nf,nf)/np.sqrt(2.)
    data_2d = signal_2d + noise_2d
    #now, only filter fringe-rate domain. The fringe rate for a source
    #overhead should be roughly 0.0036 for this baseline.
    filtered_data_fr, _ = dspec.dayenu_filter(data_2d, np.ones_like(data_2d), delta_data = dt,
                        filter_centers = [0.], filter_half_widths = [0.004], filter_factors = [1e-10],
                        filter_dimensions = [0], cache = TEST_CACHE)

    np.testing.assert_almost_equal(np.sqrt(np.mean(np.abs(filtered_data_fr.flatten())**2.)),
                                    1., decimal = 1)

    #only filter in the delay-domain.

    filtered_data_df, _ = dspec.dayenu_filter(data_2d, np.ones_like(data_2d), delta_data=100e3,
                        filter_centers = [0.], filter_half_widths=[100e-9], filter_factors=[1e-10],
                        filter_dimensions = [1], cache = TEST_CACHE)

    np.testing.assert_almost_equal(np.sqrt(np.mean(np.abs(filtered_data_df.flatten())**2.)),
                                                            1., decimal = 1)
    #supply an int for filter dimensions
    filtered_data_df, _ = dspec.dayenu_filter(data_2d, np.ones_like(data_2d), delta_data=100e3,
                        filter_centers = [0.], filter_half_widths=[100e-9], filter_factors=[1e-10],
                        filter_dimensions = 1, cache = TEST_CACHE)

    np.testing.assert_almost_equal(np.sqrt(np.mean(np.abs(filtered_data_df.flatten())**2.)),
                                    1., decimal = 1)

    #filter in both domains. I use a smaller filter factor
    #for each domain since they multiply in the target region.

    filtered_data_df_fr, _ = dspec.dayenu_filter(data_2d, np.ones_like(data_2d), delta_data = [dt,100e3],
                    filter_centers = [[0.002],[0.]], filter_half_widths = [[0.001],[100e-9]], filter_factors = [[1e-5],[1e-5]],
                    filter_dimensions = [0,1],cache = TEST_CACHE)

    np.testing.assert_almost_equal(np.sqrt(np.mean(np.abs(filtered_data_df_fr.flatten())**2.)),
                                    1., decimal = 1)

    #test error messages if we do not provide lists of lists.
    nt.assert_raises(ValueError,dspec.dayenu_filter,data_2d, np.ones_like(data_2d),
                        delta_data = [dt,100e3],
                        filter_centers = [[0.002],0.],
                        filter_half_widths = [[0.001],[100e-9]],
                        filter_factors = [1e-5,[1e-5]],
                        filter_dimensions = [0,1],cache = {})



    #test linear algebra error
    #wbad = np.ones(32,dtype=complex)
    #wbad[2:30] = 0.
    #d_fail, info_fail = dspec.dayenu_filter(np.zeros(32,dtype=complex), wbad, 1e5, [0.], [32/1e5/32], [1e-9], cache = {},
    #                        filter_dimensions = [False, True])
    #np.testing.assert_array_equal(d_fail, np.zeros_like(d_fail))
    #np.testing.assert_array_equal(np.array(info_fail['skipped_channels']), np.array([0]))

def test_dayenu_filter_user_frequencies():
    nf = 100
    df = 100e3
    freqs = np.arange(-nf//2, nf//2) * df
    noise = (np.random.randn(nf) + 1j * np.random.randn(nf))/np.sqrt(2.)
    #a foreground tone and a signal tone
    fg_tone = 1e4 * np.exp(2j * np.pi * freqs * 50e-9)
    sg_tone = 1e2 * np.exp(2j * np.pi * freqs * 1000e-9)
    fg_ns = noise + fg_tone
    fg_sg =  fg_tone + sg_tone
    filter_centers = [0.]
    filter_half_widths = [200e-9]
    filter_factors = [1e-9]
    wghts_1d = np.ones(nf)
    data_1d = fg_ns

    d1,_ = dspec.dayenu_filter(data_1d, wghts_1d, [1], np.array(filter_centers), np.array(filter_half_widths),
                            np.array(filter_factors), delta_data=df)
    d2,_ = dspec.dayenu_filter(data_1d, wghts_1d, [1], np.array(filter_centers), np.array(filter_half_widths),
                            np.array(filter_factors), user_frequencies=freqs)
    #test whether user provided frequencies gives same answer as non user provided frequencies.
    np.testing.assert_almost_equal(d1, d2)


def test_dayenu_mat_inv():
    cmat = dspec.dayenu_mat_inv(32, 100e3, filter_centers = [], filter_half_widths = [], filter_factors = [])
    #verify that the inverse cleaning matrix without cleaning windows is the identity!
    np.testing.assert_array_equal(cmat, np.identity(32).astype(np.complex128))
    #next, test with a single filter window with list and float arguments supplied
    cmat1 = dspec.dayenu_mat_inv(32, 100e3, filter_centers = 0., filter_half_widths = 112e-9, filter_factors = 1e-9)
    cmat2 = dspec.dayenu_mat_inv(32, 100e3, filter_centers = [0.], filter_half_widths = [112e-9], filter_factors = [1e-9])
    x,y = np.meshgrid(np.arange(-16,16), np.arange(-16,16))
    cmata = np.identity(32).astype(np.complex128) + 1e9 * np.sinc( (x-y) * 100e3 * 224e-9 ).astype(np.complex128)
    np.testing.assert_array_equal(cmat1, cmat2)
    #next test that the array is equal to what we expect
    np.testing.assert_almost_equal(cmat1, cmata)
    #now test no_regularization
    cmat1 = dspec.dayenu_mat_inv(32, 100e3, filter_centers = 0.,
            filter_half_widths = 112e-9, filter_factors = 1, no_regularization = True)
    np.sinc( (x-y) * 100e3 * 224e-9 ).astype(np.complex128)
    np.testing.assert_almost_equal(cmat1, cmata / 1e9)
    #now test wrap!
    cmat1 = dspec.dayenu_mat_inv(32, 100e3, filter_centers = 0.,
            filter_half_widths = 112e-9, filter_factors = 1, wrap = True,
             no_regularization = True)
    cmata = np.zeros_like(cmat1)
    for m in range(-500,500):
        cmata += np.sinc((x-y - 32 * m) * 100e3 * 224e-9)
    np.testing.assert_almost_equal(cmat1, cmata)



def test_vis_filter():
    # load file
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, "zen.2458042.17772.xx.HH.uvXA"), bls=[(24, 25)])

    freqs = uvd.freq_array.squeeze()
    times = np.unique(uvd.time_array) * 24 * 3600
    times -= np.mean(times)
    sdf = np.median(np.diff(freqs))
    dt = np.median(np.diff(times))
    frs = np.fft.fftfreq(uvd.Ntimes, d=dt)
    dlys = np.fft.fftfreq(uvd.Nfreqs, d=sdf) * 1e9

    # simulate some data in fringe-rate and delay space
    np.random.seed(0)
    dfr, ddly = frs[1] - frs[0], dlys[1] - dlys[0]
    d = 200 * np.exp(-2j*np.pi*times[:, None]*(frs[2]+dfr/4) - 2j*np.pi*freqs[None, :]*(dlys[2]+ddly/4)/1e9)
    d += 50 * np.exp(-2j*np.pi*times[:, None]*(frs[20]) - 2j*np.pi*freqs[None, :]*(dlys[20])/1e9)
    d += 10 * ((np.random.normal(0, 1, uvd.Nfreqs * uvd.Ntimes).astype(np.complex) \
         + 1j * np.random.normal(0, 1, uvd.Nfreqs * uvd.Ntimes)).reshape(uvd.Ntimes, uvd.Nfreqs))

    def get_snr(clean, fftax=1, avgax=0, modes=[2, 20]):
        cfft = np.fft.ifft(clean, axis=fftax)
        cavg = np.median(np.abs(cfft), axis=avgax)
        std = np.median(cavg)
        return [cavg[m] / std for m in modes]

    # get snr of modes
    freq_snr1, freq_snr2 = get_snr(d, fftax=1, avgax=0, modes=[2, 20])
    time_snr1, time_snr2 = get_snr(d, fftax=0, avgax=1, modes=[2, 20])
    # simulate some flags
    f = np.zeros_like(d, dtype=np.bool)
    d[:, 20:22] += 1e3
    f[:, 20:22] = True
    d[20, :] += 1e3
    f[20, :] = True
    w = (~f).astype(np.float)
    bl_len = 70.0 / 2.99e8

    # delay filter basic execution
    mdl, res, info = dspec.delay_filter(d, w, bl_len, sdf, standoff=0, horizon=1.0, min_dly=0.0,
                                             tol=1e-4, window='none', skip_wgt=0.1, gain=0.1)
    cln = mdl + res
    # assert recovered snr of input modes
    snrs = get_snr(cln, fftax=1, avgax=0)
    nt.assert_true(np.isclose(snrs[0], freq_snr1, atol=3))
    nt.assert_true(np.isclose(snrs[1], freq_snr2, atol=3))

    # test vis filter is the same
    mdl2, res2, info2 = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, standoff=0, horizon=1.0, min_dly=0.0,
                                               tol=1e-4, window='none', skip_wgt=0.1, gain=0.1)
    nt.assert_true(np.isclose(mdl - mdl2, 0.0).all())

    # fringe filter basic execution
    mdl, res, info = dspec.fringe_filter(d, w, frs[15], dt, tol=1e-4, window='none', skip_wgt=0.1, gain=0.1)
    cln = mdl + res

    # assert recovered snr of input modes
    snrs = get_snr(cln, fftax=0, avgax=1)
    nt.assert_true(np.isclose(snrs[0], time_snr1, atol=3))
    nt.assert_true(np.isclose(snrs[1], time_snr2, atol=3))

    # test vis filter is the same
    mdl2, res2, info2 = dspec.vis_filter(d, w, max_frate=frs[15], dt=dt, tol=1e-4, window='none', skip_wgt=0.1, gain=0.1)
    cln2 = mdl2 + res2
    nt.assert_true(np.isclose(mdl - mdl2, 0.0).all())

    # try non-symmetric filter
    mdl, res, info = dspec.fringe_filter(d, w, (frs[-20], frs[10]), dt, tol=1e-4, window='none', skip_wgt=0.1, gain=0.1)
    cln = mdl + res

    # assert recovered snr of input modes
    snrs = get_snr(cln, fftax=0, avgax=1)
    nt.assert_true(np.isclose(snrs[0], time_snr1, atol=3))
    nt.assert_true(np.isclose(snrs[1], time_snr2, atol=3))

    # 2d clean
    mdl, res, info = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, max_frate=frs[15], dt=dt, tol=1e-4, window='none', maxiter=100, gain=1e-1)
    cln = mdl + res
    # assert recovered snr of input modes
    snrs = get_snr(cln, fftax=1, avgax=0)
    nt.assert_true(np.isclose(snrs[0], freq_snr1, atol=3))
    nt.assert_true(np.isclose(snrs[1], freq_snr2, atol=3))

    # non-symmetric 2D clean
    mdl, res, info = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, max_frate=(frs[-20], frs[10]), dt=dt, tol=1e-4, window='none', maxiter=100, gain=1e-1)
    cln = mdl + res

    # assert recovered snr of input modes
    snrs = get_snr(cln, fftax=1, avgax=0)
    nt.assert_true(np.isclose(snrs[0], freq_snr1, atol=3))
    nt.assert_true(np.isclose(snrs[1], freq_snr2, atol=3))

    # try plus filtmode on 2d clean
    mdl, res, info = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, max_frate=(frs[10], frs[10]), dt=dt, tol=1e-4, window=('none', 'none'), edgecut_low=(0, 5), edgecut_hi=(2, 5), maxiter=100, gain=1e-1, filt2d_mode='plus')
    mfft = np.fft.ifft2(mdl)
    cln = mdl + res

    # assert clean components fall only in plus area
    clean_comp = np.where(~np.isclose(np.abs(mfft), 0.0))
    for cc in zip(*clean_comp):
        nt.assert_true(0 in cc)

    # exceptions
    nt.assert_raises(ValueError, dspec.vis_filter, d, w, bl_len=bl_len, sdf=sdf, max_frate=(frs[-20], frs[10]), dt=dt, filt2d_mode='foo')

    # test add_clean_residual: test res of filtered modes are lower when add_residual is True
    mdl, res, info = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, max_frate=frs[15], dt=dt, tol=1e-6, window='none', maxiter=100, gain=1e-1, add_clean_residual=False)
    mdl2, res2, info = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, max_frate=frs[15], dt=dt, tol=1e-6, window='none', maxiter=100, gain=1e-1, add_clean_residual=True)
    rfft = np.fft.ifft2(res)
    rfft2 = np.fft.ifft2(res2)
    nt.assert_true(np.median(np.abs(rfft2[:15, :23] / rfft[:15, :23])) < 1)

def test_delay_interpolation_matrix():
    """
    Test inerpolation matrix.

    Test creates some simple data with a few underlying delays.
    Flagged data is generated from underlying data.
    The flagged channels are interpolated over using the delay_interpolation_matrix
    and the interpolation is compared to the original.

    """
    MYCACHE={}
    fs = np.arange(-10,10)
    #here is some underlying data
    data = np.exp(2j * np.pi * 3/20. * fs) + 5*np.exp(2j * np.pi * 4./20 * fs)
    data += np.exp(-2j * np.pi * 3/20. * fs) + 5*np.exp(2j * np.pi * -4/20. * fs)
    #here are some weights with flags
    wgts = np.ones_like(data)
    wgts[6] = 0
    wgts[17] = 0
    dw = data*wgts
    #here is some flagged data.
    #interpolate data and see if it matches true data.
    data_interp = np.dot(dspec.delay_interpolation_matrix(nchan=20, ndelay=5, wgts=wgts, fundamental_period=20, cache=MYCACHE), dw)
    #check that interpolated data agrees with original data.
    nt.assert_true( np.all(np.isclose(data_interp, data, atol=1e-6)))
    #test error raising.
    nt.assert_raises(ValueError, dspec.delay_interpolation_matrix, 10, 2, np.ones(5))
    nt.assert_raises(ValueError, dspec.delay_interpolation_matrix, 5, 2, np.asarray([0., 0., 0., 0., 0.]))
    #test diagnostic mode.
    data_interp1, _, _, _ = dspec.delay_interpolation_matrix(nchan=20, ndelay=5, wgts=wgts,
     fundamental_period=20, cache={}, return_diagnostics=True)
    data_interp1 = np.dot(data_interp1, dw)
    nt.assert_true(np.all(np.isclose(data_interp, data_interp1, atol=1e-6)))
     #test warning
    with warnings.catch_warnings(record=True) as w:
        wgtpc = np.ones(100)
        randflags = np.asarray([29, 49,  6, 47, 68, 98, 69, 70, 32,  3]).astype(int)
        wgtpc[randflags]=0.
        amat_pc = dspec.delay_interpolation_matrix(nchan=100, ndelay=25, wgts=wgtpc, fundamental_period=200)
        print(len(w))
        nt.assert_true(len(w) > 0)

def test_vis_filter_dayenu():
    # load file
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, "zen.2458042.17772.xx.HH.uvXA"), bls=[(24, 25)])

    freqs = uvd.freq_array.squeeze()
    times = np.unique(uvd.time_array) * 24 * 3600
    times -= np.mean(times)
    sdf = np.median(np.diff(freqs))
    dt = np.median(np.diff(times))
    frs = np.fft.fftfreq(uvd.Ntimes, d=dt)
    dlys = np.fft.fftfreq(uvd.Nfreqs, d=sdf) * 1e9
    # simulate some data in fringe-rate and delay space
    np.random.seed(0)
    dfr, ddly = frs[1] - frs[0], dlys[1] - dlys[0]
    d = 200 * np.exp(-2j*np.pi*times[:, None]*(frs[2]+dfr/4) - 2j*np.pi*freqs[None, :]*(dlys[2]+ddly/4)/1e9)
    d += 50 * np.exp(-2j*np.pi*times[:, None]*(frs[20]) - 2j*np.pi*freqs[None, :]*(dlys[20])/1e9)
    n = 10 * ((np.random.normal(0, 1, uvd.Nfreqs * uvd.Ntimes).astype(np.complex) \
         + 1j * np.random.normal(0, 1, uvd.Nfreqs * uvd.Ntimes)).reshape(uvd.Ntimes, uvd.Nfreqs))
    d += n
    print(uvd.Nfreqs)
    def get_snr(clean, fftax=1, avgax=0, modes=[2, 20]):
        cfft = np.fft.ifft(clean, axis=fftax)
        cavg = np.median(np.abs(cfft), axis=avgax)
        std = np.median(cavg)
        return [cavg[m] / std for m in modes]

    # get snr of modes
    freq_snr1, freq_snr2 = get_snr(d, fftax=1, avgax=0, modes=[2, 20])
    time_snr1, time_snr2 = get_snr(d, fftax=0, avgax=1, modes=[2, 20])

    # simulate some flags
    f = np.zeros_like(d, dtype=np.bool)
    d[:, 20:22] += 1e3
    f[:, 20:22] = True
    d[20, :] += 1e3
    f[20, :] = True
    w = (~f).astype(np.float)
    bl_len = 70.0 / 2.99e8

    # delay filter basic execution 1d with and without leastsq
    mdl1, res1, info1 = dspec.delay_filter(d[0], w[0], bl_len, sdf, standoff=0, horizon=1.0, min_dly=0.,
                                             tol=1e-8, window='none', skip_wgt=0.1, gain=1e-1, mode='dayenu', deconv_dayenu_foregrounds=True, fg_deconv_method='leastsq')

    mdl2, res2, info2 = dspec.delay_filter(d[0], w[0], bl_len, sdf, standoff=0, horizon=1.0, min_dly=0.,
                                             tol=1e-8, window='none', skip_wgt=0.1, gain=1e-1, mode='dayenu', deconv_dayenu_foregrounds=True, fg_deconv_method='clean')
    #residuals should be same
    nt.assert_true(np.isclose(res1 - res2, 0.0, atol=10).all())

    # delay filter basic execution with leastsq
    mdl, res, info = dspec.delay_filter(d, w, bl_len, sdf, standoff=0, horizon=1.0, min_dly=0.,
                                             tol=1e-8, window='none', skip_wgt=0.1, gain=1e-1, mode='dayenu', deconv_dayenu_foregrounds=True, fg_deconv_method='leastsq')
    cln = mdl + res
    snrs = get_snr(cln, fftax=1, avgax=0)
    nt.assert_true(np.isclose(snrs[0], freq_snr1, atol=4))
    nt.assert_true(np.isclose(snrs[1], freq_snr2, atol=4))
    # delay filter basic execution
    mdl, res, info = dspec.delay_filter(d, w, bl_len, sdf, standoff=0, horizon=1.0, min_dly=0.,
                                             tol=1e-8, window='none', skip_wgt=0.1, gain=1e-1, mode='dayenu', deconv_dayenu_foregrounds=True, fg_deconv_method='clean')
    cln = mdl + res
    snrs = get_snr(cln, fftax=1, avgax=0)
    nt.assert_true(np.isclose(snrs[0], freq_snr1, atol=3))
    nt.assert_true(np.isclose(snrs[1], freq_snr2, atol=3))
    # test vis filter is the same
    mdl2, res2, info2 = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, standoff=0, horizon=1.0, min_dly=0.,
                                               tol=1e-8, window='none', skip_wgt=0.1, gain=0.1, mode='dayenu', deconv_dayenu_foregrounds=True, fg_deconv_method='clean')
    nt.assert_true(np.isclose(mdl - mdl2, 0.0).all())
    # fringe filter basic execution
    mdl, res, info = dspec.fringe_filter(d, w, frs[15] * 1.5, dt, tol=1e-8, window='none', skip_wgt=0.1, gain=0.1, mode='dayenu', deconv_dayenu_foregrounds=True, fg_deconv_method='clean')
    cln = mdl + res
    # assert recovered snr of input modes
    snrs = get_snr(cln, fftax=0, avgax=1)
    nt.assert_true(np.isclose(snrs[0], time_snr1, atol=3))
    nt.assert_true(np.isclose(snrs[1], time_snr2, atol=3))

    # test vis filter is the same
    mdl2, res2, info2 = dspec.vis_filter(d, w, max_frate=frs[15] * 1.5, dt=dt, tol=1e-8, window='none', skip_wgt=0.1, gain=0.1, mode='dayenu', deconv_dayenu_foregrounds=True, fg_deconv_method='clean')
    cln2 = mdl2 + res2
    nt.assert_true(np.isclose(mdl - mdl2, 0.0).all())

    # try non-symmetric filter
    mdl, res, info = dspec.fringe_filter(d, w, (frs[-20]*2, frs[10]*2), dt, tol=1e-8, window='none', skip_wgt=0.1, gain=0.1, mode='dayenu', deconv_dayenu_foregrounds=True, fg_deconv_method='clean')
    cln = mdl + res

    # assert recovered snr of input modes
    snrs = get_snr(cln, fftax=0, avgax=1)
    nt.assert_true(np.isclose(snrs[0], time_snr1, atol=3))
    nt.assert_true(np.isclose(snrs[1], time_snr2, atol=3))
    # 2d clean
    mdl, res, info = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, max_frate=1.5*frs[15], dt=dt, tol=1e-8, window='none', maxiter=100, gain=1e-1, mode='dayenu', filt2d_mode='plus', deconv_dayenu_foregrounds=True, fg_deconv_method='clean')
    cln = mdl + res
    # assert recovered snr of input modes
    snrs = get_snr(cln, fftax=1, avgax=0)
    nt.assert_true(np.isclose(snrs[0], freq_snr1, atol=3))
    nt.assert_true(np.isclose(snrs[1], freq_snr2, atol=3))

    #2d linear clean with least squares fitting of foregrounds
    mdl2, res2, info = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, max_frate=1.5*frs[15], dt=dt, tol=1e-8, window='none', maxiter=100, gain=1e-1, mode='dayenu', filt2d_mode='plus', deconv_dayenu_foregrounds=True, fg_deconv_method='leastsq')
    #compare residuals
    nt.assert_true(np.isclose(res - res2, 0.0).all())


    # non-symmetric 2D clean
    mdl, res, info = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, max_frate=(frs[-20], frs[10]), dt=dt, tol=1e-8, window='none', maxiter=100, gain=1e-1, mode='dayenu', filt2d_mode='plus', deconv_dayenu_foregrounds=True, fg_deconv_method='clean')
    cln = mdl + res
    # assert recovered snr of input modes
    snrs = get_snr(cln, fftax=1, avgax=0)
    nt.assert_true(np.isclose(snrs[0], freq_snr1, atol=3))
    nt.assert_true(np.isclose(snrs[1], freq_snr2, atol=3))


    # try plus filtmode on 2d clean
    mdl, res, info = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, max_frate=(frs[10], frs[10]), dt=dt, tol=1e-8, window=('none', 'none'), edgecut_low=(0, 5), edgecut_hi=(2, 5), maxiter=100, gain=1e-1, filt2d_mode='plus' ,mode='dayenu', deconv_dayenu_foregrounds=True, fg_deconv_method='clean')
    mfft = np.fft.ifft2(mdl)
    cln = mdl + res

    # exceptions
    nt.assert_raises(ValueError, dspec.vis_filter, d, w, bl_len=bl_len, sdf=sdf, max_frate=(frs[-20], frs[10]), dt=dt, filt2d_mode='foo')
    nt.assert_raises(ValueError, dspec.vis_filter, d, w, bl_len=bl_len, sdf=sdf, max_frate=(frs[-20],frs[10]),dt=dt, mode='foo')

def test_vis_filter_dft_interp():
    # load file
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, "zen.2458042.17772.xx.HH.uvXA"), bls=[(24, 25)])

    freqs = uvd.freq_array.squeeze()
    times = np.unique(uvd.time_array) * 24 * 3600
    times -= np.mean(times)
    sdf = np.median(np.diff(freqs))
    dt = np.median(np.diff(times))
    frs = np.fft.fftfreq(uvd.Ntimes, d=dt)
    dlys = np.fft.fftfreq(uvd.Nfreqs, d=sdf) * 1e9
    # simulate some data in fringe-rate and delay space
    np.random.seed(0)
    dfr, ddly = frs[1] - frs[0], dlys[1] - dlys[0]
    d = 200 * np.exp(-2j*np.pi*times[:, None]*(frs[2]+dfr/4) - 2j*np.pi*freqs[None, :]*(dlys[2]+ddly/4)/1e9)
    d += 50 * np.exp(-2j*np.pi*times[:, None]*(frs[20]) - 2j*np.pi*freqs[None, :]*(dlys[20])/1e9)
    n = 10 * ((np.random.normal(0, 1, uvd.Nfreqs * uvd.Ntimes).astype(np.complex) \
         + 1j * np.random.normal(0, 1, uvd.Nfreqs * uvd.Ntimes)).reshape(uvd.Ntimes, uvd.Nfreqs))
    d += n
    print(uvd.Nfreqs)
    def get_snr(clean, fftax=1, avgax=0, modes=[2, 20]):
        cfft = np.fft.ifft(clean, axis=fftax)
        cavg = np.median(np.abs(cfft), axis=avgax)
        std = np.median(cavg)
        return [cavg[m] / std for m in modes]

    # get snr of modes
    freq_snr1, freq_snr2 = get_snr(d, fftax=1, avgax=0, modes=[2, 20])
    time_snr1, time_snr2 = get_snr(d, fftax=0, avgax=1, modes=[2, 20])

    # simulate some flags
    f = np.zeros_like(d, dtype=np.bool)
    d[:, 20:22] += 1e3
    f[:, 20:22] = True
    d[20, :] += 1e3
    f[20, :] = True
    w = (~f).astype(np.float)
    bl_len = 70.0 / 2.99e8


    # delay filter basic execution with leastsq
    mdl, res, info = dspec.delay_filter(d, w, bl_len, sdf, standoff=0, horizon=1.0, min_dly=0.,
                                             tol=1e-8, window='none', skip_wgt=0.1, gain=1e-1, mode='dft_interp',
                                             deconv_dayenu_foregrounds=True, fg_deconv_method='leastsq')
    cln = mdl + res

    snrs = get_snr(cln, fftax=1, avgax=0)

    nt.assert_true(np.isclose(snrs[0], freq_snr1, atol=4))
    nt.assert_true(np.isclose(snrs[1], freq_snr2, atol=4))
    # delay filter basic execution
    mdl, res, info = dspec.delay_filter(d, w, bl_len, sdf, standoff=0, horizon=1.0, min_dly=0.,
                                             tol=1e-8, window='none', skip_wgt=0.1, gain=1e-1,
                                             mode='dft_interp', deconv_dayenu_foregrounds=True, fg_deconv_method='leastsq')
    cln = mdl + res
    snrs = get_snr(cln, fftax=1, avgax=0)
    nt.assert_true(np.isclose(snrs[0], freq_snr1, atol=4))
    nt.assert_true(np.isclose(snrs[1], freq_snr2, atol=4))
    # test vis filter is the same
    mdl2, res2, info2 = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, standoff=0, horizon=1.0, min_dly=0.,
                                               tol=1e-8, window='none', skip_wgt=0.1, gain=0.1, mode='dft_interp',
                                               deconv_dayenu_foregrounds=True, fg_deconv_method='leastsq')
    nt.assert_true(np.isclose(mdl - mdl2, 0.0).all())
    # fringe filter basic execution
    mdl, res, info = dspec.fringe_filter(d, w, frs[15] * 1.5, dt, tol=1e-8, window='none', skip_wgt=0.1, gain=0.1, mode='dft_interp')
    cln = mdl + res
    # assert recovered snr of input modes
    snrs = get_snr(cln, fftax=0, avgax=1)
    nt.assert_true(np.isclose(snrs[0], time_snr1, atol=3))
    nt.assert_true(np.isclose(snrs[1], time_snr2, atol=3))

    # test vis filter is the same
    mdl2, res2, info2 = dspec.vis_filter(d, w, max_frate=frs[15] * 1.5, dt=dt, tol=1e-8, window='none', skip_wgt=0.1, gain=0.1, mode='dft_interp')
    cln2 = mdl2 + res2
    nt.assert_true(np.isclose(mdl - mdl2, 0.0).all())

    # try non-symmetric filter
    mdl, res, info = dspec.fringe_filter(d, w, (frs[-20], frs[10]), dt, tol=1e-8, window='none', skip_wgt=0.1, gain=0.1, mode='dft_interp')
    cln = mdl + res

    # assert recovered snr of input modes
    snrs = get_snr(cln, fftax=0, avgax=1)
    nt.assert_true(np.isclose(snrs[0], time_snr1, atol=4))
    nt.assert_true(np.isclose(snrs[1], time_snr2, atol=4))

    # exceptions
    nt.assert_raises(ValueError, dspec.vis_filter, d, w, bl_len=bl_len,
                    sdf=sdf, max_frate=1.5*frs[15], dt=dt,
                    tol=1e-8, window='none', maxiter=100,
                    gain=1e-1, mode='dft_interp',
                    filt2d_mode='plus')



if __name__ == '__main__':
    unittest.main()
