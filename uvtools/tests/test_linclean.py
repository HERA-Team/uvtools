"""Tests for linclean.py"""

from __future__ import division
import nose.tools as nt
import numpy as np
from astropy import units
import uvtools.linclean as linclean

class TestMethods(object):
    def setUp(self):
        self.nfreq = 1024
        self.freq_min = 0.1  # GHz
        self.freq_max = 0.2
        fmin = self.freq_min + self.freq_min / (2 * self.nfreq)
        f, df = np.linspace(fmin, self.freq_max, num=self.nfreq, retstep=True)
        self.freqs = f
        self.tau = np.fft.fftfreq(self.nfreq, d=df)
        self.bl_len = 14.
        return

    def test_tau_select(self):
        # get array of tau values
        taus, imax = linclean.tau_select(self.tau, self.bl_len, tau_pad=160.)

        # make sure taus is the right size
        nt.assert_equal(len(taus), imax)

        # check for specific values too
        nt.assert_true(np.isclose(taus[0], 9.995, atol=1e-3))

        return

    def test_build_A(self):
        # get tau values
        taus, imax = linclean.tau_select(self.tau, self.bl_len, tau_pad=160.)

        f = self.freqs
        A = linclean.build_A(taus, f)

        # make sure matrix is the right shape
        nt.assert_equal(A.shape, (1024, 41))

        return

    def test_linCLEAN(self):
        # build fake waterfall
        ntimes = 60
        wf = np.zeros((ntimes, self.nfreq), dtype=np.complex128)
        for i in range(ntimes):
            # build a simple model for the foregrounds
            wf[i, :] = np.cos(self.freqs * self.tau) \
                       + 1j * np.sin(self.freqs * self.tau)
        flags = np.zeros_like(wf, dtype=np.bool)
        model, res = linclean.linCLEAN(wf, flags, self.bl_len, self.freq_min, self.freq_max,
                                       tau_pad=160.)

        # make sure the sizes match
        nt.assert_equal(model.shape, res.shape)

        return
