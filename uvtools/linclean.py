# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 The HERA Collaboration

from __future__ import print_function, division, absolute_import
import numpy as np
import astropy.constants as const
from astropy import units


def tau_select(tau, bl_len, tau_pad=160.):
    """
    Function for selecting the tau values in the delay transform.

    Arguments
    ====================
    tau -- array of initial delay values, in ns
    bl_len -- baseline length, in m
    tau_pad -- amount to increase maximum tau value, in ns

    Returns
    ====================
    taus -- array of delays to use in delay transform, in ns
    imax -- index corresponding to maximum delay value
    """
    if not isinstance(bl_len, units.quantity.Quantity):
        # assume bl_len is in meters
        bl_len *= units.m
    tau_max = (bl_len / const.c).to(units.ns).value
    imax = np.where((tau > 0) * (tau <= tau_max + tau_pad))[0].max()
    taus = tau[1:imax + 1]
    return taus, imax


def build_A(taus, f):
    """
    Construct matrix of coefficients to solve through a linear fit.

    Arguments
    ====================
    taus -- array of delays provided by tau_select function, in ns
    f -- array of frequencies of corresponding delays, in GHz

    Returns
    ====================
    A -- matrix of coefficients to solve via a linear fit.
    """
    nfreq = len(f)
    arg = 2 * np.pi * np.outer(f, taus)
    A = np.concatenate((np.ones([nfreq, 1]), np.cos(arg), np.sin(arg)), axis=1)
    return A


def linCLEAN(wf, fl, bl_len, freq_min, freq_max, tau_pad=160.):
    """
    Run the LinCLEAN algorithm to find large-scale foregrounds in a waterfall.

    Arguments
    ====================
    wf -- two-dimensional array of complex visibilites of size Ntimes x Nfreq (a waterfall)
    fl -- two-dimensional array of corresponding flags
    bl_len -- baseline length, in m
    freq_min -- minimum frequency of waterfall, in GHz
    freq_max -- maximum frequency of waterfall, in GHz
    tau_pad -- amount to increase maximum tau value, in ns

    Returns
    ====================
    model -- two-dimensional array of best-fit foreground (low delay) model of data
    resid -- residual (waterfall - model)
    """
    model = np.zeros_like(wf)
    ntimes, nfreq = wf.shape

    # make array of frequencies
    # make smallest frequency the input minimum + fundamental mode
    fmin = freq_min + freq_min / (2. * nfreq)
    f, df = np.linspace(fmin, freq_max, num=nfreq, retstep=True)

    # build array of tau values from frequencies
    tau = np.fft.fftfreq(nfreq, d=df)
    taus, imax = tau_select(tau, bl_len, tau_pad)

    for i in range(ntimes):
        # solve the matrix for each spectra
        whf = np.where(np.logical_not(fl[i, :]))[0]
        f_tofit = f[whf]
        A = build_A(taus, f_tofit)

        # solve for real and imaginary components separately
        xreal, residuals, rank, singular = np.linalg.lstsq(A, wf[i, whf].real)
        ximag, residuals, rank, singular = np.linalg.lstsq(A, wf[i, whf].imag)
        model[i, whf] = np.dot(A, xreal) + 1j * np.dot(A, ximag)
    resid = wf - model

    return model, resid


