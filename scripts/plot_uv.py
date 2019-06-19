#!/usr/bin/env python

"""
Creates waterfall plots for visibility datasets.  Can tile multiple plots
on one window, or plot just a single baseline.  When taking the delay
transform (-d), channels are interepreted as selecting delays after the
the transform operation.  Similarly, the time select (-t) will be interpreted
as selecting fringe rates if the fringe-rate transform (-f) has been selected.
In both cases, the ranges specified are intepreted to be the units of the
output plot (i.e. as specified by --chan_axis and --time_axis).
"""

from __future__ import print_function, division, absolute_import

import aipy as a, numpy as np, sys, optparse
import uvtools
import pyuvdata
import hera_cal
from matplotlib import pylab as plt

# XXX TODO: support chan and time indexing

o = optparse.OptionParser()
o.set_usage('plot_uv.py [options] *.uv')
o.set_description(__doc__)
a.scripting.add_standard_options(o, ant=True, chan=True,
    cmap=True, max=True, drng=True)
o.add_option('-m', '--mode', dest='mode', default='log',
    help='Plot mode can be log (logrithmic), lin (linear), phs (phase), real, or imag.')
#o.add_option('-t', '--time', dest='time', default='all', help='Select which time sample to plot. Options are: "all" (default), "<time1 #>_<time2 #>" (a range of times to plot), or "<time1 #>,<time2 #>" (a list of times to plot). If "all" or a range are selected, a 2-d image will be plotted. If a list of times is selected an xy plot will be generated.')
o.add_option('-u', '--unmask', dest='unmask', action='store_true',
    help='Plot masked data, too.')
o.add_option('-o', '--out_file', dest='out_file', default='',
    help='If provided, will save the figure to the specified file instead of popping up a window.')
o.add_option('--time_axis', dest='time_axis', default='jd',
    help='Choose time axis to be integration index (cnt), julian date (jd) or local sidereal time (lst).  Default is jd.')
o.add_option('--freq_axis', dest='freq_axis', default='freq',
    help='Choose specral axis to be channel index (chan) or frequency (freq).  Default is freq.')
o.add_option('--nolegend', dest='nolegend', action='store_true',
    help='Omit legend in last plot.')
o.add_option('--share', dest='share', action='store_true',
    help='Share plots in a single frame.')
o.add_option('--xlim', dest='xlim',
    help='Limits on the x axis (channel/delay) for plotting.')
o.add_option('--ylim', dest='ylim',
    help='Limits on the x axis (time/delay-rate) for plotting.')
o.add_option('--plot_each', dest='plot_each',
    help='Instead of a waterfall plot, plot each of the specified axis (chan,time)')

FILETYPES = ('uvh5', 'miriad', 'uvfits')

YLABELS = {'cnt': 'Time (integrations)',
           'lst': 'Local Sidereal Time (radians)',
           'jd' : 'Time (Julian Date)',
}

XLABELS = {'chan': 'Frequency (channel)',
           'freq': 'Frequency (Hz)',
}

def parse_ants(antstr):
    rv = [s.split('_') for s in antstr.split(',')]
    rv = [(int(i[:-1]), int(j[:-1]), i[-1]+j[-1]) for i,j in rv]
    return rv

def convert_arg_range(arg):
    """Split apart command-line lists/ranges into a list of numbers."""
    arg = arg.split(',')
    return [map(float, option.split('_')) for option in arg]

def gen_times(timeopt, uv, coords, decimate):
    if timeopt == 'all':
        def time_selector(t, cnt): return True
    else:
        timeopt = convert_arg_range(timeopt)
        if len(timeopt[0]) != 1:
            def time_selector(t, cnt):
                if coords == 'index': t = cnt
                for opt in timeopt:
                    if (t >= opt[0]) and (t < opt[1]): return True
                return False
        else:
            timeopt = [opt[0] for opt in timeopt]
            inttime = uv['inttime'] / a.const.s_per_day * decimate
            def time_selector(t, cnt):
                if coords == 'index': return cnt in timeopt
                for opt in timeopt:
                    if (t >= opt) and (t < opt + inttime): return True
                return False
    return time_selector

def data_mode(data, mode='abs'):
    if mode.startswith('phs'): data = np.angle(data.filled(0))
    elif mode.startswith('lin'):
        data = np.ma.absolute(data.filled(0))
        data = np.ma.masked_less_equal(data, 0)
    elif mode.startswith('real'): data = data.real
    elif mode.startswith('imag'): data = data.imag
    elif mode.startswith('log'):
        data = np.ma.absolute(data.filled(0))
        data = np.ma.masked_less_equal(data, 0)
        data = np.ma.log10(data)
    else: raise ValueError('Unrecognized plot mode.')
    return data

opts, args = o.parse_args(sys.argv[1:])

# Parse command-line options
cmap = plt.get_cmap(opts.cmap)
if not opts.xlim is None:
    opts.xlim = map(float, opts.xlim.split('_'))
if not opts.ylim is None:
    opts.ylim = map(float, opts.ylim.split('_'))
is_chan_range, is_time_range = True, True
if opts.plot_each == 'chan': is_chan_range = False
elif opts.plot_each == 'time': is_time_range = False
#time_sel = gen_times(opts.time, uv, opts.time_axis, opts.decimate)
#inttime = uv['inttime'] * opts.decimate

# Loop through UV files collecting relevant data
plot_f = {}
plot_t = {'jd':[], 'lst':[]}

# Hold plotting handles
plots = {}
plt_data = {}

bls = parse_ants(opts.ant)
data, flgs = [], []

intcnt = 0
for filecnt, uvfile in enumerate(args):
    print('Reading', uvfile)
    if filecnt == 0:
        for filetype in FILETYPES:
            try:
                uvf = hera_cal.io.HERAData(uvfile, filetype=filetype)
                break
            except(IOError):
                continue
    else:
        uvf = hera_cal.io.HERAData(uvfile, filetype=filetype)
    dat, flg, _ = uvf.read(bls)
    data.append(dat); flgs.append(flg) 
    meta = uvf.get_metadata_dict()
    plot_f['freq'] = meta['freqs'].flatten()
    plot_f['chan'] = np.arange(meta['freqs'].size)
    plot_t['lst'].append(meta['lsts'])
    plot_t['jd'].append(meta['times'])
if len(data) > 1:
    data = data[0].concatenate(data[1:])
    flgs = flgs[0].concatenate(flgs[1:])
    plot_t = {k:np.concatenate(v) for k,v in plot_t.items()}
else:
    data = data[0]
    flgs = flgs[0]
    plot_t = {k:v[0] for k,v in plot_t.items()}
plot_t['cnt'] = np.arange(plot_t['jd'].size)

def sort_func(a, b):
    ai,aj,pa = map(int, a.split(','))
    bi,bj,pb = map(int, b.split(','))
    if bi > ai or (bi == ai and bj > aj) or (bi == ai and bj == aj and pb < pa): return -1
    return 1

#import IPython; IPython.embed()
bls = data.keys()
bls.sort(cmp=sort_func)
if len(bls) == 0:
    print('No data to plot.')
    sys.exit(0)
m2 = int(np.sqrt(len(bls)))
m1 = int(np.ceil(float(len(bls)) / m2))

# Generate all the plots
dmin,dmax = None, None
fig = plt.figure()
for cnt, bl in enumerate(bls):
    d,f = data[bl], flgs[bl]
    if not opts.unmask:
        d = np.where(f, np.nan, d)
    plt_data[cnt+1] = d
    d = uvtools.plot.data_mode(d, opts.mode)
    if not opts.share:
        plt.subplot(m2, m1, cnt+1)
        dmin,dmax = None,None
        label = ''
    else:
        label = str(bl)
    if is_chan_range and is_time_range: # Need to plot a waterfall
        t = plot_t[opts.time_axis]
        step = np.median(np.diff(t))
        t1,t2 = t[0]-0.5*step, t[-1]+0.5*step
        ylabel = YLABELS[opts.time_axis]
        f = plot_f[opts.freq_axis]
        step = np.median(np.diff(f))
        f1,f2 = f[0]-0.5*step, f[-1]+0.5*step
        xlabel = XLABELS[opts.freq_axis]
        plots[cnt+1] = uvtools.plot.waterfall(d, extent=(f1,f2,t2,t1), cmap=cmap, mode=opts.mode, mx=opts.max, drng=opts.drng)
        plt.colorbar(shrink=0.5)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if not opts.xlim == None:
            plt.xlim(*opts.xlim)
        if not opts.ylim == None:
            plt.ylim(opts.ylim[1],opts.ylim[0]) # Reverse b/c otherwise ylim flips origin for unknown reasons

    elif is_chan_range and not is_time_range:
        plot_chans = plot_f[opts.freq_axis]
        xlabel = XLABELS[opts.freq_axis]
        if opts.time_axis == 'cnt':
            if cnt == 0: plot_t = plot_t['cnt']
            label += '#%d'
        else:
            if cnt == 0: plot_t = plot_t['jd']
            label += 'jd%f'
        for ti,t in enumerate(plot_t):
            plt.plot(plot_chans, d[ti,:], '-', label=label % t)
        plt.xlabel(xlabel)
        if not opts.xlim == None:
            plt.xlim(*opts.xlim)
        if not opts.ylim == None:
            plt.ylim(*opts.ylim)

    elif not is_chan_range and is_time_range:
        plot_times = plot_t[opts.time_axis]
        xlabel = YLABELS[opts.time_axis] # Y/X mismatch on purpose
        if opts.chan_axis == 'cnt':
            chans = plot_f['chan']
            label += '#%d'
        else:
            chans = plot_f['freq']
            label += '%f GHz'
        for c, chan in enumerate(chans):
            plt.plot(plot_times, d[:,c], '-', label=label % chan)
        plt.xlabel(xlabel)
        if not opts.xlim == None:
            plt.xlim(*opts.xlim)
        if not opts.ylim == None:
            plt.ylim(*opts.ylim)

    else: raise ValueError('Either time or chan needs to be a range.')

    if not opts.share:
        title = str(bl)
        plt.title(title)
if not opts.nolegend and (not is_time_range or not is_chan_range):
    plt.legend(loc='best')

# Save to a file or pop up a window
if opts.out_file != '': plt.savefig(opts.out_file)
else:
    def click(event):
        print([event.key])
        if event.key == 'm':
            mode = raw_input('Enter new mode: ')
            for k in plots:
                try:
                    d = data_mode(plt_data[k], mode)
                    plots[k].set_data(d)
                except(ValueError):
                    print('Unrecognized plot mode')
            plt.draw()
        elif event.key == 'd':
            max = raw_input('Enter new max: ')
            try: max = float(max)
            except(ValueError): max = None
            drng = raw_input('Enter new drng: ')
            try: drng = float(drng)
            except(ValueError): drng = None
            for k in plots:
                _max,_drng = max, drng
                if _max is None or _drng is None:
                    d = plots[k].get_array()
                    if _max is None: _max = d.max()
                    if _drng is None: _drng = _max - d.min()
                plots[k].set_clim(vmin=_max-_drng, vmax=_max)
            print('Replotting...')
            plt.draw()
    plt.connect('key_press_event', click)
    plt.show()
