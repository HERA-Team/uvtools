import aipy
import numpy as np
from scipy.stats import binned_statistic_2d

from . import utils

def data_mode(data, mode='abs'):
    """
    Apply filter to data according to a chosen plotting mode.
    
    Parameters
    ----------
    data : array_like
        Array of data to be plotted (normally complex floats).
    
    mode : str, optional
        Which transformation to apply to the data. Options are: 
          - 'phs':  Phase angle.
          - 'abs':  Absolute value.
          - 'real': Real value.
          - 'imag': Imaginary value.
          - 'log':  Log (base-10) of absolute value.
        Default: 'abs'.
    
    Returns
    -------
    data : array_like
        Data transformed according to the value of `mode`.
    """
    if mode.startswith('phs'):
        data = np.angle(data)
    elif mode.startswith('abs'):
        data = np.absolute(data)
    elif mode.startswith('real'):
        data = data.real
    elif mode.startswith('imag'):
        data = data.imag
    elif mode.startswith('log'):
        data = np.absolute(data)
        data = np.log10(data)
    else:
        raise ValueError('Unrecognized plot mode.')
    return data


def waterfall(d, mode='log', vmin=None, vmax=None, drng=None, mx=None, 
              recenter=False, **kwargs):
    """
    Generate a 2D waterfall plot.
    
    Parameters
    ----------
    d : array_like
        2D array of data.
    
    mode : str, optional
        Which transformation to apply to the data before plotting. Options are: 
          - 'phs':  Phase angle.
          - 'abs':  Absolute value.
          - 'real': Real value.
          - 'imag': Imaginary value.
          - 'log':  Log (base-10) of absolute value.
        Default: 'log'.
    
    vmin, vmax : float, optional
        Minimum and maximum values of the color scale. If not set (and `mx` and 
        `drng` are not set), the min. and max. values of the data will be used.
        
        Note that that the min. and max. values are the ones _after_ the data 
        have been transformed according to `mode`. So, if `mode='log'`, these 
        values are the min. and max. of log_10(data).
        
    mx : float, optional
        The max. value of the color scale in the plot (equivalent to vmax). 
        Cannot be specified at the same time as `vmin` and `vmax`.
    
    drng : float, optional
        The difference between the min. and max. values of the color scale in 
        the plot, `drng = mx - min`, where these are the min/max values after 
        applying the transformation specified by `mode`.
        
        Cannot be specified at the same time as `vmin` and `vmax`.
    
    recenter : bool, optional
        Recenter the image data by shifting by 50% around a circle (useful for 
        recentering Fourier-transformed data). Default: False.
    
    Returns
    -------
    plot : matplotlib.imshow
        Waterfall plot.
    """
    # import matplotlib
    import pylab as plt
    
    # Check validity of inputs
    validity_msg = "Must specify either `vmin` and `vmax` *or* `mx` and `drng`."
    if mx is not None or drng is not None:
        assert vmin is None and vmax is None, validity_msg
        
    if vmin is not None or vmax is not None:
        assert mx is None and drng is None, validity_msg
        mx = vmax
        drng = vmax - vmin
    
    # Fill masked array and recenter if requested
    if np.ma.isMaskedArray(d):
        d = d.filled(0)
    if recenter:
        d = aipy.img.recenter(d, np.array(d.shape)/2)
    
    # Apply requested transform to data
    d = data_mode(d, mode=mode)
    
    # Get min/max values for color scale
    if mx is None:
        mx = d.max()
    if drng is None:
        drng = mx - d.min()
    mn = mx - drng
    
    # Choose aspect ratio
    if 'aspect' not in kwargs.keys():
        kwargs['aspect'] = 'auto'
    return plt.imshow(d, vmax=mx, vmin=mn, interpolation='nearest', **kwargs)


def plot_hmap_ortho(h, cmap='jet', mode='log', mx=None, drng=None, 
                    res=0.25, verbose=False, normalize=False):
    """
    Plot a Healpix map in ortho projection.
    
    Parameters
    ----------
    h : aipy HealpixMap object
        HEALPIX map.
    
    cmap : str, optional
        Which matplotlib colormap to use. Default: 'jet'.
    
    mode : str, optional
        Which transform to apply to the data before plotting. See the 
        `data_mode` function for available options. Default: 'log'.
    
    vmin, vmax : float, optional
        Minimum and maximum values of the color scale. If not set (and `mx` and 
        `drng` are not set), the min. and max. values of the data will be used.
        
        Note that that the min. and max. values are the ones _after_ the data 
        have been transformed according to `mode`. So, if `mode='log'`, these 
        values are the min. and max. of log_10(data).
        
    mx : float, optional
        The max. value of the color scale in the plot (equivalent to vmax). 
        Cannot be specified at the same time as `vmin` and `vmax`.
    
    drng : float, optional
        The difference between the min. and max. values of the color scale in 
        the plot, `drng = mx - min`, where these are the min/max values after 
        applying the transformation specified by `mode`.
        
        Cannot be specified at the same time as `vmin` and `vmax`.
    
    res : float, optional
        Resolution of pixel grid, in degrees. Default: 0.25.
    
    verbose : bool, optional
        Whether to print basic debugging information. Default: False. 
    
    normalize : bool, optional
        Whether to normalize the data by the value at coordinates lat, long = 
        (0, 0). Default: False.
    
    Returns
    -------
    plot : matplotlib.imshow
        Healpix map in ortho projection.
    """
    from mpl_toolkits.basemap import Basemap
    
    # Check validity of inputs
    validity_msg = "Must specify either `vmin` and `vmax` *or* `mx` and `drng`."
    if mx is not None or drng is not None:
        assert vmin is None and vmax is None, validity_msg
        
    if vmin is not None or vmax is not None:
        assert mx is None and drng is None, validity_msg
        mx = vmax
        drng = vmax - vmin
    
    # Create new Basemap
    m = Basemap(projection='ortho', lat_0=90, lon_0=180, rsphere=1.)
    if verbose:
        print('SCHEME:', h.scheme())
        print('NSIDE:', h.nside())
    
    # Make grid of lat/long coords
    lons, lats, x, y = m.makegrid(int(360/res), int(180/res), returnxy=True)
    lons = 360 - lons
    lats *= aipy.img.deg2rad
    lons *= aipy.img.deg2rad
    
    # Convert coordinates
    y,x,z = aipy.coord.radec2eq(np.array([lons.flatten(), lats.flatten()]))
    ax,ay,az = aipy.coord.latlong2xyz(np.array([0,0]))
    data = h[x,y,z]
    data.shape = lats.shape
    
    # Normalize data and apply requested transformation
    if normalize:
        data /= h[0,0,1]
    data = data_mode(data, mode)
    
    # Draw marker lines on map
    m.drawmapboundary()
    m.drawmeridians(np.arange(0, 360, 30))
    m.drawparallels(np.arange(0, 90, 10))
    
    if mx is None:
        mx = data.max()
    if drng is None:
        mn = data.min()
    else:
        mn = mx - drng
    
    return m.imshow(data, vmax=mx, vmin=mn, cmap=cmap)
    

def plot_antpos(antpos, ants=None, xants=None, aspect_equal=True, 
                ant_numbers=True):
    """
    Plot antenna x,y positions from a dictionary of antenna positions.
    
    Parameters
    ----------
    antpos : dict
        Dictionary of antenna positions
    
    ants : list, optional
        A list of which antennas to plot. If not specified, all of the antennas 
        in `antpos` will be plotted. Default: None.
    
    xants : list, optional
        List of antennas to exclude from the plot. Default: None.
    
    aspect_equal : bool, optional
        Whether to make the width and height of the plot equal.
        Default: True.
    
    ant_numbers : bool, optional
        Whether to add the antenna numbers to the plot.
        Default: True
    
    Returns
    -------
    plot : matplotlib.Axes
        Plot of antenna x,y positions.
    """
    import pylab as plt
    
    if ants is None:
        ants = antpos.keys()
    if xants is not None:
        ants = [ant for ant in ants if ant not in xants]
    xs = [antpos[ant][0] for ant in ants]
    ys = [antpos[ant][1] for ant in ants]
    
    # Plot the antenna positions with black circles
    plt.figure()
    plt.scatter(xs, ys, marker='.', color='k', s=3000)
    
    # Add antenna numbers
    if ant_numbers:
        for i, ant in enumerate(ants):
            plt.text(xs[i], ys[i], ant, color='w', va='center', ha='center')
    
    # Axis labels
    plt.xlabel('X-position (m)')
    plt.ylabel('Y-position (m)')
    
    # Aspect ratio
    if aspect_equal:
        plt.axis('equal')
    ax = plt.gca()
    return ax


def plot_phase_ratios(data, cmap='twilight'):
    """
    Plot grid of waterfalls, each showing the phase of the product (V_1 V_2^*) 
    for bls 1 and 2.
    
    Parameters
    ----------
    data : dict
        Nested dictionary of data; first key is baseline, second key is pol.
    
    cmap : str, optional
        Colormap to use for plots. Default: 'twilight'.
    """
    import pylab as plt
    
    bls = data.keys()
    nbls = len(bls)
    pol = data[bls[0]].keys()[0]
    
    # Calculate no. rows and columns
    nratios = (nbls * (nbls-1))/2
    r = int(divmod(nratios,3)[0] + np.ceil(divmod(nratios,3)[1]/3.))
    c = 3
    
    # Construct list of blpairs
    ncross = []
    for k in range(nbls): 
        for i in range(k+1,nbls): 
            ncross.append((bls[k], bls[i]))
    
    # Plot waterfall 
    fig = plt.figure(figsize=(16,12))
    for i,k in enumerate(ncross):
        ax = plt.subplot(r,c,i+1)
        plt.title(str(k), color='magenta')
        g = 1.0
        waterfall(data[k[0]][pol]*np.conj(data[k[-1]][pol])*g, 
                  mode='phs', cmap=cmap, mx=np.pi, drng=2*np.pi)
        plt.grid(0)
        if divmod(i,c)[-1] != 0:
            ax.yaxis.set_visible(False) 
        if divmod(i,c)[0] != r-1:
            ax.xaxis.set_visible(False)
    cax = fig.add_axes([0.2, 0.06, 0.6, 0.01])
    plt.colorbar(cax=cax, orientation='horizontal')


def omni_view(reds, vis, pol, integration=10, chan=500, norm=False, 
              cursor=True, save=None, colors=None, symbols=None, ex_ants=[], 
              title=''):
    """
    Scatter plot of the real vs imaginary parts of all visibilities in a set 
    of redundant groups.
    
    Parameters
    ----------
    reds : list of lists
        List of redundant baseline groups. Each group should be a list of 
        baselines, specified as an antenna-pair tuple.
    
    vis : nested dict of array_like
        Nested dictionary containing visibility data. The structure is defined 
        as: `vis[baseline][pol][integration, chan]`, where `integration` is 
        the index of a time sample and `chan` is the index of a frequency 
        channel.
    
    pol : str
        Which polarization to plot from the `vis` dict.
    
    integration : int, optional
        Which time integration to plot from the `vis` dict. Default: 10.
    
    chan : int, optional
        Which frequency channel to plot from the `vis` dict. Default: 500
    
    norm : bool, optional
        Whether to normalize each point by its absolute value. Default: False.
    
    cursor : bool, optional
        Whether to include an interactive cursor label in the plot. 
        Default: True
    
    save : str, optional
        If specified, the filename to save the plot to. Default: None
    
    colors : list of str, optional
        List of colors to cycle through.
        Default: None (will use a default list).
    
    symbols : list of str, optional
        List of symbols to cycle through.
        Default: None (will use a default list).
    
    ex_ants : list, optional
        List of antennas to skip plotting. Default: [].
    
    title : str, optional
        Figure title. Default: ''.
    """
    import pylab as plt
    
    # Set default values for colors and symbols
    if not colors:
        colors = ["#006BA4", "#FF7F0E", "#2CA02C", "#D61D28", "#9467BD", 
                  "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
    if not symbols: 
        symbols = ["o", "v", "^", "<", ">", "*"]
    points = []
    sym = []
    col = []
    bl = []
    ngps = len(reds)
    if save:
        plt.clf()
        plt.cla()
    
    # Loop over redundant groups
    for i, gp in enumerate(reds):
        c = colors[i%len(colors)]
        s = symbols[i/len(colors)]
        for r in gp:
            if np.any([ant in r for ant in ex_ants]): continue
            try:
                points.append(vis[r][pol][integration,chan])
                bl.append(r)
            except(KeyError):
                points.append(np.conj(vis[r[::-1]][pol][integration,chan]))
                bl.append(r[::-1])
            sym.append(s)
            col.append(c)
    points = np.array(points)
    max_x = 0
    max_y = 0
    fig, ax = plt.subplots(nrows=1, ncols=1)
    
    # Loop over points
    for i, pt in enumerate(points):
        if norm:
            ax.scatter(pt.real/np.abs(pt), pt.imag/np.abs(pt), c=col[i], 
                       marker=sym[i], s=50, label='{}'.format(bl[i]))
        else:        
            ax.scatter(pt.real, pt.imag, c=col[i], marker=sym[i], s=50, 
                       label='{}'.format(bl[i]))
            if np.abs(pt.real) > max_x: max_x = np.abs(pt.real)
            if np.abs(pt.imag) > max_y: max_y = np.abs(pt.imag)
    plt.suptitle(title)
    
    # Choose scale according to whether normalized
    if norm:         
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
    else: 
        plt.xlim(-1.1 * max_x, 1.1 * max_x)
        plt.ylim(-1.1 * max_y, 1.1 * max_y)
    plt.ylabel('imag(V)')
    plt.xlabel('real(V)')
    
    if cursor:
        from mpldatacursor import datacursor
        datacursor(formatter='{label}'.format)
    if save:
        plt.savefig(save)


def omni_view_gif(filenames, name='omni_movie.gif'):
    """
    Create a gif from a list of images. Uses the `imageio` library.
    
    Parameters
    ----------
    filenames : list
        Ordered list of full paths to images that will be added to the 
        animation.
    
    name : str, optional
        Output filename for animation. Default: 'omni_movie.gif'.
    """
    import imageio
    
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(name, images)
    
def plot_diff_waterfall(uvd1, uvd2, antpairpol, plot_type="all", 
                        skip_check=False, save_path=None):
    """Produce waterfall plot(s) of differenced visibilities.

    Parameters
    ----------
    uvd1, uvd2 : pyuvdata.UVData
        UVData objects which store visibilities to be differenced and their
        associated metadata. They should have the same number of frequencies,
        same baselines, and same times as each other.

    antpairpol : tuple
        Tuple specifying which baseline and polarization to use to compare 
        visibility waterfalls. See pyuvdata.UVData.get_data method docstring 
        for information on accepted tuples.
    
    plot_type : str, tuple of str, or list of str, optional
        Which spaces to use for investigating differences. Available options 
        are as follows: time and frequency ('time_vs_freq'); time and delay 
        ('time_vs_dly'); fringe rate and frequency ('fr_vs_freq'); fringe 
        rate and delay ('fr_vs_dly'). Default is to use all plot types.
    
    skip_check : bool, optional
        Whether to check that the metadata in `uvd1` and `uvd2` match.
        Default behavior is to check the metadata.

    save_path : str, optional
        Path specifying where to save the figure. Can be absolute or relative. 
        Default behavior does not save the figure.

    """
    # check that metadata agrees, unless specified otherwise
    if not skip_check:
        utils.check_uvd_pair_metadata(uvd1, uvd2)

    # get visibility data
    vis1 = uvd1.get_data(antpairpol)
    vis2 = uvd2.get_data(antpairpol)

    # get important metadata
    times = np.unique(uvd1.time_array) # days
    lsts = np.unique(uvd1.lst_array) # radians
    freqs = uvd1.freq_array[0] # choose 0th spectral window; Hz

    # import astropy.units for conversion from days to seconds
    import astropy.units as u
    frs = utils.get_fourier_freqs(times * u.day.to('s')) # Hz
    dlys = utils.get_fourier_freqs(freqs) # s

    # make dictionary of plotting parameters; keys chosen for ease-of-use
    plot_params = {"time" : lsts, 
                   "freq" : freqs / 1e6, # MHz
                   "fr" : frs * 1e3, # mHz
                   "dly" : dlys * 1e9, # ns
                   }

    # make some axis labels; use LST instead of time b/c time is clunky
    labels = {"time" : "LST [radians]",
              "freq" : "Frequency [MHz]",
              "fr" : "Fringe Rate [mHz]",
              "dly" : "Delay [ns]",
              }

    # map plot types to transforms needed
    plot_types = {"time_vs_freq" : lambda data : data, # do nothing
                  "time_vs_dly" : lambda data : utils.FFT(data, 1), # FFT in freq
                  "fr_vs_freq" : lambda data : utils.FFT(data, 0), # FFT in time
                  "fr_vs_dly" : lambda data : utils.FFT(utils.FFT(data, 0), 1), # both
                  }

    # convert plot type to tuple
    if isinstance(plot_type, str):
        plot_type = tuple(plot_types.keys()) if plot_type == "all" else (plot_type,)

    # check that chosen plot type(s) OK
    assert all([plot in plot_types.keys() for plot in plot_type]), \
            "Please ensure the plot type chosen is supported. The supported " \
            "types are : {types}".format(types=list(plot_types.keys()))

    # now make a dictionary of the transformed visibilities
    visibilities = {plot : {label : xform(vis)
                            for label, vis in zip(("vis1", "vis2"), (vis1, vis2))}
                            for plot, xform in plot_types.items()
                            if plot in plot_type} # but only use desired transforms


    # import matplotlib, setup the figure
    import matplotlib.pyplot as plt
    figsize = (4 * 3, 3 * len(plot_type)) # (4,3) figsize for each plot
    fig = plt.figure(figsize=figsize)
    axes = fig.subplots(len(plot_type), 3)
    axes = [axes,] if len(axes.shape) == 1 else axes # avoid bug for single row
    axes[0][0].set_title("Amplitude Difference", fontsize=12)
    axes[0][1].set_title("Phase Difference", fontsize=12)
    axes[0][2].set_title("Amplitude of Complex Difference", fontsize=12)

    # helper function for getting the extent of axes
    extent = lambda xvals, yvals : (xvals[0], xvals[-1], yvals[-1], yvals[0])

    # loop over items in visibilities and plot them
    for i, item in enumerate(visibilities.items()):
        # extract visibilities, get diffs
        visA, visB = item[1].values()
        diffs = (utils.diff(visA, visB, 'abs'), 
                 utils.diff(visA, visB, 'phs'),
                 utils.diff(visA, visB, 'complex'))

        # extract parameters
        ykey, xkey = item[0].split("_vs_") # keys for choosing parameters
        xvals, yvals = plot_params[xkey], plot_params[ykey]

        # get labels
        xlabel, ylabel = labels[xkey], labels[ykey]

        # plot stuff
        for ax, diff in zip(axes[i], diffs):
            # set labels
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

            # plot waterfall and add a colorbar
            fig.sca(ax)
            cax = waterfall(diff, mode="real", cmap='viridis', 
                            extent=extent(xvals, yvals))
            fig.colorbar(cax)

    # display the figure
    plt.tight_layout()
    plt.show()

    # save if desired
    if save_path is not None:
        utils.savefig(fig, save_path)

def plot_diff_uv(uvd1, uvd2, pol=None, speedup=True,
                 skip_check=False, save_path=None,
                 resolution=50):
    """Summary plot for difference between visibilities.

    Parameters
    ----------
    uvd1, uvd2 : pyuvdata.UVData
        Input UVData objects which contain the visibilities to be differenced 
        and any relevant metadata.

    pol : str, None, optional
        String specifying which polarization to be used. Must be one of the
        polarizations listed in the UVData.get_pols() method for both
        `uvd1` and `uvd2`. Default is to use the 0th polarization.

    speedup : bool, optional
        Whether to use the fast implementation of this plotting tool or the
        slower implementation. The faster implementation does not produce
        figures as nice as the slower implementation, but the slower
        implementation is only really usable for small sets of data.
        Default is to use the fast implementation.

    skip_check : bool, optional
        Whether to check that the metadata for `uvd1` and `uvd2` match.
        Default behavior is to check the metadata.

    save_path : str, optional
        Path to where the figure should be saved; may be absolute or relative.
        Default is to not save the figure.
    
    resolution : int, optional
        Number of bins to use for regridding the u and v arrays.

    """
    # check the metadata unless instructed otherwise
    if not skip_check:
        utils.check_uvd_pair_metadata(uvd1, uvd2)

    # convert polarization to index
    pol = 0 if pol is None else uvd1.get_pols().index(pol)

    # load in relevant metadata
    bl_vecs = uvd1.uvw_array
    freqs = uvd1.freq_array[0]

    # import astropy constants to convert freq to wavelength
    from astropy.constants import c
    wavelengths = c.value / freqs

    # get uvw vectors; shape = (Nfreq, Nblts, 3)
    uvw_vecs = np.array([bl_vecs / wavelength for wavelength in wavelengths])
    
    # reshape uvw vectors to (Nblts, Nfreq, 3)
    uvw_vecs = np.einsum("ijk->jik", uvw_vecs)

    # get the u and v arrays, flattened
    uvals, vvals = uvw_vecs[:,:,0].flatten(), uvw_vecs[:,:,1].flatten()

    # get the regridded u and v arrays' bin edges
    u_regrid = np.linspace(uvals.min(), uvals.max(), resolution+1)
    v_regrid = np.linspace(vvals.min(), vvals.max(), resolution+1)

    # make an alias for regridding an array and taking the complex mean
    # this also takes the transpose so that axis0 is along the v-axis
    bin_2d = lambda arr : binned_statistic_2d(
                            uvals, vvals, arr, statistic='mean', 
                            bins=[u_regrid, v_regrid])[0].T

    # regrid the visibilities
    # need to do real/imag separately or information is lost
    vis1 = uvd1.data_array[:,0,:,pol].flatten()
    vis2 = uvd2.data_array[:,0,:,pol].flatten()
    vis1 = bin_2d(vis1.real) + 1j*bin_2d(vis1.imag)
    vis2 = bin_2d(vis2.real) + 1j*bin_2d(vis2.imag)

    # calculate differences of amplitudes and phases as masked arrays
    absdiff_ma = utils.diff(vis1, vis2, "abs")
    phsdiff_ma = utils.diff(vis1, vis2, "phs")
    cabsdiff_ma = utils.diff(vis1, vis2, "complex")

    # make the arrays into proper masked arrays
    mask = lambda arr : np.ma.MaskedArray(arr, np.isnan(arr))
    absdiff_ma = mask(absdiff_ma)
    phsdiff_ma = mask(phsdiff_ma)
    cabsdiff_ma = mask(cabsdiff_ma)

    # remove nans so that the data can actually be normalized
    unnan = lambda arr : arr[np.where(np.logical_not(np.isnan(arr)))]
    absdiff = unnan(absdiff_ma)
    phsdiff = unnan(phsdiff_ma)
    cabsdiff = unnan(cabsdiff_ma)

    # import matplotlib to  set things up and make the plot
    import matplotlib.pyplot as plt
    
    # get norms for generating colormaps for difference arrays
    absnorm = plt.cm.colors.SymLogNorm(0.1, vmin=absdiff.min(), vmax=absdiff.max())
    phsnorm = plt.cm.colors.Normalize(vmin=phsdiff.min(), vmax=phsdiff.max())
    cabsnorm = plt.cm.colors.LogNorm(vmin=cabsdiff.min(), vmax=cabsdiff.max())

    # setup the figure
    fig = plt.figure(figsize=(15,4.5))
    axes = fig.subplots(1,3)
    
    # add labels
    for ax, label in zip(axes, ("Amplitude", "Phase", "Amplitude of Complex")):
        ax.set_xlabel(r'$u$', fontsize=12)
        ax.set_ylabel(r'$v$', fontsize=12)
        ax.set_title(" ".join([label, "Difference"]), fontsize=12)

    extent = (uvals.min(), uvals.max(), vvals.max(), vvals.min())
    plot_iterable = zip(axes, 
                        (absdiff_ma, phsdiff_ma, cabsdiff_ma), 
                        (absnorm, phsnorm, cabsnorm))
    for ax, diff, norm in plot_iterable:
        cax = ax.imshow(diff, norm=norm, aspect="auto", 
                        cmap='viridis', extent=extent)
        fig.sca(ax)
        fig.colorbar(cax)

    # tidy up and display
    plt.tight_layout()
    plt.show()

    # save if desired
    if save_path is not None:
        utils.savefig(fig, save_path)
