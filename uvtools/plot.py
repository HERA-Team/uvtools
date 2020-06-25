import aipy
import numpy as np
from astropy import units
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
    try:
        from mpl_toolkits.basemap import Basemap
    except ModuleNotFoundError:
        raise ModuleNotFoundError("plot_hmap_ortho requires Basemap. Try running 'pip install --user git+https://github.com/matplotlib/basemap.git'")
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
    
def labeled_waterfall(
    data,
    antpairpol=None,
    freqs=None,
    times=None,
    lsts=None,
    time_or_lst="lst",
    mode="log",
    ax=None,
    figsize=(10,7),
    dpi=100,
    aspect="auto",
    fontsize=None,
    draw_colorbar=True,
    cmap="best",
    dynamic_range=None,
    fft_axis=None,
    freq_taper=None,
    freq_taper_kwargs=None,
    time_taper=None,
    time_taper_kwargs=None,
):
    """Make a waterfall plot with axes labeled.

    Parameters
    ----------
    data: array-like of complex, or :class:`pyuvdata.UVData` instance
        Object containing visibility data. If an array is passed, then ``freqs``
        and either ``times`` or ``lsts`` must be provided, and the array must
        have shape (``lsts.size``, ``freqs.size``). Otherwise, an ``antpairpol``
        key must be provided.
    antpairpol: tuple
        (ant1, ant2, pol) tuple specifying the baseline and polarization to
        pull data for if ``data`` is a :class:`pyuvdata.UVData` instance. Ignored
        if ``data`` is an array-like object.
    freqs: array-like of float
        Frequencies corresponding to the observed data, in Hz. Required if ``data``
        is an array-like object; ignored otherwise.
    times: array-like of float
        Observation times, in JD. Required if ``data`` is an array-like object and
        ``lsts`` is not provided.
    lsts: array-like of float
        Observed LSTs, in radians. Required if ``data`` is an array-like object
        and ``times`` is not provided.
    time_or_lst: str, optional
        Either "time" or "lst". Used to specify whether the time axis should be
        in JD or LST. If ``data`` is an array-like object and only one of ``times``
        or ``lsts`` is provided, then this parameter is ignored.
    mode: str, optional
        Plotting mode to use; must be a mode accepted by :func:`data_mode`. Default
        is "log", which plots the base-10 logarithm of the absolute value of the data.
    ax: :class:`plt.Axes` instance, optional
        :class:`plt.Axes` object to use for plotting the waterfall. If not provided,
        then a new :class:`plt.Figure` object and :class:`plt.Axes` instance is created.
    figsize: tuple of int, optional
        Length-2 tuple specifying figure dimensions in inches. Ignored if ``ax`` 
        is provided.
    dpi: int, optional
        Dots per inch to be used in creating the figure. Ignored if ``ax`` is provided.
    aspect: str or float, optional
        Aspect ratio of the plot. Should be either "equal", "auto", or a number.
        Default is to use "auto".
    fontsize: float, optional
        Font size for labels, in points.
    draw_colorbar: bool, optional
        Whether to draw a colorbar. Default is to draw a colorbar.
    cmap: str or :class:`plt.cm.colors.Colormap` instance, optional
        Colormap to use for plotting the waterfall. Default is to choose a colormap
        appropriate for the plotting mode chosen ("twilight" for plotting phases,
        and "inferno" otherwise).
    dynamic_range: float, optional
        Number of orders of magnitude of dynamic range to plot. For example, setting
        ``dynamic_range=5`` limits the colorbar to range from the maximum value to
        five orders of magnitude below the maximum. If ``mode=="phs"``, then this
        parameter is ignored.
    fft_axis: int or str, optional
        Axis over which to perform a Fourier transform. May be specified with one
        of three strings ("freq", "time", "both") or one of three integers (-1, 0,
        1), with -1 indicating to transform over both axes. Default is to not
        perform a Fourier transform over any axis.
    freq_taper: str, optional
        Taper to use when performing a Fourier transform along the frequency axis.
        Must be one of the tapers supported by :func:`dspec.gen_window`.
    freq_taper_kwargs: dict, optional
        Keyword arguments to use in generating the taper for the frequency axis.
    time_taper: str, optional
        Taper to use when performing a Fourier transform along the time axis.
        Must be one of the tapers supported by :func:`dspec.gen_window`.
    time_taper_kwargs: dict, optional
        Keyword arguments to use in generating the taper for the time axis.

    Returns
    -------
    fig or ax: :class:`plt.Figure` instance or :class:`plt.Axes` instance
        Figure or axes object containing the waterfall plot. If ``ax`` was
        provided in the function all, then the same :class:`plt.Axes` instance
        that was passed is returned with the waterfall drawn into it, as well
        as labels and a colorbar added.
    """
    import matplotlib.pyplot as plt

    # Validate parameters.
    if time_or_lst not in ("time", "lst"):
        raise ValueError("time_or_lst must be 'time' or 'lst'.")
    if np.array(data).dtype != np.dtype('O'):
        data = np.atleast_2d(data)
        if not np.iscomplexobj(data):
            raise TypeError("array-like data must consist of complex numbers.")
        if data.ndim != 2 or (data.ndim == 2 and 1 in data.shape):
            raise ValueError("array-like data must be 2-dimensional.")
    if isinstance(data, np.ndarray):
        if freqs is None or (times is None or lsts is None):
            raise ValueError(
                "freqs, times, and lsts must be provided for plotting an array."
            )
        if times is None:
            time_or_lst = "lst"
            times = lsts / (2 * np.pi) # For Fourier transform purposes
        else:
            time_or_lst = "time"
    else:
        try:
            from pyuvdata import UVData
            # In case UVData is installed and a non-UVData object was passed.
            if type(data) is not UVData:
                raise ImportError
        except ImportError:
            raise TypeError("data must either be an ndarray or UVData object.")
        if antpairpol is None:
            raise ValueError(
                "You must provide an antpairpol key if data is a UVData object."
            )
        freqs = np.unique(data.freq_array)
        times = np.unique(data.time_array)
        lsts = np.unique(data.lst_array)
        data = data.get_data(antpairpol)
    
    # Prepare axis labels.
    xvals = freqs / 1e6
    xlabel = "Frequency [MHz]"
    if time_or_lst == "time":
        yvals = times - int(times[0])
        ylabel = f"JD - {int(times[0]):d}"
    else:
        yvals = lsts * units.day.to("hr") / (2 * np.pi)
        ylabel = "LST [hours]"

    # Do any requested Fourier transforms and update axis labels.
    if fft_axis is not None:
        freq_taper_kwargs = freq_taper_kwargs or {}
        time_taper_kwargs = time_taper_kwargs or {}
        if fft_axis not in ("freq", "time", "both", -1, 0, 1):
            raise ValueError("fft_axis not recognized.")
        if type(fft_axis) is int:
            fft_axis = ("freq", "time", "both")[fft_axis]
        if fft_axis in ("freq", "both"):
            delays = utils.fourier_freqs(freqs) * 1e9 # ns
            data = utils.FFT(data, axis=1, taper=freq_taper, **freq_taper_kwargs)
            xvals = delays
            xlabel = "Delay [ns]"
        if fft_axis in ("time", "both"):
            fringe_rates = utils.fourier_freqs(times * units.day.to("s")) * 1e3 # mHz
            data = utils.FFT(data, axis=0, taper=time_taper, **time_taper_kwargs)
            yvals = fringe_rates
            ylabel = "Fringe Rate [mHz]"

    # Update data for plotting.
    data = data_mode(data, mode=mode)
    vmin, vmax = None, None

    # Prepare colorbar parameters.
    if mode == "phs":
        cbar_label = "Phase [radians]"
        vmin, vmax = -np.pi, np.pi
        if cmap == "best":
            cmap = "twilight"
    else:
        if fft_axis == "freq":
            base_label = r"$\tilde{V}(t,\tau)$"
            unit_label = "[Jy Hz]"
        elif fft_axis == "time":
            base_label = r"$\tilde{V}(f,\nu)$"
            unit_label = "[Jy s]"
        elif fft_axis == "both":
            base_label = r"$\tilde{V}(f,\tau)$"
            unit_label = "[Jy Hz s]"
        else:
            base_label = r"$V(t,\nu)$"
            unit_label = "[Jy]"

        if mode == "abs":
            cbar_label = f"|{base_label}| {unit_label}"
        elif mode == "real":
            cbar_label = r"$\mathfrak{Re}$" + f"({base_label}) {unit_label}"
        elif mode == "imag":
            cbar_label = r"$\mathfrak{Im}$" + f"({base_label}) {unit_label}"
        else:
            cbar_label = r"$\log_{10}$" + f"|{base_label}| {unit_label}"

        if cmap == "best":
            cmap = "inferno"

    # Limit the dynamic range if desired.
    if dynamic_range is not None and mode != "phs":
        vmax = data.max()
        if mode == "log":
            vmin = vmax - dynamic_range
        else:
            vmin = vmax / 10 ** dynamic_range

    # Setup mappable for drawing colorbar.
    if draw_colorbar:
        norm = plt.cm.colors.Normalize(vmin=vmin, vmax=vmax)
        scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Prepare the figure.
    return_value = ax
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        return_value = fig

    # Finish setup, then plot.
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.imshow(
        data,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        extent=(xvals.min(), xvals.max(), yvals.max(), yvals.min()),
    )
    if draw_colorbar:
        if return_value is ax:
            fig = ax.get_figure()

        extend = "neither" if dynamic_range is None else "min"
        cbar = fig.colorbar(mappable=scalar_map, ax=ax, extend=extend)
        cbar.set_label(cbar_label, fontsize=fontsize)

    return return_value

def fourier_transform_waterfalls(
    data,
    antpairpol=None,
    freqs=None,
    times=None,
    lsts=None,
    time_or_lst="lst",
    mode="log",
    figsize=(14,10),
    dpi=100,
    aspect="auto",
    fontsize=None,
    cmap="best",
    dynamic_range=None,
    plot_limits=None,
    freq_taper=None,
    freq_taper_kwargs=None,
    time_taper=None,
    time_taper_kwargs=None,
):
    """
    Plot a 2x2 grid of waterfalls showing all possible Fourier transforms.

    Moving clockwise from the top-left, the plots are as follows:
        time vs frequency
        fringe-rate vs frequency
        fringe-rate vs delay
        time vs delay

    Time units used for plotting are either JD or LST hours.
    Frequency units used for plotting are MHz.
    Fringe-rate units used for plotting are mHz.
    Delay units used for plotting are ns.
    
    Note that these units may not correspond exactly to the units of the related
    parameters passed. The parameter units for ``freqs`` and ``times`` or ``lsts``
    are chosen to match the units typically used for recording metadata in
    :class:`pyuvdata.UVData` objects.

    Parameters
    ----------
    data: array-like of complex, or :class:`pyuvdata.UVData` instance
        Object containing visibility data. If an array is passed, then ``freqs``
        and either ``times`` or ``lsts`` must be provided, and the array must
        have shape (``lsts.size``, ``freqs.size``). Otherwise, an ``antpairpol``
        key must be provided.
    antpairpol: tuple
        (ant1, ant2, pol) tuple specifying the baseline and polarization to
        pull data for if ``data`` is a :class:`pyuvdata.UVData` instance. Ignored
        if ``data`` is an array-like object.
    freqs: array-like of float
        Frequencies corresponding to the observed data, in Hz. Required if ``data``
        is an array-like object; ignored otherwise.
    times: array-like of float
        Observation times, in JD. Required if ``data`` is an array-like object and
        ``lsts`` is not provided.
    lsts: array-like of float
        Observed LSTs, in radians. Required if ``data`` is an array-like object
        and ``times`` is not provided.
    time_or_lst: str, optional
        Either "time" or "lst". Used to specify whether the time axis should be
        in JD or LST. If ``data`` is an array-like object and only one of ``times``
        or ``lsts`` is provided, then this parameter is ignored.
    mode: str, optional
        Plotting mode to use; must be a mode accepted by :func:`data_mode`. Default
        is "log", which plots the base-10 logarithm of the absolute value of the data.
    figsize: tuple of float, optional
        Size of the figure to be produced, in inches. Default is 14x10.
    dpi: float, optional
        Dots-per-inch of the figure. Default is 100.
    aspect: float or str, optional
        Aspect ratio to use for each subplot. Default is "auto".
    fontsize: float, optional
        Font size to use for plotting labels, in points.
    cmap: str or :class:`plt.cm.colors.Colormap` instance
        Color map to be used when drawing waterfalls. Default is to have the choice
        be based on the data mode selected: if ``mode`` is "phs", then "twilight"
        is used; otherwise, "inferno" is used.
    dynamic_range: dict, optional
        Dictionary mapping strings to number of orders-of-magnitude to restrict
        the plot dynamic range to. Accepted strings are as follows:
            ("time", "freq", "delay", "fringe-rate"): specifying one of these
            will limit the dynamic range for the associated row or column. For
            example, passing {"time": 5} will limit the dynamic range of the left
            column to five orders-of-magnitude, clipping values on the low-end.
            
            Any length-2 combination of an entry from the following pairs:
                ("time", "fringe-rate"), ("freq", "delay")
            This type of mapping will limit the dynamic range for a single plot
            with axes specified by the pair specified. For example, passing
            {("fringe-rate", "delay"): 5} will only limit the dynamic range for
            the bottom-right plot.
    plot_limits: dict, optional
        Dictionary mapping strings to length-2 tuples. The keys designate the
        dimension ("time", "freq", "fringe-rate", "delay") to crop, and the values
        give the lower- and upper-bounds of the cropped region. For example, passing
        {"delay": (-500, 500)} will crop the delay axis to only show delays between
        -500 ns and +500 ns. Units should be the same as the units used in the plot
        display. See the beginning of the function documentation for details.
    freq_taper: str, optional
        Name of the taper to be applied along the frequency-axis when performing
        Fourier transforms. Must be a taper supported by :func:`dspec.gen_window`.
        Default is no taper (an implicit top-hat or boxcar).
    freq_taper_kwargs: dict, optional
        Keyword arguments to be used in generating the frequency taper.
    time_taper: str, optional
        Name of the taper to be applied along the time-axis when performing Fourier
        transforms. Default is the same as for the frequency taper.
    time_taper_kwargs: dict, optional
        Keyword arguments to be used in generating the time taper.
    
    Returns
    -------
    fig: :class:`plt.Figure` instance
        Figure containing 2x2 grid of plots visualizing the data in the selected
        mode for all possible Fourier transforms, with axis labels and colorbars.
    """
    import matplotlib.pyplot as plt

    # Convert potential None-types to empty dictionaries where needed.
    dynamic_range = dynamic_range or {}
    plot_limits = plot_limits or {}

    # Figure setup
    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes = fig.subplots(2,2)
    transform_axes = (None, 0, 1, -1)
    axes_dims = (
        ("time", "freq"),
        ("fringe-rate", "freq"),
        ("time", "delay"),
        ("fringe-rate", "delay")
    )

    # Make the plots.
    for i, ax in enumerate(axes.ravel()):
        # Determine any adjustments to be made to axes in plotting routine.
        x_dim, y_dim = axes_dims[i]
        possible_drng_keys = (x_dim, y_dim, (x_dim, y_dim), (y_dim, x_dim))
        transform_axis = transform_axes[i]
        limit_dynamic_range = list(
            key in dynamic_range.keys()
            for key in possible_drng_keys
        )
        if any(limit_dynamic_range):
            drng = dynamic_range[possible_drng_keys[limit_dynamic_range.index(True)]]
        else:
            drng = None

        # Actually make the plot.
        ax = labeled_waterfall(
            data=data,
            antpairpol=antpairpol,
            freqs=freqs,
            times=times,
            lsts=lsts,
            time_or_lst=time_or_lst,
            mode=mode,
            ax=ax,
            aspect=aspect,
            fontsize=fontsize,
            draw_colorbar=True,
            cmap=cmap,
            dynamic_range=drng,
            fft_axis=transform_axis,
            freq_taper=freq_taper,
            freq_taper_kwargs=freq_taper_kwargs,
            time_taper=time_taper,
            time_taper_kwargs=time_taper_kwargs,
        )

        # Adjust the plot boundaries if requested.
        if x_dim in plot_limits:
            ax.set_xlim(*plot_limits[x_dim])
        if y_dim in plot_limits:
            ax.set_ylim(*plot_limits[y_dim])

    return fig


def plot_diff_waterfall(uvd1, uvd2, antpairpol, plot_type="all", 
                        check_metadata=True, freq_taper=None, 
                        freq_taper_kwargs=None, time_taper=None,
                        time_taper_kwargs=None):
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
    
    check_metadata : bool, optional
        Whether to check that the metadata for `uvd1` and `uvd2` match.
        See ``utils.check_uvd_pair_metadata`` docstring for details on 
        how the metadata are compared. If `check_metadata` is set to 
        False, but the metadata don't agree, then the plotter may or 
        may not error out, depending on how the metadata disagree. 
        Default behavior is to check the metadata.

    freq_taper : str, optional
        Choice of tapering function to use along the frequency axis. Default 
        is to use no taper.

    freq_taper_kwargs : dict, optional
        Keyword arguments to be used with the taper for the frequency-axis. 
        These are ultimately passed to ``dspec.gen_window``. Default behavior 
        is to use an empty dictionary.

    time_taper : str, optional
        Choice of tapering function to use along the time axis. Default is to 
        use no taper.

    time_taper_kwargs : dict, optional
        Keyword arguments to be used with the taper for the time axis. These 
        are ultimately passed to ``dspec.gen_window``. Default behavior is to 
        use an empty dictionary.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing all of the desired plots.
    """
    # check that metadata agrees, unless specified otherwise
    if check_metadata:
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
    frs = utils.fourier_freqs(times * u.day.to('s')) # Hz
    dlys = utils.fourier_freqs(freqs) # s

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

    # convert taper kwargs to empty dictionaries if not specified
    freq_taper_kwargs = freq_taper_kwargs or {}
    time_taper_kwargs = time_taper_kwargs or {}

    # map plot types to transforms needed
    plot_types = {
        "time_vs_freq" : lambda data : data, # do nothing
        "time_vs_dly" : lambda data : utils.FFT(data, 1, freq_taper, **freq_taper_kwargs),
        "fr_vs_freq" : lambda data : utils.FFT(data, 0, time_taper, **time_taper_kwargs),
        "fr_vs_dly" : lambda data : utils.FFT(
            utils.FFT(data, 0, time_taper, **time_taper_kwargs), 
            1, freq_taper, **freq_taper_kwargs
        ),
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

    return fig

def plot_diff_uv(uvd1, uvd2, pol=None, check_metadata=True, bins=50):
    """Summary plot for difference between visibilities.

    This function produces three plots which summarize the differences 
    between the data arrays in `uvd1` and `uvd2`. Each summary plot is 
    shown in a regridded uv-plane, with a resolution set by the `bins` 
    parameter. This function uses ``scipy.stats.binned_statistic_2d`` 
    to perform a complex average in the uv-plane for each visibility 
    array before performing any further operations. After taking the 
    complex average in the uv-plane, the following plots are produced: 
    first, the difference of the amplitudes of each array; second, the 
    difference of the phases of each array; third, the amplitude of the 
    complex difference of the visibility arrays.

    Parameters
    ----------
    uvd1, uvd2 : pyuvdata.UVData
        Input UVData objects which contain the visibilities to be differenced 
        and any relevant metadata.

    pol : str, None, optional
        String specifying which polarization to be used. Must be one of the
        polarizations listed in the UVData.get_pols() method for both
        `uvd1` and `uvd2`. Default is to use the 0th polarization.

    check_metadata : bool, optional
        Whether to check that the metadata for `uvd1` and `uvd2` match.
        See ``utils.check_uvd_pair_metadata`` docstring for details on 
        how the metadata are compared. If `check_metadata` is set to 
        False, but the metadata don't agree, then the plotter may or 
        may not error out, depending on how the metadata disagree. 
        Default behavior is to check the metadata.

    bins : int, optional
        Number of bins to use for regridding the u and v arrays.

    """
    # check the metadata unless instructed otherwise
    if check_metadata:
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
    uvw_vecs = np.swapaxes(uvw_vecs, 0, 1)

    # get the u and v arrays, flattened
    uvals, vvals = uvw_vecs[:,:,0].flatten(), uvw_vecs[:,:,1].flatten()

    # get the regridded u and v arrays' bin edges
    u_regrid = np.linspace(uvals.min(), uvals.max(), bins+1)
    v_regrid = np.linspace(vvals.min(), vvals.max(), bins+1)

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

    return fig

def plot_diff_1d(uvd1, uvd2, antpairpol, plot_type="both", 
                 check_metadata=True, dimension=None,
                 taper=None, taper_kwargs=None,
                 average_mode=None, **kwargs):
    """Produce plots of visibility differences along a single axis.

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
    
    plot_type : str, optional
        A string identifying which quantities to plot. Accepted values are 
        as follows:
            - normal
                - Single row of plots in the usual basis (time or frequency).
            - fourier
                - Single row of plots in Fourier space (fringe rate or delay).
            - both
                - Two rows of plots in the usual and Fourier domains.
        Default behavior is to use the 'both' setting.

    check_metadata : bool, optional
        Whether to check that the metadata for `uvd1` and `uvd2` match.
        See ``utils.check_uvd_pair_metadata`` docstring for details on 
        how the metadata are compared. If `check_metadata` is set to 
        False, but the metadata don't agree, then the plotter may or 
        may not error out, depending on how the metadata disagree. 
        Default behavior is to check the metadata.
    
    dimension : str, optional
        String specifying which dimension is used for the normal domain. This 
        may be either 'time' or 'freq'. Default is to determine which axis has
        more entries and to use that axis.

    taper : str, optional
        Sting specifying which taper to use; must be a taper supported by 
        ``dspec.gen_window``. Default is to use no taper.

    taper_kwargs : dict, optional
        Dictionary of keyword arguments and their values, passed downstream to 
        ``dspec.gen_window``. Default is to use an empty dictionary (i.e. 
        default parameter values for whatever window is generated).

    average_mode : str, optional
        String specifying which ``numpy`` averaging function to use. Default 
        behavior is to use ``np.mean``.

    **kwargs
        These are passed directly to the averaging function used. Refer to 
        the documentation of the averaging function you want to use for 
        information regarding what parameters may be specified here.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Figure object containing the plots. The plots have their axes and 
        titles automatically set depending on what quantities are being 
        plotted.

    Notes
    -----
    This function extracts the visibility waterfall corresponding to the 
    provided antpairpol and flattens it by taking the average along the axis 
    not being used. The averaging function used may be specified with the 
    `average_mode` parameter, and weights (or optional parameters to be 
    passed to the averaging function) may be specified in the variable 
    keyword parameter `kwargs`. Any flags relevant for the data are 
    currently ignored, but this functionality may be introduced in a future 
    update.
    """
    if check_metadata:
        utils.check_uvd_pair_metadata(uvd1, uvd2)

    if plot_type not in ("normal", "fourier", "both"):
        raise ValueError(
            "You must specify whether to make one or two plots with "
            "the ``plot_type`` parameter. You may choose to plot the "
            "visibility difference as a function of frequency/time by "
            "setting ``plot_type`` to 'normal', or you can choose to "
            "plot the difference in Fourier space by setting "
            "``plot_type`` to 'fourier'. If you would like to plot both, "
            "then set ``plot_type`` to 'both'."
        )

    dimensions_to_duals = {"time" : "fr", "freq" : "dly"} 

    if dimension is None:
        dimension = "time" if uvd1.Ntimes > uvd1.Nfreqs else "freq"
        if uvd1.Ntimes == uvd1.Nfreqs:
            warnings.warn(
                "The UVData objects passed have the same number of " \
                "times as they do frequencies. You did not specify " \
                "which dimension to use, so the difference plots " \
                "will be made along the time axis."
            )

    if dimension not in ("time", "freq"):
        raise ValueError(
            "You must specify whether the visibilities are a function "
            "of time or frequency by setting the ``dimension`` "
            "parameter to 'time' or 'freq', respectively."
        )
    
    dual = dimensions_to_duals[dimension]

    use_axis = 0 if dimension == "time" else 1
    proj_axis = (use_axis + 1) % 2

    # choose an averaging function
    if average_mode is not None:
        try:
            average = getattr(np, average_mode)
        except AttributeError as err:
            err_msg = err.args[0] + "\nDefaulting to using np.mean"
            warnings.warn(err_msg)
            average = np.mean
    else:
        average = np.mean

    # get visibility data
    vis1 = average(uvd1.get_data(antpairpol), axis=proj_axis, **kwargs)
    vis2 = average(uvd2.get_data(antpairpol), axis=proj_axis, **kwargs)

    # use same approach as in plot_diff_waterfall
    # get important metadata
    times = np.unique(uvd1.time_array) # days
    lsts = np.unique(uvd1.lst_array) # radians
    freqs = np.unique(uvd1.freq_array) # Hz

    # import astropy for unit conversions
    import astropy.units as u
    frs = utils.fourier_freqs(times * u.day.to('s')) # Hz
    dlys = utils.fourier_freqs(freqs) # s

    # make dictionary of plotting parameters
    plot_params = {"time" : lsts, # radians
                   "freq" : freqs / 1e6, # MHz
                   "fr" : frs * 1e3, # mHz
                   "dly" : dlys * 1e9 # ns
                   }

    # now do the same for abscissa labels
    labels = {"time" : "LST [radians]",
              "freq" : "Frequency [MHz]",
              "fr" : "Fringe Rate [mHz]",
              "dly" : "Delay [ns]"
              }

    # and now for ordinate labels
    vis_labels = {"time" : r"$V(t)$ [{vis_units}]",
                  "freq" : r"$V(\nu)$ [{vis_units}]",
                  "fr" : r"$\tilde{V}(f)$ [{vis_units}$\cdot$s]",
                  "dly" : r"$\tilde{V}(\tau)$ [{vis_units}$\cdot$Hz]"
                  }

    # make sure the taper kwargs are a dictionary
    taper_kwargs = taper_kwargs or {}

    # make some mappings for plot types
    plot_types = {dimension : lambda data : data, # no fft
                  dual : lambda data : utils.FFT(data, 0, taper, **taper_kwargs)
                  }

    # update the plot_type parameter to something useful
    if plot_type == "normal":
        plot_type = (dimension,)
    elif plot_type == "fourier":
        plot_type = (dual,)
    else:
        plot_type = (dimension, dual)

    # make a dictionary of visibilities to plot
    visibilities = {
        plot : [xform(vis) for vis in (vis1, vis2)]
        for plot, xform in plot_types.items()
        if plot in plot_type
    }

    # XXX make a helper function for this
    # now setup the figure
    import matplotlib.pyplot as plt
    figsize = (4 * 3, 3 * len(plot_type))
    fig = plt.figure(figsize=figsize)
    axes = fig.subplots(len(plot_type), 3)
    axes = [axes,] if axes.ndim == 1 else axes
    axes[0][0].set_title("Amplitude Difference", fontsize=12)
    axes[0][1].set_title("Phase Difference", fontsize=12)
    axes[0][2].set_title("Amplitude of Complex Difference", fontsize=12)

    # plot the visibilities
    for i, item in enumerate(visibilities.items()):
        # get the differences
        visA, visB = item[1]
        diffs = (
            utils.diff(visA, visB, 'abs'),
            utils.diff(visA, visB, 'phs'),
            utils.diff(visA, visB, 'complex')
        )

        xdim = item[0]
        xlabel = labels[xdim]
        # to ensure appropriate LaTeX formatting and visibility units
        ylabel = vis_labels[xdim].format(V="{V}", vis_units=uvd1.vis_units)

        # actually plot it
        for ax, diff in zip(axes[i], diffs):
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

            ax.plot(plot_params[xdim], diff, marker='o', color='k', lw=0)

    return fig
