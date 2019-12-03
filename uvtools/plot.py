import aipy
import numpy as np

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
    except:
        print("plot_hmap_ortho requires Basemap. Try running 'pip install --user git+https://github.com/matplotlib/basemap.git'")

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
