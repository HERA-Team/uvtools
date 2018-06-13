import aipy, numpy as np, pylab as plt
import sys, scipy

def data_mode(data, mode='abs'):
    if mode.startswith('phs'): data = np.angle(data)
    elif mode.startswith('lin'):
        data = np.absolute(data)
        #data = np.masked_less_equal(data, 0)
    elif mode.startswith('real'): data = data.real
    elif mode.startswith('imag'): data = data.imag
    elif mode.startswith('log'):
        data = np.absolute(data)
        data = np.log10(data)
    else: raise ValueError('Unrecognized plot mode.')
    return data

def waterfall(d, mode='log', mx=None, drng=None, recenter=False, **kwargs):
    if np.ma.isMaskedArray(d): d = d.filled(0)
    if recenter: d = aipy.img.recenter(d, np.array(d.shape)/2)
    d = data_mode(d, mode=mode)
    if mx is None: mx = d.max()
    if drng is None: drng = mx - d.min()
    mn = mx - drng
    if not kwargs.has_key('aspect'): kwargs['aspect'] = 'auto'
    return plt.imshow(d, vmax=mx, vmin=mn, interpolation='nearest', **kwargs)

def plot_hmap_ortho(h, cmap='jet', mode='log', mx=None, drng=None, 
        res=0.25, verbose=False, normalize=False):
    from mpl_toolkits.basemap import Basemap
    m = Basemap(projection='ortho',lat_0=90,lon_0=180,rsphere=1.)
    if verbose:
        print 'SCHEME:', h.scheme()
        print 'NSIDE:', h.nside()
    lons,lats,x,y = m.makegrid(int(360/res),int(180/res), returnxy=True)
    lons = 360 - lons
    lats *= aipy.img.deg2rad; lons *= aipy.img.deg2rad
    y,x,z = aipy.coord.radec2eq(np.array([lons.flatten(), lats.flatten()]))
    ax,ay,az = aipy.coord.latlong2xyz(np.array([0,0]))
    data = h[x,y,z]
    data.shape = lats.shape
    if normalize: data /= h[0,0,1]
    data = data_mode(data, mode)
    m.drawmapboundary()
    m.drawmeridians(np.arange(0, 360, 30))
    m.drawparallels(np.arange(0, 90, 10))
    if mx is None: mx = data.max()
    if drng is None:
        mn = data.min()
    #    if min < (max - 10): min = max-10
    else: mn = mx - drng
    return m.imshow(data, vmax=mx, vmin=mn, cmap=cmap)
    #step = (mx - mn) / 10
    #levels = np.arange(mn-step, mx+step, step)
    #map.contourf(cx,cy,data,levels,linewidth=0,cmap=cmap)

def plot_antpos(aa, ants=None, ex_ants=None, aspect_equal=True, 
        ant_numbers=True, elevation=False, grid=True):
    if ants is None: ants = range(len(aa.ants))
    if ex_ants is not None: ants = [i for i in ants if i not in ex_ants]
    antpos = [aa.get_baseline(0,i,src='z') for i in ants]
    antpos = np.array(antpos) * aipy.const.len_ns / 100.
    x,y,z = antpos[:,0], antpos[:,1], antpos[:,2]
    x -= np.average(x)
    y -= np.average(y)
    plt.plot(x,y, 'k.')
    for (ant,xa,ya,za) in zip(ants,x,y,z):
        if elevation:
            hx,hy = r*za*np.cos(th)+xa, r*za*np.sin(th)+ya
            if za > 0: fmt = '#eeeeee'
            else: fmt = '#a0a0a0'
            plt.fill(hx,hy, fmt)
        if ant_numbers: plt.text(xa,ya, str(ant))
    if grid: plt.grid()
    plt.xlabel("East-West Antenna Position (m)")
    plt.ylabel("North-South Antenna Position (m)")
    ax = plt.gca()
    if aspect_equal: ax.set_aspect('equal')
    return ax
    
def plot_phase_ratios(data):
    '''Plots ratios of baselines given in data. 
       Data is a nested dictionary. First key is baseline, second key is pol. 
    '''
    bls = data.keys()
    nbls = len(bls)
    pol = data[bls[0]].keys()[0]

    nratios = (nbls * (nbls-1))/2
    r = int(divmod(nratios,3)[0] + np.ceil(divmod(nratios,3)[1]/3.))
    c = 3
    ncross = []
    for k in range(nbls): 
        for i in range(k+1,nbls): 
            ncross.append((bls[k],bls[i]))

    fig = plt.figure(figsize=(16,12))
    for i,k in enumerate(ncross):
        ax = plt.subplot(r,c,i+1)
        plt.title(str(k),color='magenta')
        g = 1.0
        waterfall(data[k[0]][pol]*np.conj(data[k[-1]][pol])*g, mode='phs', cmap='jet', mx=np.pi, drng=2*np.pi)
        plt.grid(0)
        if divmod(i,c)[-1] != 0:  ax.yaxis.set_visible(False) 
        if divmod(i,c)[0] != r-1: ax.xaxis.set_visible(False)
    cax = fig.add_axes([0.2, 0.06, 0.6, 0.01])
    plt.colorbar(cax=cax, orientation='horizontal')

def omni_view(reds,vis,pol,integration=10,chan=500,norm=False,cursor=True,save=None,colors=None,symbols=None, ex_ants=[], title=''):
    if not colors:
        colors = ["#006BA4", "#FF7F0E", "#2CA02C", "#D61D28", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
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
    for i,gp in enumerate(reds):
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
    max_x=0
    max_y=0
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for i,pt in enumerate(points):
        if norm:
            ax.scatter(pt.real/np.abs(pt), pt.imag/np.abs(pt), c=col[i], marker=sym[i], s=50, label='{}'.format(bl[i]))
        else:        
            ax.scatter(pt.real, pt.imag, c=col[i], marker=sym[i], s=50, label='{}'.format(bl[i]))
            if np.abs(pt.real) > max_x: max_x = np.abs(pt.real)
            if np.abs(pt.imag) > max_y: max_y = np.abs(pt.imag)
    plt.suptitle(title)

    if norm:         
        plt.xlim(-1,1)
        plt.ylim(-1,1)
    else: 
        plt.xlim(-max_x-.1*max_x,max_x+.1*max_x)
        plt.ylim(-max_y-.1*max_y,max_y+.1*max_y)
    plt.ylabel('imag(V)')
    plt.xlabel('real(V)')
    if cursor:
        from mpldatacursor import datacursor
        datacursor(formatter='{label}'.format)
    if save:
        plt.savefig(save)
    return None

def omni_view_gif(filenames, name='omni_movie.gif'):
    '''Give it a full path for images and all the images you want to use in order to use. list format.'''
    import imageio
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(name, images)
