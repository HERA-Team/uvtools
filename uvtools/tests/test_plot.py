import nose.tools as nt
import matplotlib
import matplotlib.pyplot as plt
import unittest
import uvtools as uvt
import numpy as np
from itertools import combinations
from astropy import units


def axes_contains(ax, obj_list):
    """Check that a matplotlib.Axes instance contains certain elements.

    This function was taken directly from the ``test_plot`` module of 
    ``hera_pspec``.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes instance.
    obj_list : list of tuples
        List of tuples, one for each type of object to look for. The tuple
        should be of the form (matplotlib.object, int), where int is the
        number of instances of that object that are expected.
    """
    # Get plot elements
    elems = ax.get_children()

    # Loop over list of objects that should be in the plot
    contains_all = False
    for obj in obj_list:
        objtype, num_expected = obj
        num = 0
        for elem in elems:
            if isinstance(elem, objtype): num += 1
        if num != num_expected:
            return False

    # Return True if no problems found
    return True

class TestMethods(unittest.TestCase):
    
    def test_data_mode(self):
        data = np.ones(100) - 1j*np.ones(100)
        d = uvt.plot.data_mode(data, mode='abs')
        self.assertTrue(np.all(d == np.sqrt(2)))
        d = uvt.plot.data_mode(data, mode='log')
        self.assertTrue(np.all(d == np.log10(np.sqrt(2))))
        d = uvt.plot.data_mode(data, mode='phs')
        self.assertTrue(np.all(d == -np.pi/4))
        d = uvt.plot.data_mode(data, mode='real')
        self.assertTrue(np.all(d == 1))
        d = uvt.plot.data_mode(data, mode='imag')
        self.assertTrue(np.all(d == -1))
        self.assertRaises(ValueError, uvt.plot.data_mode, data, mode='')
    
    def test_waterfall(self):
        import matplotlib
        data = np.ones((10,10)) - 1j*np.ones((10,10))
        for mode in ('abs','log','phs','real','imag'):
            uvt.plot.waterfall(data, mode=mode)
            #matplotlib.pyplot.show()
            matplotlib.pyplot.clf()
    
    def test_plot_antpos(self):
        antpos = {i: [i,i,0] for i in range(10)}
        import matplotlib
        uvt.plot.plot_antpos(antpos)
        #matplotlib.pyplot.show()
        
class TestFancyPlotters(unittest.TestCase):
    def setUp(self):
        import hera_sim
        sim = hera_sim.Simulator(
            n_freq=100,
            n_times=50,
            antennas={0: [0,0,0], 1: [14.6,0,0]}
        )
        sim.add_eor("noiselike_eor")
        self.uvd = sim.data

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_labeled_waterfall(self):
        # Most of this functionality is tested in the next test function,
        # so this will test some features not exposed in
        # plot.fourier_transform_waterfalls
        uvd = self.uvd
        
        # Dynamic range setting.
        fig, ax = uvt.plot.labeled_waterfall(
            uvd,
            antpairpol=(0,1,"xx"),
            mode="phs",
            vmin=-1,
            vmax=1,
        )
        image = ax.get_images()[0]
        assert image.cmap.name == "RdBu"
        assert np.allclose(image.get_clim(), (-1, 1))

        fig, ax = uvt.plot.labeled_waterfall(
            uvd,
            antpairpol=(0,1,"xx"),
            mode="log",
            vmin=-7,
            dynamic_range=2,
        )
        image = ax.get_images()[0]
        assert image.cmap.name == "inferno"
        assert np.allclose(image.get_clim(), (-7, -5))

        fig, ax = uvt.plot.labeled_waterfall(
            uvd,
            antpairpol=(0,1,"xx"),
            mode="log",
            vmax=-5,
            dynamic_range=1,
        )
        image = ax.get_images()[0]
        assert np.allclose(image.get_clim(), (-6, -5))

    def test_fourier_transform_waterfalls(self):
        uvd = self.uvd
        data = uvd.get_data(0,1,'xx')
        freqs = np.unique(uvd.freq_array) # Hz
        times = np.unique(uvd.time_array) # JD
        lsts = np.unique(uvd.lst_array) * 24 / (2 * np.pi) # hours
        delays = uvt.utils.fourier_freqs(freqs) * 1e9 # ns
        fringe_rates = uvt.utils.fourier_freqs(times * 24 * 3600) * 1e3 # mHz

        f1, f2 = freqs[10] / 1e6, freqs[90] / 1e6
        lst1, lst2 = lsts[10], lsts[40]
        dly1, dly2 = delays[10], delays[90]
        fr1, fr2 = fringe_rates[10], fringe_rates[40]

        plot_limits = {
            "time": (lst2, lst1),
            "freq": (f1, f2),
            "fringe-rate": (fr2, fr1),
            "delay": (dly1, dly2)
        }

        # Test that it works passing a UVData object.
        fig = uvt.plot.fourier_transform_waterfalls(
            data=uvd,
            antpairpol=(0,1,'xx'),
            plot_limits=plot_limits,
            time_or_lst="time",
        )
        axes = fig.get_axes()
        xlabels = list(ax.get_xlabel() for ax in axes)
        ylabels = list(ax.get_ylabel() for ax in axes)
        xlimits = list(ax.get_xlim() for ax in axes)
        ylimits = list(ax.get_ylim() for ax in axes)

        assert sum("JD" in ylabel for ylabel in ylabels) == 2
        assert sum("Fringe Rate" in ylabel for ylabel in ylabels) == 2
        assert sum("Frequency" in xlabel for xlabel in xlabels) == 2
        assert sum("Delay" in xlabel for xlabel in xlabels) == 2

        assert sum(np.allclose((lst2, lst1), ylim) for ylim in ylimits) == 2
        assert sum(np.allclose((f1, f2), xlim) for xlim in xlimits) == 2
        assert sum(np.allclose((fr2, fr1), ylim) for ylim in ylimits) == 2
        assert sum(np.allclose((dly1, dly2), xlim) for xlim in xlimits) == 2

        # Now test with an array.
        plot_times = times - int(times[0]) # For bound checking later
        lsts = np.unique(uvd.lst_array) # Ensure they're in radians
        fringe_rates = uvt.utils.fourier_freqs(times * units.day.to("s")) # Hz
        delays = uvt.utils.fourier_freqs(freqs) # s

        fig = uvt.plot.fourier_transform_waterfalls(data=data, freqs=freqs, lsts=lsts)
        axes = fig.get_axes()
        ylabels = list(ax.get_ylabel() for ax in axes)
        assert sum("LST" in ylabel for ylabel in ylabels) == 2

        fig = uvt.plot.fourier_transform_waterfalls(data=data, freqs=freqs, times=times)
        axes = fig.get_axes()
        ylabels = list(ax.get_ylabel() for ax in axes)
        assert sum("JD" in ylabel for ylabel in ylabels) == 2

        # Check custom data units.
        fig = uvt.plot.fourier_transform_waterfalls(
            data=data, freqs=freqs, lsts=lsts, data_units="mK sr"
        )
        axes = fig.get_axes()
        ylabels = list(ax.get_ylabel() for ax in axes)
        assert sum("mK sr" in ylabel for ylabel in ylabels) == 4
        
        # Check custom plot units.
        plot_units = {
            "time": "hour",
            "lst": "deg",
            "freq": "GHz",
            "fringe-rate": "Hz",
            "delay": "us",
        }
        lstmin, lstmax = np.array([lsts.min(), lsts.max()]) * units.rad.to("deg")
        tmin, tmax = np.array([plot_times.min(), plot_times.max()]) * units.day.to("hr")
        fmin, fmax = np.array([freqs.min(), freqs.max()]) * 1e-9  # GHz
        frmin, frmax = np.array([fringe_rates.min(), fringe_rates.max()])  # Hz
        dlymin, dlymax = np.array([delays.min(), delays.max()]) * 1e6  # us

        fig = uvt.plot.fourier_transform_waterfalls(
            data=data, freqs=freqs, lsts=lsts, plot_units=plot_units
        )
        axes = fig.get_axes()
        xlabels = list(ax.get_xlabel() for ax in axes)
        ylabels = list(ax.get_ylabel() for ax in axes)
        for xdim in ("freq", "delay"):
            assert sum(f"[{plot_units[xdim]}]" in xlabel for xlabel in xlabels) == 2
        for ydim in ("lst", "fringe-rate"):
            assert sum(f"[{plot_units[ydim]}]" in ylabel for ylabel in ylabels) == 2

        xlimits = list(ax.get_xlim() for ax in axes)
        ylimits = list(ax.get_ylim() for ax in axes)
        assert sum(np.allclose(xlims, (fmin, fmax)) for xlims in xlimits) == 2
        assert sum(np.allclose(xlims, (dlymin, dlymax)) for xlims in xlimits) == 2
        assert sum(np.allclose(ylims, (frmax, frmin), rtol=0.01) for ylims in ylimits) == 2
        assert sum(np.allclose(ylims, (lstmax, lstmin)) for ylims in ylimits) == 2

        fig = uvt.plot.fourier_transform_waterfalls(
            data=data, freqs=freqs, times=times, plot_units=plot_units
        )
        axes = fig.get_axes()
        # Already checked everything but time units, so only check that.
        ylabels = list(ax.get_ylabel() for ax in axes)
        assert sum(f"[{plot_units['time']}]" in ylabel for ylabel in ylabels) == 2
        
        ylimits = list(ax.get_ylim() for ax in axes)
        assert sum(np.allclose(ylims, (tmax, tmin)) for ylims in ylimits) == 2

        # Do some exception raising checking.
        with nt.assert_raises(ValueError):
            uvt.plot.fourier_transform_waterfalls(
                data=uvd, antpairpol=(0,1,'xx'), time_or_lst="nan"
            )

        with nt.assert_raises(TypeError):
            uvt.plot.fourier_transform_waterfalls(data={})

        with nt.assert_raises(TypeError):
            uvt.plot.fourier_transform_waterfalls(
                data=data, freqs=freqs, lsts=lsts, plot_units="bad_type"
            )

        with nt.assert_raises(ValueError):
            uvt.plot.fourier_transform_waterfalls(data=np.ones((3,5,2), dtype=np.complex))

        with nt.assert_raises(ValueError):
            uvt.plot.fourier_transform_waterfalls(data=data, freqs=freqs)

        with nt.assert_raises(ValueError):
            uvt.plot.fourier_transform_waterfalls(data=data, times=times)

        with nt.assert_raises(ValueError):
            uvt.plot.fourier_transform_waterfalls(data=uvd)

        with nt.assert_raises(TypeError):
            uvt.plot.fourier_transform_waterfalls(data=np.ones((15,20), dtype=np.float))


class TestDiffPlotters(unittest.TestCase):

    def setUp(self):
        # make some mock data
        import hera_sim
        import copy
        # first, make an array
        antennas = hera_sim.antpos.hex_array(3)
        # now make a slightly offset array
        dx = lambda : np.random.random(3) * 0.1
        offset_ants = {ant : pos + dx() for ant, pos in antennas.items()}
        # now make an array with mismatched number of antennas
        bad_ants = hera_sim.antpos.hex_array(3, split_core=False)
        # choose two different integration times
        dt1 = 10.7
        dt2 = 15.3
        # choose two different channel widths
        df1 = 1e8 / 1024
        df2 = 2e8 / 1024
        # actually mock up the data
        sim = hera_sim.Simulator(n_freq=10, n_times=10, 
                                 antennas=antennas,
                                 integration_time=dt1, 
                                 channel_width=df1)
        self.uvd1 = copy.deepcopy(sim.data)
        sim.add_eor("noiselike_eor")
        self.uvd2 = copy.deepcopy(sim.data)
        # now just make some things with metadata that will raise exceptions

        # make the visibility units disagree
        sim.data.vis_units = 'mK'
        self.uvd_bad_vis_units = copy.deepcopy(sim.data)
        # mismatched baselines
        sim = hera_sim.Simulator(n_freq=10, n_times=10, 
                                 antennas=offset_ants,
                                 integration_time=dt1,
                                 channel_width=df1)
        self.uvd_bad_bls = copy.deepcopy(sim.data)
        # wrong number of antennas
        sim = hera_sim.Simulator(n_freq=10, n_times=10, 
                                 antennas=bad_ants,
                                 integration_time=dt1, 
                                 channel_width=df1)
        self.uvd_bad_ants = copy.deepcopy(sim.data)
        # bad Nfreq
        sim = hera_sim.Simulator(n_freq=50, n_times=10,
                                 antennas=antennas,
                                 integration_time=dt1,
                                 channel_width=df1)
        self.uvd_bad_Nfreq = copy.deepcopy(sim.data)
        # bad Ntimes
        sim = hera_sim.Simulator(n_freq=10, n_times=50, 
                                 antennas=antennas,
                                 integration_time=dt1, 
                                 channel_width=df1)
        self.uvd_bad_Ntimes = copy.deepcopy(sim.data)
        # bad integration time
        sim = hera_sim.Simulator(n_freq=10, n_times=10, 
                                 antennas=antennas,
                                 integration_time=dt2, 
                                 channel_width=df1)
        self.uvd_bad_int_time = copy.deepcopy(sim.data)
        # bad channel width
        sim = hera_sim.Simulator(n_freq=10, n_times=10, 
                                 antennas=antennas,
                                 integration_time=dt1, 
                                 channel_width=df2)
        self.uvd_bad_chan_width = copy.deepcopy(sim.data)

        # choose an antenna pair and polarization for later
        self.antpairpol = (0, 1, "xx")

        # make a simulation for the plot_diff_1d test
        sim = hera_sim.Simulator(
            n_freq=100, n_times=2, antennas=antennas
        )
        sim.add_eor("noiselike_eor")
        self.sim = copy.deepcopy(sim)

        # make some UVData objects collapsed along a single axis
        sim = hera_sim.Simulator(n_freq=10, n_times=1,
                                 antennas=antennas,
                                 integration_time=dt1,
                                 channel_width=df1)
        self.uvd_1d_times = copy.deepcopy(sim.data)

        sim = hera_sim.Simulator(n_freq=1, n_times=10,
                                 antennas=antennas,
                                 integration_time=dt1,
                                 channel_width=df1)
        self.uvd_1d_freqs = copy.deepcopy(sim.data)

        antennas_ = {0 : [0, 0, 0], 1 : [10, 0, 0]}
        sim = hera_sim.Simulator(n_freq=10, n_times=1,
                                 antennas=antennas_,
                                 integration_time=dt1,
                                 channel_width=df1,
                                 no_autos=True)
        self.uvd_1d_uvws = copy.deepcopy(sim.data)

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_plot_diff_1d(self):
        # list possible plot types and dimensions
        plot_types = ("normal", "fourier", "both")
        dimensions = ("time", "freq")
        duals = {"time" : "fringe rate", "freq" : "delay"}

        # loop over all the choices
        for plot_type in plot_types:
            Nplots = 6 if plot_type == "both" else 3
            elements = [(plt.Subplot, Nplots),]
            for dimension in dimensions:
                fig = uvt.plot.plot_diff_1d(
                    self.uvd1, self.uvd2, self.antpairpol, 
                    plot_type=plot_type, dimension=dimension
                )

                # check the number of plots is correct
                self.assertTrue(axes_contains(fig, elements))

                # check that the plots are labeled correctly
                for i, ax in enumerate(fig.axes):
                    xlabel = ax.get_xlabel().lower()

                    # find out what the dimension should be
                    if plot_type == "normal":
                        dim = dimension
                    elif plot_type == "fourier":
                        dim = duals[dimension]
                    else:
                        dim = dimension if i // 3 == 0 else duals[dimension]
                    
                    # account for the fact that it plots against lst if
                    # plotting along the time axis
                    dim = "lst" if dim == "time" else dim

                    # make sure that the label is correct
                    self.assertTrue(xlabel.startswith(dim))

        plt.close(fig)

        # now test the auto-dimension-choosing feature

        # make just one row of plots
        fig = uvt.plot.plot_diff_1d(
            self.sim.data, self.sim.data, self.antpairpol, plot_type="normal"
        )

        # make sure that it's plotting in frequency space
        ax = fig.axes[0]
        xlabel = ax.get_xlabel().lower()
        self.assertTrue(xlabel.startswith('freq'))

        # check that it works when an axis has length 1
        fig = uvt.plot.plot_diff_1d(
                self.uvd_1d_freqs, self.uvd_1d_freqs, self.antpairpol, 
                plot_type="normal"
        )

    def test_plot_diff_uv(self):
        # plot something
        fig = uvt.plot.plot_diff_uv(self.uvd1, self.uvd2)
        # check for six instances of subplots, one per image and
        # one per colorbar
        elements = [(plt.Subplot, 6),]
        self.assertTrue(axes_contains(fig, elements))
        
        # now check that we get three images and three colorbars
        Nimages = 0
        Ncbars = 0
        for ax in fig.axes:
            image = [(matplotlib.image.AxesImage, 1),]
            cbar = [(matplotlib.collections.QuadMesh, 1),]
            contains_image = axes_contains(ax, image)
            contains_cbar = axes_contains(ax, cbar)
            self.assertTrue(contains_image or contains_cbar)
            Nimages += int(contains_image)
            Ncbars += int(contains_cbar)
        
        self.assertTrue(Nimages == 3)
        self.assertTrue(Ncbars == 3)

        # close the figure
        plt.close(fig)


    def test_plot_diff_waterfall(self):
        plot_types = ("time_vs_freq", "time_vs_dly", 
                      "fr_vs_freq", "fr_vs_dly")
        # get all combinations
        plot_types = [list(combinations(plot_types, r)) 
                      for r in range(1, len(plot_types) + 1)]

        # unpack the nested list
        plot_types = [item for items in plot_types for item in items]

        # loop over all combinations of plot types, check that the 
        # right number of subplots are made, noting how many different 
        # differences are taken and how many plot types there are
        # also account for colorbars techincally being subplots
        for plot_type in plot_types:
            # each plot_type is a list; 3 differences
            Nplots = 3 * len(plot_type)
            # each plot consists of an image and a colorbar
            # so we need to count the colorbars as well
            Nsubplots = 2 * Nplots
            # make the list of objects to search for
            elements = [(plt.Subplot, Nsubplots),]
            
            # actually make the plot
            fig = uvt.plot.plot_diff_waterfall(self.uvd1, self.uvd2, 
                                               self.antpairpol,
                                               plot_type=plot_type)
            
            # check that the correct number of subplots are made
            self.assertTrue(axes_contains(fig, elements))

            Nimages = 0
            Ncbars = 0
            for ax in fig.axes:
                # check that each Axes object contains either an 
                # AxesImage (from imshow) or a QuadMesh (from colorbar)
                image = [(matplotlib.image.AxesImage, 1),]
                cbar = [(matplotlib.collections.QuadMesh, 1),]
                contains_image = axes_contains(ax, image)
                contains_cbar = axes_contains(ax, cbar)
                Nimages += int(contains_image)
                Ncbars += int(contains_cbar)
                self.assertTrue(contains_image or contains_cbar)
            
            # check that the amount of colorbars and images is correct
            self.assertTrue(Nimages == Nplots)
            self.assertTrue(Ncbars == Nplots)

            # now close the figure
            plt.close(fig)

    def test_plot_diff_waterfall_with_tapers(self):
        # since the above test makes sure the figures are correctly configured,
        # this one will just make sure nothing breaks when a taper is specified
        fig = uvt.plot.plot_diff_waterfall(
            self.uvd1, self.uvd2, self.antpairpol, freq_taper='blackman-harris',
            time_taper='hann'
        )

        plt.close(fig)

    def test_check_metadata(self):
        for attr, value in self.__dict__.items():
            if attr.startswith("uvd_1d"):
                uvt.utils.check_uvd_pair_metadata(value, value)
                continue
            if not attr.startswith("uvd_bad"):
                continue
            nt.assert_raises(uvt.utils.MetadataError, 
                             uvt.plot.plot_diff_uv,
                             self.uvd1, value,
                             check_metadata=True)


if __name__ == '__main__':
    unittest.main()

