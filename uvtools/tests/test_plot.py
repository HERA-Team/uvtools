import matplotlib.pyplot as plt
import unittest
import uvtools as uvt
import numpy as np
from itertools import combinations


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
        
    
class TestDiffPlotters(unittest.TestCase):

    def setUp(self):

        # mock up some UVData objects
        pass

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_plot_diff_uv(self):
        # tests to make sure this function works as expected
        pass

    def test_plot_diff_waterfall(self):
        plot_types = ("time_vs_freq", "time_vs_dly", 
                      "fr_vs_freq", "fr_vs_dly")
        # get all combinations
        plot_types = [list(combinations(plot_types, r)) 
                      for r in range(len(plot_types))]

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
                                               self.antpairpol)
            
            # check that the correct number of subplots are made
            self.assertTrue(axes_contains(fig, elements))

            Nimages = 0
            Ncbars = 0
            for ax in fig.axes:
                # check that each Axes object contains either an 
                # AxesImage (from imshow) or a QuadMesh (from colorbar)
                image = [(matplotlib.image.AxesImage, 1),]
                cbar = [(matplotlib.collections.QuadMesh, 1),]
                contains_image = axes_contain(ax, image)
                contains_cbar = axes_contain(ax, cbar)
                Nimages += int(contains_image)
                Ncbars += int(contains_cbar)
                self.assertTrue(contains_image or contains_cbar)
            
            # check that the amount of colorbars and images is correct
            self.assertTrue(Nimages == Nplots)
            self.assertTrue(Ncbars == Nplots)

            # now close the figure
            plt.close(fig)

    def test_bad_metadata(self):
        # test for freq arrays not matching
        # test for time arrays not matching
        # test for baseline arrays not matching
        # anything else?
        pass


    pass


if __name__ == '__main__':
    unittest.main()

