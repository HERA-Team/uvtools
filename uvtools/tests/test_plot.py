import matplotlib
import matplotlib.pyplot as plt
import unittest
import uvtools as uvt
import numpy as np


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
        # test all different plotting modes
        pass

    def test_bad_metadata(self):
        # test for freq arrays not matching
        # test for time arrays not matching
        # test for baseline arrays not matching
        # anything else?
        pass


    pass


if __name__ == '__main__':
    unittest.main()

