import unittest
import uvtools as uvt
import numpy as np

BACKEND = 'Agg'
#BACKEND = 'MacOSX'

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
        matplotlib.use(BACKEND)
        data = np.ones((10,10)) - 1j*np.ones((10,10))
        for mode in ('abs','log','phs','real','imag'):
            uvt.plot.waterfall(data, mode=mode)
            matplotlib.pyplot.show()
            matplotlib.pyplot.clf()
        
    
if __name__ == '__main__':
    unittest.main()

