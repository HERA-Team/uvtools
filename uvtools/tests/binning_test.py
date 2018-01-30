import unittest
import uvtools.binning as binning
import aipy
import numpy as np, random

random.seed(0)

class TestMethods(unittest.TestCase):
    def test_gen_lst_res(self):
        lstres = binning.gen_lst_res(nbins=100)
        self.assertEqual(lstres, 2*np.pi/100)
        lstres2 = binning.gen_lst_res(dlst=lstres)
        self.assertEqual(lstres, lstres2)
        lstres = binning.gen_lst_res(dlst=.01)
        self.assertEqual(lstres, 2*np.pi / 628)
        lstres = binning.gen_lst_res(secs=1.)
        self.assertEqual(lstres, 2*np.pi / int(aipy.const.sidereal_day))
    def test_get_lstsbins(self):
        lstres = binning.gen_lst_res(nbins=100)
        bins = binning.get_lstbins(lstres)
        self.assertEqual(bins.size, 100)
        np.testing.assert_allclose(bins % lstres, lstres/2)
        self.assertEqual(bins[0], lstres/2)
    def test_lstbin(self):
        lsts = np.linspace(0, 4*np.pi, 5000)
        lst_res = binning.gen_lst_res(nbins=1000)
        lstb = binning.lstbin(lsts, lst_res=lst_res)
        self.assertEqual(len(set(lstb)), 1000)
        np.testing.assert_array_less(lstb, 2*np.pi)
        self.assertTrue(np.all(lstb >= 0))
        np.testing.assert_allclose(lstb % lst_res, lst_res/2)
        lsts = np.linspace(0, 2*np.pi, 5000) % (2*np.pi)
        lst_res = binning.gen_lst_res(nbins=1000)
        self.assertEqual(binning.lstbin(0., lst_res=lst_res), lst_res/2)
        self.assertEqual(binning.lstbin(lsts[-2], lst_res=lst_res), 2*np.pi - lst_res/2)
        lstb = binning.lstbin(lsts, lst_res=lst_res)
        np.testing.assert_allclose(lsts - lstb, 0, atol=lst_res/2)
    def test_uv2bin(self):
        lst_res = binning.gen_lst_res(nbins=1000)
        u,v,lst = binning.uv2bin(0.,0.,0, uv_res=1., lst_res=lst_res)
        self.assertEqual((u,v,lst), (0., 0., lst_res/2.))
        u,v,lst = binning.uv2bin(0.4,0.4,0, uv_res=1., lst_res=lst_res)
        self.assertEqual((u,v,lst), (0., 0., lst_res/2.))
        u,v,lst = binning.uv2bin(0.6,0.6,0, uv_res=1., lst_res=lst_res)
        self.assertEqual((u,v,lst), (1., 1., lst_res/2.))
        

if __name__ == '__main__':
    unittest.main()
