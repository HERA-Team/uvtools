import nose.tools as nt
import uvtools as uvt
import numpy as np
import glob
import os
import sys

def test_search_data():
    # setup data
    pols = ['xx', 'yy']
    files = ["zen.foo.{pol}.uv", "zen.bar.{pol}.uv"]
    allfiles = []

    # create data
    for p in pols:
        for f in files:
            df = f.format(pol=p)
            allfiles.append(df)
            with open(df, "w"):
                pass

    # search data
    datafiles = uvt.utils.search_data(files, pols)
    nt.assert_equal(len(datafiles), 2)
    nt.assert_equal(len(datafiles[0]), len(datafiles[1]), 2)
    nt.assert_true(np.all(['.xx.' in df for df in datafiles[0]]))

    # matched pols
    datafiles = uvt.utils.search_data(files, pols, matched_pols=True)
    nt.assert_equal(len(datafiles), 2)
    nt.assert_equal(len(datafiles[0]), len(datafiles[1]), 2)
    nt.assert_true(np.all(['.xx.' in df for df in datafiles[0]]))
    datafiles = uvt.utils.search_data(files, pols + ['pI'], matched_pols=True)
    nt.assert_equal(len(datafiles), 0)

    # reverse nesting
    datafiles = uvt.utils.search_data(files, pols, reverse_nesting=True)
    nt.assert_equal(len(datafiles), 2)
    nt.assert_equal(len(datafiles[0]), len(datafiles[1]), 2)
    nt.assert_true(np.all(['.bar.' in df for df in datafiles[0]]))

    # flatten
    datafiles = uvt.utils.search_data(files, pols, flatten=True)
    nt.assert_equal(len(datafiles), 4)
    nt.assert_true(isinstance(datafiles[0], (str, np.str)))

    for f in allfiles:
        if os.path.exists(f):
            os.remove(f)



