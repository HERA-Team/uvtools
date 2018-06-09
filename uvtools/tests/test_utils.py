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

    templates = sorted(files + ["zen.inp.{pol}.uv"])

    # search data
    dfs, dps = uvt.utils.search_data(templates, pols)
    nt.assert_equal(len(dfs), 2)
    nt.assert_equal(len(dfs[0]), len(dfs[1]), 2)
    nt.assert_equal(len(dps), 2)
    nt.assert_equal(len(dps[0]), len(dps[1]), 2)
    nt.assert_equal(dps[0], ['xx', 'xx'])
    nt.assert_true(np.all(['.xx.' in df for df in dfs[0]]))

    # matched pols
    dfs, dps = uvt.utils.search_data(templates, pols, matched_pols=True)
    nt.assert_equal(len(dfs), 2)
    nt.assert_equal(len(dfs[0]), len(dfs[1]), 2)
    nt.assert_true(np.all(['.xx.' in df for df in dfs[0]]))
    dfs, dps = uvt.utils.search_data(files, pols + ['pI'], matched_pols=True)
    nt.assert_equal(len(dfs), 0)

    # reverse nesting
    dfs, dps = uvt.utils.search_data(templates, pols, reverse_nesting=True)
    nt.assert_equal(len(dfs), 2)
    nt.assert_equal(len(dfs[0]), len(dfs[1]), 2)
    nt.assert_true(np.all(['.bar.' in df for df in dfs[0]]))

    # flatten
    dfs, dps = uvt.utils.search_data(templates, pols, flatten=True)
    nt.assert_equal(len(dfs), 4)
    nt.assert_true(isinstance(dfs[0], (str, np.str)))

    for f in allfiles:
        if os.path.exists(f):
            os.remove(f)



