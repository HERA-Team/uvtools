import glob
import os
import sys

import numpy as np
import pytest

from .. import utils


def test_search_data():
    # setup data
    pols = ["xx", "yy"]
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
    dfs, dps = utils.search_data(templates, pols)
    assert len(dfs) == 2
    assert len(dfs[0]) == len(dfs[1])
    assert len(dfs[1]) == 2
    assert len(dps) == 2
    assert len(dps[0]) == len(dps[1])
    assert len(dps[0]) == 2
    assert dps[0], ["xx" == "xx"]
    assert np.all([".xx." in df for df in dfs[0]])

    # matched pols
    dfs, dps = utils.search_data(templates, pols, matched_pols=True)
    assert len(dfs) == 2
    assert len(dfs[0]) == len(dfs[1])
    assert len(dfs[0]) == 2
    assert np.all([".xx." in df for df in dfs[0]])
    dfs, dps = utils.search_data(files, pols + ["pI"], matched_pols=True)
    assert len(dfs) == 0

    # reverse nesting
    dfs, dps = utils.search_data(templates, pols, reverse_nesting=True)
    assert len(dfs) == 2
    assert len(dfs[0]) == len(dfs[1])
    assert len(dfs[1]) == 2
    assert np.all([".bar." in df for df in dfs[0]])

    # flatten
    dfs, dps = utils.search_data(templates, pols, flatten=True)
    assert len(dfs) == 4
    assert isinstance(dfs[0], str)

    for f in allfiles:
        if os.path.exists(f):
            os.remove(f)
