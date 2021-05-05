# uvtools
[![Build Status](https://travis-ci.org/HERA-Team/uvtools.svg?branch=master)](https://travis-ci.org/HERA-Team/uvtools)
[![codecov](https://codecov.io/gh/HERA-Team/uvtools/branch/main/graph/badge.svg?token=BImdA3Oz6u)](https://codecov.io/gh/HERA-Team/uvtools)

Tools useful for the handling, visualization, and analysis of interferometric data.

## Installation
Preferred method is `pip install .` (or `pip install git+https://github.com/HERA-Team/uvtools`).
This should install all dependencies.

If you use `conda` (preferred), then you may wish to install the following packages
manually before installing `uvtools` (if you don't have them already)::

    $ conda install -c conda-forge numpy scipy "aipy>=3.0rc2"
    
If you are developing `uvtools`, you will also require `nose` and `pyuvdata` to run 
tests. All of these packages can be installed with the following commands::

    $ conda create -n uvtools python=3
    $ conda activate uvtools
    $ conda env update -n uvtools -f environment.yml
    $ pip install -e . 
    
To test the package, execute the following command::

    $ nosetests uvtools/tests/test_dspec.py uvtools/tests/test_utils.py 
