from setuptools import setup
import glob
import os
import sys
import json


sys.path.append("uvtools")
from branch_scheme import branch_scheme



def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths

data_files = package_files('uvtools', 'data')

setup_args = {
    'name': 'uvtools',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/uvtools',
    'license': 'BSD',
    'description': 'Tools useful for the handling, visualization, and analysis of interferometric data.',
    'package_dir': {'uvtools': 'uvtools'},
    'packages': ['uvtools'],
    'package_data': {'uvtools': data_files},
    'include_package_data': True,
    'scripts': glob.glob('scripts/*'),
    'use_scm_version': {"local_scheme":branch_scheme},
    'install_requires':[
        'numpy',
        'six',
        'scipy',
        'setuptools_scm',
    ],
    'extras_require': {'aipy':['aipy>=3.0rc2']}
}


if __name__ == '__main__':
    setup(**setup_args)
