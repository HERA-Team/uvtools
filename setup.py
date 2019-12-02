from setuptools import setup
import glob
import os
import sys
import json

sys.path.append("uvtools")
import version

data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(os.path.join('uvtools', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)

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
    'version': version.version,
    'include_package_data': True,
    'scripts': glob.glob('scripts/*'),
    'install_requires':[
        'numpy',
        'aipy>=3.0rc2',
        'six',
        'scipy',
        'basemap'
    ]
}


if __name__ == '__main__':
    setup(**setup_args)
