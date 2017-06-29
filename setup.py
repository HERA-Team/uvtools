from setuptools import setup
import glob
import os.path as path

__version__ = '0.0.0'

setup_args = {
    'name': 'uvtools',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/uvtools',
    'license': 'BSD',
    'description': 'Tools useful for the handling, visualization, and analysis of interferometric data.',
    'package_dir': {'uvtools': 'uvtools'},
    'packages': ['uvtools'],
    'version': __version__,
}


if __name__ == '__main__':
    apply(setup, (), setup_args)
