from . import dspec, plot, utils

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # pragma: no cover
    try:
        __version__ = version("uvtools")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "unknown"
