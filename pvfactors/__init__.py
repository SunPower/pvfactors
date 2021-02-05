# -*- coding: utf-8 -*-

from pvfactors.version import __version__
import logging
logging.basicConfig()

try:
    from shapely.geos import lgeos
except OSError as err:
    # https://github.com/SunPower/pvfactors/issues/109
    msg = (
        "pvfactors encountered an error when importing the shapely package. "
        "This often happens when a binary dependency is missing because "
        "shapely was installed from PyPI using pip. Try reinstalling shapely "
        "from another source like conda-forge with "
        "`conda install -c conda-forge shapely`, or alternatively from "
        "Christoph Gohlke's website if you're on Windows: "
        "https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely"
    )
    err.strerror += "; " + msg
    raise err


class PVFactorsError(Exception):
    pass
