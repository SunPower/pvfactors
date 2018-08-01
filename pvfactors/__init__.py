# -*- coding: utf-8 -*-

import logging
logging.basicConfig()

from pvfactors.version import __version__


class PVFactorsError(Exception):
    pass


class PVFactorsEdgePointDoesNotExist(Exception):
    pass


class PVFactorsArrayUpdateException(Exception):
    pass
