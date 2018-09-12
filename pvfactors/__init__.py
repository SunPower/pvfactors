# -*- coding: utf-8 -*-

import logging
import sys
logging.basicConfig()

from pvfactors.version import __version__


class PVFactorsError(Exception):
    pass


class PVFactorsEdgePointDoesNotExist(Exception):
    pass


class PVFactorsArrayUpdateException(Exception):
    pass


# Define function used for progress bar when running long simulations
# Borrowed from: https://gist.github.com/aubricus
def print_progress(iteration, total, prefix='', suffix='', decimals=1,
                   bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent
        complete (Int)
        bar_length   - Optional  : character length of bar (Int)
    """
    format_str = "{0:." + str(decimals) + "f}"
    percents = format_str.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%',
                                            suffix)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()
