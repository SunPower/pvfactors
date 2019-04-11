""" Some utility functions and classes for pvfactors """

from pvfactors.config import Y_GROUND
from shapely.geometry import Point
from pvfactors import PVFactorsError
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# TODO: hard coding these values is not ideal
THRESHOLD_EDGE_POINT = 1e3


class LinePVArray(dict):
    """``LinePVArray`` is the general class that is used to instantiate all
    the initial line objects of the pv array before putting them into the
    surface registry.
    It is a sub-class of a dictionary with already defined keys.

    Parameters
    ----------
    geometry : ``shapely`` geometry object
        Python representation of pv line geometry
    style : str
        matplotlib`` plotting style for the line. E.g. '--'
    line_type : str
        type of surface in :py:class:`~pvfactors.pvarray.Array`,
        e.g. 'pvrow' or 'ground'
    shaded : bool
        specifies if surface is shaded (from direct shading)
    pvrow_index : int
        if the surface's ``line_type`` is a 'pvrow', this
        will be its pv row index (which is different from its
        ``surface_registry`` index in :py:class:`~pvfactors.pvarray.Array`)

    """

    _list_line_types = ['pvrow', 'ground', None]

    def __init__(self, geometry=None, style='-', line_type=None,
                 shaded=None, pvrow_index=None):
        if line_type in self._list_line_types:
            super(LinePVArray, self).__init__(geometry=geometry, style=style,
                                              line_type=line_type,
                                              shaded=shaded,
                                              pvrow_index=pvrow_index)
        else:
            raise PVFactorsError("'line_type' cannot be: %s, \n possible "
                                 "values are: %s" % (
                                     str(line_type),
                                     str(self._list_line_types)))


def find_edge_point(b1_pvrow, b2_pvrow):
    """Return edge point formed by pv row line and ground line. This assumes a
    flat ground surface, located at the hard-coded elevation ``Y_GROUND`` value

    Parameters
    ----------
    b1_pvrow : ``shapely.Point``
        first boundary point of the pv row line.
    b2_pvrow : ``shapely.Point``
        second boundary point of the pv row line.

    Returns
    -------
    ``shapely.Point``
        intersection point of pvrow continued line and ground line

    """

    u_vector = [b1_pvrow.x - b2_pvrow.x, b1_pvrow.y - b2_pvrow.y]
    n_vector = [u_vector[1], -u_vector[0]]
    intercept = - (n_vector[0] * b1_pvrow.x + n_vector[1] * b1_pvrow.y)

    if n_vector[0]:
        x_edge_point = - (intercept +
                          n_vector[1] * Y_GROUND) / np.float64(n_vector[0])
    else:
        x_edge_point = np.inf

    # TODO: need to find a better way to deal with this case
    if np.abs(x_edge_point) > THRESHOLD_EDGE_POINT:
        LOGGER.debug("find_edge_point: it looks like the tilt should be "
                     "approximated with a flat case")

    return Point(x_edge_point, Y_GROUND)
