"""Utility functions for geometrical calculations."""

import numpy as np
from pvfactors import PVFactorsError
from pvfactors.config import TOL_COLLINEAR, DISTANCE_TOLERANCE


def is_collinear(list_elements):
    """Check that all :py:class:`~pvfactors.pvsurface.PVSegment`
    or :py:class:`~pvfactors.pvsurface.PVSurface` objects in list
    are collinear"""
    is_col = True
    u_direction = None  # will be orthogonal to normal vector
    for element in list_elements:
        if not element.is_empty:
            if u_direction is None:
                # set u_direction if not defined already
                u_direction = np.array([- element.n_vector[1],
                                        element.n_vector[0]])
            else:
                # check that collinear
                dot_prod = u_direction.dot(element.n_vector)
                is_col = np.abs(dot_prod) < TOL_COLLINEAR
                if not is_col:
                    return is_col
    return is_col


def check_collinear(list_elements):
    """Raise error if all :py:class:`~pvfactors.pvsurface.PVSegment`
    or :py:class:`~pvfactors.pvsurface.PVSurface` objects in list
    are not collinear"""
    is_col = is_collinear(list_elements)
    if not is_col:
        msg = "All elements should be collinear"
        raise PVFactorsError(msg)


def are_2d_vecs_collinear(u1, u2):
    """Check that two 2D vectors are collinear"""
    n1 = np.array([-u1[1], u1[0]])
    dot_prod = n1.dot(u2)
    return np.abs(dot_prod) < TOL_COLLINEAR
