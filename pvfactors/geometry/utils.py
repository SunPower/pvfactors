"""Utility functions for geometrical calculations."""

import numpy as np
from pvfactors import PVFactorsError
from pvfactors.config import TOL_COLLINEAR
from shapely.geometry import Point, GeometryCollection


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


def projection(point, vector, linestring):
    """Projection of point along vector onto a linestring.
    Define equations of two lines:
    - one defined by point and vector: a*x + b*y + c = 0
    - one defined by linestring: d*x + e*y + f = 0

    then if the lines are not parallel, the interesction is defined by:
    X = - W^(-1) . B, where X = [x, y], W = [[a, b], [c, d]], B = [c, f]

    Do not use if the two lines are parallel (determinant is 0, W not
    invertible)
    """
    # Define equation a*x + b*y + c = 0
    a, b = -vector[1], vector[0]
    c = - (a * point.x + b * point.y)
    # Define equation d*x + e*y +f = 0
    b1, b2 = linestring.boundary
    d, e = - (b2.y - b1.y), b2.x - b1.x
    f = - (d * b1.x + e * b1.y)
    # TODO: check that two lines are not parallel
    W = [[a, b], [d, e]]
    B = [c, f]
    x, y = - np.linalg.inv(W).dot(B)
    pt_intersection = Point(x, y)
    if linestring.contains(pt_intersection):
        return pt_intersection
    else:
        return GeometryCollection()
