"""Utility functions for geometrical calculations."""

import numpy as np
from pvfactors import PVFactorsError
from pvfactors.config import \
    TOL_COLLINEAR, DISTANCE_TOLERANCE, THRESHOLD_DISTANCE_TOO_CLOSE
from shapely.geometry import \
    Point, GeometryCollection, LineString, MultiLineString, MultiPoint


def difference(u, v):
    """Calculate difference between two lines, avoiding shapely float
    precision errors"""
    ub1, ub2 = u.boundary
    vb1, vb2 = v.boundary
    u_contains_vb1 = contains(u, vb1)
    u_contains_vb2 = contains(u, vb2)
    v_contains_ub1 = contains(v, ub1)
    v_contains_ub2 = contains(v, ub2)

    if u_contains_vb1:
        if u_contains_vb2:
            l_tmp = LineString([ub1, vb1])
            if contains(l_tmp, vb2):
                coords = [MultiPoint([ub1, vb2]), MultiPoint([vb1, ub2])]
            else:
                coords = [MultiPoint([ub1, vb1]), MultiPoint([vb2, ub2])]
            return MultiLineString(coords)
        elif v_contains_ub1:
            return LineString([vb1, ub2])
        elif v_contains_ub2:
            return LineString([ub1, vb1])
        else:
            return u
    elif u_contains_vb2:
        if v_contains_ub1:
            return LineString([vb2, ub2])
        elif v_contains_ub2:
            return LineString([ub1, vb2])
        else:
            return u
    else:
        return u


def contains(linestring, point, tol_distance=DISTANCE_TOLERANCE):
    """Fixing floating point errors obtained in shapely for contains"""
    return linestring.distance(point) < tol_distance


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
    length_linestring = linestring.length
    # For the following, using linestring.contains(pt_intersection) leads
    # to wrong assessments sometimes, probably because of round off errors
    distance_to_b1 = b1.distance(pt_intersection)
    distance_to_b2 = b2.distance(pt_intersection)
    contained_by_linestring = (
        (linestring.distance(pt_intersection) < DISTANCE_TOLERANCE) and
        (distance_to_b1 <= length_linestring) and
        (distance_to_b2 <= length_linestring))
    if contained_by_linestring:
        # Check that the intersection is not too close to a boundary: if it
        # is it can create a "memory access error" it seems
        too_close_to_b1 = distance_to_b1 < DISTANCE_TOLERANCE
        too_close_to_b2 = distance_to_b2 < DISTANCE_TOLERANCE
        if too_close_to_b1:
            return b1
        elif too_close_to_b2:
            return b2
        else:
            # Need this, otherwise all future intersection and contain
            # assertions will fail
            shapely_contains = linestring.contains(pt_intersection)
            if not shapely_contains:
                # project pt_intersection onto linestring
                print("######## PROJECTING")
                projected_distance = linestring.project(pt_intersection)
                new_point = linestring.interpolate(projected_distance)
                return new_point
            else:
                return pt_intersection
    else:
        return GeometryCollection()