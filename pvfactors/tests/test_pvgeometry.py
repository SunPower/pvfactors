# -*- coding: utf-8 -*-

"""
Test the implementation of the pvgeometry helper functions
"""

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from pvfactors.pvgeometry import PVGeometry

LIST_GEOMETRIES = [Point(1, 2), Point(3, 2),
                   LineString([Point(0, 0), Point(3, 2)]),
                   LineString([Point(0, 1), Point(1, 0)])]


def test_point_boundaries():
    """ Check that the correct boundary points are returned """
    list_points = [Point(0, 0), Point(3, 2)]
    linestring = LineString(list_points)
    registry = pd.DataFrame({'geometry': [linestring]})

    # ``boundary`` returns ``MultiPoint`` objects, and property ``geoms``
    # allows to get the actual points
    assert registry.pvgeometry.boundary[0].geoms[0] == list_points[0]
    assert registry.pvgeometry.boundary[0].geoms[1] == list_points[1]


def test_centroids():
    """ Testing that the centroid values are correctly calculated """

    list_linestrings = [LineString([Point(0, 0), Point(3, 2)]),
                        LineString([Point(0, 1), Point(1, 0)])]
    registry = pd.DataFrame({'geometry': list_linestrings})

    assert registry.pvgeometry.centroid.values[0] == Point(1.5, 1)
    assert registry.pvgeometry.centroid.values[1] == Point(0.5, 0.5)
