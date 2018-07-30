# -*- coding: utf-8 -*-

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

    assert registry.pvgeometry.boundary[0].geoms[0] == list_points[0]
    assert registry.pvgeometry.boundary[0].geoms[1] == list_points[1]
