# -*- coding: utf-8 -*-

"""
Test the implementation of the pvgeometry helper functions
"""

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from pvfactors.pvgeometry import PVGeometry
from pvfactors.pvarray import ArrayBase
from pvfactors.pvcore import LinePVArray
import warnings
warnings.filterwarnings("always")

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


def test_add_linepvarray_to_registry():
    """ Testing that the lines are added correctly to the registry """

    linestring = LineString([Point(0, 1), Point(1, 0)])
    registry = ArrayBase.initialize_line_registry()
    line_pvarray = LinePVArray(
        geometry=linestring,
        line_type='ground',
        shaded=True)
    _ = registry.pvgeometry.add([line_pvarray])

    assert registry.geometry.values[0] == linestring


def test_bounds():
    """ Testing that the geometry bounds are calculated correctly """

    linestring = LineString([Point(0, 1), Point(1, 0)])
    registry = ArrayBase.initialize_line_registry()
    line_pvarray = LinePVArray(
        geometry=linestring,
        line_type='ground',
        shaded=True)
    _ = registry.pvgeometry.add([line_pvarray])
    assert registry.pvgeometry.bounds.values[0].tolist() == [0., 0., 1., 1.]


def test_split_ground_geometry_from_edge_points():
    """ Testing that the geometry bounds are calculated correctly """

    linestring = LineString([Point(0, 0), Point(2, 0)])
    registry = ArrayBase.initialize_line_registry()
    line_pvarray = LinePVArray(
        geometry=linestring,
        line_type='ground',
        shaded=True)
    _ = registry.pvgeometry.add([line_pvarray])

    list_edge_points = [Point(1, 0)]
    registry.pvgeometry.split_ground_geometry_from_edge_points(
        list_edge_points)

    assert (registry.geometry.values[0] == LineString([(0, 0), (1, 0)]))
    assert (registry.geometry.values[1] == LineString([(1, 0), (2, 0)]))


def test_split_pvrow_geomtry():
    """ Testing that the pvrow geometry is split up correctly by line """

    linestring = LineString([Point(0, 0), Point(2, 2)])
    registry = ArrayBase.initialize_line_registry()
    line_pvarray = LinePVArray(
        geometry=linestring,
        line_type='pvrow',
        shaded=True)
    _ = registry.pvgeometry.add([line_pvarray])
    surface_side = 'front'
    registry.loc[:, 'surface_side'] = surface_side
    registry.loc[:, 'pvrow_index'] = 0.

    linestring_shadow = LineString([Point(0, 2), Point(2, 0)])

    # Split pvrow using shadow linestring
    registry.pvgeometry.split_pvrow_geometry(0, linestring_shadow,
                                             Point(2, 2), surface_side)

    assert registry.geometry.values[1] == LineString([(1, 1), (2, 2)])
    assert registry.geometry.values[0] == LineString([(0, 0), (1, 1)])


def test_cut_pvrow_geometry():
    """ Testing that the pvrow geometry is discretized as expected """

    linestring = LineString([Point(0, 0), Point(2, 2)])
    registry = ArrayBase.initialize_line_registry()
    line_pvarray = LinePVArray(
        geometry=linestring,
        line_type='pvrow',
        shaded=True)
    _ = registry.pvgeometry.add([line_pvarray])
    registry['surface_side'] = 'front'
    registry['pvrow_index'] = 0

    list_points = [Point(1, 1)]
    pvrow_index = 0
    side = 'front'

    registry.pvgeometry.cut_pvrow_geometry(list_points, pvrow_index,
                                           side)

    assert registry.geometry.values[1] == LineString([(1, 1), (2, 2)])
    assert registry.geometry.values[0] == LineString([(0, 0), (1, 1)])
