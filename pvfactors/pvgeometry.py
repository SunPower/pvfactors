# -*- coding: utf-8 -*-

"""
In ``pvfactors``, we need a way to store efficiently ``shapely`` geometries in
Dataframes and perform only basic geometric operations. So in this case,
using a complete package like ``geopandas`` may be overkill
since it brings with it a lot of dependencies, and may complicate
the integration of the package in other open-source projects.
Here we implement the basic functionalities of geopandas that are useful in
pvfactors.

"""

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from pvfactors import PVFactorsError
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


# TODO: remove hard-coding if possible
DISTANCE_TOLERANCE = 1e-8
THRESHOLD_DISTANCE_TOO_CLOSE = 1e-10


@pd.api.extensions.register_dataframe_accessor("pvgeometry")
class PVGeometry(object):

    def __init__(self, pandas_obj):
        """
        Lightweight extension of ``pandas.DataFrame`` that copies
        functionalities of ``geopandas.GeoSeries``

        :param pandas_obj: ``pandas`` object to perform operations on
        :type pandas_obj: ``pandas.DataFrame``
        """
        self._obj = pandas_obj

    def _series_op(self, this, other, op, **kwargs):
        """
        Geometric operation that returns a pandas Series

        :param this: dataframe that *must* have a `geometry` column
            containing the geometry objects
        :type this: :class:``pandas.DataFrame``
        :param other: generally a ``shapely`` object
        :param str op: operation to perform
        :return: a :class:``pandas.Series`` object
        """
        null_val = False if op not in ['distance', 'project'] else np.nan

        return pd.Series([getattr(s, op)(other, **kwargs) if s else null_val
                          for s in this.geometry],
                         index=this.index, dtype=np.dtype(type(null_val)))

    def _geo_unary_op(self, this, op):
        """
        Unary operation that returns a pandas Series

        :param this: dataframe that *must* have a `geometry` column
            containing the geometry objects
        :type this: :class:``pandas.DataFrame``
        :param str op: operation to perform
        :return: a :class:``pandas.Series`` object
        """
        # TODO: may need a dtype argument in the returned series
        return pd.Series([getattr(geom, op) for geom in this.geometry],
                         index=this.index)

    def _series_unary_op(self, this, op, null_value=False):
        """
        Unary operation that returns a pandas Series

        :param this: dataframe that *must* have a `geometry` column
            containing the geometry objects
        :type this: :class:``pandas.DataFrame``
        :param str op: operation to perform
        :return: a :class:``pandas.Series`` object
        """
        return pd.Series([getattr(geom, op, null_value)
                          for geom in this.geometry],
                         index=this.index, dtype=np.dtype(type(null_value)))

    @property
    def area(self):
        """Returns a ``Series`` containing the area of each geometry in the
        ``GeoSeries``."""
        return self._series_unary_op(self._obj, 'area', null_value=np.nan)

    @property
    def geom_type(self):
        """Returns a ``Series`` of strings specifying the `Geometry Type` of each
        object."""
        return self._series_unary_op(self._obj, 'geom_type', null_value=None)

    @property
    def length(self):
        """Returns a ``Series`` containing the length of each geometry."""
        return self._series_unary_op(self._obj, 'length', null_value=np.nan)

    @property
    def boundary(self):
        """Returns a ``GeoSeries`` of lower dimensional objects representing
        each geometries's set-theoretic `boundary`."""
        return self._geo_unary_op(self._obj, 'boundary')

    @property
    def centroid(self):
        """Returns a ``GeoSeries`` of points representing the centroid of each
        geometry."""
        return self._geo_unary_op(self._obj, 'centroid')

    @property
    def bounds(self):
        """
        Returns a ``DataFrame`` with columns ``minx``, ``miny``, ``maxx``,
        ``maxy`` values containing the bounds for each geometry.
        """
        bounds = np.array([geom.bounds for geom in self._obj.geometry])
        return pd.DataFrame(bounds,
                            columns=['minx', 'miny', 'maxx', 'maxy'],
                            index=self._obj.index)

    def contains(self, other):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each geometry that contains `other`.
        An object is said to contain `other` if its `interior` contains the
        `boundary` and `interior` of the other object and their boundaries do
        not touch at all.

        :param other: geometric object to check if contained
        :type other: ``shapely`` object
        """
        return self._series_op(self._obj, other, 'contains')

    def distance(self, other):
        """Returns a ``Series`` containing the distance to `other`.

        :param other: geometric object to calculate distance to
        :type other: ``shapely`` object
        """
        return self._series_op(self._obj, other, 'distance')

    def add(self, list_lines_pvarray):
        """
        Add list of objects of class :class:`pvcore.LinePVArray` to the
        registry.

        :param list_lines_pvarray: list of objects of type
            :class:`pvcore.LinePVArray`
        :return: ``idx_list`` -- ``int``, the list of registry indices that
            were added to the registry
        """
        # Find the start index that will be used to add entries to the registry
        if len(self._obj.index) > 0:
            start_index = self._obj.index[-1] + 1
        else:
            start_index = 0
        idx_list = []
        # Loop through list of PV array lines
        for counter, line_pvarray in enumerate(list_lines_pvarray):
            idx = start_index + counter
            for key in line_pvarray.keys():
                self._obj.loc[idx, key] = line_pvarray[key]
            idx_list.append(idx)

        return idx_list

    def split_ground_geometry_from_edge_points(self, edge_points):
        """
        Break up ground lines into multiple ones at the pv row "edge points",
        which are the intersections of pv row lines and ground lines. This is
        important to separate the ground lines that a pv row's front surface
         sees from the ones its back surface does.

        :param edge_points: points the specify location where to break lines
        :type edge_points: list of :class:`shapely.Point` objects
        :return: None
        """
        for point in edge_points:
            df_ground = self._obj.loc[self._obj.loc[:, 'line_type']
                                      == 'ground', :]
            geoentry_to_break_up = df_ground.loc[df_ground.pvgeometry
                                                 .contains(point)]
            if geoentry_to_break_up.shape[0] == 1:
                self.break_and_add_entries(geoentry_to_break_up, point)
            elif geoentry_to_break_up.shape[0] > 1:
                raise PVFactorsError("geoentry_to_break_up.shape[0] cannot be"
                                     " larger than 1")

    def break_and_add_entries(self, geoentry_to_break_up, point,
                              pvrow_segment_index=None):
        """
        Break up a surface geometry into two objects at a point location.

        :param geoentry_to_break_up: registry entry to break up
        :type geoentry_to_break_up: ``pandas.DataFrame`` with `geometry`
            column and ``pvgeometry`` extension
        :param point: point used to decide where to break up entry.
        :type point: :class:`shapely.Point` object
        :param int pvrow_segment_index: (optional) index of segment being
            broken up. Default is None.
        :return: None
        """
        # Get geometry
        idx = geoentry_to_break_up.index
        geometry = geoentry_to_break_up.geometry.values[0]
        line_1, line_2 = self.cut_linestring(geometry, point)
        geometry_1 = pd.Series(line_1)
        geometry_2 = pd.Series(line_2)
        self._obj.at[idx, 'geometry'] = geometry_1.values
        new_registry_entry = self._obj.loc[idx, :].copy()
        new_registry_entry['geometry'] = geometry_2.values
        # Add a value to "pvrow segment index" if provided
        if pvrow_segment_index is not None:
            self._obj.at[idx, 'pvrow_segment_index'] = pvrow_segment_index
            new_registry_entry['pvrow_segment_index'] = pvrow_segment_index + 1
        self._obj.loc[self._obj.shape[0], :] = new_registry_entry.values[0]

    @staticmethod
    def cut_linestring(line, point):
        """
        Adapted from shapely documentation. Cuts a line in two at a calculated
        distance from its starting point

        :param line: :class:`shapely.LineString` object to cut
        :param point: :class:`shapely.Point` object to use for the cut
        :return: list of two :class:`shapely.LineString` objects
        """

        distance = line.project(point)
        assert ((distance >= 0.0) & (distance <= line.length)), (
            "cut_linestring: the lines didn't intersect")
        # There could be multiple points in a line
        coords = list(line.coords)
        for i, p in enumerate(coords):
            pd = line.project(Point(p))
            if pd == distance:
                return [
                    LineString(coords[:i + 1]),
                    LineString(coords[i:])]
            if pd > distance:
                cp = line.interpolate(distance)
                return [
                    LineString(coords[:i] + [(cp.x, cp.y)]),
                    LineString([(cp.x, cp.y)] + coords[i:])]

    def split_pvrow_geometry(self, idx_pvrow, line_shadow, pvrow_top_point,
                             surface_side):
        """
        Break up pv row line into two pv row lines, a shaded one and an
        unshaded one. This function requires knowing the pv row line index in
        the registry, the "shadow line" that intersects with the pv row, and
        the top point of the pv row in order to decide which pv row line will
        be shaded or not after break up.

        :param int idx_pvrow: index of shaded pv row entry
        :param line_shadow: :class:`shapely.LineString` object representing the
            "shadow line" intersecting with the pv row line
        :param pvrow_top_point: the highest point of the pv row line (in the
            elevation direction)
        :param pvrow_top_point: :class:``shapely.Point`` object
        :param str surface_side: surface side of the pvrow
        :return: None
        """
        # Define geometry to work on
        df_pvrow = self._obj.loc[(self._obj.pvrow_index == idx_pvrow)
                                 & (self._obj.surface_side == surface_side),
                                 :]
        geometry = df_pvrow.loc[:, 'geometry'].values[0]
        # Find intersection point of line shadow and pvrow geometry
        point_intersect = geometry.intersection(line_shadow)
        # Check that the intersection is not too close to a boundary: if it
        # is it can create a "memory access error" it seems
        is_too_close = [
            point.distance(point_intersect) < THRESHOLD_DISTANCE_TOO_CLOSE
            for point in geometry.boundary]
        if True in is_too_close:
            # Leave it as it is and do not split geometry: it should not
            # affect the calculations for a good threshold
            pass
        else:
            # Cut geometry in two pieces
            self.cut_pvrow_geometry([point_intersect], idx_pvrow, surface_side)
            # Find the geometries that should be marked as shaded
            df_pvrow = self._obj.loc[(self._obj.pvrow_index == idx_pvrow)
                                     & (self._obj.surface_side == surface_side),
                                     :]
            centroids = df_pvrow.pvgeometry.centroid.to_frame()
            centroids.columns = ['geometry']
            is_below_shading_interesect = (centroids.pvgeometry.bounds['miny']
                                           < point_intersect.y)
            self._obj.loc[(self._obj.pvrow_index == idx_pvrow)
                          & (self._obj.surface_side == surface_side)
                          & is_below_shading_interesect,
                          'shaded'] = True

    def cut_pvrow_geometry(self, list_points, pvrow_index, side,
                           count_segments=False):
        """
        Break up pv row lines into multiple segments based on the list of
        points specified. This is the "discretization" of the pvrow segments.
        For now, it only works for pv rows.

        :param list_points: list of :class:`shapely.Point`, breaking points for
            the pv row lines.
        :param int pvrow_index: pv row index to specify the PV row to
            discretize; note that this could return multiple entries from the
            registry.
        :param str side: only do it for one side of the selected PV row. This
            can only be 'front' or 'back'.
        :param bool count_segments: (optional, default False) add pvrow segment idx
        :return: None
        """
        # TODO: is currently not able to work for other surfaces than pv rows..
        for idx, point in enumerate(list_points):
            if count_segments:
                pvrow_segment_index = idx
            else:
                pvrow_segment_index = None
            df_selected = self._obj.loc[
                (self._obj['pvrow_index'] == pvrow_index)
                & (self._obj['surface_side'] == side), :]
            geoentry_to_break_up = df_selected.loc[
                df_selected.pvgeometry.distance(point) < DISTANCE_TOLERANCE]
            if geoentry_to_break_up.shape[0] == 1:
                self.break_and_add_entries(geoentry_to_break_up, point,
                                           pvrow_segment_index=pvrow_segment_index)
            elif geoentry_to_break_up.shape[0] > 1:
                raise PVFactorsError("geoentry_to_break_up.shape[0] cannot be"
                                     "larger than 1")
