# -*- coding: utf-8 -*-

"""
``Geopandas`` is an amazing open-source package, but it brings with it
a lot of dependencies that are not useful in all use cases of the package.
In pvfactors, we need a way to store efficiently shapely geometries in
Dataframes and perform basic geometric operations. So in this case,
using the whole ``Geopandas`` package may be overkill and may complicate
the integration of the package in other open-source projects.

"""

import numpy as np
import pandas as pd


def _series_op(this, other, op, **kwargs):
    """Geometric operation that returns a pandas Series"""
    null_val = False if op not in ['distance', 'project'] else np.nan

    return pd.Series([getattr(s, op)(other, **kwargs) if s else null_val
                      for s in this.geometry],
                     index=this.index, dtype=np.dtype(type(null_val)))


def _geo_unary_op(this, op):
    """Unary operation that returns a GeoSeries"""

    # TODO: may need a dtype argument in the returned series
    return pd.Series([getattr(geom, op) for geom in this.geometry],
                     index=this.index)


def _series_unary_op(this, op, null_value=False):
    """Unary operation that returns a Series"""
    return pd.Series([getattr(geom, op, null_value) for geom in this.geometry],
                     index=this.index, dtype=np.dtype(type(null_value)))


# @pd.api.extensions.register_dataframe_accessor("pvgeometry")
@pd.api.extensions.register_dataframe_accessor("pvgeometry")
class PVGeometry(object):
    """ Lightweight version of ``geopandas.GeoSeries``"""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def area(self):
        """Returns a ``Series`` containing the area of each geometry in the
        ``GeoSeries``."""
        return _series_unary_op(self._obj, 'area', null_value=np.nan)

    @property
    def geom_type(self):
        """Returns a ``Series`` of strings specifying the `Geometry Type` of each
        object."""
        return _series_unary_op(self._obj, 'geom_type', null_value=None)

    @property
    def type(self):
        """Return the geometry type of each geometry in the GeoSeries"""
        return self.geom_type

    @property
    def length(self):
        """Returns a ``Series`` containing the length of each geometry."""
        return _series_unary_op(self._obj, 'length', null_value=np.nan)

    @property
    def boundary(self):
        """Returns a ``GeoSeries`` of lower dimensional objects representing
        each geometries's set-theoretic `boundary`."""
        return _geo_unary_op(self._obj, 'boundary')

    @property
    def centroid(self):
        """Returns a ``GeoSeries`` of points representing the centroid of each
        geometry."""
        return _geo_unary_op(self._obj, 'centroid')

    def contains(self, other):
        """Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
        each geometry that contains `other`.
        An object is said to contain `other` if its `interior` contains the
        `boundary` and `interior` of the other object and their boundaries do
        not touch at all.
        This is the inverse of :meth:`within` in the sense that the expression
        ``a.contains(b) == b.within(a)`` always evaluates to ``True``.
        Parameters
        ----------
        other : GeoSeries or geometric object
            The GeoSeries (elementwise) or geometric object to test if is
            contained.
        """
        return _series_op(self._obj, other, 'contains')

    def add(self, list_lines_pvarray):
        """
        Add list of objects of class :class:`pvcore.LinePVArray` to the
        registry.

        :param list_lines_pvarray: list of objects of type
            :class:`pvcore.LinePVArray`
        :return: ``idx_list`` -- ``int``, the list of registry indices that were
        added to the registry
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
