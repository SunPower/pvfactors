"""Timeseries geometry classes. They allow the vectorization of geometry
calculations."""

import numpy as np
from pvlib.tools import cosd, sind
from pvfactors import PVFactorsError
from pvfactors.config import \
    DISTANCE_TOLERANCE, COLOR_DIC, Y_GROUND, MIN_X_GROUND, MAX_X_GROUND
from pvfactors.geometry.base import (
    PVSurface, ShadeCollection, PVSegment, BaseSide)
from pvfactors.geometry.pvrow import PVRow
from pvfactors.geometry.pvground import PVGround
from shapely.geometry import GeometryCollection, LineString
from copy import deepcopy


class TsPVRow(object):
    """Timeseries PV row class: this class is a vectorized version of the
    PV row geometries. The coordinates and attributes (front and back sides)
    are all vectorized."""

    def __init__(self, ts_front_side, ts_back_side, xy_center, index=None,
                 full_pvrow_coords=None):
        """Initialize timeseries PV row with its front and back sides.

        Parameters
        ----------
        ts_front_side : :py:class:`~pvfactors.geometry.timeseries.TsSide`
            Timeseries front side of the PV row
        ts_back_side : :py:class:`~pvfactors.geometry.timeseries.TsSide`
            Timeseries back side of the PV row
        xy_center : tuple of float
            x and y coordinates of the PV row center point (invariant)
        index : int, optional
            index of the PV row (Default = None)
        full_pvrow_coords : :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`, optional
            Timeseries coordinates of the full PV row, end to end
            (Default = None)
        """
        self.front = ts_front_side
        self.back = ts_back_side
        self.xy_center = xy_center
        self.index = index
        self.full_pvrow_coords = full_pvrow_coords

    @classmethod
    def from_raw_inputs(cls, xy_center, width, rotation_vec,
                        cut, shaded_length_front, shaded_length_back,
                        index=None, param_names=None):
        """Create timeseries PV row using raw inputs.
        Note: shading will always be zero when pv rows are flat.

        Parameters
        ----------
        xy_center : tuple of float
            x and y coordinates of the PV row center point (invariant)
        width : float
            width of the PV rows [m]
        rotation_vec : np.ndarray
            Timeseries rotation values of the PV row [deg]
        cut : dict
            Discretization scheme of the PV row. Eg {'front': 2, 'back': 4}.
            Will create segments of equal length on the designated sides.
        shaded_length_front : np.ndarray
            Timeseries values of front side shaded length [m]
        shaded_length_back : np.ndarray
            Timeseries values of back side shaded length [m]
        index : int, optional
            Index of the pv row (default = None)
        param_names : list of str, optional
            List of names of surface parameters to use when creating geometries
            (Default = None)

        Returns
        -------
        New timeseries PV row object
        """
        # Calculate full pvrow coords
        pvrow_coords = TsPVRow._calculate_full_coords(
            xy_center, width, rotation_vec)
        # Calculate normal vectors
        dx = pvrow_coords.b2.x - pvrow_coords.b1.x
        dy = pvrow_coords.b2.y - pvrow_coords.b1.y
        normal_vec_front = np.array([-dy, dx])
        # Calculate front side coords
        ts_front = TsSide.from_raw_inputs(
            xy_center, width, rotation_vec, cut.get('front', 1),
            shaded_length_front, n_vector=normal_vec_front,
            param_names=param_names)
        # Calculate back side coords
        ts_back = TsSide.from_raw_inputs(
            xy_center, width, rotation_vec, cut.get('back', 1),
            shaded_length_back, n_vector=-normal_vec_front,
            param_names=param_names)

        return cls(ts_front, ts_back, xy_center, index=index,
                   full_pvrow_coords=pvrow_coords)

    @staticmethod
    def _calculate_full_coords(xy_center, width, rotation):
        """Method to calculate the full PV row coordinaltes.

        Parameters
        ----------
        xy_center : tuple of float
            x and y coordinates of the PV row center point (invariant)
        width : float
            width of the PV rows [m]
        rotation : np.ndarray
            Timeseries rotation values of the PV row [deg]

        Returns
        -------
        coords: :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Timeseries coordinates of full PV row
        """
        x_center, y_center = xy_center
        radius = width / 2.
        # Calculate coords
        x1 = radius * cosd(rotation + 180.) + x_center
        y1 = radius * sind(rotation + 180.) + y_center
        x2 = radius * cosd(rotation) + x_center
        y2 = radius * sind(rotation) + y_center
        coords = TsLineCoords.from_array(np.array([[x1, y1], [x2, y2]]))
        return coords

    def surfaces_at_idx(self, idx):
        """Get all PV surface geometries in timeseries PV row for a certain
        index.

        Parameters
        ----------
        idx : int
            Index to use to generate PV surface geometries

        Returns
        -------
        list of :py:class:`~pvfactors.geometry.base.PVSurface` objects
            List of PV surfaces
        """
        pvrow = self.at(idx)
        return pvrow.all_surfaces

    def plot_at_idx(self, idx, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
                    color_illum=COLOR_DIC['pvrow_illum']):
        """Plot timeseries PV row at a certain index.

        Parameters
        ----------
        idx : int
            Index to use to plot timeseries PV rows
        ax : :py:class:`matplotlib.pyplot.axes` object
            Axes for plotting
        color_shaded : str, optional
            Color to use for plotting the shaded surfaces (Default =
            COLOR_DIC['pvrow_shaded'])
        color_shaded : str, optional
            Color to use for plotting the illuminated surfaces (Default =
            COLOR_DIC['pvrow_illum'])
        """
        pvrow = self.at(idx)
        pvrow.plot(ax, color_shaded=color_shaded,
                   color_illum=color_illum, with_index=False)

    def at(self, idx):
        """Generate a PV row geometry for the desired index.

        Parameters
        ----------
        idx : int
            Index to use to generate PV row geometry

        Returns
        -------
        pvrow : :py:class:`~pvfactors.geometry.pvrow.PVRow`
        """
        front_geom = self.front.at(idx)
        back_geom = self.back.at(idx)
        original_line = LineString(
            self.full_pvrow_coords.as_array[:, :, idx])
        pvrow = PVRow(front_side=front_geom, back_side=back_geom,
                      index=self.index, original_linestring=original_line)
        return pvrow

    def update_params(self, new_dict):
        """Update timeseries surface parameters of the PV row.

        Parameters
        ----------
        new_dict : dict
            Parameters to add or update for the surfaces
        """
        self.front.update_params(new_dict)
        self.back.update_params(new_dict)

    @property
    def n_ts_surfaces(self):
        """Number of timeseries surfaces in the ts PV row"""
        return self.front.n_ts_surfaces + self.back.n_ts_surfaces

    @property
    def all_ts_surfaces(self):
        """List of all timeseries surfaces"""
        return self.front.all_ts_surfaces + self.back.all_ts_surfaces

    @property
    def centroid(self):
        """Centroid point of the timeseries pv row"""
        return self.full_pvrow_coords.centroid

    @property
    def length(self):
        """Length of both sides of the timeseries PV row"""
        return self.front.length + self.back.length


class TsSide(object):
    """Timeseries side class: this class is a vectorized version of the
    BaseSide geometries. The coordinates and attributes (list of segments,
    normal vector) are all vectorized."""

    def __init__(self, segments, n_vector=None):
        """Initialize timeseries side using list of timeseries segments.

        Parameters
        ----------
        segments : list of :py:class:`~pvfactors.geometry.timeseries.TsSegment`
            List of timeseries segments of the side
        n_vector : np.ndarray, optional
            Timeseries normal vectors of the side (Default = None)
        """
        self.list_segments = segments
        self.n_vector = n_vector

    @classmethod
    def from_raw_inputs(cls, xy_center, width, rotation_vec, cut,
                        shaded_length, n_vector=None, param_names=None):
        """Create timeseries side using raw PV row inputs.
        Note: shading will always be zero when PV rows are flat.

        Parameters
        ----------
        xy_center : tuple of float
            x and y coordinates of the PV row center point (invariant)
        width : float
            width of the PV rows [m]
        rotation_vec : np.ndarray
            Timeseries rotation values of the PV row [deg]
        cut : int
            Discretization scheme of the PV side.
            Will create segments of equal length.
        shaded_length : np.ndarray
            Timeseries values of side shaded length from lowest point [m]
        n_vector : np.ndarray, optional
            Timeseries normal vectors of the side
        param_names : list of str, optional
            List of names of surface parameters to use when creating geometries
            (Default = None)

        Returns
        -------
        New timeseries side object
        """

        mask_tilted_to_left = rotation_vec >= 0

        # Create Ts segments
        x_center, y_center = xy_center
        radius = width / 2.
        segment_length = width / cut
        is_not_flat = rotation_vec != 0.

        # Calculate coords of shading point
        r_shade = radius - shaded_length
        x_sh = np.where(
            mask_tilted_to_left,
            r_shade * cosd(rotation_vec + 180.) + x_center,
            r_shade * cosd(rotation_vec) + x_center)
        y_sh = np.where(
            mask_tilted_to_left,
            r_shade * sind(rotation_vec + 180.) + y_center,
            r_shade * sind(rotation_vec) + y_center)

        # Calculate coords
        list_segments = []
        for i in range(cut):
            # Calculate segment coords
            r1 = radius - i * segment_length
            r2 = radius - (i + 1) * segment_length
            x1 = r1 * cosd(rotation_vec + 180.) + x_center
            y1 = r1 * sind(rotation_vec + 180.) + y_center
            x2 = r2 * cosd(rotation_vec + 180) + x_center
            y2 = r2 * sind(rotation_vec + 180) + y_center
            segment_coords = TsLineCoords.from_array(
                np.array([[x1, y1], [x2, y2]]))
            # Determine lowest and highest points of segment
            x_highest = np.where(mask_tilted_to_left, x2, x1)
            y_highest = np.where(mask_tilted_to_left, y2, y1)
            x_lowest = np.where(mask_tilted_to_left, x1, x2)
            y_lowest = np.where(mask_tilted_to_left, y1, y2)
            # Calculate illum and shaded coords
            x2_illum, y2_illum = x_highest, y_highest
            x1_shaded, y1_shaded, x2_shaded, y2_shaded = \
                x_lowest, y_lowest, x_lowest, y_lowest
            mask_all_shaded = (y_sh > y_highest) & (is_not_flat)
            mask_partial_shaded = (y_sh > y_lowest) & (~ mask_all_shaded) \
                & (is_not_flat)
            # Calculate second boundary point of shade
            x2_shaded = np.where(mask_all_shaded, x_highest, x2_shaded)
            x2_shaded = np.where(mask_partial_shaded, x_sh, x2_shaded)
            y2_shaded = np.where(mask_all_shaded, y_highest, y2_shaded)
            y2_shaded = np.where(mask_partial_shaded, y_sh, y2_shaded)
            x1_illum = x2_shaded
            y1_illum = y2_shaded
            illum_coords = TsLineCoords.from_array(
                np.array([[x1_illum, y1_illum], [x2_illum, y2_illum]]))
            shaded_coords = TsLineCoords.from_array(
                np.array([[x1_shaded, y1_shaded], [x2_shaded, y2_shaded]]))
            # Create illuminated and shaded collections
            illum = TsShadeCollection(
                [TsSurface(illum_coords, n_vector=n_vector,
                           param_names=param_names)], False)
            shaded = TsShadeCollection(
                [TsSurface(shaded_coords, n_vector=n_vector,
                           param_names=param_names)], True)
            # Create segment
            segment = TsSegment(segment_coords, illum, shaded,
                                n_vector=n_vector)
            list_segments.append(segment)

        return cls(list_segments, n_vector=n_vector)

    def surfaces_at_idx(self, idx):
        """Get all PV surface geometries in timeseries side for a certain
        index.

        Parameters
        ----------
        idx : int
            Index to use to generate PV surface geometries

        Returns
        -------
        list of :py:class:`~pvfactors.geometry.base.PVSurface` objects
            List of PV surfaces
        """
        side_geom = self.at(idx)
        return side_geom.all_surfaces

    def at(self, idx):
        """Generate a side geometry for the desired index.

        Parameters
        ----------
        idx : int
            Index to use to generate side geometry

        Returns
        -------
        side : :py:class:`~pvfactors.geometry.base.BaseSide`
        """
        list_geom_segments = []
        for ts_seg in self.list_segments:
            list_geom_segments.append(ts_seg.at(idx))
        side = BaseSide(list_geom_segments)
        return side

    def plot_at_idx(self, idx, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
                    color_illum=COLOR_DIC['pvrow_illum']):
        """Plot timeseries side at a certain index.

        Parameters
        ----------
        idx : int
            Index to use to plot timeseries side
        ax : :py:class:`matplotlib.pyplot.axes` object
            Axes for plotting
        color_shaded : str, optional
            Color to use for plotting the shaded surfaces (Default =
            COLOR_DIC['pvrow_shaded'])
        color_shaded : str, optional
            Color to use for plotting the illuminated surfaces (Default =
            COLOR_DIC['pvrow_illum'])
        """
        side_geom = self.at(idx)
        side_geom.plot(ax, color_shaded=color_shaded, color_illum=color_illum,
                       with_index=False)

    @property
    def shaded_length(self):
        """Timeseries shaded length of the side."""
        length = 0.
        for seg in self.list_segments:
            length += seg.shaded.length
        return length

    @property
    def length(self):
        """Timeseries length of side."""
        length = 0.
        for seg in self.list_segments:
            length += seg.length
        return length

    def get_param_weighted(self, param):
        """Get timeseries parameter for the side, after weighting by
        surface length.

        Parameters
        ----------
        param : str
            Name of parameter

        Returns
        -------
        np.ndarray
            Weighted parameter values
        """
        return self.get_param_ww(param) / self.length

    def get_param_ww(self, param):
        """Get timeseries parameter from the side's surfaces with weight, i.e.
        after multiplying by the surface lengths.

        Parameters
        ----------
        param: str
            Surface parameter to return

        Returns
        -------
        np.ndarray
            Timeseries parameter values multiplied by weights

        Raises
        ------
        KeyError
            if parameter name not in a surface parameters
        """
        value = 0.
        for seg in self.list_segments:
            value += seg.get_param_ww(param)
        return value

    def update_params(self, new_dict):
        """Update timeseries surface parameters of the side.

        Parameters
        ----------
        new_dict : dict
            Parameters to add or update for the surfaces
        """
        for seg in self.list_segments:
            seg.update_params(new_dict)

    @property
    def n_ts_surfaces(self):
        """Number of timeseries surfaces in the ts side"""
        n_ts_surfaces = 0
        for ts_segment in self.list_segments:
            n_ts_surfaces += ts_segment.n_ts_surfaces
        return n_ts_surfaces

    @property
    def all_ts_surfaces(self):
        """List of all timeseries surfaces"""
        all_ts_surfaces = []
        for ts_segment in self.list_segments:
            all_ts_surfaces += ts_segment.all_ts_surfaces
        return all_ts_surfaces


class TsSegment(object):
    """A TsSegment is a timeseries segment that has a timeseries shaded
    collection and a timeseries illuminated collection."""

    def __init__(self, coords, illum_collection, shaded_collection,
                 index=None, n_vector=None):
        """Initialize timeseries segment using segment coordinates and
        timeseries illuminated and shaded surfaces.

        Parameters
        ----------
        coords : :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Timeseries coordinates of full segment
        illum_collection :
        :py:class:`~pvfactors.geometry.timeseries.TsShadeCollection`
            Timeseries collection for illuminated part of segment
        shaded_collection :
        :py:class:`~pvfactors.geometry.timeseries.TsShadeCollection`
            Timeseries collection for shaded part of segment
        index : int, optional
            Index of segment (Default = None)
        n_vector : np.ndarray, optional
            Timeseries normal vectors of the side (Default = None)
        """
        self.coords = coords
        self.illum = illum_collection
        self.shaded = shaded_collection
        self.index = index
        self.n_vector = n_vector

    def surfaces_at_idx(self, idx):
        """Get all PV surface geometries in timeseries segment for a certain
        index.

        Parameters
        ----------
        idx : int
            Index to use to generate PV surface geometries

        Returns
        -------
        list of :py:class:`~pvfactors.geometry.base.PVSurface` objects
            List of PV surfaces
        """
        segment = self.at(idx)
        return segment.all_surfaces

    def plot_at_idx(self, idx, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
                    color_illum=COLOR_DIC['pvrow_illum']):
        """Plot timeseries segment at a certain index.

        Parameters
        ----------
        idx : int
            Index to use to plot timeseries segment
        ax : :py:class:`matplotlib.pyplot.axes` object
            Axes for plotting
        color_shaded : str, optional
            Color to use for plotting the shaded surfaces (Default =
            COLOR_DIC['pvrow_shaded'])
        color_shaded : str, optional
            Color to use for plotting the illuminated surfaces (Default =
            COLOR_DIC['pvrow_illum'])
        """
        segment = self.at(idx)
        segment.plot(ax, color_shaded=color_shaded, color_illum=color_illum,
                     with_index=False)

    def at(self, idx):
        """Generate a PV segment geometry for the desired index.

        Parameters
        ----------
        idx : int
            Index to use to generate PV segment geometry

        Returns
        -------
        segment : :py:class:`~pvfactors.geometry.base.PVSegment`
        """
        # Create illum collection
        illum_collection = self.illum.at(idx)
        # Create shaded collection
        shaded_collection = self.shaded.at(idx)
        # Create PV segment
        segment = PVSegment(illum_collection=illum_collection,
                            shaded_collection=shaded_collection,
                            index=self.index)
        return segment

    @property
    def length(self):
        """Timeseries length of segment."""
        return self.illum.length + self.shaded.length

    @property
    def shaded_length(self):
        """Timeseries length of shaded part of segment."""
        return self.shaded.length

    @property
    def centroid(self):
        """Timeseries point coordinates of the segment's centroid"""
        return self.coords.centroid

    def get_param_weighted(self, param):
        """Get timeseries parameter for the segment, after weighting by
        surface length.

        Parameters
        ----------
        param : str
            Name of parameter

        Returns
        -------
        np.ndarray
            Weighted parameter values
        """
        return self.get_param_ww(param) / self.length

    def get_param_ww(self, param):
        """Get timeseries parameter from the segment's surfaces with weight,
        i.e. after multiplying by the surface lengths.

        Parameters
        ----------
        param: str
            Surface parameter to return

        Returns
        -------
        np.ndarray
            Timeseries parameter values multiplied by weights
        """
        return self.illum.get_param_ww(param) + self.shaded.get_param_ww(param)

    def update_params(self, new_dict):
        """Update timeseries surface parameters of the segment.

        Parameters
        ----------
        new_dict : dict
            Parameters to add or update for the surfaces
        """
        self.illum.update_params(new_dict)
        self.shaded.update_params(new_dict)

    @property
    def highest_point(self):
        """Timeseries point coordinates of highest point of segment"""
        return self.coords.highest_point

    @property
    def lowest_point(self):
        """Timeseries point coordinates of lowest point of segment"""
        return self.coords.lowest_point

    @property
    def all_ts_surfaces(self):
        """List of all timeseries surfaces in segment"""
        return self.illum.list_ts_surfaces + self.shaded.list_ts_surfaces

    @property
    def n_ts_surfaces(self):
        """Number of timeseries surfaces in the segment"""
        return self.illum.n_ts_surfaces + self.shaded.n_ts_surfaces


class TsShadeCollection(object):
    """Collection of timeseries surfaces that are all either shaded or
    illuminated"""

    def __init__(self, list_ts_surfaces, shaded):
        self._list_ts_surfaces = list_ts_surfaces
        self.shaded = shaded
        # TODO: maybe the ts surfaces should have a "shaded" attribute

    @property
    def list_ts_surfaces(self):
        return self._list_ts_surfaces

    @property
    def length(self):
        length = 0.
        for ts_surf in self._list_ts_surfaces:
            length += ts_surf.length
        return length

    @property
    def n_ts_surfaces(self):
        """Number of timeseries surfaces in the collection"""
        return len(self._list_ts_surfaces)

    def add(self, new_list_ts_surfaces):
        # TODO: need to check that all new surfaces have same shading status
        self._list_ts_surfaces += new_list_ts_surfaces

    def get_param_weighted(self, param):
        """Get timeseries parameter for the collection, after weighting by
        surface length.

        Parameters
        ----------
        param : str
            Name of parameter

        Returns
        -------
        np.ndarray
            Weighted parameter values
        """
        return self.get_param_ww(param) / self.length

    def get_param_ww(self, param):
        """Get timeseries parameter from the collection with weight,
        i.e. after multiplying by the surface lengths.

        Parameters
        ----------
        param: str
            Surface parameter to return

        Returns
        -------
        np.ndarray
            Timeseries parameter values multiplied by weights
        """

        value = 0
        for ts_surf in self._list_ts_surfaces:
            value += ts_surf.length * ts_surf.get_param(param)
        return value

    def update_params(self, new_dict):
        """Update timeseries surface parameters of the segment.

        Parameters
        ----------
        new_dict : dict
            Parameters to add or update for the surfaces
        """
        for ts_surf in self._list_ts_surfaces:
            ts_surf.params.update(new_dict)

    def at(self, idx):
        """Generate a ponctual shade collection for the desired index.

        Parameters
        ----------
        idx : int
            Index to use to generate shade collection

        Returns
        -------
        collection : :py:class:`~pvfactors.geometry.base.ShadeCollection`
        """
        list_surfaces = [ts_surf.at(idx, shaded=self.shaded)
                         for ts_surf in self._list_ts_surfaces
                         if not ts_surf.at(idx, shaded=self.shaded).is_empty]
        return ShadeCollection(list_surfaces, shaded=self.shaded)


class TsGround(object):
    """Timeseries ground class: this class is a vectorized version of the
    PV ground geometry class, and it will store timeseries shaded ground
    and illuminated ground elements, as well as pv row cut points."""

    def __init__(self, shadow_elements, illum_elements, param_names=None,
                 flag_overlap=None, cut_point_coords=None, y_ground=None):
        """Initialize timeseries ground using list of timeseries surfaces
        for the ground shadows

        Parameters
        ----------
        shadow_elements : list of :py:class:`~pvfactors.geometry.timeseries.TsGroundElement`
            Timeseries shaded ground elements
        illum_elements : list of :py:class:`~pvfactors.geometry.timeseries.TsGroundElement`
            Timeseries illuminated ground elements
        param_names : list of str, optional
            List of names of surface parameters to use when creating geometries
            (Default = None)
        flag_overlap : list of bool, optional
            Flags indicating if the ground shadows are overlapping, for all
            time steps (Default=None). I.e. is there direct shading on pv rows?
        cut_point_coords : list of :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`, optional
            List of cut point coordinates, as calculated for timeseries PV rows
            (Default = None)
        y_ground : float, optional
            Y coordinate of flat ground [m] (Default=None)
        """
        # Lists of timeseries ground elements
        self.shadow_elements = shadow_elements
        self.illum_elements = illum_elements
        # Shade collections
        list_shaded_surf = []
        list_illum_surf = []
        for shadow_el in shadow_elements:
            list_shaded_surf += shadow_el.all_ts_surfaces
        for illum_el in illum_elements:
            list_illum_surf += illum_el.all_ts_surfaces
        self.illum = TsShadeCollection(list_illum_surf, False)
        self.shaded = TsShadeCollection(list_shaded_surf, True)
        # Other ground attributes
        self.param_names = [] if param_names is None else param_names
        self.flag_overlap = flag_overlap
        self.cut_point_coords = [] if cut_point_coords is None \
            else cut_point_coords
        self.y_ground = y_ground
        self.shaded_params = dict.fromkeys(self.param_names)
        self.illum_params = dict.fromkeys(self.param_names)

    @classmethod
    def from_ts_pvrows_and_angles(cls, list_ts_pvrows, alpha_vec, rotation_vec,
                                  y_ground=Y_GROUND, flag_overlap=None,
                                  param_names=None):
        """Create timeseries ground from list of timeseries PV rows, and
        PV array and solar angles.

        Parameters
        ----------
        list_ts_pvrows : list of :py:class:`~pvfactors.geometry.timeseries.TsPVRow`
            Timeseries PV rows to use to calculate timeseries ground shadows
        alpha_vec : np.ndarray
            Angle made by 2d solar vector and PV array x-axis [rad]
        rotation_vec : np.ndarray
            Timeseries rotation values of the PV row [deg]
        y_ground : float, optional
            Fixed y coordinate of flat ground [m] (Default = Y_GROUND constant)
        flag_overlap : list of bool, optional
            Flags indicating if the ground shadows are overlapping, for all
            time steps (Default=None). I.e. is there direct shading on pv rows?
        param_names : list of str, optional
            List of names of surface parameters to use when creating geometries
            (Default = None)
        """
        rotation_vec = np.deg2rad(rotation_vec)
        n_steps = len(rotation_vec)
        # Calculate coords of ground shadows and cutting points
        ground_shadow_coords = []
        cut_point_coords = []
        for ts_pvrow in list_ts_pvrows:
            # Get pvrow coords
            x1s_pvrow = ts_pvrow.full_pvrow_coords.b1.x
            y1s_pvrow = ts_pvrow.full_pvrow_coords.b1.y
            x2s_pvrow = ts_pvrow.full_pvrow_coords.b2.x
            y2s_pvrow = ts_pvrow.full_pvrow_coords.b2.y
            # --- Shadow coords calculation
            # Calculate x coords of shadow
            x1s_shadow = x1s_pvrow - (y1s_pvrow - y_ground) / np.tan(alpha_vec)
            x2s_shadow = x2s_pvrow - (y2s_pvrow - y_ground) / np.tan(alpha_vec)
            # Order x coords from left to right
            x1s_on_left = x1s_shadow <= x2s_shadow
            xs_left_shadow = np.where(x1s_on_left, x1s_shadow, x2s_shadow)
            xs_right_shadow = np.where(x1s_on_left, x2s_shadow, x1s_shadow)
            # Append shadow coords to list
            ground_shadow_coords.append(
                [[xs_left_shadow, y_ground * np.ones(n_steps)],
                 [xs_right_shadow, y_ground * np.ones(n_steps)]])
            # --- Cutting points coords calculation
            dx = (y1s_pvrow - y_ground) / np.tan(rotation_vec)
            cut_point_coords.append(
                TsPointCoords(x1s_pvrow - dx, y_ground * np.ones(n_steps)))

        ground_shadow_coords = np.array(ground_shadow_coords)
        return cls.from_ordered_shadows_coords(
            ground_shadow_coords, flag_overlap=flag_overlap,
            cut_point_coords=cut_point_coords, param_names=param_names,
            y_ground=y_ground)

    @classmethod
    def from_ordered_shadows_coords(cls, shadow_coords, flag_overlap=None,
                                    param_names=None, cut_point_coords=None,
                                    y_ground=Y_GROUND):
        """Create timeseries ground from list of ground shadow coordinates.

        Parameters
        ----------
        shadow_coords : np.ndarray
            List of ordered ground shadow coordinates (from left to right)
        flag_overlap : list of bool, optional
            Flags indicating if the ground shadows are overlapping, for all
            time steps (Default=None). I.e. is there direct shading on pv rows?
        param_names : list of str, optional
            List of names of surface parameters to use when creating geometries
            (Default = None)
        cut_point_coords : list of :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`, optional
            List of cut point coordinates, as calculated for timeseries PV rows
            (Default = None)
        y_ground : float, optional
            Fixed y coordinate of flat ground [m] (Default = Y_GROUND constant)
        """

        # Get cut point coords if any
        cut_point_coords = cut_point_coords or []
        # Create shadow coordinate objects
        list_shadow_coords = [TsLineCoords.from_array(coords)
                              for coords in shadow_coords]
        # If the overlap flags were passed, make sure shadows don't overlap
        if flag_overlap is not None:
            if len(list_shadow_coords) > 1:
                for idx, coords in enumerate(list_shadow_coords[:-1]):
                    coords.b2.x = np.where(flag_overlap,
                                           list_shadow_coords[idx + 1].b1.x,
                                           coords.b2.x)
        # Create shaded ground elements
        ts_shadows_elements = cls._shadow_elements_from_coords_and_cut_pts(
            list_shadow_coords, cut_point_coords, param_names)
        # Create illuminated ground elements
        ts_illum_elements = cls._illum_elements_from_coords_and_cut_pts(
            ts_shadows_elements, cut_point_coords, param_names, y_ground)
        return cls(ts_shadows_elements, ts_illum_elements,
                   param_names=param_names, flag_overlap=flag_overlap,
                   cut_point_coords=cut_point_coords, y_ground=y_ground)

    def at(self, idx, x_min_max=None, merge_if_flag_overlap=True,
           with_cut_points=True):
        """Generate a PV ground geometry for the desired index. This will
        only return non-point surfaces within the ground bounds, i.e.
        surfaces that are not points, and which are within x_min and x_max.

        Parameters
        ----------
        idx : int
            Index to use to generate PV ground geometry
        x_min_max : tuple, optional
            List of minimum and maximum x coordinates for the flat surface [m]
            (Default = None)
        merge_if_flag_overlap : bool, optional
            Decide whether to merge all shadows if they overlap or not
            (Default = True)
        with_cut_points :  bool, optional
            Decide whether to include the saved cut points in the created
            PV ground geometry (Default = True)

        Returns
        -------
        pvground : :py:class:`~pvfactors.geometry.pvground.PVGround`
        """
        # Get shadow elements that are not points at the given index
        non_pt_shadow_elements = [
            shadow_el for shadow_el in self.shadow_elements
            if shadow_el.coords.length[idx] > DISTANCE_TOLERANCE]

        if with_cut_points:
            # We want the ground surfaces broken up at the cut points
            if merge_if_flag_overlap:
                # We want to merge the shadow surfaces when they overlap
                list_shadow_surfaces = self._merge_shadow_surfaces(
                    idx, non_pt_shadow_elements)
            else:
                # No need to merge the shadow surfaces
                list_shadow_surfaces = []
                for shadow_el in non_pt_shadow_elements:
                    list_shadow_surfaces += \
                        shadow_el.non_point_surfaces_at(idx)
            # Get the illuminated surfaces
            list_illum_surfaces = []
            for illum_el in self.illum_elements:
                list_illum_surfaces += illum_el.non_point_surfaces_at(idx)
        else:
            # No need to break up the surfaces at the cut points
            # We will need to build up new surfaces (since not done by classes)

            # Get the parameters at the given index
            illum_params = _get_params_at_idx(idx, self.illum_params)
            shaded_params = _get_params_at_idx(idx, self.shaded_params)

            if merge_if_flag_overlap and (self.flag_overlap is not None):
                # We want to merge the shadow surfaces when they overlap
                is_overlap = self.flag_overlap[idx]
                if is_overlap and (len(non_pt_shadow_elements) > 1):
                    coords = [non_pt_shadow_elements[0].b1.at(idx),
                              non_pt_shadow_elements[-1].b2.at(idx)]
                    list_shadow_surfaces = [PVSurface(
                        coords, shaded=True, param_names=self.param_names,
                        params=shaded_params)]
                else:
                    # No overlap for the given index or config
                    list_shadow_surfaces = [
                        PVSurface(shadow_el.coords.at(idx),
                                  shaded=True, params=shaded_params,
                                  param_names=self.param_names)
                        for shadow_el in non_pt_shadow_elements
                        if shadow_el.coords.length[idx]
                        > DISTANCE_TOLERANCE]
            else:
                # No need to merge the shadow surfaces
                list_shadow_surfaces = [
                    PVSurface(shadow_el.coords.at(idx),
                              shaded=True, params=shaded_params,
                              param_names=self.param_names)
                    for shadow_el in non_pt_shadow_elements
                    if shadow_el.coords.length[idx]
                    > DISTANCE_TOLERANCE]
            # Get the illuminated surfaces
            list_illum_surfaces = [PVSurface(illum_el.coords.at(idx),
                                             shaded=False, params=illum_params,
                                             param_names=self.param_names)
                                   for illum_el in self.illum_elements
                                   if illum_el.coords.length[idx]
                                   > DISTANCE_TOLERANCE]

        # Pass the created lists to the PVGround builder
        return PVGround.from_lists_surfaces(
            list_shadow_surfaces, list_illum_surfaces,
            param_names=self.param_names, y_ground=self.y_ground,
            x_min_max=x_min_max)

    def plot_at_idx(self, idx, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
                    color_illum=COLOR_DIC['pvrow_illum'], x_min_max=None,
                    merge_if_flag_overlap=True, with_cut_points=True):
        """Plot timeseries ground at a certain index.

        Parameters
        ----------
        idx : int
            Index to use to plot timeseries side
        ax : :py:class:`matplotlib.pyplot.axes` object
            Axes for plotting
        color_shaded : str, optional
            Color to use for plotting the shaded surfaces (Default =
            COLOR_DIC['pvrow_shaded'])
        color_shaded : str, optional
            Color to use for plotting the illuminated surfaces (Default =
            COLOR_DIC['pvrow_illum'])
        x_min_max : tuple, optional
            List of minimum and maximum x coordinates for the flat surface [m]
            (Default = None)
        merge_if_flag_overlap : bool, optional
            Decide whether to merge all shadows if they overlap or not
            (Default = True)
        with_cut_points :  bool, optional
            Decide whether to include the saved cut points in the created
            PV ground geometry (Default = True)
        """
        pvground = self.at(idx, x_min_max=x_min_max,
                           merge_if_flag_overlap=merge_if_flag_overlap,
                           with_cut_points=with_cut_points)
        pvground.plot(ax, color_shaded=color_shaded, color_illum=color_illum,
                      with_index=False)

    def update_params(self, new_dict):
        """Update the illuminated parameters with new ones, not only for the
        timeseries ground, but also for its ground elements and the timeseries
        surfaces of the ground elements, so that they are all synced.

        Parameters
        ----------
        new_dict : dict
            New parameters
        """
        self.update_illum_params(new_dict)
        self.update_shaded_params(new_dict)

    def update_illum_params(self, new_dict):
        """Update the illuminated parameters with new ones, not only for the
        timeseries ground, but also for its ground elements and the timeseries
        surfaces of the ground elements, so that they are all synced.

        Parameters
        ----------
        new_dict : dict
            New parameters
        """
        self.illum_params.update(new_dict)
        for illum_el in self.illum_elements:
            illum_el.params.update(new_dict)
            for surf in illum_el.surface_list:
                surf.params.update(new_dict)

    def update_shaded_params(self, new_dict):
        """Update the shaded parameters with new ones, not only for the
        timeseries ground, but also for its ground elements and the timeseries
        surfaces of the ground elements, so that they are all synced.

        Parameters
        ----------
        new_dict : dict
            New parameters
        """
        self.shaded_params.update(new_dict)
        for shaded_el in self.shadow_elements:
            shaded_el.params.update(new_dict)
            for surf in shaded_el.surface_list:
                surf.params.update(new_dict)

    def get_param_weighted(self, param):
        """Get timeseries parameter for the ts ground, after weighting by
        surface length.

        Parameters
        ----------
        param : str
            Name of parameter

        Returns
        -------
        np.ndarray
            Weighted parameter values
        """
        return self.get_param_ww(param) / self.length

    def get_param_ww(self, param):
        """Get timeseries parameter from the ground's surfaces with weight,
        i.e. after multiplying by the surface lengths.

        Parameters
        ----------
        param: str
            Surface parameter to return

        Returns
        -------
        np.ndarray
            Timeseries parameter values multiplied by weights

        Raises
        ------
        KeyError
            if parameter name not in a surface parameters
        """
        value = 0.
        for shadow_el in self.shadow_elements:
            value += shadow_el.get_param_ww(param)
        for illum_el in self.illum_elements:
            value += illum_el.get_param_ww(param)
        return value

    def shadow_coords_left_of_cut_point(self, idx_cut_pt):
        """Get coordinates of shadows located on the left side of the cut point
        with given index. The coordinates of the shadows will be bounded
        by the coordinates of the cut point and the default minimum
        ground x values.

        Parameters
        ----------
        idx_cut_pt : int
            Index of the cut point of interest

        Returns
        -------
        list of :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Coordinates of the shadows on the left side of the cut point
        """
        cut_pt_coords = self.cut_point_coords[idx_cut_pt]
        return [shadow_el._coords_left_of_cut_point(shadow_el.coords,
                                                    cut_pt_coords)
                for shadow_el in self.shadow_elements]

    def shadow_coords_right_of_cut_point(self, idx_cut_pt):
        """Get coordinates of shadows located on the right side of the cut
        point with given index. The coordinates of the shadows will be bounded
        by the coordinates of the cut point and the default maximum
        ground x values.

        Parameters
        ----------
        idx_cut_pt : int
            Index of the cut point of interest

        Returns
        -------
        list of :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Coordinates of the shadows on the right side of the cut point
        """
        cut_pt_coords = self.cut_point_coords[idx_cut_pt]
        return [shadow_el._coords_right_of_cut_point(shadow_el.coords,
                                                     cut_pt_coords)
                for shadow_el in self.shadow_elements]

    def ts_surfaces_side_of_cut_point(self, side, idx_cut_pt):
        """Get a list of all the ts ground surfaces an a request side of
        a cut point

        Parameters
        ----------
        side : str
            Side of the cut point, either 'left' or 'right'
        idx_cut_pt : int
            Index of the cut point, on whose side we want to get the ground
            surfaces

        Returns
        -------
        list
            List of timeseries ground surfaces on the side of the cut point
        """
        list_ts_surfaces = []
        for shadow_el in self.shadow_elements:
            list_ts_surfaces += shadow_el.surface_dict[idx_cut_pt][side]
        for illum_el in self.illum_elements:
            list_ts_surfaces += illum_el.surface_dict[idx_cut_pt][side]
        return list_ts_surfaces

    @property
    def n_ts_surfaces(self):
        """Number of timeseries surfaces in the ts ground"""
        return self.n_ts_shaded_surfaces + self.n_ts_illum_surfaces

    @property
    def n_ts_shaded_surfaces(self):
        """Number of shaded timeseries surfaces in the ts ground"""
        n_ts_surfaces = 0
        for shadow_el in self.shadow_elements:
            n_ts_surfaces += shadow_el.n_ts_surfaces
        return n_ts_surfaces

    @property
    def n_ts_illum_surfaces(self):
        """Number of illuminated timeseries surfaces in the ts ground"""
        n_ts_surfaces = 0
        for illum_el in self.illum_elements:
            n_ts_surfaces += illum_el.n_ts_surfaces
        return n_ts_surfaces

    @property
    def all_ts_surfaces(self):
        """Number of timeseries surfaces in the ts ground"""
        all_ts_surfaces = []
        for shadow_el in self.shadow_elements:
            all_ts_surfaces += shadow_el.all_ts_surfaces
        for illum_el in self.illum_elements:
            all_ts_surfaces += illum_el.all_ts_surfaces
        return all_ts_surfaces

    @property
    def length(self):
        """Length of the timeseries ground"""
        length = 0
        for shadow_el in self.shadow_elements:
            length += shadow_el.length
        for illum_el in self.illum_elements:
            length += illum_el.length
        return length

    @property
    def shaded_length(self):
        """Length of the timeseries ground"""
        length = 0
        for shadow_el in self.shadow_elements:
            length += shadow_el.length
        return length

    def non_point_shaded_surfaces_at(self, idx):
        """Return a list of shaded surfaces, that are not points
        at given index

        Parameters
        ----------
        idx : int
            Index at which we want the surfaces not to be points

        Returns
        -------
        list of :py:class:`~pvfactors.geometry.base.PVSurface`
        """
        list_surfaces = []
        for shadow_el in self.shadow_elements:
            list_surfaces += shadow_el.non_point_surfaces_at(0)
        return list_surfaces

    def non_point_illum_surfaces_at(self, idx):
        """Return a list of illuminated surfaces, that are not
        points at given index

        Parameters
        ----------
        idx : int
            Index at which we want the surfaces not to be points

        Returns
        -------
        list of :py:class:`~pvfactors.geometry.base.PVSurface`
        """
        list_surfaces = []
        for illum_el in self.illum_elements:
            list_surfaces += illum_el.non_point_surfaces_at(0)
        return list_surfaces

    def non_point_surfaces_at(self, idx):
        """Return a list of all surfaces that are not
        points at given index

        Parameters
        ----------
        idx : int
            Index at which we want the surfaces not to be points

        Returns
        -------
        list of :py:class:`~pvfactors.geometry.base.PVSurface`
        """
        return self.non_point_illum_surfaces_at(idx) \
            + self.non_point_shaded_surfaces_at(idx)

    def n_non_point_surfaces_at(self, idx):
        """Return the number of :py:class:`~pvfactors.geometry.base.PVSurface`
        that are not points at given index

        Parameters
        ----------
        idx : int
            Index at which we want the surfaces not to be points

        Returns
        -------
        int
        """
        return len(self.non_point_surfaces_at(idx))

    @staticmethod
    def _shadow_elements_from_coords_and_cut_pts(
            list_shadow_coords, cut_point_coords, param_names):
        """Create ground shadow elements from a list of ordered shadow
        coordinates (from left to right), and the ground cut point coordinates.

        Notes
        -----
        This method will clip the shadow coords to the limit of ground,
        i.e. the shadow coordinates shouldn't be outside of the range
        [MIN_X_GROUND, MAX_X_GROUND].

        Parameters
        ----------
        list_shadow_coords : list of :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            List of ordered ground shadow coordinates (from left to right)
        cut_point_coords : list of :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            List of cut point coordinates (from left to right)
        param_names : list
            List of parameter names for the ground elements

        Returns
        -------
        list_shadow_elements : list of :py:class:`~pvfactors.geometry.timeseries.TsGroundElement`
            Ordered list of shadow elements (from left to right)
        """

        list_shadow_elements = []
        # FIXME: x_min and x_max should be passed as inputs
        for shadow_coords in list_shadow_coords:
            shadow_coords.b1.x = np.clip(shadow_coords.b1.x, MIN_X_GROUND,
                                         MAX_X_GROUND)
            shadow_coords.b2.x = np.clip(shadow_coords.b2.x, MIN_X_GROUND,
                                         MAX_X_GROUND)
            list_shadow_elements.append(
                TsGroundElement(shadow_coords,
                                list_ordered_cut_pts_coords=cut_point_coords,
                                param_names=param_names, shaded=True))

        return list_shadow_elements

    @staticmethod
    def _illum_elements_from_coords_and_cut_pts(
            list_shadow_elements, cut_pt_coords, param_names, y_ground):
        """Create ground illuminated elements from a list of ordered shadow
        elements (from left to right), and the ground cut point coordinates.
        This method will make sure that the illuminated ground elements are
        all within the ground limits [MIN_X_GROUND, MAX_X_GROUND].

        Parameters
        ----------
        list_shadow_coords : list of :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            List of ordered ground shadow coordinates (from left to right)
        cut_point_coords : list of :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            List of cut point coordinates (from left to right)
        param_names : list
            List of parameter names for the ground elements

        Returns
        -------
        list_shadow_elements : list of :py:class:`~pvfactors.geometry.timeseries.TsGroundElement`
            Ordered list of shadow elements (from left to right)
        """

        list_illum_elements = []
        if len(list_shadow_elements) == 0:
            msg = """There must be at least one shadow element on the ground,
            otherwise it probably means that no PV rows were created, so
            there's no point in running a simulation..."""
            raise PVFactorsError(msg)
        n_steps = len(list_shadow_elements[0].coords.b1.x)
        y_ground_vec = y_ground * np.ones(n_steps)
        # FIXME: x_min and x_max should be passed as inputs
        next_x = MIN_X_GROUND * np.ones(n_steps)
        # Build the groud elements from left to right, starting at x_min
        # and covering the ground with illuminated elements where there's no
        # shadow
        for shadow_element in list_shadow_elements:
            x1 = next_x
            x2 = shadow_element.coords.b1.x
            coords = TsLineCoords.from_array(
                np.array([[x1, y_ground_vec], [x2, y_ground_vec]]))
            list_illum_elements.append(TsGroundElement(
                coords, list_ordered_cut_pts_coords=cut_pt_coords,
                param_names=param_names, shaded=False))
            next_x = shadow_element.coords.b2.x
        # Add the last illuminated element to the list
        coords = TsLineCoords.from_array(
            np.array([[next_x, y_ground_vec],
                      [MAX_X_GROUND * np.ones(n_steps), y_ground_vec]]))
        list_illum_elements.append(TsGroundElement(
            coords, list_ordered_cut_pts_coords=cut_pt_coords,
            param_names=param_names, shaded=False))

        return list_illum_elements

    def _merge_shadow_surfaces(self, idx, non_pt_shadow_elements):
        """Merge the shadow surfaces in a list of shadow elements
        at the shadow boundaries only, at a given index, but keep the shadow
        surfaces broken up at the cut points.

        Parameters
        ----------
        idx : int
            Index at which we want to merge the surfaces
        non_pt_shadow_elements : list of :py:class:`~pvfactors.geometry.timeseries.TsGroundElement`
            List of non point shadow elements

        Returns
        -------
        list_shadow_surfaces : list of :py:class:`~pvfactors.geometry.base.PVSurface`
            List of shadow surfaces at a given index
            (ordered from left to right)
        """
        # TODO: check if it would be faster to merge the ground elements first,
        # and then break it down with the cut points

        # Decide whether to merge all shadows or not
        list_shadow_surfaces = []
        if self.flag_overlap is not None:
            # Get the overlap flags
            is_overlap = self.flag_overlap[idx]
            n_shadow_elements = len(non_pt_shadow_elements)
            if is_overlap and (n_shadow_elements > 1):
                # If there's only one shadow, not point in going through this

                # Now go from left to right and merge shadow surfaces
                surface_to_merge = None
                for i_el, shadow_el in enumerate(non_pt_shadow_elements):
                    surfaces = shadow_el.non_point_surfaces_at(idx)
                    n_surf = len(surfaces)
                    for i_surf, surface in enumerate(surfaces):
                        if i_surf == n_surf - 1:
                            # last surface, could also be first
                            if i_surf == 0:
                                # Need to merge with preceding if exists
                                if surface_to_merge is not None:
                                    coords = [surface_to_merge.boundary[0],
                                              surface.boundary[1]]
                                    surface = PVSurface(
                                        coords, shaded=True,
                                        param_names=self.param_names,
                                        params=surface.params)
                            if i_el == n_shadow_elements - 1:
                                # last surface of last shadow element
                                list_shadow_surfaces.append(surface)
                            else:
                                # keep for merging with next element
                                surface_to_merge = surface
                        elif i_surf == 0:
                            # first surface but definitely not last either
                            if surface_to_merge is not None:
                                coords = [surface_to_merge.boundary[0],
                                          surface.boundary[1]]
                                list_shadow_surfaces.append(
                                    PVSurface(coords, shaded=True,
                                              param_names=self.param_names,
                                              params=surface.params))
                            else:
                                list_shadow_surfaces.append(surface)
                        else:
                            # not first nor last surface
                            list_shadow_surfaces.append(surface)
            else:
                # There's no need to merge anything
                for shadow_el in non_pt_shadow_elements:
                    list_shadow_surfaces += \
                        shadow_el.non_point_surfaces_at(idx)
        else:
            # There's no need to merge anything
            for shadow_el in non_pt_shadow_elements:
                list_shadow_surfaces += shadow_el.non_point_surfaces_at(idx)

        return list_shadow_surfaces


class TsGroundElement(object):
    """Special class for timeseries ground elements: a ground element has known
    timeseries coordinate boundaries, but it will also have a break down of
    its area into n+1 timeseries surfaces located in the n+1 ground zones
    defined by the n ground cutting points.
    This is crucial to calculate view factors in a vectorized way."""

    def __init__(self, coords, list_ordered_cut_pts_coords=None,
                 param_names=None, shaded=False):
        """Initialize the timeseries ground element using its timeseries
        line coordinates, and build the timeseries surfaces for all the
        cut point zones.

        Parameters
        ----------
        coords : :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Timeseries line coordinates of the ground element
        list_ordered_cut_pts_coords : list, optional
            List of all the cut point timeseries coordinates
            (Default = [])
        param_names : list of str, optional
            List of names of surface parameters to use when creating geometries
            (Default = None)
        shaded : bool, optional
            Flag specifying is element is a shadow or not (Default = False)
        """
        self.coords = coords
        self.param_names = param_names or []
        self.params = dict.fromkeys(self.param_names)
        self.shaded = shaded
        self.surface_dict = None  # will be necessary for view factor calcs
        self.surface_list = []  # will be necessary for vf matrix formation
        list_ordered_cut_pts_coords = list_ordered_cut_pts_coords or []
        if len(list_ordered_cut_pts_coords) > 0:
            self._create_all_ts_surfaces(list_ordered_cut_pts_coords)
        self.n_ts_surfaces = len(self.surface_list)

    @property
    def b1(self):
        """Timeseries coordinates of first boundary point"""
        return self.coords.b1

    @property
    def b2(self):
        """Timeseries coordinates of second boundary point"""
        return self.coords.b2

    @property
    def centroid(self):
        """Timeseries point coordinates of the element's centroid"""
        return self.coords.centroid

    @property
    def length(self):
        """Timeseries length of the ground"""
        return self.coords.length

    @property
    def all_ts_surfaces(self):
        """List of all ts surfaces making up the ts ground element"""
        return self.surface_list

    def surfaces_at(self, idx):
        """Return list of surfaces (from left to right) at given index that
        make up the ground element.

        Parameters
        ----------
        idx : int
            Index of interest

        Returns
        -------
        list of :py:class:`~pvfactors.geometry.base.PVSurface`
        """
        return [surface.at(idx, shaded=self.shaded)
                for surface in self.surface_list]

    def non_point_surfaces_at(self, idx):
        """Return list of non-point surfaces (from left to right) at given
        index that make up the ground element.

        Parameters
        ----------
        idx : int
            Index of interest

        Returns
        -------
        list of :py:class:`~pvfactors.geometry.base.PVSurface`
        """
        return [surface.at(idx, shaded=self.shaded)
                for surface in self.surface_list
                if surface.length[idx] > DISTANCE_TOLERANCE]

    def get_param_weighted(self, param):
        """Get timeseries parameter for the ground element, after weighting by
        surface length.

        Parameters
        ----------
        param : str
            Name of parameter

        Returns
        -------
        np.ndarray
            Weighted parameter values
        """
        return self.get_param_ww(param) / self.length

    def get_param_ww(self, param):
        """Get timeseries parameter from the ground element with weight,
        i.e. after multiplying by the surface lengths.

        Parameters
        ----------
        param: str
            Surface parameter to return

        Returns
        -------
        np.ndarray
            Timeseries parameter values multiplied by weights

        Raises
        ------
        KeyError
            if parameter name not in a surface parameters
        """
        value = 0.
        for ts_surf in self.surface_list:
            value += ts_surf.length * ts_surf.get_param(param)
        return value

    def _create_all_ts_surfaces(self, list_ordered_cut_pts):
        """Create all the n+1 timeseries surfaces that make up the timeseries
        ground element, and which are located in the n+1 zones defined by
        the n cut points.

        Parameters
        ----------
        list_ordered_cut_pts : list of :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            List of timeseries coordinates of all cut points, ordered from
            left to right
        """
        # Initialize dict
        self.surface_dict = {i: {'right': [], 'left': []}
                             for i in range(len(list_ordered_cut_pts))}
        n_cut_pts = len(list_ordered_cut_pts)

        next_coords = self.coords
        for idx_pt, cut_pt_coords in enumerate(list_ordered_cut_pts):
            # Get coords on left of cut pt
            coords_left = self._coords_left_of_cut_point(next_coords,
                                                         cut_pt_coords)
            # Save that surface in the required structures
            surface_left = TsSurface(coords_left, param_names=self.param_names)
            self.surface_list.append(surface_left)
            for i in range(idx_pt, n_cut_pts):
                self.surface_dict[i]['left'].append(surface_left)
            for j in range(0, idx_pt):
                self.surface_dict[j]['right'].append(surface_left)
            next_coords = self._coords_right_of_cut_point(next_coords,
                                                          cut_pt_coords)
        # Save the right most portion
        next_surface = TsSurface(next_coords, param_names=self.param_names)
        self.surface_list.append(next_surface)
        for j in range(0, n_cut_pts):
            self.surface_dict[j]['right'].append(next_surface)

    @staticmethod
    def _coords_right_of_cut_point(coords, cut_pt_coords):
        """Calculate timeseries line coordinates that are right of the given
        cut point coordinates, but still within the ground area

        Parameters
        ----------
        coords : :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Original timeseries coordinates
        cut_pt_coords :
        :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Timeseries coordinates of cut point
        Returns
        -------
        :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Timeseries line coordinates that are located right of the cut
            point
        """
        coords = deepcopy(coords)
        # FIXME: should be using x_min x_max inputs instead of global constant
        coords.b1.x = np.maximum(coords.b1.x, cut_pt_coords.x)
        coords.b1.x = np.minimum(coords.b1.x, MAX_X_GROUND)
        coords.b2.x = np.maximum(coords.b2.x, cut_pt_coords.x)
        coords.b2.x = np.minimum(coords.b2.x, MAX_X_GROUND)
        return coords

    @staticmethod
    def _coords_left_of_cut_point(coords, cut_pt_coords):
        """Calculate timeseries line coordinates that are left of the given
        cut point coordinates, but still within the ground area

        Parameters
        ----------
        coords : :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Original timeseries coordinates
        cut_pt_coords :
        :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Timeseries coordinates of cut point
        Returns
        -------
        :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Timeseries line coordinates that are located left of the cut
            point
        """
        coords = deepcopy(coords)
        # FIXME: should be using x_min x_max inputs instead of global constant
        coords.b1.x = np.minimum(coords.b1.x, cut_pt_coords.x)
        coords.b1.x = np.maximum(coords.b1.x, MIN_X_GROUND)
        coords.b2.x = np.minimum(coords.b2.x, cut_pt_coords.x)
        coords.b2.x = np.maximum(coords.b2.x, MIN_X_GROUND)
        return coords


class TsSurface(object):
    """Timeseries surface class: vectorized representation of PV surface
    geometries."""

    def __init__(self, coords, n_vector=None, param_names=None, index=None):
        """Initialize timeseries surface using timeseries coordinates.

        Parameters
        ----------
        coords : :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Timeseries coordinates of full segment
        index : int, optional
            Index of segment (Default = None)
        n_vector : np.ndarray, optional
            Timeseries normal vectors of the side (Default = None)
        index : int, optional
            Index of the timeseries surfaces (Default = None)
        """
        # TODO: ts surfaces should have a shaded attribute
        self.coords = coords
        self.param_names = [] if param_names is None else param_names
        # TODO: the following should probably be turned into properties,
        # because if the coords change, they won't be altered. But speed...
        self.n_vector = n_vector
        self.params = dict.fromkeys(self.param_names)
        self.index = index

    def at(self, idx, shaded=None):
        """Generate a PV segment geometry for the desired index.

        Parameters
        ----------
        idx : int
            Index to use to generate PV segment geometry

        Returns
        -------
        segment : :py:class:`~pvfactors.geometry.base.PVSurface` or :py:class:`~shapely.geometry.GeometryCollection`
            The returned object will be an empty geometry if its length is
            really small, otherwise it will be a PV surface geometry
        """
        if self.length[idx] < DISTANCE_TOLERANCE:
            # return an empty geometry
            return GeometryCollection()
        else:
            # Get normal vector at idx
            n_vector = (self.n_vector[:, idx] if self.n_vector is not None
                        else None)
            # Get params at idx
            # TODO: should find faster solution
            params = _get_params_at_idx(idx, self.params)
            # Return a pv surface geometry with given params
            return PVSurface(self.coords.at(idx), shaded=shaded,
                             normal_vector=n_vector,
                             param_names=self.param_names,
                             params=params)

    def plot_at_idx(self, idx, ax, color):
        """Plot timeseries PV row at a certain index, only if it's not
        too small.

        Parameters
        ----------
        idx : int
            Index to use to plot timeseries PV surface
        ax : :py:class:`matplotlib.pyplot.axes` object
            Axes for plotting
        color_shaded : str, optional
            Color to use for plotting the PV surface
        """
        if self.length[idx] > DISTANCE_TOLERANCE:
            self.at(idx).plot(ax, color=color)

    @property
    def b1(self):
        """Timeseries coordinates of first boundary point"""
        return self.coords.b1

    @property
    def b2(self):
        """Timeseries coordinates of second boundary point"""
        return self.coords.b2

    @property
    def centroid(self):
        """Timeseries point coordinates of the surface's centroid"""
        return self.coords.centroid

    def get_param(self, param):
        """Get timeseries parameter values of surface

        Parameters
        ----------
        param: str
            Surface parameter to return

        Returns
        -------
        np.ndarray
            Timeseries parameter values
        """
        return self.params[param]

    def update_params(self, new_dict):
        """Update timeseries surface parameters.

        Parameters
        ----------
        new_dict : dict
            Parameters to add or update for the surface
        """
        self.params.update(new_dict)

    @property
    def length(self):
        """Timeseries length of the surface"""
        return self.coords.length

    @property
    def highest_point(self):
        """Timeseries point coordinates of highest point of surface"""
        return self.coords.highest_point

    @property
    def lowest_point(self):
        """Timeseries point coordinates of lowest point of surface"""
        return self.coords.lowest_point


class TsLineCoords(object):
    """Timeseries line coordinates class: will provide a helpful shapely-like
    API to invoke timeseries coordinates."""

    def __init__(self, b1_ts_coords, b2_ts_coords, coords=None):
        """Initialize timeseries line coordinates using the timeseries
        coordinates of its boundaries.

        Parameters
        ----------
        b1_ts_coords : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Timeseries coordinates of first boundary point
        b2_ts_coords : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Timeseries coordinates of second boundary point
        coords : np.ndarray, optional
            Timeseries coordinates as numpy array
        """
        self.b1 = b1_ts_coords
        self.b2 = b2_ts_coords

    def at(self, idx):
        """Get coordinates at a given index

        Parameters
        ----------
        idx : int
            Index to use to get coordinates
        """
        return self.as_array[:, :, idx]

    @classmethod
    def from_array(cls, coords_array):
        """Create timeseries line coordinates from numpy array of coordinates.

        Parameters
        ----------
        coords_array : np.ndarray
            Numpy array of coordinates.
        """
        b1 = TsPointCoords.from_array(coords_array[0, :, :])
        b2 = TsPointCoords.from_array(coords_array[1, :, :])
        return cls(b1, b2)

    @property
    def length(self):
        """Timeseries length of the line."""
        return np.sqrt((self.b2.y - self.b1.y)**2
                       + (self.b2.x - self.b1.x)**2)

    @property
    def as_array(self):
        """Timeseries line coordinates as numpy array"""
        return np.array([[self.b1.x, self.b1.y], [self.b2.x, self.b2.y]])

    @property
    def centroid(self):
        """Timeseries point coordinates of the line coordinates"""
        dy = self.b2.y - self.b1.y
        dx = self.b2.x - self.b1.x
        return TsPointCoords(self.b1.x + 0.5 * dx, self.b1.y + 0.5 * dy)

    @property
    def highest_point(self):
        """Timeseries point coordinates of highest point of timeseries
        line coords"""
        is_b1_highest = self.b1.y >= self.b2.y
        x = np.where(is_b1_highest, self.b1.x, self.b2.x)
        y = np.where(is_b1_highest, self.b1.y, self.b2.y)
        return TsPointCoords(x, y)

    @property
    def lowest_point(self):
        """Timeseries point coordinates of lowest point of timeseries
        line coords"""
        is_b1_highest = self.b1.y >= self.b2.y
        x = np.where(is_b1_highest, self.b2.x, self.b1.x)
        y = np.where(is_b1_highest, self.b2.y, self.b1.y)
        return TsPointCoords(x, y)

    def __repr__(self):
        """Use the numpy array representation of the coords"""
        return str(self.as_array)


class TsPointCoords(object):
    """Timeseries point coordinates: provides a shapely-like API for timeseries
    point coordinates."""

    def __init__(self, x, y):
        """Initialize timeseries point coordinates using numpy array of coords.

        Parameters
        ----------
        coords : np.ndarray
            Numpy array of timeseries point coordinates
        """
        self.x = x
        self.y = y

    def at(self, idx):
        """Get coordinates at a given index

        Parameters
        ----------
        idx : int
            Index to use to get coordinates
        """
        return self.as_array[:, idx]

    @property
    def as_array(self):
        """Timeseries point coordinates as numpy array"""
        return np.array([self.x, self.y])

    @classmethod
    def from_array(cls, coords_array):
        """Create timeseries point coords from numpy array of coordinates.

        Parameters
        ----------
        coords_array : np.ndarray
            Numpy array of coordinates.
        """
        return cls(coords_array[0, :], coords_array[1, :])

    def __repr__(self):
        """Use the numpy array representation of the point"""
        return str(self.as_array)


def _get_params_at_idx(idx, params_dict):
    """Get the parameter values at given index. Return the whole parameter
    when it's None, a scalar, or a dictionary

    Parameters
    ----------
    idx : int
        Index at which we want the parameter values
    params_dict : dict
        Dictionary of parameters

    Returns
    -------
    Parameter value at index
    """
    if params_dict is None:
        return None
    else:
        return {k: (val if (val is None) or np.isscalar(val)
                    or isinstance(val, dict) else val[idx])
                for k, val in params_dict.items()}
