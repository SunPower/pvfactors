"""Timeseries geometry classes. They allow the vectorization of geometry
calculations."""

import numpy as np
from pvlib.tools import cosd, sind
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


class TsSide(object):
    """Timeseries side class: this class is a vectorized version of the
    BaseSide geometries. The coordinates and attributes (list of segments,
    normal vector) are all vectorized."""

    def __init__(self, segments, n_vector=None):
        """Initialize timeseries side using list of timeseries segments.

        Parameters
        ----------
        segments : list of :py:class:`~pvfactors.geometry.timeseries.TsDualSegment`
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
            # Create illuminated and shaded surfaces
            illum = TsSurface(illum_coords, n_vector=n_vector,
                              param_names=param_names)
            shaded = TsSurface(shaded_coords, n_vector=n_vector,
                               param_names=param_names)
            # Create dual segment
            segment = TsDualSegment(segment_coords, illum, shaded,
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


class TsDualSegment(object):
    """A TsDualSegment is a timeseries segment that can only have
    1 shaded surface and 1 illuminated surface. This allows indexing of the
    object."""

    def __init__(self, coords, illum_ts_surface, shaded_ts_surface,
                 index=None, n_vector=None):
        """Initialize timeseries dual segment using segment coordinates and
        timeseries illuminated and shaded surfaces.

        Parameters
        ----------
        coords : :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Timeseries coordinates of full segment
        illum_ts_surface : :py:class:`~pvfactors.geometry.timeseries.TsSurface`
            Timeseries surface for illuminated part of dual segment
        shaded_ts_surface : :py:class:`~pvfactors.geometry.timeseries.TsSurface`
            Timeseries surface for shaded part of dual segment
        index : int, optional
            Index of segment (Default = None)
        n_vector : np.ndarray, optional
            Timeseries normal vectors of the side (Default = None)
        """
        self.coords = coords
        self.illum = illum_ts_surface
        self.shaded = shaded_ts_surface
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
        illum_surface = self.illum.at(idx, shaded=False)
        list_illum_surfaces = [] if illum_surface.is_empty \
            else [illum_surface]
        illum_collection = ShadeCollection(
            list_surfaces=list_illum_surfaces, shaded=False,
            param_names=None)
        # Create shaded collection
        shaded_surface = self.shaded.at(idx, shaded=True)
        list_shaded_surfaces = [] if shaded_surface.is_empty \
            else [shaded_surface]
        shaded_collection = ShadeCollection(
            list_surfaces=list_shaded_surfaces, shaded=True,
            param_names=None)
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

        value = 0
        value += self.illum.get_param(param) * self.illum.length
        value += self.shaded.get_param(param) * self.shaded.length
        return value

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


class TsGround(object):
    """Timeseries ground class: this class is a vectorized version of the
    PV ground geometry class, and it will store timeseries coordinates
    for ground shadows and pv row cut points."""

    def __init__(self, shadow_surfaces, param_names=None,
                 flag_overlap=None, cut_point_coords=None, y_ground=None):
        """Initialize timeseries ground using list of timeseries surfaces
        for the ground shadows

        Parameters
        ----------
        shadow_surfaces : list of :py:class:`~pvfactors.geometry.timeseries.TsSurface`
            Timeseries surfaces for ground shadows
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
        self.shadows = shadow_surfaces
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
            List of ground shadow coordinates
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
        cut_point_coords = [] if cut_point_coords is None else cut_point_coords
        # Create shadow surfaces
        list_coords = [TsLineCoords.from_array(coords)
                       for coords in shadow_coords]
        # If the overlap flags were passed, make sure shadows don't overlap
        if flag_overlap is not None:
            if len(list_coords) > 1:
                for idx, coords in enumerate(list_coords[:-1]):
                    coords.b2.x = np.where(flag_overlap,
                                           list_coords[idx + 1].b1.x,
                                           coords.b2.x)
        # Create shadow surfaces
        ts_shadows = [TsSurface(coords) for coords in list_coords]
        return cls(ts_shadows, param_names=param_names,
                   flag_overlap=flag_overlap,
                   cut_point_coords=cut_point_coords, y_ground=y_ground)

    def at(self, idx, x_min_max=None, merge_if_flag_overlap=True,
           with_cut_points=True):
        """Generate a PV ground geometry for the desired index.

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
        # Get cut point coords
        cut_point_coords = ([cut_point.at(idx)
                             for cut_point in self.cut_point_coords]
                            if with_cut_points else [])
        # Decide whether to merge all shadows or not
        if merge_if_flag_overlap and (self.flag_overlap is not None):
            is_overlap = self.flag_overlap[idx]
            if is_overlap and (len(self.shadows) > 1):
                ordered_shadow_coords = [[self.shadows[0].coords.b1.at(idx),
                                          self.shadows[-1].coords.b2.at(idx)]]
            else:
                ordered_shadow_coords = [shadow.coords.at(idx)
                                         for shadow in self.shadows]
        else:
            ordered_shadow_coords = [shadow.coords.at(idx)
                                     for shadow in self.shadows]
        # Create parameters at given idx
        # TODO: should find faster solution
        illum_params = {k: (val if (val is None) or np.isscalar(val)
                            or isinstance(val, dict) else val[idx])
                        for k, val in self.illum_params.items()}
        shaded_params = {k: (val if (val is None) or np.isscalar(val)
                             or isinstance(val, dict) else val[idx])
                         for k, val in self.shaded_params.items()}
        # Return ground geometry
        return PVGround.from_ordered_shadow_and_cut_pt_coords(
            x_min_max=x_min_max, ordered_shadow_coords=ordered_shadow_coords,
            cut_point_coords=cut_point_coords, param_names=self.param_names,
            illum_params=illum_params, shaded_params=shaded_params)

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
        shadow_coords = []
        cut_pt_coords = self.cut_point_coords[idx_cut_pt]
        for shadow in self.shadows:
            coords = deepcopy(shadow.coords)
            coords.b1.x = np.minimum(coords.b1.x, cut_pt_coords.x)
            coords.b1.x = np.maximum(coords.b1.x, MIN_X_GROUND)
            coords.b2.x = np.minimum(coords.b2.x, cut_pt_coords.x)
            coords.b2.x = np.maximum(coords.b2.x, MIN_X_GROUND)
            shadow_coords.append(coords)
        return shadow_coords

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
        shadow_coords = []
        cut_pt_coords = self.cut_point_coords[idx_cut_pt]
        for shadow in self.shadows:
            coords = deepcopy(shadow.coords)
            coords.b1.x = np.maximum(coords.b1.x, cut_pt_coords.x)
            coords.b1.x = np.minimum(coords.b1.x, MAX_X_GROUND)
            coords.b2.x = np.maximum(coords.b2.x, cut_pt_coords.x)
            coords.b2.x = np.minimum(coords.b2.x, MAX_X_GROUND)
            shadow_coords.append(coords)
        return shadow_coords


class TsSurface(object):
    """Timeseries surface class: vectorized representation of PV surface
    geometries."""

    def __init__(self, coords, n_vector=None, param_names=None):
        """Initialize timeseries surface using timeseries coordinates.

        Parameters
        ----------
        coords : :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Timeseries coordinates of full segment
        index : int, optional
            Index of segment (Default = None)
        n_vector : np.ndarray, optional
            Timeseries normal vectors of the side (Default = None)
        """
        self.coords = coords
        self.param_names = [] if param_names is None else param_names
        # TODO: the following should probably be turned into properties,
        # because if the coords change, they won't be altered. But speed...
        self.n_vector = n_vector
        self.params = dict.fromkeys(self.param_names)

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
            params = {k: (val if (val is None) or np.isscalar(val)
                          or isinstance(val, dict) else val[idx])
                      for k, val in self.params.items()}
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
        # TODO: the following should probably be turned into properties,
        # because if the coords change, they won't be altered. But speed...
        self.length = np.sqrt((b2_ts_coords.y - b1_ts_coords.y)**2
                              + (b2_ts_coords.x - b1_ts_coords.x)**2)

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
