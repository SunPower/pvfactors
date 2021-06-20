"""Module will classes related to PV row geometries"""

import numpy as np
from pvfactors.config import COLOR_DIC
from pvfactors.geometry.timeseries import \
    TsShadeCollection, TsLineCoords, TsSurface
from pvlib.tools import cosd, sind


class TsPVRow(object):
    """Timeseries PV row class: this class is a vectorized version of the
    PV row geometries. The coordinates and attributes (front and back sides)
    are all vectorized."""

    def __init__(self, ts_front_side, ts_back_side, xy_center, index=None,
                 full_pvrow_coords=None):
        """Initialize timeseries PV row with its front and back sides.

        Parameters
        ----------
        ts_front_side : :py:class:`~pvfactors.geometry.pvrow.TsSide`
            Timeseries front side of the PV row
        ts_back_side : :py:class:`~pvfactors.geometry.pvrow.TsSide`
            Timeseries back side of the PV row
        xy_center : tuple of float
            x and y coordinates of the PV row center point (invariant)
        index : int, optional
            index of the PV row (Default = None)
        full_pvrow_coords : \
        :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`, optional
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
                    color_illum=COLOR_DIC['pvrow_illum'],
                    with_surface_index=False):
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
        with_surface_index : bool, optional
            Plot the surfaces with their index values (Default = False)
        """
        pvrow = self.at(idx)
        pvrow.plot(ax, color_shaded=color_shaded,
                   color_illum=color_illum, with_index=with_surface_index)

    # def at(self, idx):
    #     """Generate a PV row geometry for the desired index.

    #     Parameters
    #     ----------
    #     idx : int
    #         Index to use to generate PV row geometry

    #     Returns
    #     -------
    #     pvrow : :py:class:`~pvfactors.geometry.pvrow.PVRow`
    #     """
    #     front_geom = self.front.at(idx)
    #     back_geom = self.back.at(idx)
    #     original_line = LineString(
    #         self.full_pvrow_coords.as_array[:, :, idx])
    #     pvrow = PVRow(front_side=front_geom, back_side=back_geom,
    #                   index=self.index, original_linestring=original_line)
    #     return pvrow

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
        centroid = (self.full_pvrow_coords.centroid
                    if self.full_pvrow_coords is not None else None)
        return centroid

    @property
    def length(self):
        """Length of both sides of the timeseries PV row"""
        return self.front.length + self.back.length

    @property
    def highest_point(self):
        """Timeseries point coordinates of highest point of PV row"""
        high_pt = (self.full_pvrow_coords.highest_point
                   if self.full_pvrow_coords is not None else None)
        return high_pt


class TsSide(object):
    """Timeseries side class: this class is a vectorized version of the
    BaseSide geometries. The coordinates and attributes (list of segments,
    normal vector) are all vectorized."""

    def __init__(self, segments, n_vector=None):
        """Initialize timeseries side using list of timeseries segments.

        Parameters
        ----------
        segments : list of :py:class:`~pvfactors.geometry.pvrow.TsSegment`
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
            is_shaded = False
            illum = TsShadeCollection(
                [TsSurface(illum_coords, n_vector=n_vector,
                           param_names=param_names, shaded=is_shaded)],
                is_shaded)
            is_shaded = True
            shaded = TsShadeCollection(
                [TsSurface(shaded_coords, n_vector=n_vector,
                           param_names=param_names, shaded=is_shaded)],
                is_shaded)
            # Create segment
            segment = TsSegment(segment_coords, illum, shaded,
                                n_vector=n_vector, index=i)
            list_segments.append(segment)

        return cls(list_segments, n_vector=n_vector)

    # def surfaces_at_idx(self, idx):
    #     """Get all PV surface geometries in timeseries side for a certain
    #     index.

    #     Parameters
    #     ----------
    #     idx : int
    #         Index to use to generate PV surface geometries

    #     Returns
    #     -------
    #     list of :py:class:`~pvfactors.geometry.base.PVSurface` objects
    #         List of PV surfaces
    #     """
    #     side_geom = self.at(idx)
    #     return side_geom.all_surfaces

    # def at(self, idx):
    #     """Generate a side geometry for the desired index.

    #     Parameters
    #     ----------
    #     idx : int
    #         Index to use to generate side geometry

    #     Returns
    #     -------
    #     side : :py:class:`~pvfactors.geometry.base.BaseSide`
    #     """
    #     list_geom_segments = []
    #     for ts_seg in self.list_segments:
    #         list_geom_segments.append(ts_seg.at(idx))
    #     side = BaseSide(list_geom_segments)
    #     return side

    # def plot_at_idx(self, idx, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
    #                 color_illum=COLOR_DIC['pvrow_illum']):
    #     """Plot timeseries side at a certain index.

    #     Parameters
    #     ----------
    #     idx : int
    #         Index to use to plot timeseries side
    #     ax : :py:class:`matplotlib.pyplot.axes` object
    #         Axes for plotting
    #     color_shaded : str, optional
    #         Color to use for plotting the shaded surfaces (Default =
    #         COLOR_DIC['pvrow_shaded'])
    #     color_shaded : str, optional
    #         Color to use for plotting the illuminated surfaces (Default =
    #         COLOR_DIC['pvrow_illum'])
    #     """
    #     side_geom = self.at(idx)
    #     side_geom.plot(ax, color_shaded=color_shaded, color_illum=color_illum,
    #                    with_index=False)

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
        illum_collection : \
        :py:class:`~pvfactors.geometry.timeseries.TsShadeCollection`
            Timeseries collection for illuminated part of segment
        shaded_collection : \
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

    # def surfaces_at_idx(self, idx):
    #     """Get all PV surface geometries in timeseries segment for a certain
    #     index.

    #     Parameters
    #     ----------
    #     idx : int
    #         Index to use to generate PV surface geometries

    #     Returns
    #     -------
    #     list of :py:class:`~pvfactors.geometry.base.PVSurface` objects
    #         List of PV surfaces
    #     """
    #     segment = self.at(idx)
    #     return segment.all_surfaces

    # def plot_at_idx(self, idx, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
    #                 color_illum=COLOR_DIC['pvrow_illum']):
    #     """Plot timeseries segment at a certain index.

    #     Parameters
    #     ----------
    #     idx : int
    #         Index to use to plot timeseries segment
    #     ax : :py:class:`matplotlib.pyplot.axes` object
    #         Axes for plotting
    #     color_shaded : str, optional
    #         Color to use for plotting the shaded surfaces (Default =
    #         COLOR_DIC['pvrow_shaded'])
    #     color_shaded : str, optional
    #         Color to use for plotting the illuminated surfaces (Default =
    #         COLOR_DIC['pvrow_illum'])
    #     """
    #     segment = self.at(idx)
    #     segment.plot(ax, color_shaded=color_shaded, color_illum=color_illum,
    #                  with_index=False)

    # def at(self, idx):
    #     """Generate a PV segment geometry for the desired index.

    #     Parameters
    #     ----------
    #     idx : int
    #         Index to use to generate PV segment geometry

    #     Returns
    #     -------
    #     segment : :py:class:`~pvfactors.geometry.base.PVSegment`
    #     """
    #     # Create illum collection
    #     illum_collection = self.illum.at(idx)
    #     # Create shaded collection
    #     shaded_collection = self.shaded.at(idx)
    #     # Create PV segment
    #     segment = PVSegment(illum_collection=illum_collection,
    #                         shaded_collection=shaded_collection,
    #                         index=self.index)
    #     return segment

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


class PVRowSide:

    def __init__(self, list_segments=[]):
        pass


class PVRow:
    def __init__(self, front_side: PVRowSide, back_side: PVRowSide,
                 index: int = None, original_linestring=None):
        pass
