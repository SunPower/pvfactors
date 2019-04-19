"""Module will classes related to PV row geometries"""

import numpy as np
from pvfactors.config import COLOR_DIC
from pvfactors.geometry.base import \
    BaseSide, coords_from_center_tilt_length
from shapely.geometry import GeometryCollection, LineString


class PVRowSide(BaseSide):
    """A PV row side represents the whole surface of one side of a PV row.
    At its core it will contain a fixed number of
    :py:class:`~pvfactors.geometry.base.PVSegment` objects that will together
    constitue one side of a PV row: a PV row side can also be
    "discretized" into multiple segments"""

    def __init__(self, list_segments=[]):
        """Initialize PVRowSide using its base class
        :py:class:`pvfactors.geometry.base.BaseSide`

        Parameters
        ----------
        list_segments : list of :py:class:`~pvfactors.geometry.base.PVSegment`
            List of PV segments for PV row side.
        """
        super(PVRowSide, self).__init__(list_segments)


class PVRow(GeometryCollection):
    """A PV row is made of two PV row sides, a front and a back one."""

    def __init__(self, front_side=PVRowSide(), back_side=PVRowSide(),
                 index=None, original_linestring=None):
        """Initialize PV row.

        Parameters
        ----------
        front_side : :py:class:`~pvfactors.geometry.pvrow.PVRowSide`, optional
            Front side of the PV Row (Default = Empty PVRowSide)
        back_side : :py:class:`~pvfactors.geometry.pvrow.PVRowSide`, optional
            Back side of the PV Row (Default = Empty PVRowSide)
        index : int, optional
            Index of PV row (Default = None)
        original_linestring : :py:class:`shapely.geometry.LineString`, optional
            Full continuous linestring that the PV row will be made of
            (Default = None)

        """
        self.front = front_side
        self.back = back_side
        self.index = index
        self.original_linestring = original_linestring
        self._all_surfaces = None
        super(PVRow, self).__init__([self.front, self.back])

    @classmethod
    def from_linestring_coords(cls, coords, shaded=False, normal_vector=None,
                               index=None, cut={}, surface_params=[]):
        """Create a PV row with a single PV surface and using linestring
        coordinates.

        Parameters
        ----------
        coords : list
            List of linestring coordinates for the surface
        shaded : bool, optional
            Shading status desired for the PVRow sides (Default = False)
        normal_vector : list, optional
            Normal vector for the surface (Default = None)
        index : int, optional
            Index of PV row (Default = None)
        cut : dict, optional
            Scheme to decide how many segments to create on each side.
            Eg {'front': 3, 'back': 2} will lead to 3 segments on front side
            and 2 segments on back side. (Default = {})
        surface_params : list of str, optional
            Names of the surface parameters, eg reflectivity, total incident
            irradiance, temperature, etc. (Default = [])

        Returns
        -------
        :py:class:`~pvfactors.geometry.pvrow.PVRow` object
        """
        index_single_segment = 0
        front_side = PVRowSide.from_linestring_coords(
            coords, shaded=shaded, normal_vector=normal_vector,
            index=index_single_segment, n_segments=cut.get('front', 1),
            surface_params=surface_params)
        if normal_vector is not None:
            back_n_vec = - np.array(normal_vector)
        else:
            back_n_vec = - front_side.n_vector
        back_side = PVRowSide.from_linestring_coords(
            coords, shaded=shaded, normal_vector=back_n_vec,
            index=index_single_segment, n_segments=cut.get('back', 1),
            surface_params=surface_params)
        return cls(front_side=front_side, back_side=back_side, index=index,
                   original_linestring=LineString(coords))

    @classmethod
    def from_center_tilt_width(cls, xy_center, tilt, width, surface_azimuth,
                               axis_azimuth, shaded=False, normal_vector=None,
                               index=None, cut={}, surface_params=[]):
        """Create a PV row using mainly the coordinates of the line center,
        a tilt angle, and its length.

        Parameters
        ----------
        xy_center : tuple
            x, y coordinates of center point of desired linestring
        tilt : float
            surface tilt angle desired [deg]
        length : float
            desired length of linestring [m]
        surface_azimuth : float
            Surface azimuth of PV surface [deg]
        axis_azimuth : float
            Axis azimuth of the PV surface, i.e. direction of axis of rotation
            [deg]
        shaded : bool, optional
            Shading status desired for the PVRow sides (Default = False)
        normal_vector : list, optional
            Normal vector for the surface (Default = None)
        index : int, optional
            Index of PV row (Default = None)
        cut : dict, optional
            Scheme to decide how many segments to create on each side.
            Eg {'front': 3, 'back': 2} will lead to 3 segments on front side
            and 2 segments on back side. (Default = {})
        surface_params : list of str, optional
            Names of the surface parameters, eg reflectivity, total incident
            irradiance, temperature, etc. (Default = [])

        Returns
        -------
        :py:class:`~pvfactors.geometry.pvrow.PVRow` object
        """
        coords = coords_from_center_tilt_length(xy_center, tilt, width,
                                                surface_azimuth, axis_azimuth)
        return cls.from_linestring_coords(coords, shaded=shaded,
                                          normal_vector=normal_vector,
                                          index=index, cut=cut,
                                          surface_params=surface_params)

    def plot(self, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
             color_illum=COLOR_DIC['pvrow_illum'], with_index=False):
        """Plot the surfaces of the PV Row.

        Parameters
        ----------
        ax : :py:class:`matplotlib.pyplot.axes` object
            Axes for plotting
        color_shaded : str, optional
            Color to use for plotting the shaded surfaces (Default =
            COLOR_DIC['pvrow_shaded'])
        color_shaded : str, optional
            Color to use for plotting the illuminated surfaces (Default =
            COLOR_DIC['pvrow_illum'])
        with_index : bool
            Flag to annotate surfaces with their indices (Default = False)

        """
        self.front.plot(ax, color_shaded=color_shaded, color_illum=color_illum,
                        with_index=with_index)
        self.back.plot(ax, color_shaded=color_shaded, color_illum=color_illum,
                       with_index=with_index)

    @property
    def boundary(self):
        """Boundaries of the PV Row's orginal linestring."""
        return self.original_linestring.boundary

    @property
    def highest_point(self):
        """Highest point of the PV Row."""
        b1, b2 = self.boundary
        highest_point = b1 if b1.y > b2.y else b2
        return highest_point

    @property
    def lowest_point(self):
        """Lowest point of the PV Row."""
        b1, b2 = self.boundary
        lowest_point = b1 if b1.y < b2.y else b2
        return lowest_point

    @property
    def all_surfaces(self):
        """List of all the surfaces in the PV row."""
        if self._all_surfaces is None:
            self._all_surfaces = []
            self._all_surfaces += self.front.all_surfaces
            self._all_surfaces += self.back.all_surfaces
        return self._all_surfaces

    @property
    def surface_indices(self):
        """List of all surface indices in the PV Row."""
        list_indices = []
        list_indices += self.front.surface_indices
        list_indices += self.back.surface_indices
        return list_indices

    def update_params(self, new_dict):
        """Update surface parameters for both front and back sides.

        Parameters
        ----------
        new_dict : dict
            Parameters to add or update for the surface
        """
        self.front.update_params(new_dict)
        self.back.update_params(new_dict)
