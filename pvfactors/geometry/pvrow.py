"""Class related to PV row geometries"""

import numpy as np
from pvfactors.config import COLOR_DIC
from pvfactors.geometry.base import \
    BaseSide, coords_from_center_tilt_length
from shapely.geometry import GeometryCollection, LineString


class PVRowSide(BaseSide):
    """A PV row side represents the whole surface of one side of a PV row.
    At its core it will contain a fixed number of
    :py:class:`~pvfactors.pvsurfaces.PVSegment` objects that will together
    constitue one side of a PV row: a PV row side could for instance be
    "discretized" into multiple segments"""

    def __init__(self, list_segments=[]):
        super(PVRowSide, self).__init__(list_segments)


class PVRow(GeometryCollection):
    """A PV row is made of two PV row sides, a front and a back one"""

    def __init__(self, front_side=PVRowSide(), back_side=PVRowSide(),
                 index=None, original_linestring=None):
        """front and back sides are supposed to be deleted"""
        self.front = front_side
        self.back = back_side
        self.index = index
        self.original_linestring = original_linestring
        self._all_surfaces = None
        super(PVRow, self).__init__([self.front, self.back])

    @classmethod
    def from_linestring_coords(cls, coords, shaded=False, normal_vector=None,
                               index=None, cut={}, surface_params=[]):
        """
        Parameters
        ----------

        normal_vector : list
            Normal vector of the front side PV segments
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
        coords = coords_from_center_tilt_length(xy_center, tilt, width,
                                                surface_azimuth, axis_azimuth)
        return cls.from_linestring_coords(coords, shaded=shaded,
                                          normal_vector=normal_vector,
                                          index=index, cut=cut,
                                          surface_params=surface_params)

    def plot(self, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
             color_illum=COLOR_DIC['pvrow_illum'], with_index=False):

        self.front.plot(ax, color_shaded=color_shaded, color_illum=color_illum,
                        with_index=with_index)
        self.back.plot(ax, color_shaded=color_shaded, color_illum=color_illum,
                       with_index=with_index)

    @property
    def boundary(self):
        return self.original_linestring.boundary

    @property
    def highest_point(self):
        b1, b2 = self.boundary
        highest_point = b1 if b1.y > b2.y else b2
        return highest_point

    @property
    def lowest_point(self):
        b1, b2 = self.boundary
        lowest_point = b1 if b1.y < b2.y else b2
        return lowest_point

    @property
    def all_surfaces(self):
        if self._all_surfaces is None:
            self._all_surfaces = []
            self._all_surfaces += self.front.all_surfaces
            self._all_surfaces += self.back.all_surfaces
        return self._all_surfaces

    @property
    def surface_indices(self):
        list_indices = []
        list_indices += self.front.surface_indices
        list_indices += self.back.surface_indices
        return list_indices
