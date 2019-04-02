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
        super(PVRow, self).__init__([self.front, self.back])

    @classmethod
    def from_linestring_coords(cls, coords, shaded=False, normal_vector=None,
                               index=None, cut={}):
        """
        Parameters
        ----------

        normal_vector : list
            Normal vector of the front side PV segments
        """
        index_single_segment = 0
        front_side = PVRowSide.from_linestring_coords(
            coords, shaded=shaded, normal_vector=normal_vector,
            index=index_single_segment, n_segments=cut.get('front', 1))
        if normal_vector is not None:
            back_n_vec = - np.array(normal_vector)
        else:
            back_n_vec = - front_side.n_vector
        back_side = PVRowSide.from_linestring_coords(
            coords, shaded=shaded, normal_vector=back_n_vec,
            index=index_single_segment, n_segments=cut.get('back', 1))
        return cls(front_side=front_side, back_side=back_side, index=index,
                   original_linestring=LineString(coords))

    @classmethod
    def from_center_tilt_width(cls, xy_center, tilt, width, surface_azimuth,
                               axis_azimuth, shaded=False, normal_vector=None,
                               index=None, cut={}):
        coords = coords_from_center_tilt_length(xy_center, tilt, width,
                                                surface_azimuth, axis_azimuth)
        return cls.from_linestring_coords(coords, shaded=shaded,
                                          normal_vector=normal_vector,
                                          index=index, cut=cut)

    def plot(self, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
             color_illum=COLOR_DIC['pvrow_illum']):

        self.front.plot(ax, color_shaded=color_shaded, color_illum=color_illum)
        self.back.plot(ax, color_shaded=color_shaded, color_illum=color_illum)

    @property
    def boundary(self):
        return self.original_linestring.boundary
