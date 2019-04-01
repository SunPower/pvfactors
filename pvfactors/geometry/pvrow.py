import numpy as np
from pvfactors.geometry.base import \
    BaseSide, coords_from_center_tilt_length
from shapely.geometry import GeometryCollection


class PVRowSide(BaseSide):
    """A PV row side represents the whole surface of one side of a PV row.
    At its core it will contain a fixed number of
    :py:class:`~pvfactors.pvsurfaces.PVSegment` objects that will together
    constitue one side of a PV row: a PV row side could for instance be
    "discretized" into multiple segments"""

    def __init__(self, list_pvsegments=[]):
        super(PVRowSide, self).__init__(list_pvsegments)


class PVRow(GeometryCollection):
    """A PV row is made of two PV row sides, a front and a back one"""

    def __init__(self, front_side=PVRowSide(), back_side=PVRowSide(),
                 index=None):
        """front and back sides are supposed to be deleted"""
        self.front = front_side
        self.back = back_side
        self.index = index
        super(PVRow, self).__init__([self.front, self.back])

    @classmethod
    def from_linestring_coords(cls, coords, shaded=False, normal_vector=None,
                               index=None):
        """
        Parameters
        ----------

        normal_vector : list
            Normal vector of the front side PV segments
        """
        index_single_segment = 0
        front_side = PVRowSide.from_linestring_coords(
            coords, shaded=shaded, normal_vector=normal_vector,
            index=index_single_segment)
        if normal_vector is not None:
            back_n_vec = - np.array(normal_vector)
        else:
            back_n_vec = - front_side.n_vector
        back_side = PVRowSide.from_linestring_coords(
            coords, shaded=shaded, normal_vector=back_n_vec,
            index=index_single_segment)
        return cls(front_side=front_side, back_side=back_side, index=index)

    @classmethod
    def from_center_tilt_width(cls, xy_center, tilt, width, shaded=False,
                               normal_vector=None, index=None):
        coords = coords_from_center_tilt_length(xy_center, tilt, width)
        return cls.from_linestring_coords(coords, shaded=shaded,
                                          normal_vector=normal_vector,
                                          index=index)
