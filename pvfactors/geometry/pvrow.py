from pvfactors.geometry.base import BaseSide
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

    def __init__(self, front_side=PVRowSide(), back_side=PVRowSide()):
        """front and back sides are supposed to be deleted"""
        self.front = front_side
        self.back = back_side
        super(PVRow, self).__init__([self.front, self.back])
