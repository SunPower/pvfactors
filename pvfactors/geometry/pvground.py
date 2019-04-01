"""Classes for implementation of ground geometry"""
from pvfactors.config import MAX_X_GROUND, MIN_X_GROUND
from pvfactors.geometry.base import BaseSide, PVSegment


class PVGround(BaseSide):

    def __init__(self, list_segments=[]):
        super(PVGround, self).__init__(list_segments)

    @classmethod
    def as_flat(cls, x_min_max=None, shaded=False):
        # Get ground boundaries
        if x_min_max is None:
            x_min, x_max = MIN_X_GROUND, MAX_X_GROUND
        # Create PV segment for flat ground
        seg = PVSegment.from_linestring_coords(
            [(x_min, 0), (x_max, 0)], shaded=shaded, normal_vector=[0., 1.])
        return cls(list_segments=[seg])
