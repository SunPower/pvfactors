"""Classes for implementation of ground geometry"""
from pvfactors.config import MAX_X_GROUND, MIN_X_GROUND
from pvfactors.geometry.base import BaseSide, ShadeCollection
from pvfactors.geometry.pvsurface import PVSurface, PVSegment
from shapely.geometry import LineString


class PVGround(BaseSide):

    def __init__(self, list_segments=[]):
        super(PVGround, self).__init__(list_segments)

    @classmethod
    def as_flat(self, x_min_max=None):
        if x_min_max is None:
            x_min, x_max = MIN_X_GROUND, MAX_X_GROUND
        surf = PVSurface([(MIN_X_GROUND, 0), (MAX_X_GROUND, 0)])
