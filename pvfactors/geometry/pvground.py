"""Classes for implementation of ground geometry"""
from pvfactors.config import MAX_X_GROUND, MIN_X_GROUND, Y_GROUND
from pvfactors.geometry.base import BaseSide, PVSegment
from shapely.geometry import LineString


class PVGround(BaseSide):

    def __init__(self, list_segments=[], original_linestring=None):
        self.original_linestring = original_linestring
        super(PVGround, self).__init__(list_segments)

    @classmethod
    def as_flat(cls, x_min_max=None, shaded=False, y_ground=Y_GROUND,
                surface_params=[]):
        """Build a horizontal flat ground surface"""
        # Get ground boundaries
        if x_min_max is None:
            x_min, x_max = MIN_X_GROUND, MAX_X_GROUND
        else:
            x_min, x_max = x_min_max
        # Create PV segment for flat ground
        coords = [(x_min, y_ground), (x_max, y_ground)]
        seg = PVSegment.from_linestring_coords(coords, shaded=shaded,
                                               normal_vector=[0., 1.],
                                               surface_params=surface_params)
        return cls(list_segments=[seg], original_linestring=LineString(coords))

    @property
    def boundary(self):
        return self.original_linestring.boundary
