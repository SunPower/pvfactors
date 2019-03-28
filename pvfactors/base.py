import numpy as np
from pvfactors import PVFactorsError
from pvfactors.config import DEFAULT_NORMAL_VEC, TOL_COLLINEAR
from shapely.geometry import GeometryCollection


class BaseSide(GeometryCollection):
    """A side represents a fixed collection of
    :py:class:`~pvfactors.pvsurfaces.PVSegment` objects that should
    all be collinear, with the same normal vector"""

    def __init__(self, list_segments=[]):
        self._check_segments(list_segments)
        self.list_segments = tuple(list_segments)
        super(BaseSide, self).__init__(list_segments)

    def _check_segments(self, list_segments):
        """Check that all segments are collinear"""
        u_direction = None  # will be orthogonal to normal vector
        for segment in list_segments:
            if not segment.is_empty:
                if u_direction is None:
                    # set u_direction if not defined already
                    u_direction = np.array([- segment.n_vector[1],
                                            segment.n_vector[0]])
                else:
                    # check that collinear
                    dot_prod = u_direction.dot(segment.n_vector)
                    is_col = np.abs(dot_prod) < TOL_COLLINEAR
                    if not is_col:
                        msg = "All segments should be collinear in BaseSide"
                        raise PVFactorsError(msg)

    @property
    def n_vector(self):
        if len(self.list_segments):
            return self.list_segments[0].n_vector
        else:
            return DEFAULT_NORMAL_VEC

    @property
    def shaded_length(self):
        shaded_length = 0.
        for segment in self.list_segments:
            shaded_length += segment.shaded_length
        return shaded_length
