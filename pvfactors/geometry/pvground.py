"""Classes for implementation of ground geometry"""
from pvfactors.config import MAX_X_GROUND, MIN_X_GROUND, Y_GROUND
from pvfactors.geometry.base import BaseSide, PVSegment
from shapely.geometry import LineString


class PVGround(BaseSide):
    """Class that defines the ground in PV arrays."""

    def __init__(self, list_segments=[], original_linestring=None):
        """Initialize PV ground geometry.

        Parameters
        ----------
        list_segments : list of :py:class:`~pvfactors.geometry.base.PVSegment`
            List of PV segments that will constitute the ground
        original_linestring : :py:class:`shapely.geometry.LineString`, optional
            Full continuous linestring that the ground will be made of
            (Default = None)
        """
        self.original_linestring = original_linestring
        super(PVGround, self).__init__(list_segments)

    @classmethod
    def as_flat(cls, x_min_max=None, shaded=False, y_ground=Y_GROUND,
                surface_params=[]):
        """Build a horizontal flat ground surface, made of 1 PV segment.

        Parameters
        ----------
        x_min_max : tuple, optional
            List of minimum and maximum x coordinates for the flat surface [m]
            (Default = None)
        shaded : bool, optional
            Shaded status of the created PV surfaces (Default = False)
        y_ground : float, optional
            Location of flat ground on y axis in [m] (Default = Y_GROUND)
        surface_params : list of str, optional
            Names of the surface parameters, eg reflectivity, total incident
            irradiance, temperature, etc. (Default = [])
        """
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
        """Boundaries of the ground's original linestring."""
        return self.original_linestring.boundary
