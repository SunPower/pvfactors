"""Classes for implementation of ground geometry"""
from pvfactors.config import \
    MAX_X_GROUND, MIN_X_GROUND, Y_GROUND, DISTANCE_TOLERANCE
from pvfactors.geometry.base import \
    BaseSide, PVSegment, ShadeCollection, PVSurface
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

    @classmethod
    def from_ordered_shadow_coords(cls, x_min_max=None, y_ground=Y_GROUND,
                                   ordered_shadow_coords=[],
                                   surface_params=[]):
        # Get ground boundaries
        if x_min_max is None:
            x_min, x_max = MIN_X_GROUND, MAX_X_GROUND
        else:
            x_min, x_max = x_min_max
        full_extent_coords = [(x_min, y_ground), (x_max, y_ground)]

        # Create the list of illuminated and shaded PV surfaces
        list_shaded_surfaces = []
        list_illum_surfaces = []
        if len(ordered_shadow_coords) == 0:
            # The whole ground is illuminated
            illum = PVSurface(coords=full_extent_coords, shaded=False,
                              surface_params=surface_params)
            list_illum_surfaces.append(illum)
        else:
            # There are shadows and illuminated areas
            # x_illum_left is the left x coord of the potential left illum area
            # wrt the shadow
            x_illum_left = x_min
            for shadow_coord in ordered_shadow_coords:
                [[x1, y1], [x2, y2]] = shadow_coord
                if x1 < x_min:
                    if x2 > x_min + DISTANCE_TOLERANCE:
                        # Shadow is cropped in the picture
                        coord = [[x_min, y1], [x2, y2]]
                        shadow = PVSurface(coords=coord, shaded=True,
                                           surface_params=surface_params)
                        list_shaded_surfaces.append(shadow)
                        # Update x_illum_left to right x coord of shadow
                        x_illum_left = x2
                elif x2 > x_max:
                    if x1 + DISTANCE_TOLERANCE < x_max:
                        # Shadow is cropped in the picture
                        coord = [[x1, y1], [x_max, y2]]
                        shadow = PVSurface(coords=coord, shaded=True,
                                           surface_params=surface_params)
                        list_shaded_surfaces.append(shadow)
                        if x_illum_left + DISTANCE_TOLERANCE < x1:
                            coord = [[x_illum_left, y1], [x1, y1]]
                            illum = PVSurface(coords=coord, shaded=False,
                                              surface_params=surface_params)
                            list_illum_surfaces.append(illum)
                        # Update x_illum_left to right x coord of shadow
                        x_illum_left = x_max
                else:
                    # Shadow is fully in the picture
                    shadow = PVSurface(coords=shadow_coord, shaded=True,
                                       surface_params=surface_params)
                    list_shaded_surfaces.append(shadow)
                    if x_illum_left + DISTANCE_TOLERANCE < x1:
                        coord = [[x_illum_left, y1], [x1, y1]]
                        illum = PVSurface(coords=coord, shaded=False,
                                          surface_params=surface_params)
                        list_illum_surfaces.append(illum)
                    # Update x_illum_left to right x coord of shadow
                    x_illum_left = x2
            # Add right illum area if exist
            if x_illum_left + DISTANCE_TOLERANCE < x_max:
                coord = [[x_illum_left, y1], [x_max, y1]]
                illum = PVSurface(coords=coord, shaded=False,
                                  surface_params=surface_params)
                list_illum_surfaces.append(illum)

        # Create the shade collections
        shaded_collection = ShadeCollection(
            list_surfaces=list_shaded_surfaces, shaded=True,
            surface_params=surface_params)
        illum_collection = ShadeCollection(
            list_surfaces=list_illum_surfaces, shaded=False,
            surface_params=surface_params)

        # Create the ground segment
        segment = PVSegment(illum_collection=illum_collection,
                            shaded_collection=shaded_collection)

        return cls(list_segments=[segment],
                   original_linestring=LineString(full_extent_coords))

    @property
    def boundary(self):
        """Boundaries of the ground's original linestring."""
        return self.original_linestring.boundary
