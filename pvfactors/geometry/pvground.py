"""Classes for implementation of ground geometry"""
from pvfactors.config import \
    MAX_X_GROUND, MIN_X_GROUND, Y_GROUND, DISTANCE_TOLERANCE
from pvfactors.geometry.base import \
    BaseSide, PVSegment, ShadeCollection, PVSurface
from shapely.geometry import LineString


class PVGround(BaseSide):
    """Class that defines the ground geometry in PV arrays."""

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

        Returns
        -------
        PVGround object
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
    def from_ordered_shadow_and_cut_pt_coords(
        cls, x_min_max=None, y_ground=Y_GROUND, ordered_shadow_coords=[],
            cut_point_coords=[], surface_params=[]):
        """Build a horizontal flat ground surface, made of 1 PVSegment with
        shaded areas and "cut points" (reference points used in view factor
        calculations)

        Parameters
        ----------
        x_min_max : tuple, optional
            List of minimum and maximum x coordinates for the flat surface [m]
            (Default = None)
        y_ground : float, optional
            Location of flat ground on y axis in [m] (Default = Y_GROUND)
        ordered_shadow_coords : list, optional
            List of shadow coordinates, ordered from left to right.
            Shape = (# of shadows, 2 {# of points}, 2 {for xy coords})
            (Default = [])
        cut_point_coords : list, optional
            List of cut point coordinates, ordered from left to right.
            Shape = (# of cut points, 2 {# of points}, 2 {for xy coords})
            (Default = [])
        surface_params : list of str, optional
            Names of the surface parameters, eg reflectivity, total incident
            irradiance, temperature, etc. (Default = [])

        Returns
        -------
        PVGround object
            PV ground with shadows, illuminated ground, and cut points
        """
        # Get ground boundaries
        if x_min_max is None:
            x_min, x_max = MIN_X_GROUND, MAX_X_GROUND
        else:
            x_min, x_max = x_min_max
        full_extent_coords = [(x_min, y_ground), (x_max, y_ground)]

        # Create the list of illuminated and shaded PV surfaces
        list_shaded_surfaces = []
        list_illum_surfaces = []
        recurse_on_cut_points(list_shaded_surfaces, list_illum_surfaces,
                              ordered_shadow_coords, cut_point_coords,
                              x_min, x_max, y_ground, surface_params)

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


def recurse_on_cut_points(
        list_shaded_surfaces, list_illum_surfaces,
        ordered_shadow_coords, cut_point_coords,
        x_min, x_max, y_ground, surface_params):
    """Solve recursively the problem of building the lists of shaded and
    illuminated PV surfaces when cut points and shadows might exist.

    Parameters
    ----------
    list_shaded_surfaces : list
        List of shaded surfaces to add to
    list_illum_surfaces : list
        List of illuminated surfaces to add to
    ordered_shadow_coords : list
        List of shadow coordinates, ordered from left to right.
        Shape = (# of shadows, 2 {# of points}, 2 {for xy coords})
    cut_point_coords : list
        List of cut point coordinates, ordered from left to right.
        Shape = (# of cut points, 2 {# of points}, 2 {for xy coords})
    x_min : float
        Leftmost x coordinate of the considered ground [m]
    x_max : float
        Rightmost x coordinate of the considered ground [m]
    y_ground : float
        Location of flat ground on y axis in [m]
    surface_params : list of str, optional
        Names of the surface parameters, eg reflectivity, total incident
        irradiance, temperature, etc.
    """

    if len(cut_point_coords) == 0:
        build_list_illum_shadow_surfaces(
            list_shaded_surfaces, list_illum_surfaces, ordered_shadow_coords,
            x_min, x_max, y_ground, surface_params)
    else:
        # Get leftmost cut point
        cut_point = cut_point_coords[0]
        # Remove that point from the cut point list
        cut_point_coords = cut_point_coords[1:]
        # Now try to solve in the smaller picture case
        is_cut_pt_in_picture = (cut_point[0] > x_min + DISTANCE_TOLERANCE) \
            and (cut_point[0] < x_max - DISTANCE_TOLERANCE)
        if is_cut_pt_in_picture:
            # Reduce picture frame
            new_x_max = cut_point[0]
            # TODO: try to reduce list of shadow coords as well
            build_list_illum_shadow_surfaces(
                list_shaded_surfaces, list_illum_surfaces,
                ordered_shadow_coords, x_min, new_x_max, y_ground,
                surface_params)
            # New x_min should be at the cut point
            x_min = new_x_max
        # Now move to the next [x_min, x_max] frame
        recurse_on_cut_points(list_shaded_surfaces, list_illum_surfaces,
                              ordered_shadow_coords, cut_point_coords,
                              x_min, x_max, y_ground, surface_params)


def build_list_illum_shadow_surfaces(
        list_shaded_surfaces, list_illum_surfaces, ordered_shadow_coords,
        x_min, x_max, y_ground, surface_params):
    """Build lists of shaded and illuminated surfaces in the case when
    there are no cut points.

    Parameters
    ----------
    list_shaded_surfaces : list
        List of shaded surfaces to add to
    list_illum_surfaces : list
        List of illuminated surfaces to add to
    ordered_shadow_coords : list
        List of shadow coordinates, ordered from left to right.
        Shape = (# of shadows, 2 {# of points}, 2 {for xy coords})
    x_min : float
        Leftmost x coordinate of the considered ground [m]
    x_max : float
        Rightmost x coordinate of the considered ground [m]
    y_ground : float
        Location of flat ground on y axis in [m]
    surface_params : list of str, optional
        Names of the surface parameters, eg reflectivity, total incident
        irradiance, temperature, etc.
    """
    if len(ordered_shadow_coords) == 0:
        if x_min + DISTANCE_TOLERANCE < x_max:
            # The whole ground is illuminated
            full_extent_coords = [(x_min, y_ground), (x_max, y_ground)]
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
                    if x2 < x_max - DISTANCE_TOLERANCE:
                        # Shadow is cropped in the picture
                        coord = [[x_min, y1], [x2, y2]]
                        # Update x_illum_left to right x coord of shadow
                        x_illum_left = x2
                    else:
                        # Shadow is covering the whole picture
                        coord = [[x_min, y1], [x_max, y2]]
                        # Update x_illum_left to x_max
                        x_illum_left = x_max
                    # Create shadow and add it to list
                    shadow = PVSurface(coords=coord, shaded=True,
                                       surface_params=surface_params)
                    list_shaded_surfaces.append(shadow)
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
