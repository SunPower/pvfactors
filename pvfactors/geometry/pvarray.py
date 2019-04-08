"""Implement PV array classes, which will use PV rows and ground geometries"""
import numpy as np
from pvfactors.geometry.pvground import PVGround
from pvfactors.geometry.pvrow import PVRow
from pvfactors.config import X_ORIGIN_PVROWS, VIEW_DICT
from pvfactors.geometry.base import get_solar_2d_vector, BasePVArray
from pvfactors.geometry.utils import projection
from shapely.geometry import LineString, Point


class OrderedPVArray(BasePVArray):
    """An ordered PV array has a flat horizontal ground, and pv rows which
    are all at the same height, with the same surface tilt and azimuth angles,
    and also all equally spaced. These simplifications allow faster and easier
    calculations."""

    y_ground = 0.  # ground will be at height = 0 by default

    def __init__(self, list_pvrows=[], ground=None, surface_tilt=None,
                 surface_azimuth=None, axis_azimuth=None, solar_zenith=None,
                 solar_azimuth=None, gcr=None, height=None, distance=None):
        """List PV rows need to be ordered from left to right"""
        super(OrderedPVArray, self).__init__(list_pvrows=list_pvrows,
                                             ground=ground, distance=distance,
                                             height=height)
        self.gcr = gcr
        self.solar_zenith = solar_zenith
        self.solar_azimuth = solar_azimuth
        self.surface_tilt = surface_tilt
        self.surface_azimuth = surface_azimuth
        self.axis_azimuth = axis_azimuth
        self.solar_2d_vector = get_solar_2d_vector(solar_zenith, solar_azimuth,
                                                   axis_azimuth)

        # Initialize shading attributes
        self.illum_side = None
        self.has_direct_shading = False
        # For view factors
        self.edge_points = []

    @classmethod
    def from_dict(cls, parameters):
        """Create ordered PV array from dictionary of parameters"""
        # Create ground
        ground = PVGround.as_flat(y_ground=cls.y_ground)
        # Create pvrows
        list_pvrows = []
        width = parameters['pvrow_width']
        tilt = parameters['surface_tilt']
        surface_azimuth = parameters['surface_azimuth']
        axis_azimuth = parameters['axis_azimuth']
        gcr = parameters['gcr']
        y_center = parameters['pvrow_height'] + cls.y_ground
        distance = width / gcr
        # Discretization params
        cut = parameters.get('cut', {})
        # Loop for pvrow creation
        for idx in range(parameters['n_pvrows']):
            # Make equally spaced pv rows
            x_center = X_ORIGIN_PVROWS + idx * distance
            pvrow = PVRow.from_center_tilt_width(
                (x_center, y_center), tilt, width, surface_azimuth,
                axis_azimuth, index=idx, cut=cut.get(idx, {}))
            list_pvrows.append(pvrow)
        return cls(list_pvrows=list_pvrows, ground=ground, surface_tilt=tilt,
                   surface_azimuth=surface_azimuth,
                   axis_azimuth=axis_azimuth,
                   solar_zenith=parameters['solar_zenith'],
                   solar_azimuth=parameters['solar_azimuth'],
                   gcr=gcr, height=y_center, distance=distance)

    def cast_shadows(self):
        """Use calculated solar_2d_vector and array configuration to calculate
        shadows being casted in the ordered pv array.
        The logic here will be quite specific to ordered pv arrays"""
        self.illum_side = ('front' if self.pvrows[0].front.n_vector.dot(
            self.solar_2d_vector) >= 0 else 'back')
        last_gnd_2 = None
        last_gnd_1 = None
        # Cast pvrow shadows on ground
        for idx, pvrow in enumerate(self.pvrows):
            b1, b2 = pvrow.boundary
            gnd_1 = projection(b1, self.solar_2d_vector,
                               self.ground.original_linestring)
            gnd_2 = projection(b2, self.solar_2d_vector,
                               self.ground.original_linestring)
            self.ground.cast_shadow(LineString([gnd_1, gnd_2]))
            # Check if there's direct shading
            if idx == 1:
                # There's inter-row shading if ground shadows overlap
                if self.illum_side == 'front':
                    self.has_direct_shading = gnd_1.x < last_gnd_2.x
                else:
                    self.has_direct_shading = gnd_2.x < last_gnd_1.x
            last_gnd_2 = gnd_2
            last_gnd_1 = gnd_1

        # Calculate direct shading on pvrows
        sun_on_the_right = self.solar_2d_vector[0] >= 0
        if self.has_direct_shading:
            if sun_on_the_right:
                # right pvrow shades left pvrow
                shaded_pvrow = self.pvrows[0]
                front_pvrow = self.pvrows[1]
                proj_initial = projection(front_pvrow.highest_point,
                                          self.solar_2d_vector,
                                          shaded_pvrow.original_linestring)
                list_pvrows_to_shade = self.pvrows[:-1]
            else:
                # left pvrow shades right pvrow
                shaded_pvrow = self.pvrows[1]
                front_pvrow = self.pvrows[0]
                proj_initial = projection(front_pvrow.highest_point,
                                          self.solar_2d_vector,
                                          shaded_pvrow.original_linestring)
                list_pvrows_to_shade = self.pvrows[1:]

            # Use distance translation to cast shadows on pvrows
            for idx, pvrow in enumerate(list_pvrows_to_shade):
                proj = Point(proj_initial.x + idx * self.distance,
                             proj_initial.y)
                shaded_side = getattr(pvrow, self.illum_side)
                shaded_side.cast_shadow(
                    LineString([pvrow.lowest_point, proj]))

            # Merge ground shaded surfaces
            self.ground.merge_shaded_areas()

    def cuts_for_pvrow_view(self):
        """When not flat, the PV row sides will only see a part of the ground,
        so we need to mark these limits called "edge points" and cut the ground
        surface accordingly"""
        if self.surface_tilt != 0:
            # find u_vector direction of the pvrows
            b1, b2 = self.pvrows[0].boundary
            u_direction = np.array([b2.x - b1.x, b2.y - b1.y])
            for pvrow in self.pvrows:
                b1 = pvrow.boundary[0]
                # Edge point should always exist when pvrows not flat
                edge_point = projection(b1, u_direction,
                                        self.ground.original_linestring)
                self.ground.cut_at_point(edge_point)
                self.edge_points.append(edge_point)

    def _build_view_matrix(self):
        """The surface indices used in the view matrix should be the same
        as the ones in the surface_registry"""

        # Index all surfaces if not done already
        if not self._surfaces_indexed:
            self.index_all_surfaces()

        # Initialize matrices
        n_pvrows = len(self.pvrows)
        n_surfaces_array = self.n_surfaces
        n_surfaces = n_surfaces_array + 1  # counting sky
        view_matrix = np.zeros((n_surfaces, n_surfaces), dtype=int)
        obstr_matrix = np.zeros((n_surfaces, n_surfaces), dtype=object)
        obstr_matrix[:] = None

        # All surface indices need to be grouped and tracked for simplification

        indices_ground = np.array(self.ground.surface_indices)
        index_sky_dome = np.array([view_matrix.shape[0] - 1])

        # The ground will always see the sky
        view_matrix[indices_ground[:, np.newaxis],
                    index_sky_dome] = VIEW_DICT["ground_sky"]

        pvrows_are_flat = (self.surface_tilt == 0.)
        if pvrows_are_flat:
            # Getting all front and back pvrow surface indices
            indices_front_pvrows = []
            indices_back_pvrows = []
            for pvrow in self.pvrows:
                indices_front_pvrows += pvrow.front.surface_indices
                indices_back_pvrows += pvrow.back.surface_indices
            indices_front_pvrows = np.array(indices_front_pvrows)
            indices_back_pvrows = np.array(indices_back_pvrows)

            # Only back surface can see the ground
            view_matrix[indices_back_pvrows[:, np.newaxis],
                        indices_ground] = VIEW_DICT["back_gnd"]
            view_matrix[indices_ground[:, np.newaxis],
                        indices_back_pvrows] = VIEW_DICT["gnd_back"]
            # The front side only sees the sky
            view_matrix[indices_front_pvrows[:, np.newaxis],
                        index_sky_dome] = VIEW_DICT["front_sky"]
        else:
            # Setting up row neighbors
            list_left_neighbors = [None] + self.pvrows[:-1]
            list_right_neighbors = self.pvrows[1:] + [None]

            # PVRow <---> PVRow
            rotated_to_left = self.pvrows[0].front.n_vector[0] < 0
            for idx_pvrow, pvrow in enumerate(self.pvrows):
                left_neighbor = list_left_neighbors[idx_pvrow]
                right_neighbor = list_right_neighbors[idx_pvrow]
                if rotated_to_left:
                    if left_neighbor is not None:
                        front_indices = np.array(pvrow.front.surface_indices)
                        left_n_indices = np.array(left_neighbor
                                                  .back.surface_indices)
                        view_matrix[front_indices[:, np.newaxis],
                                    left_n_indices] = VIEW_DICT["pvrows"]
                    if right_neighbor is not None:
                        back_indices = np.array(pvrow.back.surface_indices)
                        right_n_indices = np.array(right_neighbor
                                                   .front.surface_indices)
                        view_matrix[back_indices[:, np.newaxis],
                                    right_n_indices] = VIEW_DICT["pvrows"]
                else:   # rotated to right
                    if left_neighbor is not None:
                        back_indices = np.array(pvrow.back.surface_indices)
                        left_n_indices = np.array(left_neighbor
                                                  .front.surface_indices)
                        view_matrix[back_indices[:, np.newaxis],
                                    left_n_indices] = VIEW_DICT["pvrows"]
                    if right_neighbor is not None:
                        front_indices = np.array(pvrow.front.surface_indices)
                        right_n_indices = np.array(right_neighbor
                                                   .back.surface_indices)
                        view_matrix[front_indices[:, np.newaxis],
                                    right_n_indices] = VIEW_DICT["pvrows"]

            # PVRow <---> PVGround

        return view_matrix
