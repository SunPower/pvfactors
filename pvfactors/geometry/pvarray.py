"""Module containing PV array classes, which will use PV rows and ground
geometries."""

import numpy as np
from pvfactors.geometry.pvground import PVGround
from pvfactors.geometry.pvrow import PVRow
from pvfactors.config import X_ORIGIN_PVROWS, VIEW_DICT, DISTANCE_TOLERANCE
from pvfactors.geometry.base import \
    get_solar_2d_vectors, BasePVArray, coords_from_center_tilt_length, \
    get_rotation_from_tilt_azimuth
from pvfactors.geometry.utils import projection
from shapely.geometry import LineString, Point
from pvfactors import PVFactorsError


class SlowOrderedPVArray(BasePVArray):
    """An ordered PV array has a flat horizontal ground, and pv rows which
    are all at the same height, with the same surface tilt and azimuth angles,
    and also all equally spaced. These simplifications allow faster and easier
    calculations."""

    y_ground = 0.  # ground will be at height = 0 by default

    def __init__(self, axis_azimuth=None, gcr=None, pvrow_height=None,
                 n_pvrows=None, pvrow_width=None, surface_params=[], cut={}):
        """Initialize ordered PV array.
        List of PV rows will be ordered from left to right.

        Parameters
        ----------
        list_pvrows : list of :py:class:`~pvfactors.geometry.pvrow.PVRow`, optional
            List of PV rows in the PV array
            (Default = [])
        ground : :py:class:`~pvfactors.geometry.pvground.PVGround`, optional
            Ground geometry for the PV array
        surface_tilt : float, optional
            Surface tilt angles, from 0 to 180 [deg] (Default = None)
        surface_azimuth : float, optional
            Surface azimuth angles [deg] (Default = None)
        axis_azimuth : float, optional
            Azimuth angle of rotation axis [deg] (Default = None)
        solar_zenith : array-like, optional
            Solar zenith angles [deg] (Default = None)
        solar_azimuth : array-like, optional
            Solar azimuth angles [deg] (Default = None)
        gcr : float, optional
            Ground coverage ratio (Default = None)
        height : float, optional
            Unique height of all PV rows (Default = None)
        distance : float, optional
            Unique distance between PV rows (Default = None)
        """
        # Initialize base parameters: common to all sorts of PV arrays
        super(SlowOrderedPVArray, self).__init__(axis_azimuth=axis_azimuth)

        # These are the invariant parameters of the PV array
        self.gcr = gcr
        self.height = pvrow_height
        self.distance = pvrow_width / gcr \
            if (pvrow_width is not None) and (gcr is not None) \
            else None
        self.width = pvrow_width
        self.n_pvrows = n_pvrows
        self.surface_params = surface_params
        self.cut = cut

        # These parameters will be defined at fitting time
        self.pvrow_coords = []
        self.ground_shadow_coords = []
        self.cut_point_coords = []
        self.n_states = None
        self.has_direct_shading = None
        self.solar_zenith = None
        self.solar_azimuth = None
        self.surface_tilt = None
        self.surface_azimuth = None

        # These attributes will be transformed at each iteration
        self.pvrows = None
        self.ground = None
        self.front_neighbors = None
        self.back_neighbors = None
        self.solar_2d_vector = None
        self.edge_points = None

    @classmethod
    def init_from_dict(cls, params, surface_params=[]):
        return cls(
            axis_azimuth=params['axis_azimuth'], gcr=params['gcr'],
            pvrow_height=params['pvrow_height'], n_pvrows=params['n_pvrows'],
            pvrow_width=params['pvrow_width'], cut=params.get('cut', {}),
            surface_params=surface_params)

    @classmethod
    def transform_from_dict_of_scalars(cls, params, surface_params=[]):

        # Create pv array
        pvarray = cls.init_from_dict(params, surface_params=surface_params)

        # Fit pv array to scalar values
        solar_zenith = np.array([params['solar_zenith']])
        solar_azimuth = np.array([params['solar_azimuth']])
        surface_tilt = np.array([params['surface_tilt']])
        surface_azimuth = np.array([params['surface_azimuth']])
        pvarray.fit(solar_zenith, solar_azimuth,
                    surface_tilt, surface_azimuth)

        # Transform pv array to first index
        pvarray.transform(0)

        return pvarray

    def fit(self, solar_zenith, solar_azimuth, surface_tilt, surface_azimuth):

        self.n_states = len(solar_zenith)
        # Save timeseries parameters that will be used later on
        self.solar_zenith = solar_zenith
        self.solar_azimuth = solar_azimuth
        self.surface_tilt = surface_tilt
        self.surface_azimuth = surface_azimuth

    def transform(self, idx):

        self.solar_2d_vector = get_solar_2d_vectors(
            self.solar_zenith[idx], self.solar_azimuth[idx],
            self.axis_azimuth)
        # Create ground and pvrows
        self.ground, self.pvrows = self._build_pvrows_ground(idx)
        # Cast shadows on ground and pvrows
        self._cast_shadows()
        # Cut ground according to pvrow views
        self.is_flat = self.surface_tilt[idx] == 0
        self._cuts_for_pvrow_view()

        # Determine front and back pvrow neighbors
        self.front_neighbors, self.back_neighbors = self._get_neighbors()

    def _build_pvrows_ground(self, idx):
        """Create ordered PV array from dictionary of parameters

        Parameters
        ----------
        parameters : dict
            The parameters defining the PV array
        surface_params : list of str, optional
            List of parameter names to pass to surfaces (Default = [])
        """
        # Create ground
        ground = PVGround.as_flat(y_ground=self.y_ground,
                                  surface_params=self.surface_params)
        # Create pvrows
        list_pvrows = []
        y_center = self.height + self.y_ground
        tilt = self.surface_tilt[idx]
        surface_azimuth = self.surface_azimuth[idx]
        # Loop for pvrow creation
        for pvrow_idx in range(self.n_pvrows):
            # Make equally spaced pv rows
            x_center = X_ORIGIN_PVROWS + pvrow_idx * self.distance
            pvrow = PVRow.from_center_tilt_width(
                (x_center, y_center), tilt, self.width, surface_azimuth,
                self.axis_azimuth, index=pvrow_idx,
                cut=self.cut.get(pvrow_idx, {}),
                surface_params=self.surface_params)
            list_pvrows.append(pvrow)

        return ground, list_pvrows

    def _get_neighbors(self):
        """Determine the pvrows indices of the neighboring pvrow for the front
        and back surface of each pvrow.
        """
        n_pvrows = len(self.pvrows)
        if n_pvrows:
            rotated_to_left = self.pvrows[0].front.n_vector[0] < 0
            if self.is_flat:
                front_neighbors = [None] * n_pvrows
                back_neighbors = [None] * n_pvrows
            elif rotated_to_left:
                front_neighbors = [None] + list(range(n_pvrows - 1))
                back_neighbors = list(range(1, n_pvrows)) + [None]
            else:  # rotated to right
                front_neighbors = list(range(1, n_pvrows)) + [None]
                back_neighbors = [None] + list(range(n_pvrows - 1))

        return front_neighbors, back_neighbors

    def _cast_shadows(self):
        """Use calculated solar_2d_vector and array configuration to calculate
        shadows being casted in the ordered pv array.
        The logic here is quite specific to ordered pv arrays"""
        self.illum_side = ('front' if self.pvrows[0].front.n_vector.dot(
            self.solar_2d_vector) >= 0 else 'back')
        last_gnd_2 = None
        last_gnd_1 = None
        # Cast pvrow shadows on ground
        a_shadow_was_casted_on_ground = False
        stop_checking_for_direct_shading = False
        for idx, pvrow in enumerate(self.pvrows):
            b1, b2 = pvrow.boundary
            gnd_1 = projection(b1, self.solar_2d_vector,
                               self.ground.original_linestring)
            gnd_2 = projection(b2, self.solar_2d_vector,
                               self.ground.original_linestring)
            # Following can happen because the ground geometry is not infinite
            shadow_is_too_far = gnd_1.is_empty or gnd_2.is_empty
            if not shadow_is_too_far:
                self.ground.cast_shadow(LineString([gnd_1, gnd_2]))
                # Check if there's direct shading
                if a_shadow_was_casted_on_ground \
                   and not stop_checking_for_direct_shading:
                    # There's inter-row shading if ground shadows overlap
                    if self.illum_side == 'front':
                        self.has_direct_shading = gnd_1.x + DISTANCE_TOLERANCE < last_gnd_2.x
                    else:
                        self.has_direct_shading = gnd_2.x + DISTANCE_TOLERANCE < last_gnd_1.x
                    stop_checking_for_direct_shading = True
                last_gnd_2 = gnd_2
                last_gnd_1 = gnd_1
                a_shadow_was_casted_on_ground = True

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

    def _cuts_for_pvrow_view(self):
        """When not flat, the PV row sides will only see a part of the ground,
        so we need to mark these limits called "edge points" and cut the ground
        surface accordingly"""
        # Initialize list of edge points
        self.edge_points = []

        # Add points only if not flat
        if not self.is_flat:
            # find u_vector direction of the pvrows
            b1, b2 = self.pvrows[0].boundary
            u_direction = np.array([b2.x - b1.x, b2.y - b1.y])
            for pvrow in self.pvrows:
                b1 = pvrow.boundary[0]
                # Edge point should always exist when pvrows not flat
                edge_point = projection(b1, u_direction,
                                        self.ground.original_linestring,
                                        must_contain=False)
                self.ground.cut_at_point(edge_point)
                self.edge_points.append(edge_point)

    def _build_view_matrix(self):
        """Calculate the pv array view matrix: using rules specific to
        ordered pv arrays to build relationship matrix"""

        # Index all surfaces if not done already
        self.index_all_surfaces()

        # Initialize matrices
        n_surfaces_array = self.n_surfaces
        n_surfaces = n_surfaces_array + 1  # counting sky
        view_matrix = np.zeros((n_surfaces, n_surfaces), dtype=int)
        obstr_matrix = np.zeros((n_surfaces, n_surfaces), dtype=object)
        obstr_matrix[:] = None

        indices_ground = np.array(self.ground.surface_indices)
        index_sky_dome = np.array([view_matrix.shape[0] - 1])

        # The ground will always see the sky
        view_matrix[indices_ground[:, np.newaxis],
                    index_sky_dome] = VIEW_DICT["ground_sky"]

        if self.is_flat:
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
            # List ground surfaces
            ground_surfaces = self.ground.all_surfaces
            gnd_centroids = [surf.centroid for surf in ground_surfaces]

            rotated_to_left = self.pvrows[0].front.n_vector[0] < 0
            for idx_pvrow, pvrow in enumerate(self.pvrows):
                front_indices = np.array(pvrow.front.surface_indices)
                back_indices = np.array(pvrow.back.surface_indices)
                left_neighbor = list_left_neighbors[idx_pvrow]
                right_neighbor = list_right_neighbors[idx_pvrow]
                edge_point = self.edge_points[idx_pvrow]

                # Will save index of pvrows obstructing front and back
                front_obstruction = None
                back_obstruction = None

                if rotated_to_left:
                    # PVRow <---> PVRow + obstruction to ground
                    if left_neighbor is not None:
                        left_n_indices = np.array(left_neighbor
                                                  .back.surface_indices)
                        view_matrix[front_indices[:, np.newaxis],
                                    left_n_indices] = VIEW_DICT["pvrows"]
                        front_obstruction = left_neighbor.index
                    if right_neighbor is not None:
                        right_n_indices = np.array(right_neighbor
                                                   .front.surface_indices)
                        view_matrix[back_indices[:, np.newaxis],
                                    right_n_indices] = VIEW_DICT["pvrows"]
                        back_obstruction = right_neighbor.index

                    # PVRow <---> Ground
                    gnd_that_front_sees = []
                    gnd_that_back_sees = []
                    for idx_gnd, gnd_surface in enumerate(ground_surfaces):
                        gnd_surface_on_the_right = \
                            gnd_centroids[idx_gnd].x > edge_point.x
                        if gnd_surface_on_the_right:
                            gnd_that_back_sees.append(gnd_surface.index)
                        else:
                            gnd_that_front_sees.append(gnd_surface.index)

                else:   # rotated to right
                    # PVRow <---> PVRow + obstruction to ground
                    if left_neighbor is not None:
                        left_n_indices = np.array(left_neighbor
                                                  .front.surface_indices)
                        view_matrix[back_indices[:, np.newaxis],
                                    left_n_indices] = VIEW_DICT["pvrows"]
                        back_obstruction = left_neighbor.index
                    if right_neighbor is not None:
                        right_n_indices = np.array(right_neighbor
                                                   .back.surface_indices)
                        view_matrix[front_indices[:, np.newaxis],
                                    right_n_indices] = VIEW_DICT["pvrows"]
                        front_obstruction = right_neighbor.index

                    # PVRow <---> Ground
                    gnd_that_front_sees = []
                    gnd_that_back_sees = []
                    for idx_gnd, gnd_surface in enumerate(ground_surfaces):
                        gnd_surface_on_the_right = \
                            gnd_centroids[idx_gnd].x > edge_point.x
                        if gnd_surface_on_the_right:
                            gnd_that_front_sees.append(gnd_surface.index)
                        else:
                            gnd_that_back_sees.append(gnd_surface.index)

                # Update views to ground
                gnd_that_back_sees = np.array(gnd_that_back_sees)
                gnd_that_front_sees = np.array(gnd_that_front_sees)
                if len(gnd_that_back_sees):
                    # PVRow <---> Ground: update views
                    view_matrix[back_indices[:, np.newaxis],
                                gnd_that_back_sees] = \
                        VIEW_DICT["back_gnd_obst"]
                    view_matrix[gnd_that_back_sees[:, np.newaxis],
                                back_indices] = VIEW_DICT["gnd_back_obst"]
                    # PVRow <---> Ground: obstruction
                    obstr_matrix[back_indices[:, np.newaxis],
                                 gnd_that_back_sees] = back_obstruction
                    obstr_matrix[gnd_that_back_sees[:, np.newaxis],
                                 back_indices] = back_obstruction
                if len(gnd_that_front_sees):
                    # PVRow <---> Ground: update views
                    view_matrix[front_indices[:, np.newaxis],
                                gnd_that_front_sees] = \
                        VIEW_DICT["front_gnd_obst"]
                    view_matrix[gnd_that_front_sees[:, np.newaxis],
                                front_indices] = VIEW_DICT["gnd_front_obst"]
                    # PVRow <---> Ground: obstruction
                    obstr_matrix[front_indices[:, np.newaxis],
                                 gnd_that_front_sees] = front_obstruction
                    obstr_matrix[gnd_that_front_sees[:, np.newaxis],
                                 front_indices] = front_obstruction

                # PVRow <---> Sky
                view_matrix[back_indices[:, np.newaxis],
                            index_sky_dome] = VIEW_DICT["back_sky"]
                view_matrix[front_indices[:, np.newaxis],
                            index_sky_dome] = VIEW_DICT["front_sky"]

        return view_matrix, obstr_matrix


class OrderedPVArray(BasePVArray):

    y_ground = 0.  # ground will be at height = 0 by default

    def __init__(self, axis_azimuth=None, gcr=None, pvrow_height=None,
                 n_pvrows=None, pvrow_width=None, surface_params=[], cut={}):
        """Initialize ordered PV array.
        List of PV rows will be ordered from left to right.

        Parameters
        ----------
        axis_azimuth : float, optional
            Azimuth angle of rotation axis [deg] (Default = None)
        gcr : float, optional
            Ground coverage ratio (Default = None)
        pvrow_height : float, optional
            Unique height of all PV rows (Default = None)
        n_pvrows : int, optional
            Number of PV rows in the PV array (Default = None)
        pvrow_width : float, optional
            Width of the PV rows in the 2D plane (Default = None)
        surface_params : list of str, optional
            List of surface parameter names for the PV surfaces
            (Default = [])
        cut : dict, optional
            Nested dictionary that tells if some PV row sides need to be
            discretized, and how (Default = {})
        """
        # Initialize base parameters: common to all sorts of PV arrays
        super(OrderedPVArray, self).__init__(axis_azimuth=axis_azimuth)

        # These are the invariant parameters of the PV array
        self.gcr = gcr
        self.height = pvrow_height
        self.distance = (pvrow_width / gcr
                         if (pvrow_width is not None) and (gcr is not None)
                         else None)
        self.width = pvrow_width
        self.n_pvrows = n_pvrows
        self.surface_params = surface_params
        self.cut = cut

        # These attributes will be updated at fitting time
        self.solar_2d_vectors = None
        self.pvrow_coords = []
        self.ground_shadow_coords = []
        self.cut_point_coords = []
        self.n_states = None
        self.has_direct_shading = None
        self.surface_tilt = None

        # These attributes will be transformed at each iteration
        self.pvrows = None
        self.ground = None
        self.front_neighbors = None
        self.back_neighbors = None
        self.edge_points = None
        self.is_flat = None

    @classmethod
    def init_from_dict(cls, params, surface_params=[]):
        return cls(
            axis_azimuth=params['axis_azimuth'], gcr=params['gcr'],
            pvrow_height=params['pvrow_height'], n_pvrows=params['n_pvrows'],
            pvrow_width=params['pvrow_width'], cut=params.get('cut', {}),
            surface_params=surface_params)

    @classmethod
    def transform_from_dict_of_scalars(cls, params, surface_params=[]):

        # Create pv array
        pvarray = cls.init_from_dict(params, surface_params=surface_params)

        # Fit pv array to scalar values
        solar_zenith = np.array([params['solar_zenith']])
        solar_azimuth = np.array([params['solar_azimuth']])
        surface_tilt = np.array([params['surface_tilt']])
        surface_azimuth = np.array([params['surface_azimuth']])
        pvarray.fit(solar_zenith, solar_azimuth,
                    surface_tilt, surface_azimuth)

        # Transform pv array to first index
        pvarray.transform(0)

        return pvarray

    def fit(self, solar_zenith, solar_azimuth, surface_tilt, surface_azimuth):

        self.n_states = len(solar_zenith)
        self.surface_tilt = surface_tilt

        # Calculate the solar 2D vectors for all timestamps
        self.solar_2d_vectors = get_solar_2d_vectors(
            solar_zenith, solar_azimuth, self.axis_azimuth)

        # Calculate the coordinates of all PV rows for all timestamps
        xy_centers = [(X_ORIGIN_PVROWS + idx * self.distance,
                       self.height + self.y_ground)
                      for idx in range(self.n_pvrows)]
        for xy_center in xy_centers:
            self.pvrow_coords.append(
                coords_from_center_tilt_length(
                    xy_center, surface_tilt, self.width, surface_azimuth,
                    self.axis_azimuth))
        self.pvrow_coords = np.array(self.pvrow_coords)

        # Calculate ground elements coordinates
        self._calculate_ground_elements_coords(surface_azimuth, surface_tilt)

        # Determine when there's direct shading
        self.has_direct_shading = np.zeros(self.n_states, dtype=bool)
        if self.n_pvrows > 1:
            # If the shadows are crossing (or close), there's direct shading
            self.has_direct_shading = (
                self.ground_shadow_coords[1][0][0] + DISTANCE_TOLERANCE
                < self.ground_shadow_coords[0][1][0])

    def transform(self, idx):

        if idx < self.n_states:
            has_direct_shading = self.has_direct_shading[idx]
            self.is_flat = self.surface_tilt[idx] == 0

            # Create list of PV rows from calculated pvrow coordinates
            self.pvrows = [
                PVRow.from_linestring_coords(
                    pvrow_coord[:, :, idx],
                    index=pvrow_idx, cut=self.cut.get(pvrow_idx, {}),
                    surface_params=self.surface_params)
                for pvrow_idx, pvrow_coord in enumerate(self.pvrow_coords)]

            # Create ground geometry with its shadows and cut points
            shadow_coords = self.ground_shadow_coords[:, :, :, idx]
            cut_pt_coords = self.cut_point_coords[:, :, idx]
            if has_direct_shading:
                # Shadows are overlaping, so merge them into 1 big shadow
                shadow_coords = [[shadow_coords[0][0][:],
                                  shadow_coords[-1][1][:]]]
            self.ground = PVGround.from_ordered_shadow_and_cut_pt_coords(
                y_ground=self.y_ground, surface_params=self.surface_params,
                ordered_shadow_coords=shadow_coords,
                cut_point_coords=cut_pt_coords)
            self.edge_points = [Point(coord) for coord in cut_pt_coords]

            # Calculate inter row shading if any
            if has_direct_shading:
                self._calculate_interrow_shading(idx)

            # Build lists of pv row neighbors, used to calculate view matrix
            self.front_neighbors, self.back_neighbors = \
                self._get_neighbors(self.surface_tilt[idx])

        else:
            msg = "Step index {} is out of range: [0 to {}]".format(
                idx, self.n_states - 1)
            raise PVFactorsError(msg)

    def _calculate_ground_elements_coords(self, surface_azimuth, surface_tilt):

        # Calculate the angle made by 2D sun vector and x-axis
        alpha_vec = np.arctan2(self.solar_2d_vectors[1],
                               self.solar_2d_vectors[0])
        # Calculate rotation angles
        rotation_vec = get_rotation_from_tilt_azimuth(
            surface_azimuth, self.axis_azimuth, surface_tilt)
        rotation_vec = np.deg2rad(rotation_vec)
        # Calculate coords of ground shadows and cutting points
        for pvrow_coord in self.pvrow_coords:
            # Get pvrow coords
            x1s_pvrow = pvrow_coord[0][0]
            y1s_pvrow = pvrow_coord[0][1]
            x2s_pvrow = pvrow_coord[1][0]
            y2s_pvrow = pvrow_coord[1][1]
            # --- Shadow coords calculation
            # Calculate x coords of shadow
            x1s_shadow = x1s_pvrow \
                - (y1s_pvrow - self.y_ground) / np.tan(alpha_vec)
            x2s_shadow = x2s_pvrow \
                - (y2s_pvrow - self.y_ground) / np.tan(alpha_vec)
            # Order x coords from left to right
            x1s_on_left = x1s_shadow <= x2s_shadow
            xs_left_shadow = np.where(x1s_on_left, x1s_shadow, x2s_shadow)
            xs_right_shadow = np.where(x1s_on_left, x2s_shadow, x1s_shadow)
            # Append shadow coords to list
            self.ground_shadow_coords.append(
                [[xs_left_shadow, self.y_ground * np.ones(self.n_states)],
                 [xs_right_shadow, self.y_ground * np.ones(self.n_states)]])
            # --- Cutting points coords calculation
            dx = (y1s_pvrow - self.y_ground) / np.tan(rotation_vec)
            self.cut_point_coords.append(
                [x1s_pvrow - dx, self.y_ground * np.ones(self.n_states)])

        self.ground_shadow_coords = np.array(self.ground_shadow_coords)
        self.cut_point_coords = np.array(self.cut_point_coords)

    def _calculate_interrow_shading(self, idx):

        solar_2d_vector = self.solar_2d_vectors[:, idx]
        illum_side = ('front' if self.pvrows[0].front.n_vector.dot(
            solar_2d_vector) >= 0 else 'back')
        sun_on_the_right = solar_2d_vector[0] >= 0
        if sun_on_the_right:
            # right pvrow shades left pvrow
            shaded_pvrow = self.pvrows[0]
            front_pvrow = self.pvrows[1]
            proj_initial = projection(front_pvrow.highest_point,
                                      solar_2d_vector,
                                      shaded_pvrow.original_linestring)
            list_pvrows_to_shade = self.pvrows[:-1]
        else:
            # left pvrow shades right pvrow
            shaded_pvrow = self.pvrows[1]
            front_pvrow = self.pvrows[0]
            proj_initial = projection(front_pvrow.highest_point,
                                      solar_2d_vector,
                                      shaded_pvrow.original_linestring)
            list_pvrows_to_shade = self.pvrows[1:]

        # Use distance translation to cast shadows on pvrows
        for idx, pvrow in enumerate(list_pvrows_to_shade):
            proj = Point(proj_initial.x + idx * self.distance,
                         proj_initial.y)
            shaded_side = getattr(pvrow, illum_side)
            shaded_side.cast_shadow(
                LineString([pvrow.lowest_point, proj]))

    def _get_neighbors(self, surface_tilt):
        """Determine the pvrows indices of the neighboring pvrow for the front
        and back surface of each pvrow.
        """
        n_pvrows = len(self.pvrows)
        if n_pvrows:
            flat = surface_tilt == 0
            rotated_to_left = self.pvrows[0].front.n_vector[0] < 0
            if flat:
                front_neighbors = [None] * n_pvrows
                back_neighbors = [None] * n_pvrows
            elif rotated_to_left:
                front_neighbors = [None] + list(range(n_pvrows - 1))
                back_neighbors = list(range(1, n_pvrows)) + [None]
            else:  # rotated to right
                front_neighbors = list(range(1, n_pvrows)) + [None]
                back_neighbors = [None] + list(range(n_pvrows - 1))

        return front_neighbors, back_neighbors

    def _build_view_matrix(self):
        """Calculate the pv array view matrix: using rules specific to
        ordered pv arrays to build relationship matrix"""

        # Index all surfaces if not done already
        self.index_all_surfaces()

        # Initialize matrices
        n_surfaces_array = self.n_surfaces
        n_surfaces = n_surfaces_array + 1  # counting sky
        view_matrix = np.zeros((n_surfaces, n_surfaces), dtype=int)
        obstr_matrix = np.zeros((n_surfaces, n_surfaces), dtype=object)
        obstr_matrix[:] = None

        indices_ground = np.array(self.ground.surface_indices)
        index_sky_dome = np.array([view_matrix.shape[0] - 1])

        # The ground will always see the sky
        view_matrix[indices_ground[:, np.newaxis],
                    index_sky_dome] = VIEW_DICT["ground_sky"]

        if self.is_flat:
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
            # List ground surfaces
            ground_surfaces = self.ground.all_surfaces
            gnd_centroids = [surf.centroid for surf in ground_surfaces]

            rotated_to_left = self.pvrows[0].front.n_vector[0] < 0
            for idx_pvrow, pvrow in enumerate(self.pvrows):
                front_indices = np.array(pvrow.front.surface_indices)
                back_indices = np.array(pvrow.back.surface_indices)
                left_neighbor = list_left_neighbors[idx_pvrow]
                right_neighbor = list_right_neighbors[idx_pvrow]
                edge_point = self.edge_points[idx_pvrow]

                # Will save index of pvrows obstructing front and back
                front_obstruction = None
                back_obstruction = None

                if rotated_to_left:
                    # PVRow <---> PVRow + obstruction to ground
                    if left_neighbor is not None:
                        left_n_indices = np.array(left_neighbor
                                                  .back.surface_indices)
                        view_matrix[front_indices[:, np.newaxis],
                                    left_n_indices] = VIEW_DICT["pvrows"]
                        front_obstruction = left_neighbor.index
                    if right_neighbor is not None:
                        right_n_indices = np.array(right_neighbor
                                                   .front.surface_indices)
                        view_matrix[back_indices[:, np.newaxis],
                                    right_n_indices] = VIEW_DICT["pvrows"]
                        back_obstruction = right_neighbor.index

                    # PVRow <---> Ground
                    gnd_that_front_sees = []
                    gnd_that_back_sees = []
                    for idx_gnd, gnd_surface in enumerate(ground_surfaces):
                        gnd_surface_on_the_right = \
                            gnd_centroids[idx_gnd].x > edge_point.x
                        if gnd_surface_on_the_right:
                            gnd_that_back_sees.append(gnd_surface.index)
                        else:
                            gnd_that_front_sees.append(gnd_surface.index)

                else:   # rotated to right
                    # PVRow <---> PVRow + obstruction to ground
                    if left_neighbor is not None:
                        left_n_indices = np.array(left_neighbor
                                                  .front.surface_indices)
                        view_matrix[back_indices[:, np.newaxis],
                                    left_n_indices] = VIEW_DICT["pvrows"]
                        back_obstruction = left_neighbor.index
                    if right_neighbor is not None:
                        right_n_indices = np.array(right_neighbor
                                                   .back.surface_indices)
                        view_matrix[front_indices[:, np.newaxis],
                                    right_n_indices] = VIEW_DICT["pvrows"]
                        front_obstruction = right_neighbor.index

                    # PVRow <---> Ground
                    gnd_that_front_sees = []
                    gnd_that_back_sees = []
                    for idx_gnd, gnd_surface in enumerate(ground_surfaces):
                        gnd_surface_on_the_right = \
                            gnd_centroids[idx_gnd].x > edge_point.x
                        if gnd_surface_on_the_right:
                            gnd_that_front_sees.append(gnd_surface.index)
                        else:
                            gnd_that_back_sees.append(gnd_surface.index)

                # Update views to ground
                gnd_that_back_sees = np.array(gnd_that_back_sees)
                gnd_that_front_sees = np.array(gnd_that_front_sees)
                if len(gnd_that_back_sees):
                    # PVRow <---> Ground: update views
                    view_matrix[back_indices[:, np.newaxis],
                                gnd_that_back_sees] = \
                        VIEW_DICT["back_gnd_obst"]
                    view_matrix[gnd_that_back_sees[:, np.newaxis],
                                back_indices] = VIEW_DICT["gnd_back_obst"]
                    # PVRow <---> Ground: obstruction
                    obstr_matrix[back_indices[:, np.newaxis],
                                 gnd_that_back_sees] = back_obstruction
                    obstr_matrix[gnd_that_back_sees[:, np.newaxis],
                                 back_indices] = back_obstruction
                if len(gnd_that_front_sees):
                    # PVRow <---> Ground: update views
                    view_matrix[front_indices[:, np.newaxis],
                                gnd_that_front_sees] = \
                        VIEW_DICT["front_gnd_obst"]
                    view_matrix[gnd_that_front_sees[:, np.newaxis],
                                front_indices] = VIEW_DICT["gnd_front_obst"]
                    # PVRow <---> Ground: obstruction
                    obstr_matrix[front_indices[:, np.newaxis],
                                 gnd_that_front_sees] = front_obstruction
                    obstr_matrix[gnd_that_front_sees[:, np.newaxis],
                                 front_indices] = front_obstruction

                # PVRow <---> Sky
                view_matrix[back_indices[:, np.newaxis],
                            index_sky_dome] = VIEW_DICT["back_sky"]
                view_matrix[front_indices[:, np.newaxis],
                            index_sky_dome] = VIEW_DICT["front_sky"]

        return view_matrix, obstr_matrix
