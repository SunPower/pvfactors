"""Module containing PV array classes, which will use PV rows and ground
geometries."""

import numpy as np
from pvfactors.geometry.pvground import PVGround
from pvfactors.geometry.pvrow import PVRow
from pvfactors.config import X_ORIGIN_PVROWS, VIEW_DICT, DISTANCE_TOLERANCE
from pvfactors.geometry.base import \
    _get_solar_2d_vectors, BasePVArray, _coords_from_center_tilt_length, \
    _get_rotation_from_tilt_azimuth
from pvfactors.geometry.utils import projection
from shapely.geometry import LineString, Point
from pvfactors import PVFactorsError


class OrderedPVArray(BasePVArray):
    """An ordered PV array has a flat horizontal ground, and pv rows which
    are all at the same height, with the same surface tilt and azimuth angles,
    and also all equally spaced. These simplifications allow faster and easier
    calculations. In the ordered PV array, the list of PV rows must be
    ordered from left to right (along the x-axis) in the 2D geometry."""

    y_ground = 0.  # ground will be at height = 0 by default

    def __init__(self, axis_azimuth=None, gcr=None, pvrow_height=None,
                 n_pvrows=None, pvrow_width=None, surface_params=None,
                 cut=None):
        """Initialize ordered PV array.
        List of PV rows will be ordered from left to right.

        Parameters
        ----------
        axis_azimuth : float, optional
            Azimuth angle of rotation axis [deg] (Default = None)
        gcr : float, optional
            Ground coverage ratio (Default = None)
        pvrow_height : float, optional
            Unique height of all PV rows in [m] (Default = None)
        n_pvrows : int, optional
            Number of PV rows in the PV array (Default = None)
        pvrow_width : float, optional
            Width of the PV rows in the 2D plane in [m] (Default = None)
        surface_params : list of str, optional
            List of surface parameter names for the PV surfaces
            (Default = None)
        cut : dict, optional
            Nested dictionary that tells if some PV row sides need to be
            discretized, and how (Default = None).
            Example: {1: {'front': 5}}, will create 5 segments on the front
            side of the PV row with index 1
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
        self.surface_params = [] if surface_params is None else surface_params
        self.cut = {} if cut is None else cut

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
    def init_from_dict(cls, pvarray_params, surface_params=None):
        """Instantiate ordered PV array from dictionary of parameters

        Parameters
        ----------
        pvarray_params : dict
            The parameters defining the PV array
        surface_params : list of str, optional
            List of parameter names to pass to surfaces (Default = None)

        Returns
        -------
        OrderedPVArray
            Initialized Ordered PV Array
        """
        return cls(axis_azimuth=pvarray_params['axis_azimuth'],
                   gcr=pvarray_params['gcr'],
                   pvrow_height=pvarray_params['pvrow_height'],
                   n_pvrows=pvarray_params['n_pvrows'],
                   pvrow_width=pvarray_params['pvrow_width'],
                   cut=pvarray_params.get('cut', {}),
                   surface_params=surface_params)

    @classmethod
    def transform_from_dict_of_scalars(cls, pvarray_params,
                                       surface_params=None):
        """Instantiate, fit and transform ordered PV array using dictionary
        of scalar inputs.

        Parameters
        ----------
        pvarray_params : dict
            The parameters used for instantiation, fitting, and transformation
        surface_params : list of str, optional
            List of parameter names to pass to surfaces (Default = None)

        Returns
        -------
        OrderedPVArray
            Initialized, fitted, and transformed Ordered PV Array
        """

        # Create pv array
        pvarray = cls.init_from_dict(pvarray_params,
                                     surface_params=surface_params)

        # Fit pv array to scalar values
        solar_zenith = np.array([pvarray_params['solar_zenith']])
        solar_azimuth = np.array([pvarray_params['solar_azimuth']])
        surface_tilt = np.array([pvarray_params['surface_tilt']])
        surface_azimuth = np.array([pvarray_params['surface_azimuth']])
        pvarray.fit(solar_zenith, solar_azimuth,
                    surface_tilt, surface_azimuth)

        # Transform pv array to first index (since scalar values were passed)
        pvarray.transform(0)

        return pvarray

    def fit(self, solar_zenith, solar_azimuth, surface_tilt, surface_azimuth):
        """Fit the ordered PV array to the list of solar and surface angles.
        All intermediate PV array results necessary to build the geometries
        will be calculated here using vectorization as much as possible.

        Intemediate results include: PV row coordinates for all timestamps,
        ground element coordinates for all timestamps, cases of direct
        shading, ...

        Parameters
        ----------
        solar_zenith : array-like or float
            Solar zenith angles [deg]
        solar_azimuth : array-like or float
            Solar azimuth angles [deg]
        surface_tilt : array-like or float
            Surface tilt angles, from 0 to 180 [deg]
        surface_azimuth : array-like or float
            Surface azimuth angles [deg]
        """

        self.n_states = len(solar_zenith)
        self.surface_tilt = surface_tilt

        # Calculate the solar 2D vectors for all timestamps
        self.solar_2d_vectors = _get_solar_2d_vectors(
            solar_zenith, solar_azimuth, self.axis_azimuth)

        # Calculate the coordinates of all PV rows for all timestamps
        xy_centers = [(X_ORIGIN_PVROWS + idx * self.distance,
                       self.height + self.y_ground)
                      for idx in range(self.n_pvrows)]
        for xy_center in xy_centers:
            self.pvrow_coords.append(
                _coords_from_center_tilt_length(
                    xy_center, surface_tilt, self.width, surface_azimuth,
                    self.axis_azimuth))
        self.pvrow_coords = np.array(self.pvrow_coords)

        # Calculate ground elements coordinates
        self._calculate_ground_elements_coords(surface_tilt, surface_azimuth)

        # Determine when there's direct shading
        self.has_direct_shading = np.zeros(self.n_states, dtype=bool)
        if self.n_pvrows > 1:
            # If the shadows are crossing (or close), there's direct shading
            self.has_direct_shading = (
                self.ground_shadow_coords[1][0][0] + DISTANCE_TOLERANCE
                < self.ground_shadow_coords[0][1][0])

    def transform(self, idx):
        """
        Transform the ordered PV array for the given index.
        This means actually building the PV Row and Ground geometries. Note
        that the list of PV rows will be ordered from left to right in the
        geometry (along the x-axis), and indexed from 0 to n_pvrows - 1.
        This can only be run after the ``fit()`` method.

        Object attributes like ``pvrows`` and ``ground`` will be updated each
        time this method is run.

        Parameters
        ----------
        idx : int
            Index for which to build the simulation.
        """

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

    def _calculate_ground_elements_coords(self, surface_tilt, surface_azimuth):
        """This private method is run at fitting time to calculate the ground
        element coordinates: i.e. the coordinates of the ground shadows and
        cut points.
        It will update the ``ground_shadow_coords`` and ``cut_point_coords``
        attributes of the object.

        Parameters
        ----------
        surface_tilt : array-like or float
            Surface tilt angles, from 0 to 180 [deg]
        surface_azimuth : array-like or float
            Surface azimuth angles [deg]
        """

        # Calculate the angle made by 2D sun vector and x-axis
        alpha_vec = np.arctan2(self.solar_2d_vectors[1],
                               self.solar_2d_vectors[0])
        # Calculate rotation angles
        rotation_vec = _get_rotation_from_tilt_azimuth(
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
        """This private method is run at transform time to calculate
        direct shading between the PV rows, if any.
        It will update the ``pvrows`` side segments with shadows as needed.

        Parameters
        ----------
        idx : int
            Index for which to calculate inter-row shading
        """

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
        and back surfaces of each pvrow.

        Parameters
        ----------
        surface_tilt : array-like or float
            Surface tilt angles, from 0 to 180 [deg]
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
        ordered pv arrays to build relational matrix between surfaces."""

        # Index all surfaces
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
