"""Module containing PV array classes, which will use PV rows and ground
geometries."""

import numpy as np
from pvfactors.config import X_ORIGIN_PVROWS, DISTANCE_TOLERANCE
from pvfactors.geometry.base import \
    _get_solar_2d_vectors, BasePVArray, _get_rotation_from_tilt_azimuth
from pvfactors.geometry.pvrow import TsPVRow
from pvfactors.geometry.pvground import TsGround


class OrderedPVArray(BasePVArray):
    """An ordered PV array has a flat horizontal ground, and pv rows which
    are all at the same height, with the same surface tilt and azimuth angles,
    and also all equally spaced. These simplifications allow faster and easier
    calculations. In the ordered PV array, the list of PV rows must be
    ordered from left to right (along the x-axis) in the 2D geometry."""

    y_ground = 0.  # ground will be at height = 0 by default

    def __init__(self, axis_azimuth=None, gcr=None, pvrow_height=None,
                 n_pvrows=None, pvrow_width=None, param_names=None,
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
        param_names : list of str, optional
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
        self.param_names = [] if param_names is None else param_names
        self.cut = {} if cut is None else cut

        # These attributes will be updated at fitting time
        self.solar_2d_vectors = None
        self.n_states = None
        self.has_direct_shading = None
        self.rotation_vec = None
        self.shaded_length_front = None
        self.shaded_length_back = None

        # These attributes will be updated by the engine
        self.ts_vf_matrix = None
        self.ts_vf_aoi_matrix = None

    @classmethod
    def init_from_dict(cls, pvarray_params, param_names=None):
        """Instantiate ordered PV array from dictionary of parameters

        Parameters
        ----------
        pvarray_params : dict
            The parameters defining the PV array
        param_names : list of str, optional
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
                   param_names=param_names)

    @classmethod
    def fit_from_dict_of_scalars(cls, pvarray_params, param_names=None):
        """Instantiate, and fit ordered PV array using dictionary
        of scalar inputs.

        Parameters
        ----------
        pvarray_params : dict
            The parameters used for instantiation, fitting, and transformation
        param_names : list of str, optional
            List of parameter names to pass to surfaces (Default = None)

        Returns
        -------
        OrderedPVArray
            Initialized, and fitted Ordered PV Array
        """

        # Create pv array
        pvarray = cls.init_from_dict(pvarray_params,
                                     param_names=param_names)

        # Fit pv array to scalar values
        solar_zenith = np.array([pvarray_params['solar_zenith']])
        solar_azimuth = np.array([pvarray_params['solar_azimuth']])
        surface_tilt = np.array([pvarray_params['surface_tilt']])
        surface_azimuth = np.array([pvarray_params['surface_azimuth']])
        pvarray.fit(solar_zenith, solar_azimuth,
                    surface_tilt, surface_azimuth)

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
        # Calculate rotation angles
        rotation_vec = _get_rotation_from_tilt_azimuth(
            surface_azimuth, self.axis_azimuth, surface_tilt)
        # Save rotation vector
        self.rotation_vec = rotation_vec

        # Calculate the solar 2D vectors for all timestamps
        self.solar_2d_vectors = _get_solar_2d_vectors(
            solar_zenith, solar_azimuth, self.axis_azimuth)
        # Calculate the angle made by 2D sun vector and x-axis
        alpha_vec = np.arctan2(self.solar_2d_vectors[1],
                               self.solar_2d_vectors[0])

        # Calculate the coordinates of all PV rows for all timestamps
        self._calculate_pvrow_elements_coords(alpha_vec, rotation_vec)

        # Calculate ground elements coordinates for all timestamps
        self.ts_ground = TsGround.from_ts_pvrows_and_angles(
            self.ts_pvrows, alpha_vec, rotation_vec, y_ground=self.y_ground,
            flag_overlap=self.has_direct_shading,
            param_names=self.param_names)

        # Save surface rotation angles
        self.rotation_vec = rotation_vec

        # Index all timeseries surfaces
        self._index_all_ts_surfaces()

    def _calculate_pvrow_elements_coords(self, alpha_vec, rotation_vec):
        """Calculate PV row coordinate elements in a vectorized way, such as
        PV row boundary coordinates and shaded lengths.

        Parameters
        ----------
        alpha_vec : array-like or float
            Angle made by 2d solar vector and x-axis [rad]
        rotation_vec : array-like or float
            Rotation angle of the PV rows [deg]
        """
        # Initialize timeseries pv rows
        self.ts_pvrows = []

        # Calculate interrow direct shading lengths
        self._calculate_interrow_shading(alpha_vec, rotation_vec)

        # Calculate coordinates of segments of each pv row side
        xy_centers = [(X_ORIGIN_PVROWS + idx * self.distance,
                       self.height + self.y_ground)
                      for idx in range(self.n_pvrows)]
        tilted_to_left = rotation_vec > 0.
        for idx_pvrow, xy_center in enumerate(xy_centers):
            # A special treatment needs to be applied to shaded lengths for
            # the PV rows at the edge of the PV array
            if idx_pvrow == 0:
                # the leftmost row doesn't have left neighbors
                shaded_length_front = np.where(tilted_to_left, 0.,
                                               self.shaded_length_front)
                shaded_length_back = np.where(tilted_to_left,
                                              self.shaded_length_back, 0.)
            elif idx_pvrow == (self.n_pvrows - 1):
                # the rightmost row does have right neighbors
                shaded_length_front = np.where(tilted_to_left,
                                               self.shaded_length_front, 0.)
                shaded_length_back = np.where(tilted_to_left, 0.,
                                              self.shaded_length_back)
            else:
                # use calculated shaded lengths
                shaded_length_front = self.shaded_length_front
                shaded_length_back = self.shaded_length_back
            # Create timeseries PV rows and add it to the list
            self.ts_pvrows.append(TsPVRow.from_raw_inputs(
                xy_center, self.width, rotation_vec,
                self.cut.get(idx_pvrow, {}), shaded_length_front,
                shaded_length_back, index=idx_pvrow,
                param_names=self.param_names))

    def _calculate_interrow_shading(self, alpha_vec, rotation_vec):
        """Calculate the shaded length on front and back side of PV rows when
        direct shading happens, and in a vectorized way.

        Parameters
        ----------
        alpha_vec : array-like or float
            Angle made by 2d solar vector and x-axis [rad]
        rotation_vec : array-like or float
            Rotation angle of the PV rows [deg]
        """

        if self.n_pvrows > 1:
            # Calculate intermediate values for direct shading
            alpha_vec_deg = np.rad2deg(alpha_vec)
            theta_t = 90. - rotation_vec
            theta_t_rad = np.deg2rad(theta_t)
            beta = theta_t + alpha_vec_deg
            beta_rad = np.deg2rad(beta)
            delta = self.distance * (
                np.sin(theta_t_rad) - np.cos(theta_t_rad) * np.tan(beta_rad))
            # Calculate temporary shaded lengths
            tmp_shaded_length_front = np.maximum(0, self.width - delta)
            tmp_shaded_length_back = np.maximum(0, self.width + delta)
            # The shaded length can't be longer than PV row (meaning sun can't
            # be under the horizon or something...)
            self.shaded_length_front = np.where(
                tmp_shaded_length_front > self.width, 0,
                tmp_shaded_length_front)
            self.shaded_length_back = np.where(
                tmp_shaded_length_back > self.width, 0,
                tmp_shaded_length_back)
        else:
            # Since there's 1 row, there can't be any direct shading
            self.shaded_length_front = np.zeros(self.n_states)
            self.shaded_length_back = np.zeros(self.n_states)

        # Flag times when there's direct shading
        self.has_direct_shading = (
            (self.shaded_length_front > DISTANCE_TOLERANCE)
            | (self.shaded_length_back > DISTANCE_TOLERANCE))
