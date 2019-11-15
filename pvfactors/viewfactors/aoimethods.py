"""Module containing AOI loss calculation methods"""

from pvfactors.config import DISTANCE_TOLERANCE
from pvfactors.geometry.timeseries import TsPointCoords
import pvlib
from pvlib.tools import cosd
import numpy as np


class AOIMethods:
    """Class containing methods related to calculating AOI losses for
    :py:class:`~pvfactors.geometry.pvarray.OrderedPVArray` objects."""

    def __init__(self, faoi_fn, n_integral_sections=300):
        """Instantiate class with faoi function

        Parameters
        ----------
        faoi_fn : function
            Function which takes a list (or numpy array) of incidence angles
            measured from the surface horizontal
            (with values from 0 to 180 deg) and returns the fAOI values
        n_integral_sections : int, optional
            Number of integral divisions of the 0 to 180 deg interval
            to use for the fAOI loss integral (default = 300)
        """
        self.faoi_fn = faoi_fn
        self.n_integral_sections = n_integral_sections
        # The following will be updated at fitting time
        self.interval = None
        self.aoi_angles_low = None
        self.aoi_angles_high = None
        self.integrand_values = None

    def fit(self, n_timestamps):
        """Fit the AOI methods to timeseries inputs: create all the necessary
        integration attributes.

        Parameters
        ----------
        n_timestamps : int
            Number of simulation timestamps
        """
        # Will use x values at the middle of the integral sections
        aoi_angles = np.linspace(0., 180., num=self.n_integral_sections + 1)
        # Assumes that at least 2 aoi angle values, otherwise what's the point
        self.interval = aoi_angles[1] - aoi_angles[0]
        # Get integral intervals' low, high, and middle points
        aoi_angles_low = aoi_angles[:-1]
        aoi_angles_high = aoi_angles_low + self.interval
        aoi_angles_middle = aoi_angles_low + self.interval / 2.
        # Calculate faoi values using middle points of integral intervals
        faoi_values = self.faoi_fn(aoi_angles_middle)
        # Calculate small view factor values for each section
        vf_values = self._vf(aoi_angles_low, aoi_angles_high)
        # Multiply to get integrand
        integrand_values = faoi_values * vf_values
        # Replicate these values for all timestamps such that shapes
        # becomes: [n_timestamps, n_integral_sections]map
        self.aoi_angles_low = np.tile(aoi_angles_low, (n_timestamps, 1))
        self.aoi_angles_high = np.tile(aoi_angles_high, (n_timestamps, 1))
        self.integrand_values = np.tile(integrand_values, (n_timestamps, 1))

    def vf_aoi_pvrow_to_gnd(self, ts_pvrows, ts_ground, tilted_to_left,
                            vf_aoi_matrix):
        """Calculate the view factors between timeseries PV row and ground
        surfaces while accounting for non-diffuse AOI losses,
        and assign it to the passed view factor aoi matrix using
        the surface indices.

        Notes
        -----
        This assumes that the PV row surfaces are infinitesimal (very small)

        Parameters
        ----------
        ts_pvrows : list of :py:class:`~pvfactors.geometry.timeseries.TsPVRow`
            List of timeseries PV rows in the PV array
        ts_ground : :py:class:`~pvfactors.geometry.timeseries.TsGround`
            Timeseries ground of the PV array
        tilted_to_left : list of bool
            Flags indicating when the PV rows are strictly tilted to the left
        vf_aoi_matrix : np.ndarray
            View factor aoi matrix to update during calculation. Should have 3
            dimensions as follows: [n_surfaces, n_surfaces, n_timesteps]
        """
        n_pvrows = len(ts_pvrows)
        for idx_pvrow, ts_pvrow in enumerate(ts_pvrows):
            # Separate gnd surfaces depending on side
            left_gnd_surfaces = ts_ground.ts_surfaces_side_of_cut_point(
                'left', idx_pvrow)
            right_gnd_surfaces = ts_ground.ts_surfaces_side_of_cut_point(
                'right', idx_pvrow)
            # Front side
            front = ts_pvrow.front
            for pvrow_surf in front.all_ts_surfaces:
                ts_length = pvrow_surf.length
                i = pvrow_surf.index
                for gnd_surf in left_gnd_surfaces:
                    j = gnd_surf.index
                    vf_pvrow_to_gnd = (
                        self.vf_aoi_pvrow_surf_to_gnd_surf_obstruction(
                            pvrow_surf, idx_pvrow, n_pvrows,
                            tilted_to_left, ts_pvrows, gnd_surf, ts_length,
                            is_back=False, is_left=True))
                    vf_aoi_matrix[i, j, :] = vf_pvrow_to_gnd
                for gnd_surf in right_gnd_surfaces:
                    j = gnd_surf.index
                    vf_pvrow_to_gnd = (
                        self.vf_aoi_pvrow_surf_to_gnd_surf_obstruction(
                            pvrow_surf, idx_pvrow, n_pvrows,
                            tilted_to_left, ts_pvrows, gnd_surf, ts_length,
                            is_back=False, is_left=False))
                    vf_aoi_matrix[i, j, :] = vf_pvrow_to_gnd
            # Back side
            back = ts_pvrow.back
            for pvrow_surf in back.all_ts_surfaces:
                ts_length = pvrow_surf.length
                i = pvrow_surf.index
                for gnd_surf in left_gnd_surfaces:
                    j = gnd_surf.index
                    vf_pvrow_to_gnd = (
                        self.vf_aoi_pvrow_surf_to_gnd_surf_obstruction(
                            pvrow_surf, idx_pvrow, n_pvrows,
                            tilted_to_left, ts_pvrows, gnd_surf, ts_length,
                            is_back=True, is_left=True))
                    vf_aoi_matrix[i, j, :] = vf_pvrow_to_gnd
                for gnd_surf in right_gnd_surfaces:
                    j = gnd_surf.index
                    vf_pvrow_to_gnd = (
                        self.vf_aoi_pvrow_surf_to_gnd_surf_obstruction(
                            pvrow_surf, idx_pvrow, n_pvrows,
                            tilted_to_left, ts_pvrows, gnd_surf, ts_length,
                            is_back=True, is_left=False))
                    vf_aoi_matrix[i, j, :] = vf_pvrow_to_gnd

    def _vf_aoi_pvrow_surf_to_gnd_surf_obstruction(
            self, pvrow_surf, pvrow_idx, n_pvrows, tilted_to_left, ts_pvrows,
            gnd_surf, ts_length, is_back=True, is_left=True):
        """Calculate view factors from timeseries PV row surface to a
        timeseries ground surface, accounting for AOI losses.
        This will return the calculated view
        factors from the PV row surface to the ground surface.

        Notes
        -----
        This assumes that the PV row surfaces are infinitesimal (very small)

        Parameters
        ----------
        pvrow_surf : :py:class:`~pvfactors.geometry.timeseries.TsSurface`
            Timeseries PV row surface to use for calculation
        pvrow_idx : int
            Index of the timeseries PV row on the which the pvrow_surf is
        n_pvrows : int
            Number of timeseries PV rows in the PV array, and therefore number
            of shadows they cast on the ground
        tilted_to_left : list of bool
            Flags indicating when the PV rows are strictly tilted to the left
        ts_pvrows : list of :py:class:`~pvfactors.geometry.timeseries.TsPVRow`
            List of timeseries PV rows in the PV array
        gnd_surf : :py:class:`~pvfactors.geometry.timeseries.TsSurface`
            Timeseries ground surface to use for calculation
        pvrow_surf_length : np.ndarray
            Length (width) of the timeseries PV row surface [m]
        is_back : bool
            Flag specifying whether pv row surface is on  back or front surface
            (Default = True)
        is_left : bool
            Flag specifying whether gnd surface is left of pv row cut point or
            not (Default = True)

        Returns
        -------
        vf_aoi_pvrow_to_gnd_surf : np.ndarray
            View factors aoi from timeseries PV row surface to timeseries
            ground surface, dimension is [n_timesteps]
        """
        centroid = pvrow_surf.centroid
        u_vector = pvrow_surf.u_vector
        no_obstruction = (is_left & (pvrow_idx == 0)) \
            or ((not is_left) & (pvrow_idx == n_pvrows - 1))
        if no_obstruction:
            # There is no obstruction to the ground surface
            aoi_angles_1 = self._calculate_aoi_angles(u_vector, centroid,
                                                      gnd_surf.b1)
            aoi_angles_2 = self._calculate_aoi_angles(u_vector, centroid,
                                                      gnd_surf.b2)
        else:
            # Get lowest point of obstructing point
            idx_obstructing_pvrow = pvrow_idx - 1 if is_left else pvrow_idx + 1
            pt_obstr = ts_pvrows[idx_obstructing_pvrow
                                 ].full_pvrow_coords.lowest_point
            # Adjust angle seen when there is obstruction
            aoi_angles_1 = self._calculate_aoi_angles_w_obstruction(
                u_vector, centroid, gnd_surf.b1, pt_obstr, is_left)
            aoi_angles_2 = self._calculate_aoi_angles_w_obstruction(
                u_vector, centroid, gnd_surf.b2, pt_obstr, is_left)

        low_aoi_angles = np.where(aoi_angles_1 < aoi_angles_2, aoi_angles_1,
                                  aoi_angles_2)
        high_aoi_angles = np.where(aoi_angles_1 < aoi_angles_2, aoi_angles_2,
                                   aoi_angles_1)
        vf_aoi_raw = self._calculate_vf_aoi_wedge_level(low_aoi_angles,
                                                        high_aoi_angles)
        # Should be zero where either of the surfaces have zero length
        vf_aoi_raw = np.where((ts_length < DISTANCE_TOLERANCE)
                              | (gnd_surf.length < DISTANCE_TOLERANCE), 0.,
                              vf_aoi_raw)

        # Final result depends on whether front or back surface
        if is_left:
            vf_aoi = (np.where(tilted_to_left, 0., vf_aoi_raw) if is_back
                      else np.where(tilted_to_left, vf_aoi_raw, 0.))
        else:
            vf_aoi = (np.where(tilted_to_left, vf_aoi_raw, 0.) if is_back
                      else np.where(tilted_to_left, 0., vf_aoi_raw))

        return vf_aoi

    def _calculate_aoi_angles_w_obstruction(
            self, u_vector, centroid, point_gnd, point_obstr,
            gnd_surf_is_left):
        """Calculate AOI angles for a PV row surface of the
        :py:class:`~pvfactors.geometry.pvarray.OrderedPVArray` that sees
        a ground surface, while being potentially obstructed by another
        PV row

        Parameters
        ----------
        u_vector : np.ndarray
            Direction vector of the surface for which to calculate AOI angles
        centroid : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Centroid point of PV row surface for which to calculate AOI angles
        point : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Point of ground surface that will determine AOI angle
        point_obstr: :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Potentially obstructing point for the view aoi angle calculation
        gnd_surf_is_left : bool
            Flag specifying whether ground surface is left of PV row's cut
            point or not

        Returns
        -------
        np.ndarray
            AOI angles formed by remote point and centroid on surface,
            measured against surface direction vector, accounting for
            potential obstruction [degrees]
        """
        if point_obstr is None:
            # There is no obstruction
            point = point_gnd
        else:
            # Determine if there is obstruction by using the angles made by
            # specific strings with the x-axis
            alpha_pv = self._angle_with_x_axis(point_gnd, centroid)
            alpha_ob = self._angle_with_x_axis(point_gnd, point_obstr)
            if gnd_surf_is_left:
                is_obstructing = alpha_pv > alpha_ob
            else:
                is_obstructing = alpha_pv < alpha_ob
            x = np.where(is_obstructing, point_obstr.x, point_gnd.x)
            y = np.where(is_obstructing, point_obstr.y, point_gnd.y)
            point = TsPointCoords(x, y)

        aoi_angles = self._calculate_aoi_angles(u_vector, centroid, point)
        return aoi_angles

    @staticmethod
    def _calculate_aoi_angles(u_vector, centroid, point):
        """Calculate AOI angles from direction vector of surface,
        centroid point of that surface, and point from another surface

        Parameters
        ----------
        u_vector : np.ndarray
            Direction vector of the surface for which to calculate AOI angles
        centroid : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Centroid point of surface for which to calculate AOI angles
        point : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Point of remote surface that will determine AOI angle

        Returns
        -------
        np.ndarray
            AOI angles formed by remote point and centroid on surface,
            measured against surface direction vector [degrees]
        """
        v_vector = np.array([point.x - centroid.x, point.y - centroid.y])
        dot_product = u_vector[0, :] * v_vector[0, :] \
            + u_vector[1, :] * v_vector[1, :]
        u_norm = np.linalg.norm(u_vector, axis=0)
        v_norm = np.linalg.norm(v_vector, axis=0)
        cos_theta = dot_product / (u_norm * v_norm)
        aoi_angles = np.rad2deg(np.arccos(cos_theta))
        return aoi_angles

    def _calculate_vf_aoi_wedge_level(self, low_angles, high_angles):
        """Calculate faoi modified view factors for a wedge defined by
        low and high angles.

        Parameters
        ----------
        low_angles : np.ndarray
            Low AOI angles (between 0 and 180 deg), length = n_timestamps
        high_angles : np.ndarray
            High AOI angles (between 0 and 180 deg), length = n_timestamps.
            Should be bigger than ``low_angles``

        Returns
        -------
        np.ndarray
            faoi modified view factors for wedge
            shape = (n_timestamps, )
        """
        # Calculate integrand: all d_vf_aoi values
        faoi_integrand = self._calculate_vfaoi_integrand(low_angles,
                                                         high_angles)
        # Total vf_aoi will be sum of all smaller d_vf_aoi values
        total_vf_aoi = faoi_integrand.sum(axis=1)

        return total_vf_aoi

    def _calculate_vfaoi_integrand(self, low_angles, high_angles):
        """
        Calculate the timeseries view factors with aoi loss integrand
        given the low and high angles that define the surface.

        Parameters
        ----------
        low_angles : np.ndarray
            Low AOI angles (between 0 and 180 deg), length = n_timestamps
        high_angles : np.ndarray
            High AOI angles (between 0 and 180 deg), length = n_timestamps.
            Should be bigger than ``low_angles``

        Returns
        -------
        np.ndarray
            vf_aoi integrand values for all timestamps
            shape = (n_timestamps, n_integral_sections)
        """

        # Turn into dimension: [n_timestamps, n_integral_sections]
        low_angles_mat = np.tile(low_angles, (self.n_integral_sections, 1)).T
        high_angles_mat = np.tile(high_angles, (self.n_integral_sections, 1)).T

        # Filter out integrand values outside of range
        count_integral_section = ((low_angles_mat <= self.aoi_angles_high) &
                                  (high_angles_mat > self.aoi_angles_low))
        faoi_integrand = np.where(count_integral_section,
                                  self.integrand_values, 0.)

        return faoi_integrand

    @staticmethod
    def _vf(aoi_1, aoi_2):
        """Calculate view factor from infinitesimal surface to infinite band.

        See illustration: http://www.thermalradiation.net/sectionb/B-71.html
        Here we're using angles measured from the horizontal

        Parameters
        ----------
        aoi_1 : np.ndarray
            Lower angles defining the infinite band
        aoi_2 : np.ndarray
            Higher angles defining the infinite band

        Returns
        -------
        np.ndarray
            View factors from infinitesimal surface to infinite strip

        """
        return 0.5 * np.abs(cosd(aoi_1) - cosd(aoi_2))

    @staticmethod
    def _angle_with_x_axis(pt_1, pt_2):
        """Angle with x-axis of vector going from pt_1 to pt_2

        Parameters
        ----------
        pt_1 : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Timeseries point coordinates of point 1
        pt_2 : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Timeseries point coordinates of point 2

        Returns
        -------
        np.ndarray
            Angle between vector pt_1->pt_2 and x-axis
        """
        return np.arctan2(pt_2.y - pt_1.y, pt_2.x - pt_1.x)


def faoi_fn_from_pvlib_sandia(pvmodule_name):
    """Generate a faoi function from a pvlib sandia PV module name

    Parameters
    ----------
    pvmodule_name : str
        Name of PV module in pvlib Sandia module database

    Returns
    -------
    faoi_function
        Function that returns positive loss values for numeric inputs
        between 0 and 180 degrees.
    """
    # Get Sandia module database from pvlib
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    # Grab pv module sandia coeffs from database
    pvmodule = sandia_modules[pvmodule_name]

    def fn(angles):
        """fAOI loss funtion: calculate how much light is absorbed at given
        incidence angles

        Parameters
        ----------
        angles : np.ndarray or list
            Angles measured from surface horizontal, from 0 de 180 deg

        Returns
        -------
        np.ndarray
            fAOI values
        """
        angles = np.array(angles) if isinstance(angles, list) else angles
        # Transform the inputs for the SAPM function
        angles = np.where(angles >= 90, angles - 90, 90. - angles)
        # Use pvlib sapm aoi loss method
        return pvlib.pvsystem.sapm_aoi_loss(angles, pvmodule, upper=1.)

    return fn
