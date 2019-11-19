"""Module containing AOI loss calculation methods"""

from pvfactors.config import DISTANCE_TOLERANCE
from pvfactors.geometry.timeseries import (
    TsPointCoords, TsSurface, TsLineCoords)
from pvfactors import PVFactorsError
import pvlib
from pvlib.tools import cosd
import numpy as np


class AOIMethods:
    """Class containing methods related to calculating AOI losses for
    :py:class:`~pvfactors.geometry.pvarray.OrderedPVArray` objects."""

    def __init__(self, faoi_fn_front, faoi_fn_back, n_integral_sections=300):
        """Instantiate class with faoi function and number of sections to use
        to calculate integrals of view factors with faoi losses

        Parameters
        ----------
        faoi_fn_front : function
            Function which takes a list (or numpy array) of incidence angles
            measured from the surface horizontal
            (with values from 0 to 180 deg) and returns the fAOI values for
            the front side of PV rows
        faoi_fn_back : function
            Function which takes a list (or numpy array) of incidence angles
            measured from the surface horizontal
            (with values from 0 to 180 deg) and returns the fAOI values for
            the back side of PV rows
        n_integral_sections : int, optional
            Number of integral divisions of the 0 to 180 deg interval
            to use for the fAOI loss integral (default = 300)
        """
        # Check that faoi fn where passed
        faoi_fns_ok = callable(faoi_fn_front) and callable(faoi_fn_back)
        if not faoi_fns_ok:
            raise PVFactorsError("The faoi_fn passed to the AOI methods are "
                                 "not callable. Please check the fAOI "
                                 "functions again")
        self.faoi_fn_front = faoi_fn_front
        self.faoi_fn_back = faoi_fn_back
        self.n_integral_sections = n_integral_sections
        # The following will be updated at fitting time
        self.interval = None
        self.aoi_angles_low = None
        self.aoi_angles_high = None
        self.integrand_front = None
        self.integrand_back = None

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
        faoi_front = self.faoi_fn_front(aoi_angles_middle)
        faoi_back = self.faoi_fn_back(aoi_angles_middle)
        # Calculate small view factor values for each section
        vf_values = self._vf(aoi_angles_low, aoi_angles_high)
        # Multiply to get integrand
        integrand_front = faoi_front * vf_values
        integrand_back = faoi_back * vf_values
        # Replicate these values for all timestamps such that shapes
        # becomes: [n_timestamps, n_integral_sections]map
        self.aoi_angles_low = np.tile(aoi_angles_low, (n_timestamps, 1))
        self.aoi_angles_high = np.tile(aoi_angles_high, (n_timestamps, 1))
        self.integrand_front = np.tile(integrand_front, (n_timestamps, 1))
        self.integrand_back = np.tile(integrand_back, (n_timestamps, 1))

    def vf_aoi_pvrow_to_sky(self, ts_pvrows, ts_ground, tilted_to_left,
                            vf_matrix):
        """Calculate the view factors between timeseries PV row surface and sky
        while accounting for AOI losses,
        and assign values to the passed view factor matrix using
        the surface indices.

        Parameters
        ----------
        ts_pvrows : list of :py:class:`~pvfactors.geometry.timeseries.TsPVRow`
            List of timeseries PV rows in the PV array
        ts_ground : :py:class:`~pvfactors.geometry.timeseries.TsGround`
            Timeseries ground of the PV array
        tilted_to_left : list of bool
            Flags indicating when the PV rows are strictly tilted to the left
        vf_matrix : np.ndarray
            View factor matrix to update during calculation. Should have 3
            dimensions as follows: [n_surfaces, n_surfaces, n_timesteps]
        """
        sky_index = vf_matrix.shape[0] - 1
        # --- Build list of dummy sky surfaces
        # create sky left open area
        pt_1 = TsPointCoords(ts_ground.x_min * np.ones_like(tilted_to_left),
                             ts_ground.y_ground * np.ones_like(tilted_to_left))
        pt_2 = ts_pvrows[0].highest_point
        sky_left = TsSurface(TsLineCoords(pt_1, pt_2))
        # create sky right open area
        pt_1 = TsPointCoords(ts_ground.x_max * np.ones_like(tilted_to_left),
                             ts_ground.y_ground * np.ones_like(tilted_to_left))
        pt_2 = ts_pvrows[-1].highest_point
        sky_right = TsSurface(TsLineCoords(pt_2, pt_1))
        # Add sky surfaces in-between PV rows
        dummy_sky_surfaces = [sky_left]
        for idx_pvrow, ts_pvrow in enumerate(ts_pvrows[:-1]):
            right_ts_pvrow = ts_pvrows[idx_pvrow + 1]
            pt_1 = ts_pvrow.highest_point
            pt_2 = right_ts_pvrow.highest_point
            sky_surface = TsSurface(TsLineCoords(pt_1, pt_2))
            dummy_sky_surfaces.append(sky_surface)
        # Add sky right open area
        dummy_sky_surfaces.append(sky_right)

        # Now calculate vf_aoi for all PV row surfaces to sky
        for idx_pvrow, ts_pvrow in enumerate(ts_pvrows):
            # Get dummy sky surfaces
            sky_left = dummy_sky_surfaces[idx_pvrow]
            sky_right = dummy_sky_surfaces[idx_pvrow + 1]
            # Calculate vf_aoi for surfaces in PV row
            # front side
            front = ts_pvrow.front
            for front_surf in front.all_ts_surfaces:
                vf_aoi_left = self._vf_aoi_surface_to_surface(
                    front_surf, sky_left, is_back=False)
                vf_aoi_right = self._vf_aoi_surface_to_surface(
                    front_surf, sky_right, is_back=False)
                vf_aoi = np.where(tilted_to_left, vf_aoi_left, vf_aoi_right)
                vf_matrix[front_surf.index, sky_index, :] = vf_aoi
            # back side
            back = ts_pvrow.back
            for back_surf in back.all_ts_surfaces:
                vf_aoi_left = self._vf_aoi_surface_to_surface(
                    back_surf, sky_left, is_back=True)
                vf_aoi_right = self._vf_aoi_surface_to_surface(
                    back_surf, sky_right, is_back=True)
                vf_aoi = np.where(tilted_to_left, vf_aoi_right, vf_aoi_left)
                vf_matrix[back_surf.index, sky_index, :] = vf_aoi

    def vf_aoi_pvrow_to_pvrow(self, ts_pvrows, tilted_to_left, vf_matrix):
        """Calculate the view factors between timeseries PV row surfaces
        while accounting for AOI losses,
        and assign values to the passed view factor matrix using
        the surface indices.

        Parameters
        ----------
        ts_pvrows : list of :py:class:`~pvfactors.geometry.timeseries.TsPVRow`
            List of timeseries PV rows in the PV array
        tilted_to_left : list of bool
            Flags indicating when the PV rows are strictly tilted to the left
        vf_matrix : np.ndarray
            View factor matrix to update during calculation. Should have 3
            dimensions as follows: [n_surfaces, n_surfaces, n_timesteps]
        """
        for idx_pvrow, ts_pvrow in enumerate(ts_pvrows[:-1]):
            # Get the next pv row
            right_ts_pvrow = ts_pvrows[idx_pvrow + 1]
            # front side
            front = ts_pvrow.front
            for surf_i in front.all_ts_surfaces:
                i = surf_i.index
                for surf_j in right_ts_pvrow.back.all_ts_surfaces:
                    j = surf_j.index
                    # vf aoi from i to j
                    vf_i_to_j = self._vf_aoi_surface_to_surface(
                        surf_i, surf_j, is_back=False)
                    vf_i_to_j = np.where(tilted_to_left, 0., vf_i_to_j)
                    # vf aoi from j to i
                    vf_j_to_i = self._vf_aoi_surface_to_surface(
                        surf_j, surf_i, is_back=True)
                    vf_j_to_i = np.where(tilted_to_left, 0., vf_j_to_i)
                    # save results
                    vf_matrix[i, j, :] = vf_i_to_j
                    vf_matrix[j, i, :] = vf_j_to_i
            # back side
            back = ts_pvrow.back
            for surf_i in back.all_ts_surfaces:
                i = surf_i.index
                for surf_j in right_ts_pvrow.front.all_ts_surfaces:
                    j = surf_j.index
                    # vf aoi from i to j
                    vf_i_to_j = self._vf_aoi_surface_to_surface(
                        surf_i, surf_j, is_back=True)
                    vf_i_to_j = np.where(tilted_to_left, vf_i_to_j, 0.)
                    # vf aoi from j to i
                    vf_j_to_i = self._vf_aoi_surface_to_surface(
                        surf_j, surf_i, is_back=False)
                    vf_j_to_i = np.where(tilted_to_left, vf_j_to_i, 0.)
                    # save results
                    vf_matrix[i, j, :] = vf_i_to_j
                    vf_matrix[j, i, :] = vf_j_to_i

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
                        self._vf_aoi_pvrow_surf_to_gnd_surf_obstruction(
                            pvrow_surf, idx_pvrow, n_pvrows,
                            tilted_to_left, ts_pvrows, gnd_surf, ts_length,
                            is_back=False, is_left=True))
                    vf_aoi_matrix[i, j, :] = vf_pvrow_to_gnd
                for gnd_surf in right_gnd_surfaces:
                    j = gnd_surf.index
                    vf_pvrow_to_gnd = (
                        self._vf_aoi_pvrow_surf_to_gnd_surf_obstruction(
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
                        self._vf_aoi_pvrow_surf_to_gnd_surf_obstruction(
                            pvrow_surf, idx_pvrow, n_pvrows,
                            tilted_to_left, ts_pvrows, gnd_surf, ts_length,
                            is_back=True, is_left=True))
                    vf_aoi_matrix[i, j, :] = vf_pvrow_to_gnd
                for gnd_surf in right_gnd_surfaces:
                    j = gnd_surf.index
                    vf_pvrow_to_gnd = (
                        self._vf_aoi_pvrow_surf_to_gnd_surf_obstruction(
                            pvrow_surf, idx_pvrow, n_pvrows,
                            tilted_to_left, ts_pvrows, gnd_surf, ts_length,
                            is_back=True, is_left=False))
                    vf_aoi_matrix[i, j, :] = vf_pvrow_to_gnd

    def _vf_aoi_surface_to_surface(self, surf_1, surf_2, is_back=True):
        """Calculate view factor, while accounting from AOI losses, from
        surface 1 to surface 2.

        Notes
        -----
        This assumes that surf_1 is infinitesimal (very small)

        Parameters
        ----------
        surf_1 : :py:class:`~pvfactors.geometry.timeseries.TsSurface`
            Infinitesimal surface from which to calculate view factor with
            AOI losses
        surf_2 : :py:class:`~pvfactors.geometry.timeseries.TsSurface`
            Surface to which the view factor with AOI losses should be
            calculated
        is_back : bool
            Flag specifying whether pv row surface is on back or front side
            of PV row (Default = True)

        Returns
        -------
        vf_aoi : np.ndarray
            View factors with aoi losses from surface 1 to surface 2,
            dimension is [n_timesteps]
        """
        # skip calculation if either surface is empty (always zero length)
        skip = surf_1.is_empty or surf_2.is_empty
        if skip:
            vf_aoi = np.zeros_like(surf_2.length)
        else:
            # Get surface 1 params
            u_vector = surf_1.u_vector
            centroid = surf_1.centroid
            # Calculate AOI angles
            aoi_angles_1 = self._calculate_aoi_angles(u_vector, centroid,
                                                      surf_2.b1)
            aoi_angles_2 = self._calculate_aoi_angles(u_vector, centroid,
                                                      surf_2.b2)
            low_aoi_angles = np.where(aoi_angles_1 < aoi_angles_2, aoi_angles_1,
                                      aoi_angles_2)
            high_aoi_angles = np.where(aoi_angles_1 < aoi_angles_2, aoi_angles_2,
                                       aoi_angles_1)
            # Calculate vf_aoi
            vf_aoi_raw = self._calculate_vf_aoi_wedge_level(
                low_aoi_angles, high_aoi_angles, is_back=is_back)
            # Should be zero where either of the surfaces have zero length
            vf_aoi = np.where((surf_1.length < DISTANCE_TOLERANCE)
                              | (surf_2.length < DISTANCE_TOLERANCE), 0.,
                              vf_aoi_raw)
        return vf_aoi

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
            Flag specifying whether pv row surface is on back or front side
            of PV row (Default = True)
        is_left : bool
            Flag specifying whether gnd surface is left of pv row cut point or
            not (Default = True)

        Returns
        -------
        vf_aoi_pvrow_to_gnd_surf : np.ndarray
            View factors aoi from timeseries PV row surface to timeseries
            ground surface, dimension is [n_timesteps]
        """
        # skip calculation if either surface is empty (always zero length)
        skip = pvrow_surf.is_empty or gnd_surf.is_empty
        if skip:
            vf_aoi = np.zeros_like(gnd_surf.length)
        else:
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
                idx_obstructing_pvrow = (pvrow_idx - 1 if is_left
                                         else pvrow_idx + 1)
                pt_obstr = ts_pvrows[idx_obstructing_pvrow
                                     ].full_pvrow_coords.lowest_point
                # Adjust angle seen when there is obstruction
                aoi_angles_1 = self._calculate_aoi_angles_w_obstruction(
                    u_vector, centroid, gnd_surf.b1, pt_obstr, is_left)
                aoi_angles_2 = self._calculate_aoi_angles_w_obstruction(
                    u_vector, centroid, gnd_surf.b2, pt_obstr, is_left)

            low_aoi_angles = np.where(aoi_angles_1 < aoi_angles_2,
                                      aoi_angles_1, aoi_angles_2)
            high_aoi_angles = np.where(aoi_angles_1 < aoi_angles_2,
                                       aoi_angles_2, aoi_angles_1)
            vf_aoi_raw = self._calculate_vf_aoi_wedge_level(
                low_aoi_angles, high_aoi_angles, is_back=is_back)
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

    def _calculate_vf_aoi_wedge_level(self, low_angles, high_angles,
                                      is_back=True):
        """Calculate faoi modified view factors for a wedge defined by
        low and high angles.

        Parameters
        ----------
        low_angles : np.ndarray
            Low AOI angles (between 0 and 180 deg), length = n_timestamps
        high_angles : np.ndarray
            High AOI angles (between 0 and 180 deg), length = n_timestamps.
            Should be bigger than ``low_angles``
        is_back : bool
            Flag specifying whether pv row surface is on back or front side
            of PV row (Default = True)

        Returns
        -------
        np.ndarray
            faoi modified view factors for wedge
            shape = (n_timestamps, )
        """
        # Calculate integrand: all d_vf_aoi values
        faoi_integrand = self._calculate_vfaoi_integrand(
            low_angles, high_angles, is_back=is_back)
        # Total vf_aoi will be sum of all smaller d_vf_aoi values
        total_vf_aoi = faoi_integrand.sum(axis=1)
        # Make sure vf is counted as zero if the wedge is super small
        total_vf_aoi = np.where(
            np.abs(high_angles - low_angles) < DISTANCE_TOLERANCE, 0.,
            total_vf_aoi)

        return total_vf_aoi

    def _calculate_vfaoi_integrand(self, low_angles, high_angles,
                                   is_back=True):
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
        is_back : bool
            Flag specifying whether pv row surface is on back or front side
            of PV row (Default = True)
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
        include_integral_section = ((low_angles_mat <= self.aoi_angles_high) &
                                    (high_angles_mat > self.aoi_angles_low))
        # The integrand values are different for front and back sides
        if is_back:
            faoi_integrand = np.where(include_integral_section,
                                      self.integrand_back, 0.)
        else:
            faoi_integrand = np.where(include_integral_section,
                                      self.integrand_front, 0.)

        return faoi_integrand

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
        # because of round off errors, cos_theta can be slightly > 1,
        # or slightly < -1, so clip it
        cos_theta = np.clip(cos_theta, -1., 1.)
        aoi_angles = np.rad2deg(np.arccos(cos_theta))
        return aoi_angles

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

    def rho_from_faoi_fn(self, is_back):
        """Calculate global average reflectivity from faoi function
        for either side of the PV row (requires calculating view factors)

        Parameters
        ----------
        is_back : bool
            Flag specifying whether to use front or back faoi function
        Returns
        -------
        rho_average : float
            Global average reflectivity value of surface
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
        if is_back:
            faoi_values = self.faoi_fn_back(aoi_angles_middle)
        else:
            faoi_values = self.faoi_fn_front(aoi_angles_middle)
        # Calculate small view factor values for each section
        vf_values = self._vf(aoi_angles_low, aoi_angles_high)
        # Multiply to get integrand
        integrand_values = faoi_values * vf_values
        return (1. - integrand_values.sum())


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
