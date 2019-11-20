"""Module with view factor calculation tools"""

from pvfactors.config import MIN_X_GROUND, MAX_X_GROUND, DISTANCE_TOLERANCE
from pvfactors.geometry.timeseries import TsLineCoords, TsPointCoords
from pvlib.tools import cosd, sind
import numpy as np


class VFTsMethods(object):
    """This class contains all the methods used to calculate timeseries
    view factors for all the surfaces in
    :py:class:`~pvfactors.geometry.pvarray.OrderedPVArray`"""

    def vf_pvrow_gnd_surf(self, ts_pvrows, ts_ground, tilted_to_left,
                          vf_matrix):
        """Calculate the view factors between timeseries PV row and ground
        surfaces, and assign it to the passed view factor matrix using
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
                if pvrow_surf.is_empty:
                    # do no run calculation for this surface
                    continue
                ts_length = pvrow_surf.length
                i = pvrow_surf.index
                for gnd_surf in left_gnd_surfaces:
                    if gnd_surf.is_empty:
                        # do no run this calculation
                        continue
                    j = gnd_surf.index
                    vf_pvrow_to_gnd, vf_gnd_to_pvrow = (
                        self.vf_pvrow_surf_to_gnd_surf_obstruction_hottel(
                            pvrow_surf, idx_pvrow, n_pvrows,
                            tilted_to_left, ts_pvrows, gnd_surf, ts_length,
                            is_back=False, is_left=True))
                    vf_matrix[i, j, :] = vf_pvrow_to_gnd
                    vf_matrix[j, i, :] = vf_gnd_to_pvrow
                for gnd_surf in right_gnd_surfaces:
                    if gnd_surf.is_empty:
                        # do no run this calculation
                        continue
                    j = gnd_surf.index
                    vf_pvrow_to_gnd, vf_gnd_to_pvrow = (
                        self.vf_pvrow_surf_to_gnd_surf_obstruction_hottel(
                            pvrow_surf, idx_pvrow, n_pvrows,
                            tilted_to_left, ts_pvrows, gnd_surf, ts_length,
                            is_back=False, is_left=False))
                    vf_matrix[i, j, :] = vf_pvrow_to_gnd
                    vf_matrix[j, i, :] = vf_gnd_to_pvrow
            # Back side
            back = ts_pvrow.back
            for pvrow_surf in back.all_ts_surfaces:
                if pvrow_surf.is_empty:
                    # do no run calculation for this surface
                    continue
                ts_length = pvrow_surf.length
                i = pvrow_surf.index
                for gnd_surf in left_gnd_surfaces:
                    if gnd_surf.is_empty:
                        # do no run this calculation
                        continue
                    j = gnd_surf.index
                    vf_pvrow_to_gnd, vf_gnd_to_pvrow = (
                        self.vf_pvrow_surf_to_gnd_surf_obstruction_hottel(
                            pvrow_surf, idx_pvrow, n_pvrows,
                            tilted_to_left, ts_pvrows, gnd_surf, ts_length,
                            is_back=True, is_left=True))
                    vf_matrix[i, j, :] = vf_pvrow_to_gnd
                    vf_matrix[j, i, :] = vf_gnd_to_pvrow
                for gnd_surf in right_gnd_surfaces:
                    if gnd_surf.is_empty:
                        # do no run this calculation
                        continue
                    j = gnd_surf.index
                    vf_pvrow_to_gnd, vf_gnd_to_pvrow = (
                        self.vf_pvrow_surf_to_gnd_surf_obstruction_hottel(
                            pvrow_surf, idx_pvrow, n_pvrows,
                            tilted_to_left, ts_pvrows, gnd_surf, ts_length,
                            is_back=True, is_left=False))
                    vf_matrix[i, j, :] = vf_pvrow_to_gnd
                    vf_matrix[j, i, :] = vf_gnd_to_pvrow

    def vf_pvrow_surf_to_gnd_surf_obstruction_hottel(
            self, pvrow_surf, pvrow_idx, n_pvrows, tilted_to_left,
            ts_pvrows, gnd_surf, pvrow_surf_length, is_back=True,
            is_left=True):
        """Calculate view factors from timeseries PV row surface to a
        timeseries ground surface. This will return the calculated view
        factors from the PV row surface to the ground surface, AND from the
        ground surface to the PV row surface (using reciprocity).

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
        vf_pvrow_to_gnd_surf : np.ndarray
            View factors from timeseries PV row surface to timeseries ground
            surface, dimension is [n_timesteps]
        vf_gnd_to_pvrow_surf : np.ndarray
            View factors from timeseries ground surface to timeseries PV row
            surface, dimension is [n_timesteps]
        """

        pvrow_surf_lowest_pt = pvrow_surf.lowest_point
        pvrow_surf_highest_pt = pvrow_surf.highest_point
        no_obstruction = (is_left & (pvrow_idx == 0)) \
            or ((not is_left) & (pvrow_idx == n_pvrows - 1))
        if no_obstruction:
            # There is no obstruction to the gnd surface
            vf_pvrow_to_gnd_surf = self._vf_surface_to_surface(
                pvrow_surf.coords, gnd_surf, pvrow_surf_length)
        else:
            # Get lowest point of obstructing point
            idx_obstructing_pvrow = pvrow_idx - 1 if is_left else pvrow_idx + 1
            pt_obstr = ts_pvrows[idx_obstructing_pvrow
                                 ].full_pvrow_coords.lowest_point
            # Calculate vf from pv row to gnd surface
            vf_pvrow_to_gnd_surf = self._vf_hottel_gnd_surf(
                pvrow_surf_highest_pt, pvrow_surf_lowest_pt,
                gnd_surf.b1, gnd_surf.b2, pt_obstr, pvrow_surf_length,
                is_left)

        # Final result depends on whether front or back surface
        if is_left:
            vf_pvrow_to_gnd_surf = (
                np.where(tilted_to_left, 0., vf_pvrow_to_gnd_surf) if is_back
                else np.where(tilted_to_left, vf_pvrow_to_gnd_surf, 0.))
        else:
            vf_pvrow_to_gnd_surf = (
                np.where(tilted_to_left, vf_pvrow_to_gnd_surf, 0.) if is_back
                else np.where(tilted_to_left, 0., vf_pvrow_to_gnd_surf))

        # Use reciprocity to calculate ts vf from gnd surf to pv row surface
        gnd_surf_length = gnd_surf.length
        vf_gnd_to_pvrow_surf = np.where(
            gnd_surf_length > DISTANCE_TOLERANCE,
            vf_pvrow_to_gnd_surf * pvrow_surf_length / gnd_surf_length, 0.)

        return vf_pvrow_to_gnd_surf, vf_gnd_to_pvrow_surf

    def vf_pvrow_to_pvrow(self, ts_pvrows, tilted_to_left, vf_matrix):
        """Calculate the view factors between timeseries PV row surfaces,
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
                if surf_i.is_empty:
                    # do no run calculation for this surface
                    continue
                i = surf_i.index
                length_i = surf_i.length
                for surf_j in right_ts_pvrow.back.all_ts_surfaces:
                    if surf_j.is_empty:
                        # do no run calculation for this surface
                        continue
                    j = surf_j.index
                    length_j = surf_j.length
                    vf_i_to_j = self._vf_surface_to_surface(
                        surf_i.coords, surf_j.coords, length_i)
                    vf_i_to_j = np.where(tilted_to_left, 0., vf_i_to_j)
                    vf_j_to_i = np.where(
                        surf_j.length > DISTANCE_TOLERANCE,
                        vf_i_to_j * length_i / length_j, 0.)
                    vf_matrix[i, j, :] = vf_i_to_j
                    vf_matrix[j, i, :] = vf_j_to_i
            # back side
            back = ts_pvrow.back
            for surf_i in back.all_ts_surfaces:
                if surf_i.is_empty:
                    # do no run calculation for this surface
                    continue
                i = surf_i.index
                length_i = surf_i.length
                for surf_j in right_ts_pvrow.front.all_ts_surfaces:
                    if surf_j.is_empty:
                        # do no run calculation for this surface
                        continue
                    j = surf_j.index
                    length_j = surf_j.length
                    vf_i_to_j = self._vf_surface_to_surface(
                        surf_i.coords, surf_j.coords, length_i)
                    vf_i_to_j = np.where(tilted_to_left, vf_i_to_j, 0.)
                    vf_j_to_i = np.where(
                        surf_j.length > DISTANCE_TOLERANCE,
                        vf_i_to_j * length_i / length_j, 0.)
                    vf_matrix[i, j, :] = vf_i_to_j
                    vf_matrix[j, i, :] = vf_j_to_i

    def calculate_vf_to_pvrow(self, pvrow_element_coords, pvrow_idx, n_pvrows,
                              n_steps, ts_pvrows, pvrow_element_length,
                              tilted_to_left, pvrow_width, rotation_vec):
        """Calculate view factors from timeseries pvrow element to timeseries
        PV rows around it.

        Parameters
        ----------
        pvrow_element_coords :
        :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Timeseries line coordinates of pvrow_element
        pvrow_idx : int
            Index of the timeseries PV row on which the pvrow_element is
        n_pvrows : int
            Number of timeseries PV rows in the PV array
        n_steps : int
            Number of timesteps for which to calculate the pvfactors
        ts_pvrows : list of :py:class:`~pvfactors.geometry.timeseries.TsPVRow`
            Timeseries PV row geometries that will be used in the calculation
        pvrow_element_length : float or np.ndarray
            Length (width) of the timeseries pvrow element [m]
        tilted_to_left : list of bool
            Flags indicating when the PV rows are strictly tilted to the left
        pvrow_width : float
            Width of the timeseries PV rows in the PV array [m], which is
            constant
        rotation_vec : np.ndarray
            Rotation angles of the PV rows [deg]

        Returns
        -------
        vf_to_pvrow : np.ndarray
            View factors from timeseries pvrow_element to neighboring PV rows
        vf_to_shaded_pvrow : np.ndarray
            View factors from timeseries pvrow_element to shaded areas of the
            neighboring PV rows
        """
        if pvrow_idx == 0:
            vf_left_pvrow = np.zeros(n_steps)
            vf_left_shaded_pvrow = np.zeros(n_steps)
        else:
            # Get vf to full pvrow
            left_ts_pvrow = ts_pvrows[pvrow_idx - 1]
            left_ts_pvrow_coords = left_ts_pvrow.full_pvrow_coords
            vf_left_pvrow = self._vf_surface_to_surface(
                pvrow_element_coords, left_ts_pvrow_coords,
                pvrow_element_length)
            # Get vf to shaded pvrow
            shaded_coords = self._create_shaded_side_coords(
                left_ts_pvrow.xy_center, pvrow_width,
                left_ts_pvrow.front.shaded_length, tilted_to_left,
                rotation_vec, left_ts_pvrow.full_pvrow_coords.lowest_point)
            vf_left_shaded_pvrow = self._vf_surface_to_surface(
                pvrow_element_coords, shaded_coords, pvrow_element_length)

        if pvrow_idx == (n_pvrows - 1):
            vf_right_pvrow = np.zeros(n_steps)
            vf_right_shaded_pvrow = np.zeros(n_steps)
        else:
            # Get vf to full pvrow
            right_ts_pvrow = ts_pvrows[pvrow_idx + 1]
            right_ts_pvrow_coords = right_ts_pvrow.full_pvrow_coords
            vf_right_pvrow = self._vf_surface_to_surface(
                pvrow_element_coords, right_ts_pvrow_coords,
                pvrow_element_length)
            # Get vf to shaded pvrow
            shaded_coords = self._create_shaded_side_coords(
                right_ts_pvrow.xy_center, pvrow_width,
                right_ts_pvrow.front.shaded_length, tilted_to_left,
                rotation_vec, right_ts_pvrow.full_pvrow_coords.lowest_point)
            vf_right_shaded_pvrow = self._vf_surface_to_surface(
                pvrow_element_coords, shaded_coords, pvrow_element_length)

        vf_to_pvrow = np.where(tilted_to_left, vf_right_pvrow, vf_left_pvrow)
        vf_to_shaded_pvrow = np.where(tilted_to_left, vf_right_shaded_pvrow,
                                      vf_left_shaded_pvrow)

        return vf_to_pvrow, vf_to_shaded_pvrow

    def calculate_vf_to_gnd(self, pvrow_element_coords, pvrow_idx, n_pvrows,
                            n_steps, y_ground, cut_point_coords,
                            pvrow_element_length, tilted_to_left, ts_pvrows):
        """Calculate view factors from timeseries pvrow_element to the entire
        ground.

        Parameters
        ----------
        pvrow_element_coords :
        :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Timeseries line coordinates of pvrow element
        pvrow_idx : int
            Index of the timeseries PV row on the which the pvrow_element is
        n_pvrows : int
            Number of timeseries PV rows in the PV array
        n_steps : int
            Number of timesteps for which to calculate the pvfactors
        y_ground : float
            Y-coordinate of the flat ground [m]
        cut_point_coords : list of
        :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            List of cut point coordinates, as calculated for timeseries PV rows
        pvrow_element_length : float or np.ndarray
            Length (width) of the timeseries pvrow_element [m]
        tilted_to_left : list of bool
            Flags indicating when the PV rows are strictly tilted to the left
        ts_pvrows : list of :py:class:`~pvfactors.geometry.timeseries.TsPVRow`
            Timeseries PV row geometries that will be used in the calculation

        Returns
        -------
        vf_to_gnd : np.ndarray
            View factors from timeseries pvrow_element to the entire ground
        """

        pvrow_lowest_pt = ts_pvrows[pvrow_idx].full_pvrow_coords.lowest_point
        if pvrow_idx == 0:
            # There is no obstruction to view of the ground on the left
            coords_left_gnd = TsLineCoords(
                TsPointCoords(MIN_X_GROUND * np.ones(n_steps), y_ground),
                TsPointCoords(np.minimum(MAX_X_GROUND, cut_point_coords.x),
                              y_ground))
            vf_left_ground = self._vf_surface_to_surface(
                pvrow_element_coords, coords_left_gnd, pvrow_element_length)
        else:
            # The left PV row obstructs the view of the ground on the left
            left_pt_neighbor = \
                ts_pvrows[pvrow_idx - 1].full_pvrow_coords.lowest_point
            coords_gnd_proxy = TsLineCoords(left_pt_neighbor, pvrow_lowest_pt)
            vf_left_ground = self._vf_surface_to_surface(
                pvrow_element_coords, coords_gnd_proxy, pvrow_element_length)

        if pvrow_idx == (n_pvrows - 1):
            # There is no obstruction of the view of the ground on the right
            coords_right_gnd = TsLineCoords(
                TsPointCoords(np.maximum(MIN_X_GROUND, cut_point_coords.x),
                              y_ground),
                TsPointCoords(MAX_X_GROUND * np.ones(n_steps), y_ground))
            vf_right_ground = self._vf_surface_to_surface(
                pvrow_element_coords, coords_right_gnd, pvrow_element_length)
        else:
            # The right PV row obstructs the view of the ground on the right
            right_pt_neighbor = \
                ts_pvrows[pvrow_idx + 1].full_pvrow_coords.lowest_point
            coords_gnd_proxy = TsLineCoords(pvrow_lowest_pt, right_pt_neighbor)
            vf_right_ground = self._vf_surface_to_surface(
                pvrow_element_coords, coords_gnd_proxy, pvrow_element_length)

        # Merge the views of the ground for the back side
        vf_ground = np.where(tilted_to_left, vf_right_ground, vf_left_ground)

        return vf_ground

    def calculate_vf_to_shadow_obstruction_hottel(
            self, pvrow_element, pvrow_idx, n_shadows, n_steps, tilted_to_left,
            ts_pvrows, shadow_left, shadow_right, pvrow_element_length):
        """Calculate view factors from timeseries pvrow_element to the shadow
        of a specific timeseries PV row which is casted on the ground.

        Parameters
        ----------
        pvrow_element : :py:class:`~pvfactors.geometry.timeseries.TsDualSegment` or :py:class:`~pvfactors.geometry.timeseries.TsSurface`
            Timeseries pvrow_element to use for calculation
        pvrow_idx : int
            Index of the timeseries PV row on the which the pvrow_element is
        n_shadows : int
            Number of timeseries PV rows in the PV array, and therefore number
            of shadows they cast on the ground
        n_steps : int
            Number of timesteps for which to calculate the pvfactors
        tilted_to_left : list of bool
            Flags indicating when the PV rows are strictly tilted to the left
        ts_pvrows : list of :py:class:`~pvfactors.geometry.timeseries.TsPVRow`
            Timeseries PV row geometries that will be used in the calculation
        shadow_left : :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Coordinates of the shadow that are on the left side of the cut
            point of the PV row on which the pvrow_element is
        shadow_right : :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Coordinates of the shadow that are on the right side of the cut
            point of the PV row on which the pvrow_element is
        pvrow_element_length : float or np.ndarray
            Length (width) of the timeseries pvrow_element [m]

        Returns
        -------
        vf_to_shadow : np.ndarray
            View factors from timeseries pvrow_element to the ground shadow of
            a specific timeseries PV row
        """

        pvrow_element_lowest_pt = pvrow_element.lowest_point
        pvrow_element_highest_pt = pvrow_element.highest_point
        # Calculate view factors to left shadows
        if pvrow_idx == 0:
            # There is no obstruction on the left
            vf_to_left_shadow = self._vf_surface_to_surface(
                pvrow_element.coords, shadow_left, pvrow_element_length)
        else:
            # There is potential obstruction on the left
            pt_obstr = ts_pvrows[pvrow_idx - 1].full_pvrow_coords.lowest_point
            is_shadow_left = True
            vf_to_left_shadow = self._vf_hottel_gnd_surf(
                pvrow_element_highest_pt, pvrow_element_lowest_pt,
                shadow_left.b1, shadow_left.b2, pt_obstr, pvrow_element_length,
                is_shadow_left)

        # Calculate view factors to right shadows
        if pvrow_idx == n_shadows - 1:
            # There is no obstruction on the right
            vf_to_right_shadow = self._vf_surface_to_surface(
                pvrow_element.coords, shadow_right, pvrow_element_length)
        else:
            # There is potential obstruction on the right
            pt_obstr = ts_pvrows[pvrow_idx + 1].full_pvrow_coords.lowest_point
            is_shadow_left = False
            vf_to_right_shadow = self._vf_hottel_gnd_surf(
                pvrow_element_highest_pt, pvrow_element_lowest_pt,
                shadow_right.b1, shadow_right.b2, pt_obstr,
                pvrow_element_length, is_shadow_left)

        # Filter since we're considering the back surface only
        vf_to_shadow = np.where(tilted_to_left, vf_to_right_shadow,
                                vf_to_left_shadow)

        return vf_to_shadow

    def _vf_hottel_gnd_surf(self, high_pt_pv, low_pt_pv, left_pt_gnd,
                            right_pt_gnd, obstr_pt, width, shadow_is_left):
        """
        Calculate the timeseries view factors from a PV surface defined by low
        and high points, to a ground surface defined by left and right points,
        while accounting for potentially obstructing neighboring PV rows,
        defined by an obstruction point, and all of this using the Hottel
        String method.

        Parameters
        ----------
        high_pt_pv : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Highest point of the PV surface, for each timestamp
        low_pt_pv : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Lowest point of the PV surface, for each timestamp
        left_pt_gnd : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Leftmost point  of the ground surface, for each timestamp
        right_pt_gnd : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Rightmost point  of the ground surface, for each timestamp
        obstr_pt : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Obstructing point of neighboring PV row, for each timestamp
        width : float or np.ndarray
            Width of the PV row surface considered, from low to high point [m]
        shadow_is_left : bool
            Side of the considered shadow (or ground) surface with respect to
            the edge point of the PV row on which the considered PV surface is
            located

        Returns
        -------
        vf_1_to_2 : np.ndarray
            View factors from PV surface to ground (shadow) surface
        """

        if shadow_is_left:
            # When the shadow is left
            # - uncrossed strings are high_pv - left_gnd and low_pv - right_gnd
            # - crossed strings are high_pv - right_gnd and low_pv - left_gnd
            l1 = self._hottel_string_length(high_pt_pv, left_pt_gnd, obstr_pt,
                                            shadow_is_left)
            l2 = self._hottel_string_length(low_pt_pv, right_pt_gnd, obstr_pt,
                                            shadow_is_left)
            d1 = self._hottel_string_length(high_pt_pv, right_pt_gnd, obstr_pt,
                                            shadow_is_left)
            d2 = self._hottel_string_length(low_pt_pv, left_pt_gnd, obstr_pt,
                                            shadow_is_left)
        else:
            # When the shadow is right
            # - uncrossed strings are high_pv - right_gnd and low_pv - left_gnd
            # - crossed strings are high_pv - left_gnd and low_pv - right_gnd
            l1 = self._hottel_string_length(high_pt_pv, right_pt_gnd, obstr_pt,
                                            shadow_is_left)
            l2 = self._hottel_string_length(low_pt_pv, left_pt_gnd, obstr_pt,
                                            shadow_is_left)
            d1 = self._hottel_string_length(high_pt_pv, left_pt_gnd, obstr_pt,
                                            shadow_is_left)
            d2 = self._hottel_string_length(low_pt_pv, right_pt_gnd, obstr_pt,
                                            shadow_is_left)
        vf_1_to_2 = (d1 + d2 - l1 - l2) / (2. * width)
        # The formula doesn't work if surface is a point
        vf_1_to_2 = np.where(width > DISTANCE_TOLERANCE, vf_1_to_2, 0.)

        return vf_1_to_2

    def _hottel_string_length(self, pt_pv, pt_gnd, pt_obstr, shadow_is_left):
        """
        Calculate a string length as defined by the Hottel String method in the
        calculation of view factors, which allows to account for obstructions.

        Parameters
        ----------
        left_pt_gnd : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Leftmost point  of the ground surface, for each timestamp
        right_pt_gnd : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Rightmost point  of the ground surface, for each timestamp
        obstr_pt : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Obstructing point of neighboring PV row, for each timestamp
        shadow_is_left : bool
            Side of the considered shadow (or ground) surface with respect to
            the edge point of the PV row on which the considered PV surface is
            located

        Returns
        -------
        hottel_length : np.ndarray
            Return timeseries length of the string, while accounting for
            obstructions, in [m]
        """
        # Calculate length of string without obstruction
        l_pv = self._distance(pt_pv, pt_gnd)
        if pt_obstr is None:
            # There can't be any obstruction
            hottel_length = l_pv
        else:
            # Determine if there is obstruction by using the angles made by
            # specific strings with the x-axis
            alpha_pv = self._angle_with_x_axis(pt_gnd, pt_pv)
            alpha_ob = self._angle_with_x_axis(pt_gnd, pt_obstr)
            if shadow_is_left:
                is_obstructing = alpha_pv > alpha_ob
            else:
                is_obstructing = alpha_pv < alpha_ob
            # Calculate length of string with obstruction
            l_obstr = (self._distance(pt_gnd, pt_obstr)
                       + self._distance(pt_obstr, pt_pv))
            # Merge based on whether there is obstruction or not
            hottel_length = np.where(is_obstructing, l_obstr, l_pv)
        return hottel_length

    def _vf_surface_to_surface(self, line_1, line_2, width_1):
        """Calculate view factors between timeseries line coords, and using
        the Hottel String method for calculating view factors (without
        obstruction).

        Parameters
        ----------
        line_1 : :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Timeseries line coordinates of surface 1
        line_2 : :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Timeseries line coordinates of surface 2
        width_1 : float or np.ndarray
            Length of line_1 in [m]

        Returns
        -------
        vf_1_to_2 : np.ndarray
            View factors from line_1 to line_2, for each timestep
        """
        length_1 = self._distance(line_1.b1, line_2.b1)
        length_2 = self._distance(line_1.b2, line_2.b2)
        length_3 = self._distance(line_1.b1, line_2.b2)
        length_4 = self._distance(line_1.b2, line_2.b1)
        sum_1 = length_1 + length_2
        sum_2 = length_3 + length_4
        vf_1_to_2 = np.abs(sum_2 - sum_1) / (2. * width_1)
        # The formula doesn't work if the line is a point
        vf_1_to_2 = np.where(width_1 > DISTANCE_TOLERANCE, vf_1_to_2, 0.)

        return vf_1_to_2

    @staticmethod
    def _distance(pt_1, pt_2):
        """Calculate distance between two timeseries points

        Parameters
        ----------
        pt_1 : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Timeseries point coordinates of point 1
        pt_2 : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Timeseries point coordinates of point 2

        Returns
        -------
        np.ndarray
            Distance between the two points, for each timestep
        """
        return np.sqrt((pt_2.y - pt_1.y)**2 + (pt_2.x - pt_1.x)**2)

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

    @staticmethod
    def _create_shaded_side_coords(xy_center, width, shaded_length,
                                   mask_tilted_to_left, rotation_vec,
                                   side_lowest_pt):
        """
        Create the timeseries line coordinates for the shaded portion of a
        PV row side, based on inputted shaded length.

        Parameters
        ----------
        xy_center : tuple of float
            x and y coordinates of the PV row center point (invariant)
        width : float
            width of the PV rows [m]
        shaded_length : np.ndarray
            Timeseries values of side shaded length [m]
        tilted_to_left : list of bool
            Flags indicating when the PV rows are strictly tilted to the left
        rotation_vec : np.ndarray
            Timeseries rotation vector of the PV rows in [deg]
        side_lowest_pt : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Timeseries coordinates of lowest point of considered PV row side

        Returns
        -------
        side_shaded_coords : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Timeseries coordinates of the shaded portion of the PV row side
        """

        # Get invariant values
        x_center, y_center = xy_center
        radius = width / 2.

        # Calculate coords of shading point
        r_shade = radius - shaded_length
        x_sh = np.where(
            mask_tilted_to_left,
            r_shade * cosd(rotation_vec + 180.) + x_center,
            r_shade * cosd(rotation_vec) + x_center)
        y_sh = np.where(
            mask_tilted_to_left,
            r_shade * sind(rotation_vec + 180.) + y_center,
            r_shade * sind(rotation_vec) + y_center)

        side_shaded_coords = TsLineCoords(TsPointCoords(x_sh, y_sh),
                                          side_lowest_pt)

        return side_shaded_coords
