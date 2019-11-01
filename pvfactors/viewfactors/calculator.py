"""Module with classes and functions to calculate views and view factors"""

from pvfactors.config import DISTANCE_TOLERANCE
from pvfactors.viewfactors.methods import VFTsMethods
import numpy as np


class VFCalculator(object):
    """This calculator class will be used for the calculation of view factors
    for PV arrays"""

    def __init__(self, vf_ts_methods=None):
        """Initialize the view factor mapper that will be used.

        Parameters
        ----------
        vf_ts_methods : :py:class:`~pvfactors.geometry.calculator.VFTsMethods` object
            Object with methods to calculate timeseries view factors for the
            fast mode (Default = None)
        """
        self.vf_ts_methods = vf_ts_methods or VFTsMethods()

    def build_ts_vf_matrix(self, pvarray):
        """Calculate timeseries view factor matrix for the given
        ordered pv array

        Parameters
        ----------
        pvarray : :py:class:`~pvfactors.geometry.pvarray.OrderedPVArray`
            PV array whose timeseries view factor matrix to calculate

        Returns
        -------
        np.ndarray
            Timeseries view factor matrix, with 3 dimensions:
            [n_surfaces, n_surfaces, n_timesteps]
        """

        # Initialize matrix
        rotation_vec = pvarray.rotation_vec
        tilted_to_left = rotation_vec > 0
        n_steps = len(rotation_vec)
        n_ts_surfaces = pvarray.n_ts_surfaces
        vf_matrix = np.zeros((n_ts_surfaces + 1, n_ts_surfaces + 1, n_steps),
                             dtype=float)  # don't forget to include the sky

        # Get timeseries objects
        ts_ground = pvarray.ts_ground
        ts_pvrows = pvarray.ts_pvrows

        # Calculate ts view factors between pvrow and ground surfaces
        self.vf_ts_methods.vf_pvrow_gnd_surf(ts_pvrows, ts_ground,
                                             tilted_to_left, vf_matrix)
        # Calculate view factors between pv rows
        self.vf_ts_methods.vf_pvrow_to_pvrow(ts_pvrows, tilted_to_left,
                                             vf_matrix)
        # Calculate view factors to sky
        vf_matrix[:-1, -1, :] = 1. - np.sum(vf_matrix[:-1, :-1, :], axis=1)
        # This is not completely accurate yet, we need to set the sky vf
        # to zero when the surfaces have zero length
        for i, ts_surf in enumerate(pvarray.all_ts_surfaces):
            vf_matrix[i, -1, :] = np.where(ts_surf.length > DISTANCE_TOLERANCE,
                                           vf_matrix[i, -1, :], 0.)

        return vf_matrix

    def get_vf_ts_pvrow_element(self, pvrow_idx, pvrow_element, ts_pvrows,
                                ts_ground, rotation_vec, pvrow_width):
        """Calculate timeseries view factors of timeseries pvrow element
        (segment or surface) to all other elements of the PV array.

        Parameters
        ----------
        pvrow_idx : int
            Index of the timeseries PV row for which we want to calculate the
            back surface irradiance
        pvrow_element : :py:class:`~pvfactors.geometry.timeseries.TsDualSegment`
        or :py:class:`~pvfactors.geometry.timeseries.TsSurface`
            Timeseries PV row element for which to calculate view factors
        ts_pvrows : list of :py:class:`~pvfactors.geometry.timeseries.TsPVRow`
            List of timeseries PV rows in the PV array
        ts_ground : :py:class:`~pvfactors.geometry.timeseries.TsGround`
            Timeseries ground of the PV array
        rotation_vec : np.ndarray
            Timeseries rotation vector of the PV rows in [deg]
        pvrow_width : float
            Width of the timeseries PV rows in the array in [m]

        Returns
        -------
        view_factors : dict
            Dictionary of the timeseries view factors to all types of surfaces
            in the PV array. List of keys include: 'to_each_gnd_shadow',
            'to_gnd_shaded', 'to_gnd_illum', 'to_gnd_total', 'to_pvrow_total',
            'to_pvrow_shaded', 'to_pvrow_illum', 'to_sky'
        """
        tilted_to_left = rotation_vec > 0
        n_shadows = len(ts_pvrows)
        n_steps = len(rotation_vec)
        pvrow_element_coords = pvrow_element.coords
        pvrow_element_length = pvrow_element_coords.length

        # Get shadows on left and right sides of PV row
        shadows_coords_left = \
            ts_ground.shadow_coords_left_of_cut_point(pvrow_idx)
        shadows_coords_right = \
            ts_ground.shadow_coords_right_of_cut_point(pvrow_idx)
        # Calculate view factors to ground shadows
        list_vf_to_obstructed_gnd_shadows = []
        for i in range(n_shadows):
            shadow_left = shadows_coords_left[i]
            shadow_right = shadows_coords_right[i]
            # vfs to obstructed gnd shadows
            vf_obstructed_shadow = (
                self.vf_ts_methods.calculate_vf_to_shadow_obstruction_hottel(
                    pvrow_element, pvrow_idx, n_shadows, n_steps,
                    tilted_to_left, ts_pvrows, shadow_left, shadow_right,
                    pvrow_element_length))
            list_vf_to_obstructed_gnd_shadows.append(vf_obstructed_shadow)
        list_vf_to_obstructed_gnd_shadows = np.array(
            list_vf_to_obstructed_gnd_shadows)

        # Calculate view factors to shaded ground
        vf_shaded_gnd = np.sum(list_vf_to_obstructed_gnd_shadows, axis=0)

        # Calculate view factors to whole ground
        vf_gnd_total = self.vf_ts_methods.calculate_vf_to_gnd(
            pvrow_element_coords, pvrow_idx, n_shadows, n_steps,
            ts_ground.y_ground, ts_ground.cut_point_coords[pvrow_idx],
            pvrow_element_length, tilted_to_left, ts_pvrows)

        # Calculate view factors to illuminated ground
        vf_illum_gnd = vf_gnd_total - vf_shaded_gnd

        # Calculate view factors to pv rows
        vf_pvrow_total, vf_pvrow_shaded = \
            self.vf_ts_methods.calculate_vf_to_pvrow(
                pvrow_element_coords, pvrow_idx, n_shadows, n_steps, ts_pvrows,
                pvrow_element_length, tilted_to_left, pvrow_width,
                rotation_vec)
        vf_pvrow_illum = vf_pvrow_total - vf_pvrow_shaded

        # Calculate view factors to sky
        vf_to_sky = 1. - vf_gnd_total - vf_pvrow_total

        # return all timeseries view factors
        view_factors = {
            'to_each_gnd_shadow': list_vf_to_obstructed_gnd_shadows,
            'to_gnd_shaded': vf_shaded_gnd,
            'to_gnd_illum': vf_illum_gnd,
            'to_gnd_total': vf_gnd_total,
            'to_pvrow_total': vf_pvrow_total,
            'to_pvrow_shaded': vf_pvrow_shaded,
            'to_pvrow_illum': vf_pvrow_illum,
            'to_sky': vf_to_sky
        }

        return view_factors
