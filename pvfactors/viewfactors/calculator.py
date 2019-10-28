"""Module with classes and functions to calculate views and view factors"""

from pvfactors.viewfactors.mapper import VFMapperOrderedPVArray
from pvfactors.viewfactors.timeseries import VFTsMethods
import numpy as np


class VFCalculator(object):
    """This calculator class will be used for the calculation of view factors
    for PV arrays"""

    def __init__(self, mapper=None, vf_ts_methods=None):
        """Initialize the view factor mapper that will be used.

        Parameters
        ----------
        mapper : :py:class:`~pvfactors.geometry.mapper.VFMapperOrderedPVArray`, optional
            View factor mapper, which will map elements of the PV array view
            matrix to methods calculating view factors (Default =
            :py:class:`~pvfactors.viewfactors.mapper.VFMapperOrderedPVArray`)
        vf_ts_methods : :py:class:`~pvfactors.geometry.calculator.VFTsMethods` object
            Object with methods to calculate timeseries view factors for the
            fast mode (Default = None)
        """
        mapper = VFMapperOrderedPVArray() if mapper is None else mapper
        vf_ts_methods = VFTsMethods() if vf_ts_methods is None \
            else vf_ts_methods
        self.mapper = mapper
        self.vf_ts_methods = vf_ts_methods

    def get_vf_matrix_subset(
        self, geom_dict, view_matrix, obstr_matrix, list_pvrows,
            list_surface_indices):
        """Method to calculate a subset of the view factor matrix: this is
        used for fast mode calculations.


        Parameters
        ----------
        geom_dict : ``OrderedDictionary``
            Ordered dictionary of all the indexed PV surfaces from the PV
            array
        view_matrix : ``numpy.ndarray`` of int
            This matrix specifies which surfaces each surface views, and
            also what type of view it is
        obstr_matrix : ``numpy.ndarray`` of int and ``None``
            Complementing ``view_matrix`` by providing
            additional arguments for the calculations; i.e. what pvrows are
            obstructing the view between surface i and surface j
        list_pvrows : list of :py:class:`~pvfactors.geometry.pvrow.PVRow`s
            List of pvrows that can obstruct views between surfaces
        list_surface_indices : list of int
            List of indices for which to calculate view factors

        Returns
        -------
        ``numpy.ndarray`` of float
            Subset of view factor matrix calculated for given surface indices
            (values from 0 to 1 in theory)
        """

        n_all_surfaces = view_matrix.shape[0]
        n_finite_surfaces = n_all_surfaces - 1
        n_surfaces = len(list_surface_indices)
        vf_matrix_subset = np.zeros((n_surfaces, n_all_surfaces), dtype=float)

        geometries = list(geom_dict.values())
        for i, idx_i in enumerate(list_surface_indices):
            indices_views_finite = np.where(
                view_matrix[idx_i, :n_finite_surfaces])
            n_views = len(indices_views_finite[0])
            sum_finite_vfs = 0.
            for j in range(n_views):
                idx = (idx_i, indices_views_finite[0][j])
                view = self.mapper.reverse_view[view_matrix[idx]]
                line_i = geometries[idx[0]]
                line_j = geometries[idx[1]]
                obstr_index = obstr_matrix[idx]
                if obstr_index is not None:
                    obstructing_pvrow = list_pvrows[obstr_matrix[idx]]
                else:
                    obstructing_pvrow = None
                view_factor = self.mapper.function_mapping[view](
                    line_i, line_j, obstructing_pvrow)
                sum_finite_vfs += view_factor
                vf_matrix_subset[i, idx[1]] = view_factor
            # Add sky value
            vf_matrix_subset[i, -1] = 1. - sum_finite_vfs

        return vf_matrix_subset

    def get_vf_matrix(self, geom_dict, view_matrix, obstr_matrix, list_pvrow):
        """Calculate the view factors based on the surfaces of the PV array
        object, and using a mapping of "view types" in the view matrix,
        to the view factor calculation methods.
        The method uses a faster implementation by only calculating half of the
        view factors, and uses the symmetry property of the transformed view
        factor matrix to calculate the other half.
        The symmetry property comes from the reciprocity property of view
        factors: :math:`A_1 * VF_{1-2} = A_2 * VF_{2-1}`,
        where A_i is the area of surface i

        Parameters
        ----------
        geom_dict : ``OrderedDictionary``
            Ordered dictionary of all the indexed PV surfaces from the PV
            array
        view_matrix : ``numpy.ndarray`` of int
            This matrix specifies which surfaces each surface views, and
            also what type of view it is
        obstr_matrix : ``numpy.ndarray`` of int and ``None``
            Complementing ``view_matrix`` by providing
            additional arguments for the calculations; i.e. what pvrows are
            obstructing the view between surface i and surface j
        list_pvrows : list of :py:class:`~pvfactors.geometry.pvrow.PVRow`s
            List of pvrows that can obstruct views between surfaces

        Returns
        -------
        ``numpy.ndarray`` of float
            matrix of view factors (values from 0 to 1 in theory)

        """
        n_all_surfaces = view_matrix.shape[0]
        view_factors = np.zeros((n_all_surfaces, n_all_surfaces), dtype=float)

        # --- First deal with finite surfaces from the registry, and treat only
        # half of the views because symmetry will be used next
        n_finite_surfaces = n_all_surfaces - 1  # no sky
        view_matrix_upper_finite_surfaces = np.triu(
            view_matrix[:n_finite_surfaces, :n_finite_surfaces])
        indices_views_finite = np.where(view_matrix_upper_finite_surfaces)

        n_views = len(indices_views_finite[0])
        geometries = list(geom_dict.values())
        for i in range(n_views):
            idx = (indices_views_finite[0][i], indices_views_finite[1][i])
            view = self.mapper.reverse_view[view_matrix[idx]]
            line_i = geometries[idx[0]]
            line_j = geometries[idx[1]]
            obstr_index = obstr_matrix[idx]
            if obstr_index is not None:
                obstructing_pvrow = list_pvrow[obstr_matrix[idx]]
            else:
                obstructing_pvrow = None
            # The following line takes the most time to execute (looped)
            view_factors[idx] = self.mapper.function_mapping[view](
                line_i, line_j, obstructing_pvrow)

        # Use the reciprocity property of view factors to speed up the
        # vfactor calculation: A_1 * F_1-2 = A_2 * F_2-1 ==> symmetric matrx
        areas = np.array([surf.length for surf in geometries])
        matrix_areas = np.diag(areas)
        matrix_areas_inv = np.diag(1. / areas)

        upper_matrix_reciprocity = np.dot(matrix_areas,
                                          view_factors[:n_finite_surfaces,
                                                       :n_finite_surfaces])

        total_matrix_reciprocity = (upper_matrix_reciprocity +
                                    upper_matrix_reciprocity.T)
        finite_vf_matrix = np.dot(matrix_areas_inv, total_matrix_reciprocity)
        view_factors[:n_finite_surfaces, :n_finite_surfaces] = finite_vf_matrix

        # --- Then do the calculations for the sky, which is the remaining
        # portion of the hemisphere
        view_factors[:-1, -1] = 1. - np.sum(view_factors[:-1, :-1], axis=1)
        return view_factors

    def get_vf_ts_pvrow_element(self, pvrow_idx, pvrow_element, ts_pvrows,
                                ts_ground, rotation_vec, pvrow_width):
        """Calculate timeseries view factors of timeseries pvrow element
        (segment or surface) to all other elements of the PV array.

        Parameters
        ----------
        pvrow_idx : int
            Index of the timeseries PV row for which we want to calculate the
            back surface irradiance
        pvrow_element : :py:class:`~pvfactors.geometry.timeseries.TsDualSegment` or :py:class:`~pvfactors.geometry.timeseries.TsSurface`
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

    def build_ts_vf_matrix(self, pvarray):

        # Initialize matrix
        rotation_vec = pvarray.rotation_vec
        tilted_to_left = rotation_vec > 0
        n_steps = len(rotation_vec)
        n_ts_surfaces = pvarray.n_ts_surfaces
        vf_matrix = np.zeros((n_ts_surfaces, n_ts_surfaces, n_steps),
                             dtype=float)

        ts_ground = pvarray.ts_ground
        ts_pvrows = pvarray.ts_pvrows
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
                    vf_pvrow_to_gnd_surf = (
                        self.vf_ts_methods
                        .vf_pvrow_surf_to_gnd_surf_obstruction_hottel(
                            pvrow_surf, idx_pvrow, n_pvrows, n_steps,
                            tilted_to_left, ts_pvrows, gnd_surf, ts_length,
                            is_back=False, is_left=True))
                    vf_matrix[i, j, :] = vf_pvrow_to_gnd_surf
                for gnd_surf in right_gnd_surfaces:
                    j = gnd_surf.index
                    vf_pvrow_to_gnd_surf = (
                        self.vf_ts_methods
                        .vf_pvrow_surf_to_gnd_surf_obstruction_hottel(
                            pvrow_surf, idx_pvrow, n_pvrows, n_steps,
                            tilted_to_left, ts_pvrows, gnd_surf, ts_length,
                            is_back=False, is_left=False))
                    vf_matrix[i, j, :] = vf_pvrow_to_gnd_surf
            # Back side
            back = ts_pvrow.back
            for pvrow_surf in back.all_ts_surfaces:
                ts_length = pvrow_surf.length
                i = pvrow_surf.index
                for gnd_surf in left_gnd_surfaces:
                    j = gnd_surf.index
                    vf_pvrow_to_gnd_surf = (
                        self.vf_ts_methods
                        .vf_pvrow_surf_to_gnd_surf_obstruction_hottel(
                            pvrow_surf, idx_pvrow, n_pvrows, n_steps,
                            tilted_to_left, ts_pvrows, gnd_surf, ts_length,
                            is_back=True, is_left=True))
                    vf_matrix[i, j, :] = vf_pvrow_to_gnd_surf
                for gnd_surf in right_gnd_surfaces:
                    j = gnd_surf.index
                    vf_pvrow_to_gnd_surf = (
                        self.vf_ts_methods
                        .vf_pvrow_surf_to_gnd_surf_obstruction_hottel(
                            pvrow_surf, idx_pvrow, n_pvrows, n_steps,
                            tilted_to_left, ts_pvrows, gnd_surf, ts_length,
                            is_back=True, is_left=False))
                    vf_matrix[i, j, :] = vf_pvrow_to_gnd_surf

        return vf_matrix
