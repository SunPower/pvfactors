"""Module with classes and functions to calculate views and view factors"""

from pvfactors.config import DISTANCE_TOLERANCE
from pvfactors.geometry.timeseries import TsLineCoords, TsPointCoords
from pvfactors.viewfactors.mapper import VFMapperOrderedPVArray
import numpy as np
from pvlib.tools import cosd, sind


class VFCalculator(object):
    """This calculator class will be used for the calculation of view factors
    for PV arrays"""

    def __init__(self, mapper=None, vf_ts_methods=None):
        """Initialize the view factor mapper that will be used.

        Parameters
        ----------
        mapper : VF mapper object, optional
            View factor mapper, which will map elements of the PV array view
            matrix to methods calculating view factors (Default =
            :py:class:`~pvfactors.viewfactors.mapper.VFMapperOrderedPVArray`)
        """
        if mapper is None:
            mapper = VFMapperOrderedPVArray()
        if vf_ts_methods is None:
            vf_ts_methods = VFTsMethods()
        self.mapper = mapper
        self.vf_ts_methods = vf_ts_methods

    def get_vf_matrix_subset(
        self, geom_dict, view_matrix, obstr_matrix, list_pvrows,
            list_surface_indices):

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

    def get_ts_view_factors_pvrow(
            self, pvrow_idx, segment_idx, ts_pvrows, ts_ground,
            rotation_vec, distance, width):

        # TODO: check flat case
        tilted_to_left = rotation_vec > 0
        n_shadows = len(ts_pvrows)
        n_steps = len(rotation_vec)
        segment = ts_pvrows[pvrow_idx].back.list_segments[segment_idx]
        segment_coords = segment.coords
        segment_length = segment_coords.length
        strings_are_uncrossed = ts_ground.strings_are_uncrossed[pvrow_idx]

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
                    segment, pvrow_idx, n_shadows, n_steps, tilted_to_left,
                    ts_pvrows, shadow_left, shadow_right, segment_length))
            list_vf_to_obstructed_gnd_shadows.append(vf_obstructed_shadow)
        list_vf_to_obstructed_gnd_shadows = np.array(
            list_vf_to_obstructed_gnd_shadows)

        # Calculate view factors to shaded ground
        vf_shaded_gnd = np.sum(list_vf_to_obstructed_gnd_shadows, axis=0)

        # Calculate view factors to whole ground
        vf_gnd_total = self.vf_ts_methods.calculate_vf_to_gnd(
            segment_coords, pvrow_idx, n_shadows, n_steps, ts_ground.y_ground,
            ts_ground.cut_point_coords[pvrow_idx], segment_length,
            tilted_to_left, ts_pvrows)

        # Calculate view factors to illuminated ground
        vf_illum_gnd = vf_gnd_total - vf_shaded_gnd

        # Calculate view factors to complete pv rows
        vf_pvrow_total = self.vf_ts_methods.calculate_vf_to_pvrow(
            segment_coords, pvrow_idx, n_shadows, n_steps, ts_pvrows,
            segment_length, tilted_to_left)

        # return all timeseries view factors
        view_factors = {
            'to_each_gnd_shadow': list_vf_to_obstructed_gnd_shadows,
            'to_gnd_shaded': vf_shaded_gnd,
            'to_gnd_illum': vf_illum_gnd,
            'to_gnd_total': vf_gnd_total,
            'to_pvrow_total': vf_pvrow_total,
            'to_pvrow_shaded': 0.,
            'to_pvrow_illum': 0.,
            'to_sky': 0.
        }

        return view_factors


class VFTsMethods(object):

    def __init__(self):
        pass

    def calculate_vf_to_pvrow(self, segment_coords, pvrow_idx, n_pvrows,
                              n_steps, ts_pvrows, segment_width,
                              tilted_to_left):
        if pvrow_idx == 0:
            vf_left_pvrow = np.zeros(n_steps)
        else:
            left_ts_pvrow_coords = ts_pvrows[pvrow_idx - 1].full_pvrow_coords
            vf_left_pvrow = self.vf_surface_to_surface(
                segment_coords, left_ts_pvrow_coords, segment_width)

        if pvrow_idx == (n_pvrows - 1):
            vf_right_pvrow = np.zeros(n_steps)
        else:
            right_ts_pvrow_coords = ts_pvrows[pvrow_idx + 1].full_pvrow_coords
            vf_right_pvrow = self.vf_surface_to_surface(
                segment_coords, right_ts_pvrow_coords, segment_width)

        vf_to_pvrow = np.where(tilted_to_left, vf_right_pvrow, vf_left_pvrow)

        return vf_to_pvrow

    def calculate_vf_to_gnd(self, segment_coords, pvrow_idx, n_pvrows, n_steps,
                            y_ground, cut_point_coords, segment_width,
                            tilted_to_left, ts_pvrows):

        pvrow_lowest_pt = ts_pvrows[pvrow_idx].full_pvrow_coords.lowest_point
        if pvrow_idx == 0:
            coords_left_gnd = TsLineCoords(
                TsPointCoords(MIN_X_GROUND * np.ones(n_steps), y_ground),
                TsPointCoords(np.minimum(MAX_X_GROUND, cut_point_coords.x),
                              y_ground))
            vf_left_ground = self.vf_surface_to_surface(
                segment_coords, coords_left_gnd, segment_width)
        else:
            left_pt_neighbor = \
                ts_pvrows[pvrow_idx - 1].full_pvrow_coords.lowest_point
            coords_gnd_proxy = TsLineCoords(left_pt_neighbor, pvrow_lowest_pt)
            vf_left_ground = self.vf_surface_to_surface(
                segment_coords, coords_gnd_proxy, segment_width)

        if pvrow_idx == (n_pvrows - 1):
            coords_right_gnd = TsLineCoords(
                TsPointCoords(np.maximum(MIN_X_GROUND, cut_point_coords.x),
                              y_ground),
                TsPointCoords(MAX_X_GROUND * np.ones(n_steps), y_ground))
            vf_right_ground = self.vf_surface_to_surface(
                segment_coords, coords_right_gnd, segment_width)
        else:
            right_pt_neighbor = \
                ts_pvrows[pvrow_idx + 1].full_pvrow_coords.lowest_point
            coords_gnd_proxy = TsLineCoords(pvrow_lowest_pt, right_pt_neighbor)
            vf_right_ground = self.vf_surface_to_surface(
                segment_coords, coords_gnd_proxy, segment_width)

        vf_ground = np.where(tilted_to_left, vf_right_ground, vf_left_ground)

        return vf_ground

    def calculate_vf_to_shadow_obstruction_hottel(
            self, segment, pvrow_idx, n_shadows, n_steps,
            tilted_to_left, ts_pvrows,
            shadow_left, shadow_right, segment_length):

        segment_lowest_pt = segment.lowest_point
        segment_highest_pt = segment.highest_point
        # Calculate view factors to left shadows
        if pvrow_idx == 0:
            vf_to_left_shadow = np.zeros(n_steps)
        else:
            pt_obstr = ts_pvrows[pvrow_idx - 1].full_pvrow_coords.lowest_point
            is_shadow_left = True
            vf_to_left_shadow = self.vf_hottel_shadows(
                segment_highest_pt, segment_lowest_pt,
                shadow_left.b1, shadow_left.b2, pt_obstr, segment_length,
                is_shadow_left)

        # Calculate view factors to right shadows
        if pvrow_idx == n_shadows - 1:
            vf_to_right_shadow = np.zeros(n_steps)
        else:
            pt_obstr = ts_pvrows[pvrow_idx + 1].full_pvrow_coords.lowest_point
            is_shadow_left = False
            vf_to_right_shadow = self.vf_hottel_shadows(
                segment_highest_pt, segment_lowest_pt,
                shadow_right.b1, shadow_right.b2, pt_obstr, segment_length,
                is_shadow_left)

        # Filter since we're considering the back surface only
        vf_to_shadow = np.where(tilted_to_left, vf_to_right_shadow,
                                vf_to_left_shadow)

        return vf_to_shadow

    def vf_hottel_shadows(self, high_pt_pv, low_pt_pv, left_pt_gnd,
                          right_pt_gnd, obstr_pt, width, shadow_is_left):

        if shadow_is_left:
            l1 = self.hottel_string_length(high_pt_pv, left_pt_gnd, obstr_pt,
                                           shadow_is_left)
            l2 = self.hottel_string_length(low_pt_pv, right_pt_gnd, obstr_pt,
                                           shadow_is_left)
            d1 = self.hottel_string_length(high_pt_pv, right_pt_gnd, obstr_pt,
                                           shadow_is_left)
            d2 = self.hottel_string_length(low_pt_pv, left_pt_gnd, obstr_pt,
                                           shadow_is_left)
        else:
            l1 = self.hottel_string_length(high_pt_pv, right_pt_gnd, obstr_pt,
                                           shadow_is_left)
            l2 = self.hottel_string_length(low_pt_pv, left_pt_gnd, obstr_pt,
                                           shadow_is_left)
            d1 = self.hottel_string_length(high_pt_pv, left_pt_gnd, obstr_pt,
                                           shadow_is_left)
            d2 = self.hottel_string_length(low_pt_pv, right_pt_gnd, obstr_pt,
                                           shadow_is_left)
        vf_1_to_2 = (d1 + d2 - l1 - l2) / (2. * width)

        return vf_1_to_2

    def hottel_string_length(self, pt_pv, pt_gnd, pt_obstr, shadow_is_left):
        l_pv = self.distance(pt_pv, pt_gnd)
        if pt_obstr is None:
            l = l_pv
        else:
            alpha_pv = self.angle_with_x_axis(pt_gnd, pt_pv)
            alpha_ob = self.angle_with_x_axis(pt_gnd, pt_obstr)
            if shadow_is_left:
                is_obstructing = alpha_pv > alpha_ob
            else:
                is_obstructing = alpha_pv < alpha_ob
            l_obstr = (self.distance(pt_gnd, pt_obstr)
                       + self.distance(pt_obstr, pt_pv))
            l = np.where(is_obstructing, l_obstr, l_pv)
        return l

    def vf_surface_to_surface(self, line_1, line_2, width_1):
        """Calculate view factors between timeseries line coords"""
        length_1 = self.distance(line_1.b1, line_2.b1)
        length_2 = self.distance(line_1.b2, line_2.b2)
        length_3 = self.distance(line_1.b1, line_2.b2)
        length_4 = self.distance(line_1.b2, line_2.b1)
        sum_1 = length_1 + length_2
        sum_2 = length_3 + length_4
        vf_1_to_2 = np.abs(sum_2 - sum_1) / (2. * width_1)

        return vf_1_to_2

    @staticmethod
    def distance(b1, b2):
        return np.sqrt((b2.y - b1.y)**2 + (b2.x - b1.x)**2)

    @staticmethod
    def angle_with_x_axis(pt_1, pt_2):
        """Angle with x-axis of vector going from pt_1 to pt_2"""
        return np.arctan2(pt_2.y - pt_1.y, pt_2.x - pt_1.x)
