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
        tilted_to_left = rotation_vec >= 0
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
        vf_shadows_left = np.zeros(n_steps)
        vf_shadows_right = np.zeros(n_steps)
        for i in range(n_shadows):
            vf_shadows_left += self.vf_ts_methods.vf_surface_to_surface(
                segment_coords, shadows_coords_left[i], segment_length)
            vf_shadows_right += self.vf_ts_methods.vf_surface_to_surface(
                segment_coords, shadows_coords_right[i], segment_length)
        vf_full_shadows = np.where(tilted_to_left, vf_shadows_right,
                                   vf_shadows_left)

        # Calculate length of obstructions
        highest_pt_seg = segment.highest_point
        vf_to_gnd_shadow_obstrx = []
        for i in range(n_shadows):
            shadow_left = shadows_coords_left[i]
            shadow_right = shadows_coords_right[i]
            vf_to_obstrx = (
                self.vf_ts_methods.calculate_vf_to_shadow_obstruction(
                    segment, pvrow_idx, n_shadows, n_steps, tilted_to_left,
                    width, ts_pvrows,
                    shadow_left, shadow_right, highest_pt_seg,
                    rotation_vec, distance, segment_length
                ))
            vf_to_gnd_shadow_obstrx.append(vf_to_obstrx)
        vf_obstructions = np.sum(vf_to_gnd_shadow_obstrx, axis=0)

        # return all timeseries view factors
        view_factors = {
            'to_full_gnd_shadows': vf_full_shadows,
            'to_gnd_shadow_obstrx': vf_obstructions,
            'to_obstructed_gnd_shadows': 0.,
            'to_gnd_seen': 0.,
            'to_gnd_illum': 0.,
            'to_gnd_total': 0.,
            'to_pvrow_shaded': 0.,
            'to_pvrow_illum': 0.,
            'to_sky': 0.
        }

        return view_factors


class VFTsMethods(object):

    def __init__(self):
        pass

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

    def distance(self, b1, b2):
        return np.sqrt((b2.y - b1.y)**2 + (b2.x - b1.x)**2)

    def calculate_vf_to_shadow_obstruction(
            self, segment, pvrow_idx, n_shadows, n_steps,
            tilted_to_left, width, ts_pvrows,
            shadow_left, shadow_right, highest_pt_seg,
            rotation_vec, distance, segment_length):

        # Calculate view factors to left obstructions
        if pvrow_idx == 0:
            vf_to_left_obstr = np.zeros(n_steps)
        else:
            left_ts_pvrow = ts_pvrows[pvrow_idx - 1]
            vf_to_left_obstr = self._vf_to_left_obstruction(
                shadow_left, highest_pt_seg,
                rotation_vec, distance, segment_length,
                tilted_to_left, n_steps,
                left_ts_pvrow.xy_center,
                left_ts_pvrow.full_pvrow_coords.lowest_point, width,
                segment.coords)

        # Calculate view factors to right obstructions
        if pvrow_idx == n_shadows - 1:
            vf_to_right_obstr = np.zeros(n_steps)
        else:
            right_ts_pvrow = ts_pvrows[pvrow_idx + 1]
            vf_to_right_obstr = self._vf_to_right_obstruction(
                shadow_right, highest_pt_seg,
                rotation_vec, distance, segment_length,
                tilted_to_left, n_steps,
                right_ts_pvrow.xy_center,
                right_ts_pvrow.full_pvrow_coords.lowest_point, width,
                segment.coords)

        # Filter since we're considering the back surface only
        vf_to_obstruction = np.where(tilted_to_left, vf_to_right_obstr,
                                     vf_to_left_obstr)

        return vf_to_obstruction

    def _vf_to_left_obstruction(self, shadow_left, highest_pt_seg,
                                rotation_vec, distance, segment_length,
                                tilted_to_left, n_steps,
                                xy_center_left, lowest_pt_left, width,
                                segment_coords):
        # Calculate angles of chord with x-axis
        alpha_left = np.arctan2(highest_pt_seg.y - shadow_left.b1.y,
                                highest_pt_seg.x - shadow_left.b1.x)
        # Calculate length of left pv row which is obstructing
        l_left = self._length_obstr_left(
            alpha_left, rotation_vec, distance, segment_length,
            tilted_to_left=tilted_to_left)
        # Calculate obstruction coords
        obstr_coords = self._create_obstruction_coords(
            xy_center_left, lowest_pt_left, width, l_left,
            tilted_to_left, rotation_vec)
        # Calculate view factors to obstruction
        vf_left = self.vf_surface_to_surface(segment_coords, obstr_coords,
                                             width)
        return vf_left

    def _vf_to_right_obstruction(self, shadow_right, highest_pt_seg,
                                 rotation_vec, distance, segment_length,
                                 tilted_to_left, n_steps,
                                 xy_center_right, lowest_pt_right, width,
                                 segment_coords):
        # Calculate angles of chord with x-axis
        alpha_right = np.arctan2(shadow_right.b2.y - highest_pt_seg.y,
                                 shadow_right.b2.x - highest_pt_seg.x)
        # Calculate length of right pv row which is obstructing
        l_right = self._length_obstr_right(
            alpha_right, rotation_vec, distance, segment_length,
            tilted_to_left=tilted_to_left)
        # Calculate obstruction coords
        obstr_coords = self._create_obstruction_coords(
            xy_center_right, lowest_pt_right, width, l_right,
            tilted_to_left, rotation_vec)
        # Calculate view factors to obstruction
        vf_right = self.vf_surface_to_surface(segment_coords, obstr_coords,
                                              width)
        return vf_right

    @staticmethod
    def _create_obstruction_coords(xy_center, lowest_pt, width,
                                   length_obstr, mask_tilted_to_left,
                                   rotation_vec):

        # Prepare inputs
        x_center, y_center = xy_center
        radius = width / 2.

        # Calculate coords of shading point
        r_shade = radius - length_obstr
        x_sh = np.where(
            mask_tilted_to_left,
            r_shade * cosd(rotation_vec + 180.) + x_center,
            r_shade * cosd(rotation_vec) + x_center)
        y_sh = np.where(
            mask_tilted_to_left,
            r_shade * sind(rotation_vec + 180.) + y_center,
            r_shade * sind(rotation_vec) + y_center)

        line_coords = TsLineCoords(TsPointCoords(x_sh, y_sh), lowest_pt)
        return line_coords

    @staticmethod
    def _length_obstr_left(alpha, theta, d, w, tilted_to_left=None):
        if tilted_to_left is None:
            tilted_to_left = theta >= 0
        # TODO: speed boost, pass as argument
        theta_plus = np.deg2rad(90. + theta)
        theta_minus = np.deg2rad(90. - theta)
        # There can be numerical instabilities with the tan function
        # so make sure the beta values don't go beyond thresholds
        beta_plus = np.minimum(theta_minus + alpha,
                               np.pi / 2. - DISTANCE_TOLERANCE)
        beta_minus = np.maximum(theta_plus - alpha,
                                -np.pi / 2. + DISTANCE_TOLERANCE)

        # Calculate shaded length in cases where tilted left and right
        l_plus = np.maximum(
            0, w + d * (np.sin(theta_minus)
                        - np.cos(theta_minus) * np.tan(beta_plus)))
        l_minus = np.maximum(
            0, w - d * (np.sin(theta_plus)
                        - np.cos(theta_plus) * np.tan(beta_minus)))
        # Aggregate calculations
        length_left = np.where(tilted_to_left, l_plus, l_minus)
        return length_left

    @staticmethod
    def _length_obstr_right(alpha, theta, d, w, tilted_to_left=None):
        if tilted_to_left is None:
            tilted_to_left = theta >= 0
        # TODO: speed boost, pass as argument
        theta_plus = np.deg2rad(90. + theta)
        theta_minus = np.deg2rad(90. - theta)
        # There can be numerical instabilities with the tan function
        # so make sure the beta values don't go beyond thresholds
        beta_plus = np.maximum(theta_minus + alpha,
                               -np.pi / 2. + DISTANCE_TOLERANCE)
        beta_minus = np.minimum(theta_plus - alpha,
                                np.pi / 2 - DISTANCE_TOLERANCE)

        # Calculate shaded length in cases where tilted left and right
        l_plus = np.maximum(
            0, w - d * (np.sin(theta_minus)
                        - np.cos(theta_minus) * np.tan(beta_plus)))
        l_minus = np.maximum(
            0, w + d * (np.sin(theta_plus)
                        - np.cos(theta_plus) * np.tan(beta_minus)))
        # Aggregate calculations
        length_right = np.where(tilted_to_left, l_plus, l_minus)
        return length_right

    @staticmethod
    def build_obstr_coords(ts_pvrows, l_left, l_right, pvrow_idx, n_pvrows,
                           n_steps):

        # Build obstruction coords left
        if pvrow_idx == 0:
            # There is nothing on the left so return just a point
            pass
        pass

    def build_obstr_coords_left():
        pass

    def build_obstr_coords_right():
        pass
