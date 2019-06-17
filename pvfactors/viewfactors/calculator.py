"""Module with classes and functions to calculate views and view factors"""

from pvfactors.viewfactors.mapper import VFMapperOrderedPVArray
import numpy as np


class VFCalculator(object):
    """This calculator class will be used for the calculation of view factors
    for PV arrays"""

    def __init__(self, mapper=None):
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
        self.mapper = mapper

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

    def get_surface_vf_vector(self, geom_dict, view_vec, obstr_vec,
                              list_pvrows, surface_index):

        n_finite_surfaces = n_all_surfaces - 1  # no sky
        indices_views_finite = np.where(view_vec)

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
