from pvfactors.viewfactors.calculator import VFCalculator
from pvfactors.geometry import OrderedPVArray
from pvfactors.tests.test_viewfactors.test_data import \
    vf_matrix_left_cut
import numpy as np

np.set_printoptions(precision=3)


def test_vfcalculator(params):

    # Prepare pv array
    params.update({'cut': {0: {'front': 3}, 1: {'back': 2}}})
    pvarray = OrderedPVArray.from_dict(params)
    pvarray.cast_shadows()
    pvarray.cuts_for_pvrow_view()
    vm, om = pvarray._build_view_matrix()
    geom_dict = pvarray.dict_surfaces

    # Calculate view factors
    calculator = VFCalculator()
    vf_matrix = calculator.get_vf_matrix(geom_dict, vm, om,
                                         pvarray.pvrows)

    # The values where checked visually by looking at plot of pvarray
    np.testing.assert_array_equal(np.around(vf_matrix, decimals=3),
                                  vf_matrix_left_cut)
