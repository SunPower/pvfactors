from pvfactors.viewfactors.calculator import VFCalculator
from pvfactors.geometry import OrderedPVArray
from pvfactors.irradiance import HybridPerezOrdered
from pvfactors.engine import PVEngine
import datetime as dt
from pvfactors.tests.test_viewfactors.test_data import \
    vf_matrix_left_cut, vf_left_cut_sum_axis_one_rounded
import numpy as np

np.set_printoptions(precision=3)


def test_vfcalculator(params):

    # Prepare pv array
    params.update({'cut': {0: {'front': 3}, 1: {'back': 2}}})
    pvarray = OrderedPVArray.transform_from_dict_of_scalars(params)
    vm, om = pvarray._build_view_matrix()
    geom_dict = pvarray.dict_surfaces

    # Calculate view factors
    calculator = VFCalculator()
    vf_matrix = calculator.get_vf_matrix(geom_dict, vm, om,
                                         pvarray.pvrows)

    # The values where checked visually by looking at plot of pvarray
    vf_matrix_rounded = np.around(vf_matrix, decimals=3)
    np.testing.assert_almost_equal(
        np.sum(vf_matrix_rounded[:, :-1], axis=1),
        vf_left_cut_sum_axis_one_rounded)
    np.testing.assert_array_equal(vf_matrix_rounded, vf_matrix_left_cut)


def test_vf_matrix_subset_calculation(params):
    """Check that the vf matrix subset is calculated correctly"""
    # Run in fast mode
    irradiance_model = HybridPerezOrdered()
    pvarray = OrderedPVArray.init_from_dict(params)
    fast_mode_pvrow_index = 1
    eng = PVEngine(pvarray, irradiance_model=irradiance_model,
                   fast_mode_pvrow_index=fast_mode_pvrow_index)
    timestamps = dt.datetime(2019, 6, 11, 11)
    DNI = 1000.
    DHI = 100.
    eng.fit(timestamps, DNI, DHI,
            params['solar_zenith'],
            params['solar_azimuth'],
            params['surface_tilt'],
            params['surface_azimuth'],
            params['rho_ground'])
    pvarray_fast = eng.run_timestep(0)
    vf_mat_fast = pvarray_fast.vf_matrix

    # Run in full mode
    irradiance_model = HybridPerezOrdered()
    pvarray = OrderedPVArray.init_from_dict(params)
    eng = PVEngine(pvarray, irradiance_model=irradiance_model)
    timestamps = dt.datetime(2019, 6, 11, 11)
    DNI = 1000.
    DHI = 100.
    eng.fit(timestamps, DNI, DHI,
            params['solar_zenith'],
            params['solar_azimuth'],
            params['surface_tilt'],
            params['surface_azimuth'],
            params['rho_ground'])
    pvarray_full = eng.run_timestep(0)
    vf_mat_full = pvarray_full.vf_matrix

    index_of_back_side_surface = 13
    np.testing.assert_array_almost_equal(
        vf_mat_fast, vf_mat_full[[index_of_back_side_surface], :])
