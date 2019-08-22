from pvfactors.viewfactors.calculator import VFCalculator, VFTsMethods
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


def test_ts_view_factors():

    print('\n')
    # Create base params
    params = {
        'axis_azimuth': 0,
        'n_pvrows': 3,
        'pvrow_height': 1.5,
        'pvrow_width': 2.,
        'gcr': 0.7
    }

    # Timeseries parameters for testing
    solar_zenith = np.array([45., 60., 45., 60.])
    solar_azimuth = np.array([90., 90., 270., 270.])
    surface_tilt = np.array([40., 40., 40., 40.])
    surface_azimuth = np.array([90., 270., 90., 270.])

    # Plot simple ordered pv array
    pvarray = OrderedPVArray(**params)
    pvarray.fit(solar_zenith, solar_azimuth, surface_tilt,
                surface_azimuth)

    # Calculate view factors
    pvrow_idx = 1
    side = 'back'
    segment_idx = 0
    ts_pvrows = pvarray.ts_pvrows
    ts_ground = pvarray.ts_ground
    rotation_vec = pvarray.rotation_vec
    calculator = VFCalculator()
    view_factors = calculator.get_ts_view_factors_pvrow(
        pvrow_idx, side, segment_idx, ts_pvrows, ts_ground, rotation_vec,
        pvarray.distance
    )

    # print(view_factors)

    # import os
    # is_ci = os.environ.get('CI', False)

    # if not is_ci:
    #     import matplotlib.pyplot as plt

    #     # Plot it at ts 0
    #     f, ax = plt.subplots()
    #     pvarray.plot_at_idx(2, ax)
    #     # ax.set_xlim(-1, 6)
    #     plt.show()

    #     # Plot it at ts 0
    #     f, ax = plt.subplots()
    #     pvarray.plot_at_idx(3, ax, merge_if_flag_overlap=False)
    #     # ax.set_xlim(-1, 6)
    #     plt.show()


def test_length_obstr_left():
    alpha = np.array([0.40488507262489587, 0.31533385191713814,
                      1.2490068330646151, 0.6356138841900152])
    theta = np.array([-40., 40., -40., 40.])
    d = 2.857142857142857
    w = 2.
    length = VFTsMethods.length_obstr_left(alpha, theta, d, w)

    expected_length = np.array([0.7390748775279945, 0., 0., 0.])
    np.testing.assert_allclose(length, expected_length)


def test_length_obstr_right():
    alpha = np.array([-0.698131701, -2.443460953,
                      -0.698131701, -0.651525961])
    theta = np.array([-40., 40., -40., 40.])
    d = 2.857142857142857
    w = 2.
    length = VFTsMethods.length_obstr_right(alpha, theta, d, w)

    expected_length = np.array([0., 0., 0., 0.224183046])
    np.testing.assert_allclose(length, expected_length)
