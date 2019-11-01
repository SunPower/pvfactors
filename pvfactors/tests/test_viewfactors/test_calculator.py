from pvfactors.viewfactors.calculator import VFCalculator
from pvfactors.geometry import OrderedPVArray
from pvfactors.irradiance import HybridPerezOrdered
from pvfactors.engine import PVEngine
import datetime as dt
from pvfactors.tests.test_viewfactors.test_data import \
    vf_matrix_left_cut, vf_left_cut_sum_axis_one_rounded, \
    vf_left_cut_from_vectorized
import numpy as np

np.set_printoptions(precision=3)


def test_vfcalculator_vectorized(params):
    """The values from the expected matrix vf_left_cut_from_vectorized
    should be the same as in vf_matrix_left_cut, except for the ordering."""

    # Prepare pv array
    params.update({'cut': {0: {'front': 3}, 1: {'back': 2}}})
    pvarray = OrderedPVArray.fit_from_dict_of_scalars(params)

    # Calculate vf_matrix vectorized
    vfcalculator = VFCalculator()
    vf_matrix = vfcalculator.build_ts_vf_matrix(pvarray)

    # Check that correct size
    assert vf_matrix.shape == (47, 47, 1)

    # get all indices where surfaces have positive length + round up
    list_idx = [surf.index for surf in pvarray.all_ts_surfaces
                if surf.length[0] > 0]
    vf_matrix_rounded = np.around(vf_matrix, decimals=3)
    vf_matrix_rounded = vf_matrix_rounded[:, :, 0][np.ix_(list_idx, list_idx)]
    # check that the values stay consistent
    np.testing.assert_array_equal(vf_matrix_rounded,
                                  vf_left_cut_from_vectorized)


def test_ts_view_factors():
    """Test calculation of timeseries view factors for center PV row"""
    # Create base params
    params = {
        'axis_azimuth': 0,
        'n_pvrows': 3,
        'pvrow_height': 1.5,
        'pvrow_width': 2.,
        'gcr': 0.7
    }

    # Timeseries parameters for testing
    solar_zenith = np.array([45., 60., 45., 60., 0.])
    solar_azimuth = np.array([90., 90., 270., 270., 90.])
    surface_tilt = np.array([40., 40., 40., 40., 0.])
    surface_azimuth = np.array([90., 270., 90., 270., 90.])

    # Plot simple ordered pv array
    pvarray = OrderedPVArray(**params)
    pvarray.fit(solar_zenith, solar_azimuth, surface_tilt,
                surface_azimuth)

    # Calculate view factors
    pvrow_idx = 1
    segment_idx = 0
    ts_pvrows = pvarray.ts_pvrows
    ts_ground = pvarray.ts_ground
    rotation_vec = pvarray.rotation_vec
    calculator = VFCalculator()
    ts_segment = ts_pvrows[pvrow_idx].back.list_segments[segment_idx]
    # Calculate view factors for segment
    view_factors_segment = calculator.get_vf_ts_pvrow_element(
        pvrow_idx, ts_segment, ts_pvrows, ts_ground, rotation_vec,
        pvarray.width)
    # Calculate view factors for segment's illum surface (should be identical)
    view_factors_illum = calculator.get_vf_ts_pvrow_element(
        pvrow_idx, ts_segment.illum.list_ts_surfaces[0], ts_pvrows, ts_ground,
        rotation_vec, pvarray.width)

    expected_vf_to_obstructed_shadows = np.array([
        [0.053677, -0., 0.04077, 0.392779, 0.089299],
        [0.44436, -0., 0.004818, 0.323486, 0.5],
        [0.231492, 0.170844, -0., 0.030375, 0.089299]])
    expected_vf_to_gnd_total = np.array(
        [0.752735, 0.752735, 0.752735, 0.752735, 1.])
    expected_vf_to_gnd_illum = np.array(
        [0.023205, 0.581891, 0.707147, 0.006094, 0.321402])
    expected_vf_to_pvrow_total = np.array(
        [0.176386, 0.176386, 0.176386, 0.176386, 0.])
    expected_vf_to_pvrow_shaded = np.array([0., 0., 0., 0.068506, 0.])
    expected_vf_to_sky = np.array(
        [0.070879, 0.070879, 0.070879, 0.070879, -0.])
    # Test that segment and segment's illum surface values are identical
    for k, _ in view_factors_segment.items():
        np.testing.assert_almost_equal(
            view_factors_illum[k], view_factors_segment[k], decimal=6)
    # Now test that values are always consistent
    view_factors = view_factors_segment
    np.testing.assert_almost_equal(expected_vf_to_obstructed_shadows,
                                   view_factors['to_each_gnd_shadow'],
                                   decimal=6)
    np.testing.assert_almost_equal(expected_vf_to_gnd_total,
                                   view_factors['to_gnd_total'],
                                   decimal=6)
    np.testing.assert_almost_equal(expected_vf_to_gnd_illum,
                                   view_factors['to_gnd_illum'],
                                   decimal=6)
    np.testing.assert_almost_equal(expected_vf_to_pvrow_total,
                                   view_factors['to_pvrow_total'],
                                   decimal=6)
    np.testing.assert_almost_equal(expected_vf_to_pvrow_shaded,
                                   view_factors['to_pvrow_shaded'],
                                   decimal=6)
    np.testing.assert_almost_equal(expected_vf_to_sky, view_factors['to_sky'],
                                   decimal=6)
