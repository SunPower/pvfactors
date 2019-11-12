from pvfactors.viewfactors.aoimethods import \
    TsAOIMethods, faoi_fn_from_pvlib_sandia
import numpy as np
import pvlib
import pytest


@pytest.fixture(scope='function')
def pvmodule_canadian():
    yield 'Canadian_Solar_CS5P_220M___2009_'


def test_faoi_fn_from_pvlib():
    """Check that faoi function creator produces correct results"""
    module_example = 'Canadian_Solar_CS5P_220M___2009_'
    faoi_fn = faoi_fn_from_pvlib_sandia(module_example)

    # Make sure the convention to measure angles from 0 to 180 deg works
    np.testing.assert_allclose(faoi_fn([10, 20, 50]), faoi_fn([170, 160, 130]))
    # Check some value consistency
    np.testing.assert_allclose(faoi_fn([10, 20, 50, 90]),
                               [0.597472, 0.856388, 1., 1.])
    # Check values outside of acceptable range: should be zero
    np.testing.assert_allclose(faoi_fn([-10, 190]), [0., 0.])


def test_ts_aoi_methods(pvmodule_canadian):
    """Check that can create aoi methods correctly"""
    n_timestamps = 5  # using 5 timestamps
    n_points = 6  # using only 6 sections for the integral from 0 to 180 deg
    aoi_methods = TsAOIMethods(faoi_fn_from_pvlib_sandia(pvmodule_canadian),
                               n_timestamps, n_integral_sections=n_points)
    # Check that function was passed correctly
    assert callable(aoi_methods.faoi_fn)
    assert aoi_methods.aoi_angles_low.shape == (n_timestamps, n_points)
    assert aoi_methods.aoi_angles_high.shape == (n_timestamps, n_points)
    assert aoi_methods.faoi_values.shape == (n_timestamps, n_points)

    # Create some dummy angle values
    low_angles = [0., 2., 108., 72., 179.]
    high_angles = [31., 30., 144., 144., 180.]

    # Check that integrand is calculated correctly
    faoi_integrand = aoi_methods._calculate_faoi_integrand(
        low_angles, high_angles)
    expected_integrand = np.array([
        [0.75485313, 0.99750363, 0., 0., 0., 0.],
        [0.75485313, 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0.99750363, 0.],
        [0., 0., 1., 1., 0.99750363, 0.],
        [0., 0., 0., 0., 0., 0.75485313]])
    np.testing.assert_allclose(faoi_integrand, expected_integrand)

    # Check that faoi values calculated correctly
    low_angles = [0., 2., 108., 72., 179.]
    high_angles = [31., 30., 144., 144., 180.]
    pct_absorbed = aoi_methods._calculate_pct_absorbed(
        low_angles, high_angles)
    expected_pct_absorbed = [0.87617838, 0.75485313, 0.99875181,
                             0.99916788, 0.75485313]
    np.testing.assert_allclose(pct_absorbed, expected_pct_absorbed)
