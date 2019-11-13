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
    assert aoi_methods.integrand_values.shape == (n_timestamps, n_points)

    # Create some dummy angle values
    low_angles = [0., 2., 108., 72., 179.]
    high_angles = [31., 30., 144., 144., 180.]

    # Check that integrand is calculated correctly
    faoi_integrand = aoi_methods._calculate_vfaoi_integrand(
        low_angles, high_angles)
    expected_integrand = \
        [[0.05056557, 0.18255583, 0., 0., 0., 0.],
         [0.05056557, 0., 0., 0., 0., 0.],
         [0., 0., 0., 0.25, 0.18255583, 0.],
         [0., 0., 0.25, 0.25, 0.18255583, 0.],
         [0., 0., 0., 0., 0., 0.05056557]]
    np.testing.assert_allclose(faoi_integrand, expected_integrand)

    # Check that faoi values calculated correctly
    low_angles = [0., 2., 108., 72., 179.]
    high_angles = [31., 30., 144., 144., 180.]
    vf_aoi = aoi_methods._calculate_vf_aoi(low_angles, high_angles)
    expected_vf_aoi = [0.1165607, 0.05056557, 0.21627792,
                       0.22751861, 0.05056557]
    np.testing.assert_allclose(vf_aoi, expected_vf_aoi)


def test_vf():
    """Make sure that view factor from infinitesimal strip
    to parallel infinite strip is calculated correctly"""
    # Input AOI angles
    aoi_1 = [0, 90, 45, 0]
    aoi_2 = [90, 0, 135, 10]
    # Calculate view factors and check values
    vf = TsAOIMethods._vf(aoi_1, aoi_2)
    expected_vf = [0.5, 0.5, 0.70710678, 0.00759612]
    np.testing.assert_allclose(vf, expected_vf, atol=0, rtol=1e-6)
