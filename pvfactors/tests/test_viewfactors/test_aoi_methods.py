from pvfactors.viewfactors.aoimethods import \
    AOIMethods, faoi_fn_from_pvlib_sandia
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
    """Checks
    - can create aoi methods correctly
    - vf_aoi_integrand matrix makes sense and stays consistent
    - vf_aoi values stay consistent"""
    n_timestamps = 6  # using 5 timestamps
    n_points = 6  # using only 6 sections for the integral from 0 to 180 deg
    aoi_methods = AOIMethods(faoi_fn_from_pvlib_sandia(pvmodule_canadian),
                             n_integral_sections=n_points)
    aoi_methods.fit(n_timestamps)
    # Check that function was passed correctly
    assert callable(aoi_methods.faoi_fn)
    assert aoi_methods.aoi_angles_low.shape == (n_timestamps, n_points)
    assert aoi_methods.aoi_angles_high.shape == (n_timestamps, n_points)
    assert aoi_methods.integrand_values.shape == (n_timestamps, n_points)

    # Create some dummy angle values
    low_angles = [0., 2., 108., 72., 179., 0.]
    high_angles = [31., 30., 144., 144., 180., 180.]

    # Check that integrand is calculated correctly
    faoi_integrand = aoi_methods._calculate_vfaoi_integrand(
        low_angles, high_angles)
    expected_integrand = \
        [[0.05056557, 0.18255583, 0., 0., 0., 0.],
         [0.05056557, 0., 0., 0., 0., 0.],
         [0., 0., 0., 0.25, 0.18255583, 0.],
         [0., 0., 0.25, 0.25, 0.18255583, 0.],
         [0., 0., 0., 0., 0., 0.05056557],
         [0.05056557, 0.18255583, 0.25, 0.25, 0.18255583, 0.05056557]]
    np.testing.assert_allclose(faoi_integrand, expected_integrand)

    # Check that faoi values calculated correctly
    vf_aoi = aoi_methods._calculate_vf_aoi_wedge_level(low_angles, high_angles)
    expected_vf_aoi = [0.2331214, 0.05056557, 0.43255583, 0.68255583,
                       0.05056557, 0.96624281]
    np.testing.assert_allclose(vf_aoi, expected_vf_aoi)


def test_sanity_check(pvmodule_canadian):
    """Sanity check: make sure than when faoi = 1 everywhere, the calculated
    view factor values make sense"""
    n_timestamps = 3  # using 5 timestamps
    n_points = 300  # using only 6 sections for the integral from 0 to 180 deg
    aoi_methods = AOIMethods(lambda aoi_angles: np.ones_like(aoi_angles),
                             n_integral_sections=n_points)
    aoi_methods.fit(n_timestamps)
    # Create some dummy angle values
    low_angles = [0., 90., 0.]
    high_angles = [180., 180., 90.]

    # Check that faoi values calculated correctly
    vf_aoi = aoi_methods._calculate_vf_aoi_wedge_level(low_angles, high_angles)
    expected_vf_aoi = [1., 0.5, 0.5]
    np.testing.assert_allclose(vf_aoi, expected_vf_aoi)


def test_vf():
    """Make sure that view factor from infinitesimal strip
    to parallel infinite strip is calculated correctly"""
    # Input AOI angles
    aoi_1 = [0, 90, 45, 0]
    aoi_2 = [90, 0, 135, 10]
    # Calculate view factors and check values
    vf = AOIMethods._vf(aoi_1, aoi_2)
    expected_vf = [0.5, 0.5, 0.70710678, 0.00759612]
    np.testing.assert_allclose(vf, expected_vf, atol=0, rtol=1e-6)


def test_vf_aoi_pvrow_gnd_surf():
    pass
