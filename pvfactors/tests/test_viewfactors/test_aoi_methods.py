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
    aoi_methods = TsAOIMethods(faoi_fn_from_pvlib_sandia(pvmodule_canadian))
    # Check that function was passed correctly
    assert callable(aoi_methods.faoi_fn)
