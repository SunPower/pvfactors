import os
import pytest
import numpy as np
from pvfactors.geometry import OrderedPVArray, PVGround
from pvfactors.config import MAX_X_GROUND, MIN_X_GROUND


@pytest.fixture(scope='function')
def params():

    pvarray_parameters = {
        'n_pvrows': 3,
        'pvrow_height': 2.5,
        'pvrow_width': 2.,
        'surface_azimuth': 90.,  # east oriented modules
        'surface_tilt': 20.,
        'gcr': 0.4,
        'solar_zenith': 20.,
        'solar_azimuth': 90.,  # sun located in the east
        'rho_ground': 0.2,
        'rho_front_pvrow': 0.01,
        'rho_back_pvrow': 0.03
    }

    yield pvarray_parameters


@pytest.fixture(scope='function')
def pvarray(params):
    pvarray = OrderedPVArray.from_dict(params)
    yield pvarray


def test_ordered_pvarray_from_dict(params):
    """Test that can successfully create ordered pvarray from parameters dict
    """
    pvarray = OrderedPVArray.from_dict(params)

    # Test that ground is created successfully
    assert isinstance(pvarray.ground, PVGround)
    assert pvarray.ground.length == (MAX_X_GROUND - MIN_X_GROUND)

    # Test the front and back sides
    assert len(pvarray.pvrows) == 3
    np.testing.assert_array_equal(
        pvarray.pvrows[0].front.n_vector, -pvarray.pvrows[0].back.n_vector)
    assert pvarray.pvrows[0].front.shaded_length == 0
    assert pvarray.gcr == params['gcr']
    assert pvarray.surface_tilt == params['surface_tilt']
    assert pvarray.surface_azimuth == params['surface_azimuth']
    assert pvarray.solar_zenith == params['solar_zenith']
    assert pvarray.solar_azimuth == params['solar_azimuth']


def test_plot_pvarray(pvarray):
    """Test that ordered pv array plotting works correctly"""
    is_ci = os.environ.get('CI', False)
    if not is_ci:
        import matplotlib.pyplot as plt
        f, ax = plt.subplots()
        pvarray.plot(ax)
        plt.show()
