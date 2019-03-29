import pytest
from pvfactors.geometry import OrderedPVArray, PVGround
from pvfactors.config import MAX_X_GROUND, MIN_X_GROUND


@pytest.fixture(scope='function')
def params():

    pvarray_parameters = {
        'n_pvrows': 3,
        'pvrow_height': 1.75,
        'pvrow_width': 2.44,
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


def test_ordered_pvarray_from_dict(params):
    """Test that can successfully create ordered pvarray from parameters dict
    """
    pvarray = OrderedPVArray.from_dict(params)

    # Test that ground is created successfully
    assert isinstance(pvarray.ground, PVGround)
    assert pvarray.ground.length == (MAX_X_GROUND - MIN_X_GROUND)
