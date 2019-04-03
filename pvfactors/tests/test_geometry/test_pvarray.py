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
        'axis_azimuth': 0.,  # axis of rotation towards North
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
def discr_params():
    """Discretized parameters, should have 5 segments on front of first PV row,
    and 3 segments on back of second PV row"""
    params = {
        'n_pvrows': 3,
        'pvrow_height': 1.5,
        'pvrow_width': 1.,
        'surface_tilt': 20.,
        'surface_azimuth': 180.,
        'gcr': 0.4,
        'solar_zenith': 20.,
        'solar_azimuth': 90.,  # sun located in the east
        'axis_azimuth': 0.,  # axis of rotation towards North
        'rho_ground': 0.2,
        'rho_front_pvrow': 0.01,
        'rho_back_pvrow': 0.03,
        'cut': {0: {'front': 5}, 1: {'back': 3}}
    }
    yield params


@pytest.fixture(scope='function')
def params_direct_shading(params):
    params.update({'gcr': 0.6, 'surface_tilt': 60, 'solar_zenith': 60})
    yield params


@pytest.fixture(scope='function')
def ordered_pvarray(params):
    pvarray = OrderedPVArray.from_dict(params)
    yield pvarray


def test_ordered_pvarray_from_dict(params):
    """Test that can successfully create ordered pvarray from parameters dict,
    and that the axis azimuth convention works correctly (via normal vector)
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
    assert pvarray.pvrows[0].front.n_vector[0] > 0

    # Orient the array the other way
    params.update({'surface_azimuth': 270.})
    pvarray = OrderedPVArray.from_dict(params)
    assert pvarray.pvrows[0].front.n_vector[0] < 0


def test_plot_ordered_pvarray():
    """Test that ordered pv array plotting works correctly"""
    is_ci = os.environ.get('CI', False)
    if not is_ci:
        import matplotlib.pyplot as plt

        # Create base params
        params = {
            'n_pvrows': 3,
            'pvrow_height': 2.5,
            'pvrow_width': 2.,
            'surface_azimuth': 90.,  # east oriented modules / point right
            'axis_azimuth': 0.,  # axis of rotation towards North
            'surface_tilt': 20.,
            'gcr': 0.4,
            'solar_zenith': 20.,
            'solar_azimuth': 90.,  # sun located in the east
            'rho_ground': 0.2,
            'rho_front_pvrow': 0.01,
            'rho_back_pvrow': 0.03
        }

        # Plot simple ordered pv array
        ordered_pvarray = OrderedPVArray.from_dict(params)
        f, ax = plt.subplots()
        ordered_pvarray.plot(ax)
        plt.show()

        # Plot discretized ordered pv array
        params.update({'cut': {0: {'front': 5}, 1: {'back': 3}},
                       'surface_azimuth': 270.})  # point left
        ordered_pvarray = OrderedPVArray.from_dict(params)
        f, ax = plt.subplots()
        ordered_pvarray.plot(ax)
        plt.show()


def test_discretization_ordered_pvarray(discr_params):
    pvarray = OrderedPVArray.from_dict(discr_params)
    pvrows = pvarray.pvrows

    assert len(pvrows[0].front.list_segments) == 5
    assert len(pvrows[0].back.list_segments) == 1
    assert len(pvrows[1].back.list_segments) == 3


def test_ordered_pvarray_gnd_shadow_casting(params):

    # Test front shading on right
    ordered_pvarray = OrderedPVArray.from_dict(params)
    ordered_pvarray.cast_shadows()
    # Check shadow casting on ground
    assert len(ordered_pvarray.ground.list_segments[0]
               .shaded_collection.list_surfaces) == 3
    assert len(ordered_pvarray.ground.list_segments[0]
               .illum_collection.list_surfaces) == 4
    assert ordered_pvarray.ground.shaded_length == 6.385066634855475


def test_ordered_pvarray_gnd_pvrow_shadow_casting(params_direct_shading):

    # Test front shading on right
    ordered_pvarray = OrderedPVArray.from_dict(params_direct_shading)
    ordered_pvarray.cast_shadows()
    # Check shadow casting on ground
    assert len(ordered_pvarray.ground.list_segments[0]
               .shaded_collection.list_surfaces) == 3
    assert len(ordered_pvarray.ground.list_segments[0]
               .illum_collection.list_surfaces) == 4
    assert ordered_pvarray.ground.length == MAX_X_GROUND - MIN_X_GROUND

    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[0].front.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[1].front.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[2].front.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[0].back.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[1].back.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[2].back.shaded_length, 0.)
