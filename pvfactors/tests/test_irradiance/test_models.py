import pytest
from pvfactors.irradiance import IsotropicOrdered, HybridPerezOrdered
from pvfactors.geometry import OrderedPVArray
from pvlib.tools import cosd
import numpy as np


@pytest.fixture(scope='function')
def params_irr():

    pvarray_parameters = {
        'n_pvrows': 3,
        'pvrow_height': 2.5,
        'pvrow_width': 2.,
        'surface_azimuth': 90.,  # east oriented modules
        'axis_azimuth': 0.,  # axis of rotation towards North
        'surface_tilt': 20.,
        'gcr': 0.6,
        'solar_zenith': 65.,
        'solar_azimuth': 90.,  # sun located in the east
        'rho_ground': 0.2,
        'rho_front_pvrow': 0.01,
        'rho_back_pvrow': 0.03
    }
    yield pvarray_parameters


def test_isotropic_model_front(params_irr):
    """Direct shading on front surface"""

    # pvarray
    pvarray = OrderedPVArray.from_dict(params_irr,
                                       surface_params=IsotropicOrdered.params)
    pvarray.cast_shadows()
    pvarray.cuts_for_pvrow_view()
    # there should be some direct shading
    assert pvarray.pvrows[0].front.shaded_length

    # Apply irradiance model
    DNI = 1000.
    DHI = 100.
    irr_model = IsotropicOrdered()
    irr_model.fit(DNI, DHI,
                  params_irr['solar_zenith'],
                  params_irr['solar_azimuth'],
                  params_irr['surface_tilt'],
                  params_irr['surface_azimuth'])

    # Expected values
    expected_dni_pvrow = DNI * cosd(45)
    expected_dni_ground = DNI * cosd(65)

    # Check fitting
    np.testing.assert_almost_equal(irr_model.direct['ground'][0],
                                   expected_dni_ground)
    np.testing.assert_almost_equal(irr_model.direct['front_pvrow'][0],
                                   expected_dni_pvrow)
    assert irr_model.direct['back_pvrow'][0] == 0.

    # Transform
    irradiance_vec = irr_model.transform(pvarray)

    # Check transform
    expected_irradiance_vec = [
        422.61826174069944, 422.61826174069944, 422.61826174069944,
        422.61826174069944, 422.61826174069944, 0.0,
        707.10678118654744, 0.0, 0.0, 707.10678118654744, 0.0, 0.0,
        707.10678118654744, 0.0, 100.]
    # pvrow
    np.testing.assert_almost_equal(
        pvarray.pvrows[2].front.get_param_weighted('direct'),
        expected_dni_pvrow)
    np.testing.assert_almost_equal(
        pvarray.pvrows[1].front.list_segments[0]
        .illum_collection.get_param_weighted('direct'), expected_dni_pvrow)
    np.testing.assert_almost_equal(
        pvarray.pvrows[1].front.list_segments[0]
        .shaded_collection.get_param_weighted('direct'), 0.)
    np.testing.assert_almost_equal(
        pvarray.pvrows[0].back.get_param_weighted('direct'), 0.)
    # ground
    np.testing.assert_almost_equal(
        pvarray.ground.list_segments[0]
        .illum_collection.get_param_weighted('direct'), expected_dni_ground)
    np.testing.assert_almost_equal(
        pvarray.ground.list_segments[0]
        .shaded_collection.get_param_weighted('direct'), 0.)
    np.testing.assert_array_almost_equal(expected_irradiance_vec,
                                         irradiance_vec)


def test_isotropic_model_back(params_irr):
    """Direct shading on back surface"""

    params_irr.update({'surface_azimuth': 270,
                       'surface_tilt': 160})

    # pvarray
    pvarray = OrderedPVArray.from_dict(params_irr,
                                       surface_params=IsotropicOrdered.params)
    pvarray.cast_shadows()
    pvarray.cuts_for_pvrow_view()
    # there should be some direct shading
    assert pvarray.pvrows[0].back.shaded_length

    # Apply irradiance model
    DNI = 1000.
    DHI = 100.
    irr_model = IsotropicOrdered()
    irr_model.fit(DNI, DHI,
                  params_irr['solar_zenith'],
                  params_irr['solar_azimuth'],
                  params_irr['surface_tilt'],
                  params_irr['surface_azimuth'])

    # Expected values
    expected_dni_pvrow = DNI * cosd(45)
    expected_dni_ground = DNI * cosd(65)

    # Check fitting
    np.testing.assert_almost_equal(irr_model.direct['ground'][0],
                                   expected_dni_ground)
    np.testing.assert_almost_equal(irr_model.direct['back_pvrow'][0],
                                   expected_dni_pvrow)
    assert irr_model.direct['front_pvrow'][0] == 0.

    # Transform
    irradiance_vec = irr_model.transform(pvarray)

    # Check
    expected_irradiance_vec = [
        422.61826174069944, 422.61826174069944, 422.61826174069944,
        422.61826174069944, 422.61826174069944, 0.0, 0.0, 707.10678118654755,
        0.0, 0.0, 707.10678118654755, 0.0, 0.0, 707.10678118654755, 100.]
    # pvrow
    np.testing.assert_almost_equal(
        pvarray.pvrows[2].back.get_param_weighted('direct'),
        expected_dni_pvrow)
    np.testing.assert_almost_equal(
        pvarray.pvrows[1].back.list_segments[0]
        .illum_collection.get_param_weighted('direct'), expected_dni_pvrow)
    np.testing.assert_almost_equal(
        pvarray.pvrows[1].back.list_segments[0]
        .shaded_collection.get_param_weighted('direct'), 0.)
    np.testing.assert_almost_equal(
        pvarray.pvrows[0].front.get_param_weighted('direct'), 0.)
    # ground
    np.testing.assert_almost_equal(
        pvarray.ground.list_segments[0]
        .illum_collection.get_param_weighted('direct'), expected_dni_ground)
    np.testing.assert_almost_equal(
        pvarray.ground.list_segments[0]
        .shaded_collection.get_param_weighted('direct'), 0.)
    np.testing.assert_array_almost_equal(expected_irradiance_vec,
                                         irradiance_vec)


def test_hybridperez_ordered_front(params_irr):

    # pvarray
    pvarray = OrderedPVArray.from_dict(
        params_irr, surface_params=HybridPerezOrdered.params)
    pvarray.cast_shadows()
    pvarray.cuts_for_pvrow_view()
    # there should be some direct shading
    assert pvarray.pvrows[0].front.shaded_length

    # Apply irradiance model
    DNI = 1000.
    DHI = None
    irr_model = HybridPerezOrdered()
    irr_model.fit(DNI, DHI,
                  params_irr['solar_zenith'],
                  params_irr['solar_azimuth'],
                  params_irr['surface_tilt'],
                  params_irr['surface_azimuth'])

    # Expected values
    expected_dni_pvrow = DNI * cosd(45)
    expected_dni_ground = DNI * cosd(65)

    # Check fitting
    np.testing.assert_almost_equal(irr_model.direct['ground'][0],
                                   expected_dni_ground)
    np.testing.assert_almost_equal(irr_model.direct['front_pvrow'][0],
                                   expected_dni_pvrow)
    assert irr_model.direct['back_pvrow'][0] == 0.

    # Transform
    irradiance_vec = irr_model.transform(pvarray)
