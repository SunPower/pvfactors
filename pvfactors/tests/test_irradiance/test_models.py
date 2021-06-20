import pytest
from pvfactors.irradiance import IsotropicOrdered, HybridPerezOrdered
from pvfactors.irradiance.base import BaseModel
from pvfactors.geometry.pvarray import OrderedPVArray
from pvfactors.geometry.base import PVSurface
from pvfactors.geometry.pvrow import PVRow
from pvlib.tools import cosd
import numpy as np
import pandas as pd
import datetime as dt


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

    # Create and fit irradiance model
    DNI = 1000.
    DHI = 100.
    irr_model = IsotropicOrdered()
    irr_model.fit(None, DNI, DHI,
                  params_irr['solar_zenith'],
                  params_irr['solar_azimuth'],
                  params_irr['surface_tilt'],
                  params_irr['surface_azimuth'],
                  params_irr['rho_ground'])

    # Expected values
    expected_dni_pvrow = DNI * cosd(45)
    expected_dni_ground = DNI * cosd(65)

    # Check irradiance fitting
    np.testing.assert_almost_equal(irr_model.direct['ground_illum'][0],
                                   expected_dni_ground)
    np.testing.assert_almost_equal(irr_model.direct['front_illum_pvrow'][0],
                                   expected_dni_pvrow)
    assert irr_model.direct['back_illum_pvrow'][0] == 0.

    # Create, fit, and transform pv array
    pvarray = OrderedPVArray.fit_from_dict_of_scalars(
        params_irr, param_names=IsotropicOrdered.params)
    irr_model.transform(pvarray)

    # there should be some direct shading
    assert pvarray.ts_pvrows[0].front.shaded_length

    # Get modeling vectors
    irradiance_mat, rho_mat, invrho_mat, total_perez_mat = \
        irr_model.get_ts_modeling_vectors(pvarray)

    # Check transform
    expected_irradiance_vec = [
        0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0.,
        0., 0., 422.61826174, 422.61826174, 422.61826174,
        422.61826174, 422.61826174, 422.61826174, 422.61826174, 422.61826174,
        422.61826174, 422.61826174, 422.61826174, 422.61826174, 422.61826174,
        422.61826174, 422.61826174, 422.61826174, 707.10678119, 0.,
        0., 0., 707.10678119, 0., 0.,
        0., 707.10678119, 0., 0., 0.]
    # pvrow
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[2].front.get_param_weighted('direct'),
        expected_dni_pvrow)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].front.list_segments[0]
        .illum.get_param_weighted('direct'), expected_dni_pvrow)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].front.list_segments[0]
        .shaded.get_param_weighted('direct'), 0.)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[0].back.get_param_weighted('direct'), 0.)
    # ground
    np.testing.assert_almost_equal(
        pvarray.ts_ground.illum.get_param_weighted('direct'),
        expected_dni_ground)
    np.testing.assert_almost_equal(
        pvarray.ts_ground.shaded.get_param_weighted('direct'), 0.)
    np.testing.assert_array_almost_equal(expected_irradiance_vec,
                                         np.squeeze(irradiance_mat))

    # Check invrho_vec
    expected_invrho_vec = [
        5., 5., 5., 5., 5.,
        5., 5., 5., 5., 5.,
        5., 5., 5., 5., 5.,
        5., 5., 5., 5., 5.,
        5., 5., 5., 5., 5.,
        5., 5., 5., 100., 100.,
        33.33333333, 33.33333333, 100., 100., 33.33333333,
        33.33333333, 100., 100., 33.33333333, 33.33333333]
    np.testing.assert_array_almost_equal(np.squeeze(invrho_mat),
                                         expected_invrho_vec)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[0].front.get_param_weighted('rho'),
        params_irr['rho_front_pvrow'])
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[0].back.get_param_weighted('rho'),
        params_irr['rho_back_pvrow'])
    np.testing.assert_almost_equal(
        pvarray.ts_ground.get_param_weighted('rho'),
        params_irr['rho_ground'])

    # Check total perez vec
    expected_total_perez_vec = [
        100., 100., 100., 100., 100.,
        100., 100., 100., 100., 100.,
        100., 100., 522.61826174, 522.61826174, 522.61826174,
        522.61826174, 522.61826174, 522.61826174, 522.61826174, 522.61826174,
        522.61826174, 522.61826174, 522.61826174, 522.61826174, 522.61826174,
        522.61826174, 522.61826174, 522.61826174, 807.243186, 100.13640481,
        0., 0., 807.243186, 100.13640481, 0.,
        0., 807.243186, 100.13640481, 0., 0.]
    np.testing.assert_array_almost_equal(np.squeeze(total_perez_mat),
                                         expected_total_perez_vec)

    # check that 2 dimensional
    assert np.shape(irradiance_mat) == (40, 1)

    # check faoi modifiers
    assert irr_model.faoi_back['direct'] == 0.97
    assert irr_model.faoi_front['direct'] == 0.99
    assert irr_model.faoi_ground == 0.8

    # get absorbed sum of sky components
    irr_comp_absorbed = irr_model.get_summed_components(pvarray, absorbed=True)
    assert np.shape(irr_comp_absorbed) == (40, 1)
    # Check a ground surface value
    np.testing.assert_allclose(np.array(irr_comp_absorbed)[12, 0],
                               (1. - params_irr['rho_ground']) *
                               np.array(irradiance_mat)[12, 0])


def test_isotropic_model_back(params_irr):
    """Direct shading on back surface"""

    params_irr.update({'surface_azimuth': 270,
                       'surface_tilt': 160})

    # Apply irradiance model
    DNI = 1000.
    DHI = 100.
    irr_model = IsotropicOrdered()
    irr_model.fit(None, DNI, DHI,
                  params_irr['solar_zenith'],
                  params_irr['solar_azimuth'],
                  params_irr['surface_tilt'],
                  params_irr['surface_azimuth'],
                  params_irr['rho_ground'])

    # Expected values
    expected_dni_pvrow = DNI * cosd(45)
    expected_dni_ground = DNI * cosd(65)

    # Check fitting
    np.testing.assert_almost_equal(irr_model.direct['ground_illum'][0],
                                   expected_dni_ground)
    np.testing.assert_almost_equal(irr_model.direct['back_illum_pvrow'][0],
                                   expected_dni_pvrow)
    assert irr_model.direct['front_illum_pvrow'][0] == 0.

    # Create, fit, and transform pv array
    pvarray = OrderedPVArray.fit_from_dict_of_scalars(
        params_irr, param_names=IsotropicOrdered.params)
    irr_model.transform(pvarray)

    # there should be some direct shading
    assert pvarray.ts_pvrows[0].back.shaded_length

    # Get modeling vectors
    irradiance_mat, rho_mat, invrho_mat, total_perez_mat = \
        irr_model.get_ts_modeling_vectors(pvarray)

    # Check
    expected_irradiance_mat = [
        0., 0., 0., 0.,
        0., 0., 0., 0.,
        0., 0., 0., 0.,
        422.61826174, 422.61826174, 422.61826174, 422.61826174,
        422.61826174, 422.61826174, 422.61826174, 422.61826174,
        422.61826174, 422.61826174, 422.61826174, 422.61826174,
        422.61826174, 422.61826174, 422.61826174, 422.61826174,
        0., 0., 707.10678119, 0.,
        0., 0., 707.10678119, 0.,
        0., 0., 707.10678119, 0.]
    # pvrow
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[2].back.get_param_weighted('direct'),
        expected_dni_pvrow)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].back.list_segments[0]
        .illum.get_param_weighted('direct'), expected_dni_pvrow)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].back.list_segments[0]
        .shaded.get_param_weighted('direct'), 0.)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[0].front.get_param_weighted('direct'), 0.)
    # ground
    np.testing.assert_almost_equal(
        pvarray.ts_ground
        .illum.get_param_weighted('direct'), expected_dni_ground)
    np.testing.assert_almost_equal(
        pvarray.ts_ground
        .shaded.get_param_weighted('direct'), 0.)
    np.testing.assert_array_almost_equal(expected_irradiance_mat,
                                         np.squeeze(irradiance_mat))
    # Check invrho_mat
    expected_invrho_mat = [
        5., 5., 5., 5.,
        5., 5., 5., 5.,
        5., 5., 5., 5.,
        5., 5., 5., 5.,
        5., 5., 5., 5.,
        5., 5., 5., 5.,
        5., 5., 5., 5.,
        100., 100., 33.33333333, 33.33333333,
        100., 100., 33.33333333, 33.33333333,
        100., 100., 33.33333333, 33.33333333]
    np.testing.assert_array_almost_equal(np.squeeze(invrho_mat),
                                         expected_invrho_mat)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[0].front.get_param_weighted('rho'),
        params_irr['rho_front_pvrow'])
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[0].back.get_param_weighted('rho'),
        params_irr['rho_back_pvrow'])
    np.testing.assert_almost_equal(
        pvarray.ts_ground.get_param_weighted('rho'),
        params_irr['rho_ground'])

    # Check total perez vec
    expected_total_perez_mat = [
        100., 100., 100., 100.,
        100., 100., 100., 100.,
        100., 100., 100., 100.,
        522.61826174, 522.61826174, 522.61826174, 522.61826174,
        522.61826174, 522.61826174, 522.61826174, 522.61826174,
        522.61826174, 522.61826174, 522.61826174, 522.61826174,
        522.61826174, 522.61826174, 522.61826174, 522.61826174,
        104.38724754, 104.38724754, 0., 0.,
        104.38724754, 104.38724754, 0., 0.,
        104.38724754, 104.38724754, 0., 0.]
    np.testing.assert_array_almost_equal(np.squeeze(total_perez_mat),
                                         expected_total_perez_mat)

    # Get ts modeling matrices
    irradiance_mat, rho_mat, invrho_mat, total_perez_mat = \
        irr_model.get_full_ts_modeling_vectors(pvarray)

    # check that 2 dimensional
    assert irradiance_mat.shape == (41, 1)


def test_hybridperez_ordered_front(params_irr):

    # Apply irradiance model
    DNI = 1000.
    DHI = 100.
    ts = dt.datetime(2019, 6, 14, 11)
    irr_model = HybridPerezOrdered(horizon_band_angle=6.5)
    irr_model.fit(ts, DNI, DHI,
                  params_irr['solar_zenith'],
                  params_irr['solar_azimuth'],
                  params_irr['surface_tilt'],
                  params_irr['surface_azimuth'],
                  params_irr['rho_ground'])

    # Expected values
    expected_dni_pvrow = DNI * cosd(45)
    expected_dni_ground = DNI * cosd(65)
    expected_circ_pvrow = 61.542748619313045
    # FIXME: it doesn't seem right that circumsolar stronger on ground
    expected_circ_ground = 36.782407037017585
    expected_hor_pvrow_no_shad = 7.2486377533042452
    expected_hor_pvrow_w_shad = 2.1452692285058985
    horizon_shading_pct = 70.404518731426592

    # Check fitting
    np.testing.assert_almost_equal(irr_model.direct['ground_illum'][0],
                                   expected_dni_ground)
    np.testing.assert_almost_equal(irr_model.direct['front_illum_pvrow'][0],
                                   expected_dni_pvrow)
    assert irr_model.direct['back_illum_pvrow'][0] == 0.

    # Create, fit, and transform pv array
    pvarray = OrderedPVArray.fit_from_dict_of_scalars(
        params_irr, param_names=IsotropicOrdered.params)
    irr_model.transform(pvarray)

    # there should be some direct shading
    assert pvarray.ts_pvrows[0].front.shaded_length

    # Get modeling vectors
    irradiance_mat, rho_mat, invrho_mat, total_perez_mat = \
        irr_model.get_ts_modeling_vectors(pvarray)

    # Test isotropic_luminance
    np.testing.assert_almost_equal(irr_model.isotropic_luminance,
                                   63.21759296)
    # Check transform
    expected_irradiance_mat = [
        0., 0., 0., 0.,
        0., 0., 0., 0.,
        0., 0., 0., 0.,
        459.40066878, 459.40066878, 459.40066878, 459.40066878,
        459.40066878, 459.40066878, 459.40066878, 459.40066878,
        459.40066878, 459.40066878, 459.40066878, 459.40066878,
        459.40066878, 459.40066878, 459.40066878, 459.40066878,
        775.89816756, 7.24863775, 7.24863775, 7.24863775,
        775.89816756, 7.24863775, 2.14526923, 0.,
        775.89816756, 7.24863775, 2.14526923, 0.]
    # pvrow direct
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[2].front.get_param_weighted('direct'),
        expected_dni_pvrow)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].front.list_segments[0]
        .illum.get_param_weighted('direct'), expected_dni_pvrow)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].front.list_segments[0]
        .shaded.get_param_weighted('direct'), 0.)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[0].back.get_param_weighted('direct'), 0.)
    # pvrow circumsolar
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[2].front.get_param_weighted('circumsolar'),
        expected_circ_pvrow)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].front.list_segments[0]
        .illum.get_param_weighted('circumsolar'),
        expected_circ_pvrow)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].front.list_segments[0]
        .shaded.get_param_weighted('circumsolar'), 0.)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].back.list_segments[0]
        .illum.get_param_weighted('circumsolar'), 0.)
    # pvrow horizon
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].front.list_segments[0]
        .illum.get_param_weighted('horizon'),
        expected_hor_pvrow_no_shad)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].front.list_segments[0]
        .shaded.get_param_weighted('horizon'),
        expected_hor_pvrow_no_shad)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[0].back.list_segments[0]
        .illum.get_param_weighted('horizon'),
        expected_hor_pvrow_no_shad)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].back.list_segments[0]
        .illum.get_param_weighted('horizon'),
        expected_hor_pvrow_w_shad)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].back.list_segments[0]
        .illum.get_param_weighted('horizon_shd_pct'),
        horizon_shading_pct)
    # ground
    np.testing.assert_almost_equal(
        pvarray.ts_ground.get_param_weighted('horizon'), 0.)
    np.testing.assert_almost_equal(
        pvarray.ts_ground
        .illum.get_param_weighted('direct'), expected_dni_ground)
    np.testing.assert_almost_equal(
        pvarray.ts_ground
        .illum.get_param_weighted('circumsolar'),
        expected_circ_ground)
    np.testing.assert_almost_equal(
        pvarray.ts_ground
        .shaded.get_param_weighted('direct'), 0.)
    np.testing.assert_array_almost_equal(expected_irradiance_mat,
                                         np.squeeze(irradiance_mat))
    # Check invrho_mat
    expected_invrho_mat = [
        5., 5., 5., 5.,
        5., 5., 5., 5.,
        5., 5., 5., 5.,
        5., 5., 5., 5.,
        5., 5., 5., 5.,
        5., 5., 5., 5.,
        5., 5., 5., 5.,
        100., 100., 33.33333333, 33.33333333,
        100., 100., 33.33333333, 33.33333333,
        100., 100., 33.33333333, 33.33333333]
    np.testing.assert_array_almost_equal(np.squeeze(invrho_mat),
                                         expected_invrho_mat)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[0].front.get_param_weighted('rho'),
        params_irr['rho_front_pvrow'])
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[0].back.get_param_weighted('rho'),
        params_irr['rho_back_pvrow'])
    np.testing.assert_almost_equal(
        pvarray.ts_ground.get_param_weighted('rho'),
        params_irr['rho_ground'])

    # Check total perez vec
    expected_total_perez_mat = [
        63.21759296, 63.21759296, 63.21759296, 63.21759296,
        63.21759296, 63.21759296, 63.21759296, 63.21759296,
        63.21759296, 63.21759296, 63.21759296, 63.21759296,
        522.61826174, 522.61826174, 522.61826174, 522.61826174,
        522.61826174, 522.61826174, 522.61826174, 522.61826174,
        522.61826174, 522.61826174, 522.61826174, 522.61826174,
        522.61826174, 522.61826174, 522.61826174, 522.61826174,
        807.243186, 38.59365619, 0., 0.,
        807.243186, 38.59365619, 0., 0.,
        807.243186, 38.59365619, 0., 0.]
    np.testing.assert_array_almost_equal(np.squeeze(total_perez_mat),
                                         expected_total_perez_mat)

    # Get ts modeling matrices
    irradiance_mat, rho_mat, invrho_mat, total_perez_mat = \
        irr_model.get_full_ts_modeling_vectors(pvarray)

    # check that 2 dimensional
    assert irradiance_mat.shape == (41, 1)

    # check faoi modifiers
    assert irr_model.faoi_back['circumsolar'] == 0.97
    assert irr_model.faoi_back['horizon'] == 0.97
    assert irr_model.faoi_front['direct'] == 0.99
    assert irr_model.faoi_ground == 0.8

    # get absorbed sum of sky components
    irr_comp_absorbed = irr_model.get_summed_components(pvarray, absorbed=True)
    assert np.shape(irr_comp_absorbed) == (40, 1)
    # Check a ground surface value
    np.testing.assert_allclose(np.array(irr_comp_absorbed)[12, 0],
                               (1. - params_irr['rho_ground']) *
                               np.array(irradiance_mat)[12, 0])


def test_hybridperez_ordered_back(params_irr):

    params_irr.update({'surface_azimuth': 270,
                       'surface_tilt': 160})

    # Apply irradiance model
    DNI = 1000.
    DHI = 100.
    ts = dt.datetime(2019, 6, 14, 11)
    irr_model = HybridPerezOrdered(horizon_band_angle=50)
    irr_model.fit(ts, DNI, DHI,
                  params_irr['solar_zenith'],
                  params_irr['solar_azimuth'],
                  params_irr['surface_tilt'],
                  params_irr['surface_azimuth'],
                  params_irr['rho_ground'])

    # Expected values
    expected_dni_pvrow = DNI * cosd(45)
    expected_dni_ground = DNI * cosd(65)
    expected_circ_pvrow = 61.542748619313045
    # FIXME: it doesn't seem right that circumsolar stronger on ground
    expected_circ_ground = 36.782407037017585
    expected_hor_pvrow_no_shad = 7.2486377533042452
    expected_hor_pvrow_w_shad_1 = 6.0760257690033654
    expected_hor_pvrow_w_shad_2 = 3.6101632102156898
    horizon_shading_pct_1 = 16.176997998918541
    horizon_shading_pct_2 = 50.195287265251757

    # Check fitting
    np.testing.assert_almost_equal(irr_model.direct['ground_illum'][0],
                                   expected_dni_ground)
    np.testing.assert_almost_equal(irr_model.direct['back_illum_pvrow'][0],
                                   expected_dni_pvrow)
    assert irr_model.direct['front_illum_pvrow'][0] == 0.

    # Create, fit, and transform pv array
    pvarray = OrderedPVArray.fit_from_dict_of_scalars(
        params_irr, param_names=IsotropicOrdered.params)
    irr_model.transform(pvarray)

    # there should be some direct shading
    assert pvarray.ts_pvrows[0].back.shaded_length

    # Get modeling vectors
    irradiance_mat, rho_mat, invrho_mat, total_perez_mat = \
        irr_model.get_ts_modeling_vectors(pvarray)

    # Test isotropic_luminance
    np.testing.assert_almost_equal(irr_model.isotropic_luminance,
                                   63.21759296)
    # Check transform
    expected_irradiance_mat = [
        0., 0., 0., 0.,
        0., 0., 0., 0.,
        0., 0., 0., 0.,
        459.40066878, 459.40066878, 459.40066878, 459.40066878,
        459.40066878, 459.40066878, 459.40066878, 459.40066878,
        459.40066878, 459.40066878, 459.40066878, 459.40066878,
        459.40066878, 459.40066878, 459.40066878, 459.40066878,
        7.24863775, 7.24863775, 774.72555557, 3.61016321,
        7.24863775, 7.24863775, 774.72555557, 3.61016321,
        7.24863775, 7.24863775, 775.89816756, 7.24863775]

    # pvrow direct
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[2].back.get_param_weighted('direct'),
        expected_dni_pvrow)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].back.list_segments[0]
        .illum.get_param_weighted('direct'), expected_dni_pvrow)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].back.list_segments[0]
        .shaded.get_param_weighted('direct'), 0.)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[0].front.get_param_weighted('direct'), 0.)
    # pvrow circumsolar
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[2].back.get_param_weighted('circumsolar'),
        expected_circ_pvrow)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].back.list_segments[0]
        .illum.get_param_weighted('circumsolar'),
        expected_circ_pvrow)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].back.list_segments[0]
        .shaded.get_param_weighted('circumsolar'), 0.)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].front.list_segments[0]
        .illum.get_param_weighted('circumsolar'), 0.)
    # pvrow horizon
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].front.get_param_weighted('horizon'),
        expected_hor_pvrow_no_shad)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].back.list_segments[0]
        .illum.get_param_weighted('horizon'),
        expected_hor_pvrow_w_shad_1)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].back.list_segments[0]
        .shaded.get_param_weighted('horizon'),
        expected_hor_pvrow_w_shad_2)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[0].back.list_segments[0]
        .illum.get_param_weighted('horizon'),
        expected_hor_pvrow_w_shad_1)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[0].back.list_segments[0]
        .shaded.get_param_weighted('horizon'),
        expected_hor_pvrow_w_shad_2)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].back.list_segments[0]
        .illum.get_param_weighted('horizon_shd_pct'),
        horizon_shading_pct_1)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].back.list_segments[0]
        .shaded.get_param_weighted('horizon_shd_pct'),
        horizon_shading_pct_2)
    # ground
    np.testing.assert_almost_equal(
        pvarray.ts_ground.get_param_weighted('horizon'), 0.)
    np.testing.assert_almost_equal(
        pvarray.ts_ground
        .illum.get_param_weighted('direct'), expected_dni_ground)
    np.testing.assert_almost_equal(
        pvarray.ts_ground
        .illum.get_param_weighted('circumsolar'),
        expected_circ_ground)
    np.testing.assert_almost_equal(
        pvarray.ts_ground
        .shaded.get_param_weighted('direct'), 0.)
    np.testing.assert_array_almost_equal(expected_irradiance_mat,
                                         np.squeeze(irradiance_mat))
    # Check invrho_mat
    expected_invrho_mat = [
        5., 5., 5., 5.,
        5., 5., 5., 5.,
        5., 5., 5., 5.,
        5., 5., 5., 5.,
        5., 5., 5., 5.,
        5., 5., 5., 5.,
        5., 5., 5., 5.,
        100., 100., 33.33333333, 33.33333333,
        100., 100., 33.33333333, 33.33333333,
        100., 100., 33.33333333, 33.33333333]
    np.testing.assert_array_almost_equal(
        np.squeeze(invrho_mat), expected_invrho_mat)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[0].front.get_param_weighted('rho'),
        params_irr['rho_front_pvrow'])
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[0].back.get_param_weighted('rho'),
        params_irr['rho_back_pvrow'])
    np.testing.assert_almost_equal(
        pvarray.ts_ground.get_param_weighted('rho'),
        params_irr['rho_ground'])

    # Check total perez vec
    expected_total_perez_mat = [
        63.21759296, 63.21759296, 63.21759296, 63.21759296,
        63.21759296, 63.21759296, 63.21759296, 63.21759296,
        63.21759296, 63.21759296, 63.21759296, 63.21759296,
        522.61826174, 522.61826174, 522.61826174, 522.61826174,
        522.61826174, 522.61826174, 522.61826174, 522.61826174,
        522.61826174, 522.61826174, 522.61826174, 522.61826174,
        522.61826174, 522.61826174, 522.61826174, 522.61826174,
        104.38724754, 104.38724754, 0., 0.,
        104.38724754, 104.38724754, 0., 0.,
        104.38724754, 104.38724754, 0., 0.]
    np.testing.assert_array_almost_equal(np.squeeze(total_perez_mat),
                                         expected_total_perez_mat)

    # Get ts modeling matrices
    irradiance_mat, rho_mat, invrho_mat, total_perez_mat = \
        irr_model.get_full_ts_modeling_vectors(pvarray)

    # check that 2 dimensional
    assert irradiance_mat.shape == (41, 1)


# def test_hybridperez_circ_shading():
#     """Check that the function works and returns expected outputs"""
#     circumsolar_angle = 30.
#     circumsolar_model = 'uniform_disk'
#     irr_model = HybridPerezOrdered(circumsolar_angle=circumsolar_angle,
#                                    circumsolar_model=circumsolar_model)

#     surf = PVSurface(coords=[(0, -1), (0, 1)])
#     pvrows = [PVRow.from_linestring_coords([(1, -1), (1, 1)])]
#     solar_2d_vector = [1.2, 1]  # <45 deg elevation so should have >50% shading
#     idx_neighbor = 0

#     circ_shading_pct = irr_model._calculate_circumsolar_shading_pct(
#         surf, idx_neighbor, pvrows, solar_2d_vector)

#     np.testing.assert_almost_equal(circ_shading_pct, 71.5969299216)


def test_hybridperez_horizon_shading_ts():

    # Base params
    params = {
        'n_pvrows': 3,
        'pvrow_height': 1,
        'pvrow_width': 1,
        'axis_azimuth': 0.,
        'gcr': 0.3
    }
    # Timeseries inputs
    df_inputs = pd.DataFrame({
        'solar_zenith': [70., 80., 80., 70., 10.],
        'solar_azimuth': [270., 90., 270., 90., 90.],
        'surface_tilt': [20., 10., 20., 30., 0.],
        'surface_azimuth': [270., 270., 90., 90., 90.]})

    # Initialize and fit pv array
    pvarray = OrderedPVArray.init_from_dict(params)
    # Fit pv array to timeseries data
    pvarray.fit(df_inputs.solar_zenith, df_inputs.solar_azimuth,
                df_inputs.surface_tilt, df_inputs.surface_azimuth)

    # irradiance model
    model = HybridPerezOrdered(horizon_band_angle=15.)
    pvrow_idx = 1
    centroid_coords = (pvarray.ts_pvrows[pvrow_idx].back.list_segments[0]
                       .coords.centroid)
    tilted_to_left = pvarray.rotation_vec > 0
    horizon_pct_shading = model._calculate_horizon_shading_pct_ts(
        pvarray.ts_pvrows, centroid_coords, pvrow_idx, tilted_to_left,
        is_back_side=True)

    # Check that values stay consistent
    expected_pct_shading = np.array(
        [17.163813, 8.667262, 17.163813, 25.317135, 0.])
    np.testing.assert_allclose(expected_pct_shading, horizon_pct_shading)


def test_hybridperez_transform(df_inputs_clearsky_8760):

    n_points = 24
    df_inputs = df_inputs_clearsky_8760.iloc[:n_points, :]
    # Base params
    params = {
        'n_pvrows': 3,
        'pvrow_height': 1,
        'pvrow_width': 1,
        'axis_azimuth': 0.,
        'gcr': 0.3
    }
    albedo = 0.2

    # Initialize and fit pv array
    pvarray = OrderedPVArray.init_from_dict(params)
    # Fit pv array to timeseries data
    pvarray.fit(df_inputs.solar_zenith, df_inputs.solar_azimuth,
                df_inputs.surface_tilt, df_inputs.surface_azimuth)

    # irradiance model
    model = HybridPerezOrdered(horizon_band_angle=15.)
    model.fit(df_inputs.index, df_inputs.dni.values, df_inputs.dhi.values,
              df_inputs.solar_zenith.values, df_inputs.solar_azimuth.values,
              df_inputs.surface_tilt.values, df_inputs.surface_azimuth.values,
              albedo)
    model.transform(pvarray)

    # Check timeseries parameters
    expected_middle_back_horizon = np.array(
        [0., 0., 0., 0., 0., 0.,
         0., 0.8244883, 4.43051118, 6.12136418, 6.03641816, 2.75109931,
         3.15586037, 6.14709947, 6.02242241, 4.25283177, 0.58518296, 0.,
         0., 0., 0., 0., 0., 0.])
    list_idx = np.where(expected_middle_back_horizon != 0)
    np.testing.assert_allclose(
        expected_middle_back_horizon[list_idx],
        pvarray.ts_pvrows[1].back.list_segments[0]
        .illum.get_param_weighted('horizon')[list_idx])

    expected_ground_circ = np.array(
        [0., 0., 0., 0., 0.,
         0., 0., 2.19047189, 8.14152575, 13.9017384,
         18.54394777, 21.11510529, 21.00554831, 18.24251837, 13.47583799,
         7.66930532, 1.74693357, 0., 0., 0.,
         0., 0., 0., 0.])
    np.testing.assert_allclose(
        expected_ground_circ,
        pvarray.ts_ground.illum_params['circumsolar'])
    np.testing.assert_allclose(
        np.zeros(n_points),
        pvarray.ts_ground.shaded_params['circumsolar'])

    # # Check at a given time idx
    # pvrow = pvarray.ts_pvrows[1].at(7)
    # np.testing.assert_allclose(
    #     pvrow.back.list_segments[0]
    #     .illum_collection.get_param_weighted('horizon'),
    #     expected_middle_back_horizon[7])
    # pvground = pvarray.ts_ground.at(7)
    # np.testing.assert_allclose(
    #     pvground.list_segments[0].illum_collection
    #     .get_param_weighted('circumsolar'),
    #     expected_ground_circ[7])


def test_hybridperez_ordered_transparency_spacing_front(params_irr):
    """Check that module transparency and spacing params are applied
    correctly in HybridPerezOrdered"""

    # Apply irradiance model
    DNI = 1000.
    DHI = 100.
    ts = dt.datetime(2019, 6, 14, 11)
    irr_parameters = {'horizon_band_angle': 6.5,
                      'module_transparency': 0.1,
                      'module_spacing_ratio': 0.1}
    irr_model = HybridPerezOrdered(**irr_parameters)
    irr_model.fit(ts, DNI, DHI,
                  params_irr['solar_zenith'],
                  params_irr['solar_azimuth'],
                  params_irr['surface_tilt'],
                  params_irr['surface_azimuth'],
                  params_irr['rho_ground'])

    # Create, fit, and transform pv array
    pvarray = OrderedPVArray.fit_from_dict_of_scalars(
        params_irr, param_names=IsotropicOrdered.params)
    irr_model.transform(pvarray)

    gnd_seg = pvarray.ts_ground
    pvrow_front = pvarray.ts_pvrows[1].front
    # check that front is shaded
    assert pvrow_front.shaded_length > 0

    # Run some checks
    surf_gnd_shaded = gnd_seg.shaded.list_ts_surfaces[0]
    surf_gnd_illum = gnd_seg.illum.list_ts_surfaces[0]
    np.testing.assert_allclose(surf_gnd_illum.get_param('circumsolar') * 0.19,
                               surf_gnd_shaded.get_param('circumsolar'))
    np.testing.assert_allclose(surf_gnd_illum.get_param('direct') * 0.19,
                               surf_gnd_shaded.get_param('direct'))
    # Run check on pvrow surfaces
    surf_pvrow_shaded = (pvrow_front.list_segments[0]
                         .shaded.list_ts_surfaces[0])
    surf_pvrow_illum = (pvrow_front.list_segments[0]
                        .illum.list_ts_surfaces[0])
    np.testing.assert_allclose(surf_pvrow_illum.get_param('direct') * 0.19,
                               surf_pvrow_shaded.get_param('direct'))
    np.testing.assert_allclose(
        surf_pvrow_illum.get_param('circumsolar') * 0.19,
        surf_pvrow_shaded.get_param('circumsolar'))


def test_hybridperez_ordered_transparency_spacing_back(params_irr):
    """Check that module transparency and spacing params are applied
    correctly in HybridPerezOrdered"""

    params_irr.update({'surface_azimuth': 270,
                       'surface_tilt': 160})
    # Apply irradiance model
    DNI = 1000.
    DHI = 100.
    ts = dt.datetime(2019, 6, 14, 11)
    irr_parameters = {'horizon_band_angle': 6.5,
                      'module_transparency': 0.1,
                      'module_spacing_ratio': 0.1}
    irr_model = HybridPerezOrdered(**irr_parameters)
    irr_model.fit(ts, DNI, DHI,
                  params_irr['solar_zenith'],
                  params_irr['solar_azimuth'],
                  params_irr['surface_tilt'],
                  params_irr['surface_azimuth'],
                  params_irr['rho_ground'])

    # Create, fit, and transform pv array
    pvarray = OrderedPVArray.fit_from_dict_of_scalars(
        params_irr, param_names=IsotropicOrdered.params)
    irr_model.transform(pvarray)

    gnd_seg = pvarray.ts_ground
    pvrow_back = pvarray.ts_pvrows[1].back
    # check that back is shaded
    assert pvrow_back.shaded_length > 0

    # Run some checks on gnd surfaces
    surf_gnd_shaded = gnd_seg.shaded.list_ts_surfaces[0]
    surf_gnd_illum = gnd_seg.illum.list_ts_surfaces[0]
    np.testing.assert_allclose(surf_gnd_illum.get_param('circumsolar') * 0.19,
                               surf_gnd_shaded.get_param('circumsolar'))
    np.testing.assert_allclose(surf_gnd_illum.get_param('direct') * 0.19,
                               surf_gnd_shaded.get_param('direct'))
    # Run check on pvrow surfaces
    surf_pvrow_shaded = (pvrow_back.list_segments[0]
                         .shaded.list_ts_surfaces[0])
    surf_pvrow_illum = (pvrow_back.list_segments[0]
                        .illum.list_ts_surfaces[0])
    np.testing.assert_allclose(surf_pvrow_illum.get_param('direct') * 0.19,
                               surf_pvrow_shaded.get_param('direct'))
    np.testing.assert_allclose(
        surf_pvrow_illum.get_param('circumsolar') * 0.19,
        surf_pvrow_shaded.get_param('circumsolar'))


def test_isotropic_ordered_transparency_spacing(params_irr):
    """Check that module transparency and spacing params are applied
    correctly in IsotropicOrdered"""

    # Apply irradiance model
    DNI = 1000.
    DHI = 100.
    ts = dt.datetime(2019, 6, 14, 11)
    irr_parameters = {'module_transparency': 0.1,
                      'module_spacing_ratio': 0.1}
    irr_model = IsotropicOrdered(**irr_parameters)
    irr_model.fit(ts, DNI, DHI,
                  params_irr['solar_zenith'],
                  params_irr['solar_azimuth'],
                  params_irr['surface_tilt'],
                  params_irr['surface_azimuth'],
                  params_irr['rho_ground'])

    # Create, fit, and transform pv array
    pvarray = OrderedPVArray.fit_from_dict_of_scalars(
        params_irr, param_names=IsotropicOrdered.params)
    irr_model.transform(pvarray)

    gnd_seg = pvarray.ts_ground
    pvrow_front = pvarray.ts_pvrows[1].front
    # check that front is shaded
    assert pvrow_front.shaded_length > 0

    # Run some checks
    surf_gnd_shaded = gnd_seg.shaded.list_ts_surfaces[0]
    surf_gnd_illum = gnd_seg.illum.list_ts_surfaces[0]
    np.testing.assert_allclose(surf_gnd_illum.get_param('direct') * 0.19,
                               surf_gnd_shaded.get_param('direct'))
    # Run check on pvrow surfaces
    surf_pvrow_shaded = (pvrow_front.list_segments[0]
                         .shaded.list_ts_surfaces[0])
    surf_pvrow_illum = (pvrow_front.list_segments[0]
                        .illum.list_ts_surfaces[0])
    np.testing.assert_allclose(surf_pvrow_illum.get_param('direct') * 0.19,
                               surf_pvrow_shaded.get_param('direct'))


def test_initialize_rho():
    """Make sure that rho is initialized correctly"""
    model = BaseModel()
    # rho values
    rho_scalar = 0.50
    rho_default = 0.01
    rho_calculated = 0.10
    # Should use scalar input
    rho_out = model.initialize_rho(rho_scalar, rho_calculated, rho_default)
    assert rho_out == rho_scalar
    # Should use calculated
    rho_out = model.initialize_rho(None, rho_calculated, rho_default)
    np.testing.assert_allclose(rho_out, rho_calculated)
    # Should use default
    rho_out = model.initialize_rho(None, None, rho_default)
    np.testing.assert_allclose(rho_out, rho_default)
