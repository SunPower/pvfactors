import pytest
import numpy as np
from pvfactors import PVFactorsError
from pvfactors.geometry.base import \
    BaseSide, ShadeCollection, PVSurface, PVSegment, \
    _coords_from_center_tilt_length, _get_solar_2d_vectors


def test_baseside_normal_vector(pvsegments):
    side = BaseSide(pvsegments)
    np.testing.assert_array_equal(side.n_vector, [0, 1])


def test_baseside_shaded_length(pvsegments):
    side = BaseSide(pvsegments)
    assert side.shaded_length == 1.


def test_shade_collection():
    """Check that implementation of shade collection works"""
    surf_illum_1 = PVSurface([(0, 0), (1, 0)], shaded=False)
    surf_illum_2 = PVSurface([(0, 0), (0, 1)], shaded=False)
    surf_shaded = PVSurface([(1, 0), (2, 0)], shaded=True)

    col = ShadeCollection([surf_illum_1, surf_illum_2])

    assert not col.shaded
    assert col.length == 2

    with pytest.raises(PVFactorsError) as err:
        ShadeCollection([surf_illum_1, surf_shaded])

    assert str(err.value) \
        == 'All elements should have same shading'

    with pytest.raises(PVFactorsError) as err:
        col = ShadeCollection([surf_illum_1, surf_illum_2])
        col.n_vector

    assert str(err.value) \
        == "Cannot request n_vector if all elements not collinear"


def test_segment_shaded_length(shade_collections):
    """Test that calculation of shaded length is correct"""
    illum_col, shaded_col = shade_collections
    seg_1 = PVSegment(
        illum_collection=illum_col)
    assert seg_1.shaded_length == 0
    seg_2 = PVSegment(
        shaded_collection=shaded_col)
    assert seg_2.shaded_length == 1


def test_coords_from_center_tilt_length_float():
    """Test that can calculate PV row coords from inputs as scalars"""

    # Float inputs
    xy_center = (0, 0)
    length = 2.
    axis_azimuth = 0.
    tilt = 10.
    surface_azimuth = 90.

    coords = _coords_from_center_tilt_length(xy_center, tilt, length,
                                             surface_azimuth, axis_azimuth)

    expected_coords = [(-0.984807753012208, 0.17364817766693028),
                       (0.984807753012208, -0.17364817766693033)]

    np.testing.assert_almost_equal(coords, expected_coords)


def test_coords_from_center_tilt_length_vec():
    """Test that can calculate PV row coords from angle inputs as vectors"""

    # Float inputs
    xy_center = (0, 0)
    length = 2.
    axis_azimuth = 0.

    # Vector inputs
    tilt = np.array([10, 45])
    surface_azimuth = np.array([90, 270])

    coords = _coords_from_center_tilt_length(xy_center, tilt, length,
                                             surface_azimuth, axis_azimuth)

    expected_coords = [
        ([-0.98480775, -0.70710678], [0.17364818, -0.70710678]),
        ([0.98480775, 0.70710678], [-0.17364818, 0.70710678])]

    np.testing.assert_almost_equal(coords, expected_coords)


def test_solar_2d_vectors():
    """Test that can calculate solar vector with inputs as arrays"""
    # Prepare inputs
    solar_zenith = np.array([20., 45.])
    solar_azimuth = np.array([70., 200.])
    axis_azimuth = 0.

    # Calculate solar vectors for the 2 times
    solar_vectors = _get_solar_2d_vectors(solar_zenith, solar_azimuth,
                                          axis_azimuth)

    expected_solar_vectors = [[0.3213938, -0.24184476],
                              [0.93969262, 0.70710678]]

    np.testing.assert_almost_equal(solar_vectors, expected_solar_vectors)
