import pytest
import numpy as np
from pvfactors import PVFactorsError
from pvfactors.geometry.base import \
    BaseSide, ShadeCollection, PVSurface, PVSegment, \
    _coords_from_center_tilt_length, _get_solar_2d_vectors
from shapely.geometry import LineString, Point
from pvfactors.geometry.utils import projection


def test_baseside(pvsegments):
    """Test that the basic BaseSide functionalities work"""

    side = BaseSide(pvsegments)

    np.testing.assert_array_equal(side.n_vector, [0, 1])
    assert side.shaded_length == 1.


def test_shade_collection():
    """Check that implementation of shade collection works"""
    surf_illum_1 = PVSurface([(0, 0), (1, 0)], shaded=False)
    surf_illum_2 = PVSurface([(0, 0), (0, 1)], shaded=False)
    surf_illum_3 = PVSurface([(1, 0), (2, 0)], shaded=True)

    col = ShadeCollection([surf_illum_1, surf_illum_2])

    assert not col.shaded
    assert col.length == 2

    with pytest.raises(PVFactorsError) as err:
        ShadeCollection([surf_illum_1, surf_illum_3])

    assert str(err.value) \
        == 'All elements should have same shading'

    with pytest.raises(PVFactorsError) as err:
        col = ShadeCollection([surf_illum_1, surf_illum_2])
        col.n_vector

    assert str(err.value) \
        == "Cannot request n_vector if all elements not collinear"


def test_pvsegment_setter(shade_collections):
    """Test that pv segment collection updates correctly"""
    illum_col, shaded_col = shade_collections
    seg = PVSegment()
    assert seg.length == 0
    seg.illum_collection = illum_col
    assert seg.length == 1
    seg.shaded_collection = shaded_col
    assert seg.length == 2


def test_pvsegment_deleter(shade_collections):
    """Test that elements of pv segment collection get deleted
    correctly"""
    seg = PVSegment(*shade_collections)
    assert seg.length == 2
    del seg.shaded_collection
    assert seg.length == 1
    del seg.illum_collection
    assert seg.length == 0


def test_segment_shaded_length(shade_collections):
    """Test that calculation of shaded length is correct"""
    illum_col, shaded_col = shade_collections
    seg_1 = PVSegment(
        illum_collection=illum_col)
    assert seg_1.shaded_length == 0
    seg_2 = PVSegment(
        shaded_collection=shaded_col)
    assert seg_2.shaded_length == 1


def test_cast_shadow_segment():
    """Test shadow casting on PVSegment"""
    seg = PVSegment.from_linestring_coords([(0, 0), (2, 0)], shaded=False,
                                           index=0)
    shadow = LineString([(0.5, 0), (1.5, 0)])
    seg.cast_shadow(shadow)

    assert seg.length == 2
    assert seg.shaded_length == 1
    assert seg.index == 0


def test_remove_linestring_shadedcollection():
    """Check removing linestring element from shade collection"""
    # multi-part geometry difference
    coords = [(0, 0), (2, 0)]
    shaded = False
    collection_1 = ShadeCollection.from_linestring_coords(coords, shaded)
    line = LineString([(0.5, 0), (1.5, 0)])
    collection_1.remove_linestring(line)

    assert collection_1.length == 1
    assert not collection_1.shaded
    assert collection_1.is_collinear

    # single geometry difference
    coords = [(0, 0), (2, 0)]
    shaded = True
    collection_2 = ShadeCollection.from_linestring_coords(coords, shaded)
    line = LineString([(0, 0), (1, 0)])
    collection_2.remove_linestring(line)

    assert collection_2.length == 1
    assert collection_2.shaded
    assert collection_2.is_collinear


def test_add_pvsurface_shadecollection():
    """Check adding linestring element to shade collection"""
    collection = ShadeCollection(shaded=True)
    coords = [(0.5, 0), (1.5, 0)]
    surface = PVSurface(coords, shaded=True)
    collection.add_pvsurface(surface)

    assert collection.length == 1
    assert collection.shaded
    assert collection.is_collinear


def test_cast_shadow_side():
    """Cast shadow on side with 2 segments"""
    coords = [(0, 0), (2, 0)]
    side = BaseSide.from_linestring_coords(coords, shaded=False, index=0,
                                           n_segments=2)
    assert side.list_segments[0].length == 1
    assert side.list_segments[1].length == 1
    assert side.list_segments[0].shaded_length == 0
    assert side.list_segments[1].shaded_length == 0

    # Cast shadow
    shadow = LineString([(0.5, 0), (1.5, 0)])
    side.cast_shadow(shadow)

    np.testing.assert_almost_equal(side.length, 2)
    np.testing.assert_almost_equal(side.list_segments[0].length, 1)
    np.testing.assert_almost_equal(side.list_segments[1].length, 1)
    np.testing.assert_almost_equal(side.list_segments[0].shaded_length, 0.5)
    np.testing.assert_almost_equal(side.list_segments[1].shaded_length, 0.5)


def test_pvsurface_difference_precision_error():
    """This would lead to wrong result using shapely ``difference`` method"""

    surf_1 = PVSurface([(0, 0), (3, 2)])
    surf_2 = PVSurface([surf_1.interpolate(1), Point(6, 4)])
    diff = surf_1.difference(surf_2)
    assert diff == LineString([(0, 0),
                               (0.8320502943378437, 0.5547001962252291)])


def test_cast_shadow_segment_precision_error():
    """Test shadow casting on PVSegment when using inexact projection.
    In shapely, need to use ``buffer`` method for the intersection to be
    calculated correctly
    """
    coords = [(0, 0), (3, 2)]
    seg = PVSegment.from_linestring_coords(coords, shaded=False,
                                           index=0)
    # Project point on line
    pt = Point(0, 2)
    line = seg.illum_collection.list_surfaces[0]  # LineString(coords)
    vector = [1, -1]
    proj = projection(pt, vector, line)
    # Use result of projection to create shadow
    shadow = LineString([Point(0, 0), proj])
    seg.cast_shadow(shadow)

    np.testing.assert_almost_equal(seg.length, 3.60555127546)
    np.testing.assert_almost_equal(seg.shaded_collection.length, 1.44222051019)
    assert len(seg.shaded_collection.list_surfaces) == 1
    assert len(seg.illum_collection.list_surfaces) == 1
    np.testing.assert_almost_equal(
        seg.shaded_collection.list_surfaces[0].length, 1.44222051019)
    np.testing.assert_almost_equal(
        seg.illum_collection.list_surfaces[0].length, 2.1633307652783933)


def test_merge_shadecollection():
    """Test shadecollection merger feature: all contained pv surfaces should be
    merged"""
    surf_1 = PVSurface([(0, 0), (1, 0)], shaded=True)
    surf_2 = PVSurface([(1.1, 0), (2, 0)], shaded=True)
    col = ShadeCollection(list_surfaces=[surf_1, surf_2])
    col.merge_surfaces()

    assert col.length == 2
    assert len(col.list_surfaces) == 1
    assert col.shaded
    assert len(col.list_surfaces[0].coords) == 2


def test_merge_shaded_areas_side():
    """All segments should have merge shaded collections"""
    coords = [(0, 0), (2, 0)]
    side = BaseSide.from_linestring_coords(coords, shaded=False,
                                           n_segments=2)
    shadow_1 = LineString([(0.5, 0), (0.75, 0)])
    shadow_2 = LineString([(0.75, 0), (1.5, 0)])
    side.cast_shadow(shadow_1)
    side.cast_shadow(shadow_2)

    segments = side.list_segments
    assert len(segments[0].shaded_collection.list_surfaces) == 2
    assert len(segments[1].shaded_collection.list_surfaces) == 1
    np.testing.assert_almost_equal(side.shaded_length, 1)
    np.testing.assert_almost_equal(segments[0].shaded_length, 0.5)
    np.testing.assert_almost_equal(segments[1].shaded_length, 0.5)

    # Merge shaded areas
    side.merge_shaded_areas()

    assert len(segments[0].shaded_collection.list_surfaces) == 1
    assert len(segments[1].shaded_collection.list_surfaces) == 1
    np.testing.assert_almost_equal(side.shaded_length, 1)
    np.testing.assert_almost_equal(segments[0].shaded_length, 0.5)
    np.testing.assert_almost_equal(segments[1].shaded_length, 0.5)


def test_shadecol_cut_at_point():
    """Test that shade collection correctly cut surfaces at point"""
    surf_1 = PVSurface([(0, 0), (1, 1)], shaded=True)
    surf_2 = PVSurface([(1, 1), (2, 2)], shaded=True)
    col = ShadeCollection(list_surfaces=[surf_1, surf_2])
    length = col.length

    # Hitting boundary, should not cut
    point = Point(1, 1)
    col.cut_at_point(point)
    assert len(col.list_surfaces) == 2
    assert col.length == length

    # Not contained should not cut
    point = Point(1, 2)
    col.cut_at_point(point)
    assert len(col.list_surfaces) == 2
    assert col.length == length

    # Should cut
    point = Point(0.5, 0.5)
    col.cut_at_point(point)
    assert len(col.list_surfaces) == 3
    assert col.list_surfaces[0].length == length / 4
    assert col.list_surfaces[-1].length == length / 4
    assert col.list_surfaces[1].length == length / 2
    assert col.length == length


def test_side_cut_at_point():
    """Test that can cut side at point correctly"""
    coords = [(0, 0), (2, 0)]
    side = BaseSide.from_linestring_coords(coords, shaded=False,
                                           n_segments=2)

    # should cut
    point = Point(0.5, 0)
    side.cut_at_point(point)
    assert side.length == 2
    assert len(side.list_segments[0].illum_collection.list_surfaces) == 2
    assert len(side.list_segments[1].illum_collection.list_surfaces) == 1


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
