import pytest
import numpy as np
from pvfactors import PVFactorsError
from pvfactors.geometry import BaseSide, ShadeCollection, PVSurface, PVSegment
from shapely.geometry import LineString


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

    assert side.length == 2
    assert side.list_segments[0].length == 1
    assert side.list_segments[1].length == 1
    assert side.list_segments[0].shaded_length == 0.5
    assert side.list_segments[1].shaded_length == 0.5
