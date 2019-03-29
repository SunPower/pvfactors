import pytest
import numpy as np
from pvfactors import PVFactorsError
from pvfactors.geometry import BaseSide, ShadeCollection, PVSurface, PVSegment


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
