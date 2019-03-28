from pvfactors.pvsurface import PVSegment
from pvfactors.pvrow import PVRowSide, PVRow
from shapely.geometry import LineString


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


def test_shaded_length(shade_collections):
    """Test that calculation of shaded length is correct"""
    illum_col, shaded_col = shade_collections
    seg_1 = PVSegment(
        illum_collection=illum_col)
    assert seg_1.shaded_length == 0
    seg_2 = PVSegment(
        shaded_collection=shaded_col)
    assert seg_2.shaded_length == 1

    side = PVRowSide([seg_1, seg_2])
    assert side.shaded_length == 1


def test_pvrow(pvrow_side):
    """Test that can successfully create a PVRow object from 1 PVRow side,
    with a shaded pv surface"""
    pvrow = PVRow(front_side=pvrow_side)
    assert pvrow.length == 2
    assert pvrow.front.shaded_length == 1
    assert pvrow.back.length == 0

    # Check that can find out if intersection
    line = LineString([(1, 1), (-1, -1)])
    assert pvrow.intersects(line)
