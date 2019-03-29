from pvfactors.pvsurface import PVSegment


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
