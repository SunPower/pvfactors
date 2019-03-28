import pytest
from pvfactors.pvsurface import PVSurface, PVSegment
from pvfactors.pvrow import PVRowSide, PVRow
from shapely.geometry import LineString


def test_pvsegment_setter():
    """Test that pv segment collection updates correctly"""
    seg = PVSegment()
    assert seg.length == 0
    new_surf_rigid = PVSurface([(0, 0), (1, 0)], shaded=False)
    seg.illum_surface = new_surf_rigid
    assert seg.length == 1
    new_surf_rigid = PVSurface([(1, 0), (2, 0)], shaded=True)
    seg.shaded_surface = new_surf_rigid
    assert seg.length == 2


def test_pvsegment_deleter():
    """Test that elements of pv segment collection get deleted
    correctly"""
    seg = PVSegment(PVSurface([(0, 0), (1, 0)], shaded=False),
                    PVSurface([(1, 0), (2, 0)], shaded=True))
    assert seg.length == 2
    del seg.shaded_surface
    assert seg.length == 1
    del seg.illum_surface
    assert seg.length == 0


def test_shaded_length():
    """Test that calculation of shaded length is correct"""
    seg_1 = PVSegment(
        illum_surface=PVSurface([(0, 0), (1, 0)], shaded=False))
    assert seg_1.shaded_length == 0
    seg_2 = PVSegment(
        shaded_surface=PVSurface([(1, 0), (2, 0)], shaded=True))
    assert seg_2.shaded_length == 1

    side = PVRowSide([seg_1, seg_2])
    assert side.shaded_length == 1


@pytest.fixture(scope='function')
def pvsegments():
    seg_1 = PVSegment(
        illum_surface=PVSurface([(0, 0), (1, 0)], shaded=False))
    seg_2 = PVSegment(
        shaded_surface=PVSurface([(1, 0), (2, 0)], shaded=True))
    yield seg_1, seg_2


@pytest.fixture(scope='function')
def pvrow_side(pvsegments):
    side = PVRowSide(pvsegments)
    yield side


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
