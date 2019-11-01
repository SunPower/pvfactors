from pvfactors.geometry.pvrow import PVRow
from shapely.geometry import LineString


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
