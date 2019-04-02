from pvfactors.geometry.utils import projection
from shapely.geometry import Point, LineString


def test_projection():
    """Make sure that projection method is working"""
    pt = Point(1, 1)
    vector = [0, 1]
    line_1 = LineString([(0, 0), (2, 0)])
    expected_inter = Point(1, 0)
    inter = projection(pt, vector, line_1)
    assert expected_inter == inter

    line_2 = LineString([(-1, 0), (0, 0)])
    inter = projection(pt, vector, line_2)
    assert inter.is_empty
