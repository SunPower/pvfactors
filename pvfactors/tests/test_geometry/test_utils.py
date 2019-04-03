from pvfactors.geometry.utils import projection
from shapely.geometry import Point, LineString


def test_projection():
    """Make sure that projection method is working as expected"""
    pt_1 = Point(1, 1)
    vector = [0, 1]
    line_1 = LineString([(0, 0), (2, 0)])

    # projection is center
    expected_inter = Point(1, 0)
    inter = projection(pt_1, vector, line_1)
    assert expected_inter == inter

    # should be empty: way outside
    line_2 = LineString([(-1, 0), (0, 0)])
    inter = projection(pt_1, vector, line_2)
    assert inter.is_empty

    # Should be 2nd boundary
    line_3 = LineString([(0, 0), (1, 0)])
    inter = projection(pt_1, vector, line_3)
    assert inter == line_3.boundary[1]

    # Should be 1st boundary
    pt_2 = Point(0, 1)
    inter = projection(pt_2, vector, line_3)
    assert inter == line_3.boundary[0]

    # Should be 1st boundary: very close
    pt_3 = Point(0 + 1e-9, 1)
    inter = projection(pt_3, vector, line_3)
    assert inter == line_3.boundary[0]

    # Should be empty: very close
    pt_4 = Point(0 - 1e-9, 1)
    inter = projection(pt_4, vector, line_3)
    assert inter.is_empty
